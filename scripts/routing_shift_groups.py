"""lr=1e-5 试更新：按回答 token 的路由是否变化分组，比较位移。

路由快照是所有有效位置；位移只在回答位置。分组时用「预测该回答 token 的
隐状态位置」上的路由变化（与 action_log_probabilities 的错位一致）。

用法：
  PYTHONPATH=. python -m scripts.routing_shift_groups
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import torch

from grpx.advantage import compute_advantages
from grpx.command_line import build_components, load_configuration
from grpx.loop import collect_trajectories, update_policy
from grpx.routing_diagnostics import RoutingSnapshot, changed_in_any_layer
from grpx.trainer import action_log_probabilities


def valid_position_index_map(attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, L) → 同形 long，有效位置为在 flatten(valid) 中的下标，其余 -1。"""
    index_map = torch.full(attention_mask.shape, -1, dtype=torch.long)
    flat_index = 0
    for row in range(attention_mask.shape[0]):
        for column in range(attention_mask.shape[1]):
            if attention_mask[row, column]:
                index_map[row, column] = flat_index
                flat_index += 1
    return index_map


def slice_snapshot(snapshot: RoutingSnapshot, batch_index: int) -> RoutingSnapshot:
    return RoutingSnapshot(
        selected_by_layer={
            name: [pieces[batch_index]]
            for name, pieces in snapshot.selected_by_layer.items()
        }
    )


def answer_route_changed(before: RoutingSnapshot, after: RoutingSnapshot, batch) -> torch.Tensor:
    """与 (new-old)[action_mask[:, 1:]] 逐元素对齐的布尔向量。"""
    changed_valid = changed_in_any_layer(before, after)
    attention_mask = batch["attention_mask"].detach().cpu()
    action_mask = batch["action_mask"].detach().cpu()
    index_map = valid_position_index_map(attention_mask)
    logit_action = action_mask[:, 1:]
    flags = []
    for row in range(logit_action.shape[0]):
        for column in range(logit_action.shape[1]):
            if not logit_action[row, column]:
                continue
            routing_index = index_map[row, column]
            if routing_index < 0:
                raise RuntimeError(
                    f"回答位置 ({row}, {column + 1}) 的隐状态位 ({row}, {column}) "
                    "不在 attention_mask 有效集里"
                )
            flags.append(changed_valid[routing_index])
    if not flags:
        return torch.zeros(0, dtype=torch.bool)
    return torch.stack(flags)


def summarize_group(absolute_log_ratio: torch.Tensor, log_ratio: torch.Tensor, clip_range: float):
    if absolute_log_ratio.numel() == 0:
        return {
            "token_count": 0,
            "shift_median_absolute_log_ratio": float("nan"),
            "shift_percentile_95_absolute_log_ratio": float("nan"),
            "shift_clip_fraction": float("nan"),
        }
    return {
        "token_count": int(absolute_log_ratio.numel()),
        "shift_median_absolute_log_ratio": absolute_log_ratio.median().item(),
        "shift_percentile_95_absolute_log_ratio": torch.quantile(absolute_log_ratio, 0.95).item(),
        "shift_clip_fraction": ((log_ratio.exp() - 1).abs() > clip_range).float().mean().item(),
    }


def main():
    configuration = load_configuration(
        "configs/arithmetic_olmoe.toml",
        ["training.run_name=routing-shift-groups"],
    )
    (model, tokenizer, backend, train_dataset, environment_factory,
     config, _evaluation_dataset, _run_directory) = build_components(configuration)

    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = min(backend.end_token_ids)

    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    snapshot = {name: parameter.detach().clone() for name, parameter in trainable.items()}

    def restore():
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(snapshot[name])

    torch.manual_seed(config.seed)
    samples = random.Random(config.seed).sample(train_dataset, config.groups_per_iteration)
    trajectories = collect_trajectories(
        backend, tokenizer, environment_factory, samples, config, iteration=0)
    kept = compute_advantages(
        trajectories, normalize_std=config.normalize_std,
        step_advantage_weight=config.step_advantage_weight,
        discount_factor=config.discount,
    )
    print(f"kept={len(kept)} steps={sum(len(trajectory.steps) for trajectory in kept)}", flush=True)
    if not kept:
        raise SystemExit("没有有信号的组")

    learning_rate = 1e-5
    trial_config = dataclasses.replace(config, learning_rate=learning_rate)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=learning_rate)
    (history, batches, old_log_probabilities, gap_metrics, _merge, _peak, _over,
     routing_before) = update_policy(
        model, optimizer, kept, padding_token_id, trial_config, record_routing=True)
    if routing_before is None:
        raise SystemExit("没有路由快照")

    from grpx.routing_diagnostics import RoutingRecorder, RoutingSnapshot, summarize_routing

    routing_after = RoutingSnapshot()
    recorder = RoutingRecorder(model)
    log_ratios = []
    route_changed_flags = []
    model.eval()
    with torch.no_grad():
        for batch_index, (batch, old_log_probability) in enumerate(
                zip(batches, old_log_probabilities, strict=True)):
            with recorder.record(batch["attention_mask"], snapshot=routing_after):
                new_log_probability, mask = action_log_probabilities(model, batch)
            log_ratios.append((new_log_probability - old_log_probability)[mask].detach().cpu())
            before_batch = slice_snapshot(routing_before, batch_index)
            after_batch = slice_snapshot(routing_after, batch_index)
            route_changed_flags.append(answer_route_changed(before_batch, after_batch, batch))

    log_ratio = torch.cat(log_ratios).float()
    absolute = log_ratio.abs()
    changed = torch.cat(route_changed_flags)
    if changed.numel() != log_ratio.numel():
        raise RuntimeError(
            f"对齐失败：路由标记 {changed.numel()}，位移 {log_ratio.numel()}"
        )

    overall = summarize_group(absolute, log_ratio, trial_config.clip_range)
    overall.update(summarize_routing(routing_before, routing_after))
    changed_group = summarize_group(absolute[changed], log_ratio[changed], trial_config.clip_range)
    unchanged_group = summarize_group(absolute[~changed], log_ratio[~changed], trial_config.clip_range)

    results = {
        "learning_rate": learning_rate,
        "digits": configuration["task"]["train_dataset_options"].get("digits"),
        "clip_range": trial_config.clip_range,
        "first_step_max_ratio_deviation": history[0]["max_ratio_deviation"],
        "answer_token_count": int(log_ratio.numel()),
        "route_changed_answer_fraction": changed.float().mean().item(),
        "overall": overall,
        "route_changed": changed_group,
        "route_unchanged": unchanged_group,
        **gap_metrics,
    }

    restore()
    del optimizer
    if hasattr(backend, "close"):
        backend.close()

    output = Path("outputs/routing_shift_groups.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    def line(name, group):
        return (
            f"{name}: n={group['token_count']}  "
            f"median={group['shift_median_absolute_log_ratio']:.6g}  "
            f"p95={group['shift_percentile_95_absolute_log_ratio']:.6g}  "
            f"clip={group['shift_clip_fraction']:.6g}"
        )

    print(f"\nanswer tokens={results['answer_token_count']}  "
          f"route_changed_fraction={results['route_changed_answer_fraction']:.4f}", flush=True)
    print(line("overall   ", overall), flush=True)
    print(line("changed   ", changed_group), flush=True)
    print(line("unchanged ", unchanged_group), flush=True)
    print(f"\nwrote {output}", flush=True)


if __name__ == "__main__":
    main()
