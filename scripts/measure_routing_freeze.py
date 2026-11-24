"""updates_per_iteration 为 2 时，比较第二步在 routing_freeze 开和关时的差别。

同一批采样、同一份更新前的权重，各做一次学习率下的更新，然后看第二步前向：
裁剪比例，以及回答 token 上 |对数概率比| 的 95 分位。只测量，不保存检查点。
学习率用配置里的 learning_rate，不走预热。每轮测完都把权重放回起点。

用法（在仓库根目录，需要 GPU，模型须是 MoE，后端须是本地后端）：

    python scripts/measure_routing_freeze.py --config configs/arithmetic_olmoe_d2d3.toml

    python scripts/measure_routing_freeze.py --config configs/arithmetic_olmoe_d2d3.toml --rounds 3 --output outputs/measure_routing_freeze.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import torch

from grpx.advantage import compute_advantages
from grpx.command_line import (
    build_environment_factory,
    build_model_and_backend,
    build_training_config,
    call_with_options,
    import_object,
    load_configuration,
)
from grpx.loop import (
    replay_batch,
    accumulate_gradients,
    collect_trajectories,
    consume_old_policy_forward,
    split_into_row_chunks,
)
from grpx.routing_diagnostics import RoutingRecorder, RoutingSnapshot, summarize_change
from grpx.trainer import action_log_probabilities, build_batch


def trainable_checksum(model) -> float:
    total = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                total += float(parameter.detach().float().abs().sum())
    return total


def snapshot_trainable(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable(model, saved: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in saved:
                parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))
            if parameter.grad is not None:
                parameter.grad = None


def summarize_shift(log_ratio: torch.Tensor, clip_range: float) -> dict:
    if log_ratio.numel() == 0:
        return {"n_tokens": 0}
    absolute = log_ratio.abs()
    clipped = (log_ratio.exp() - 1).abs() > clip_range
    return {
        "n_tokens": int(log_ratio.numel()),
        "clip_fraction": float(clipped.float().mean()),
        "p95_absolute_log_ratio": float(torch.quantile(absolute, 0.95)),
        "median_absolute_log_ratio": float(absolute.median()),
        "max_absolute_log_ratio": float(absolute.max()),
    }


def measure_condition(model, kept, pad_token_id, config, freeze: bool) -> dict:
    """从当前权重出发更新一步，汇总第二步前向的位移。调用方负责在之后把权重放回去。"""
    condition = replace(
        config, routing_freeze=freeze, routing_replay=False, updates_per_iteration=2)
    chunks = split_into_row_chunks(
        kept, condition.max_rows_per_minibatch, condition.max_tokens_per_minibatch,
        merge_append_only=False)
    if not chunks:
        return {"kept_trajectories": len(kept), "minibatches": 0}
    batches = [
        build_batch(chunk, pad_token_id, model.device, condition.row_weighting)
        for chunk in chunks
    ]
    recorder = RoutingRecorder(model)
    if not recorder.is_available:
        raise RuntimeError("routing_freeze 的测量只适用于带路由的模型")
    before = RoutingSnapshot()
    old_log_probabilities = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            log_probability, statistics, frozen = consume_old_policy_forward(
                model, batch, recorder, before, condition)
            old_log_probabilities.append(log_probability)
            if freeze:
                if frozen is None or batch.get("routed_experts") is None:
                    raise RuntimeError("routing_freeze 打开后，更新前的前向没有记下专家")
                if statistics is not None:
                    raise RuntimeError("固定路由的那一次前向不应记成重放统计")
            elif frozen is not None or batch.get("routed_experts") is not None:
                raise RuntimeError("routing_freeze 关闭时不应写下专家记录")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=condition.learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    first_step = accumulate_gradients(model, batches, old_log_probabilities, condition)
    torch.nn.utils.clip_grad_norm_(trainable, condition.max_gradient_norm)
    optimizer.step()
    checksum = trainable_checksum(model)

    pieces = []
    after = RoutingSnapshot()
    model.eval()
    with torch.no_grad():
        for batch, old_log_probability in zip(batches, old_log_probabilities, strict=True):
            with replay_batch(model, batch):
                new_log_probability, mask = action_log_probabilities(model, batch)
            pieces.append((new_log_probability - old_log_probability)[mask].float().cpu())
        # 路由变化看模型自己的选择。第二步的对数概率可能在重放，那个钩子会换掉专家编号，不能拿来做诊断。
        for batch in batches:
            with recorder.record(batch["attention_mask"], snapshot=after):
                action_log_probabilities(model, batch)
    shift = summarize_shift(torch.cat(pieces), condition.clip_range)
    shift["first_step_clip_fraction"] = first_step["clip_fraction"]
    shift["first_step_p95_absolute_log_ratio"] = first_step["percentile_95_absolute_log_ratio"]
    shift["trainable_checksum_after_first_step"] = checksum
    shift["routing_change"] = summarize_change(before, after)
    shift["kept_trajectories"] = len(kept)
    shift["minibatches"] = len(batches)
    return shift


def pooled(parts: list[dict], key: str) -> dict | None:
    present = [part[key] for part in parts if part.get(key, {}).get("n_tokens")]
    if not present:
        return None
    # 各轮的分位数不能再平均。调用方另存了逐 token 时才池化；这里只报告各轮范围。
    clips = [part["clip_fraction"] for part in present]
    tails = [part["p95_absolute_log_ratio"] for part in present]
    tokens = sum(part["n_tokens"] for part in present)
    return {
        "rounds": len(present),
        "n_tokens": tokens,
        "clip_fraction_min": min(clips),
        "clip_fraction_max": max(clips),
        "p95_absolute_log_ratio_min": min(tails),
        "p95_absolute_log_ratio_max": max(tails),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="训练配置。后端须是本地后端，模型须是 MoE")
    parser.add_argument("--rounds", type=int, default=3, help="从同一份初始权重出发的采样轮数。默认 3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/measure_routing_freeze.json"),
        help="结果 JSON 的路径",
    )
    arguments = parser.parse_args()
    if arguments.rounds < 1:
        raise SystemExit("--rounds 至少为 1")

    configuration = load_configuration(arguments.config, [])
    training = build_training_config(configuration)
    if training.routing_replay:
        raise SystemExit("这个测量要关着 routing_replay。打开时以采样记录为准，routing_freeze 不生效")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(training.memory_fraction)
    model, tokenizer, backend = build_model_and_backend(configuration, training.seed)
    if not hasattr(backend, "record_routing"):
        raise SystemExit("这个测量只支持本地后端")
    model.grpx_output_logit_chunk_size = training.output_logit_chunk_size
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    dataset = call_with_options(
        import_object(configuration["task"]["dataset"]),
        configuration["task"].get("train_dataset_options", {}),
        "dataset",
    )
    environment_factory = build_environment_factory(configuration["task"])
    rng = random.Random(training.seed)
    initial = snapshot_trainable(model)
    rounds = []
    started = time.perf_counter()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(arguments.rounds):
        print(f"=== round {iteration} sample ===", flush=True)
        sample_started = time.perf_counter()
        samples = rng.sample(dataset, training.groups_per_iteration)
        trajectories = collect_trajectories(
            backend, tokenizer, environment_factory, samples, training, iteration)
        kept = compute_advantages(
            trajectories,
            normalize_std=training.normalize_std,
            step_advantage_weight=training.step_advantage_weight,
            discount_factor=training.discount,
        )
        print(
            f"sampled {len(trajectories)} answers, kept {len(kept)} "
            f"in {time.perf_counter() - sample_started:.1f}s",
            flush=True,
        )
        outcome = {"iteration": iteration, "sampled": len(trajectories), "kept": len(kept)}
        if kept:
            # 偶数轮先关后开，奇数轮相反，避免固定的测量顺序带来偏差
            order = (("off", False), ("on", True)) if iteration % 2 == 0 else (("on", True), ("off", False))
            outcome["order"] = [name for name, _freeze in order]
            for name, freeze in order:
                print(f"=== round {iteration} freeze {name} ===", flush=True)
                restore_trainable(model, initial)
                outcome[name] = measure_condition(model, kept, pad_token_id, training, freeze)
                restore_trainable(model, initial)
        outcome["seconds"] = time.perf_counter() - sample_started
        rounds.append(outcome)
        payload = {
            "config": arguments.config,
            "model": configuration["model"]["name"],
            "learning_rate": training.learning_rate,
            "clip_range": training.clip_range,
            "updates_per_iteration": 2,
            "note": "每轮从初始权重出发，更新一步后再看第二步。学习率不预热。",
            "rounds": rounds,
        }
        arguments.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        brief = {
            "iteration": iteration,
            "kept": len(kept),
        }
        if kept:
            brief["off"] = {
                "clip_fraction": outcome["off"]["clip_fraction"],
                "p95": outcome["off"]["p95_absolute_log_ratio"],
                "routing_changed_mean": outcome["off"]["routing_change"][
                    "routing_changed_position_fraction_mean"],
            }
            brief["on"] = {
                "clip_fraction": outcome["on"]["clip_fraction"],
                "p95": outcome["on"]["p95_absolute_log_ratio"],
                "routing_changed_mean": outcome["on"]["routing_change"][
                    "routing_changed_position_fraction_mean"],
            }
            brief["checksum_match"] = (
                outcome["off"]["trainable_checksum_after_first_step"]
                == outcome["on"]["trainable_checksum_after_first_step"]
            )
        print(json.dumps(brief, ensure_ascii=False), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload["off_across_rounds"] = pooled(rounds, "off")
    payload["on_across_rounds"] = pooled(rounds, "on")
    payload["seconds"] = time.perf_counter() - started
    arguments.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", arguments.output, flush=True)


if __name__ == "__main__":
    main()
