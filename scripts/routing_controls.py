"""OLMoE 路由对照：同一批试更新路径上测三次。

1. 同一组权重算两次（更新后还原再测位移）→ routing_changed_* 应为 0
2. 学习率 1e-8
3. 正常学习率（配置里的，默认 1e-5）

用法（WSL）：
  PYTHONPATH=. python -m scripts.routing_controls --config configs/arithmetic_olmoe.toml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

import torch

from grpx.advantage import compute_advantages
from grpx.command_line import build_components, load_configuration
from grpx.loop import collect_trajectories, measure_policy_shift, update_policy

ROUTING_KEYS = [
    "routing_changed_position_fraction_mean",
    "routing_changed_position_fraction_maximum",
    "routing_replaced_expert_fraction_mean",
    "routing_changed_in_any_layer_fraction",
    "routing_load_entropy_mean",
    "routing_load_entropy_minimum",
    "routing_load_maximum_ratio",
    "routing_unused_expert_fraction",
]

SHIFT_KEYS = [
    "shift_median_absolute_log_ratio",
    "shift_percentile_95_absolute_log_ratio",
    "shift_clip_fraction",
]


def format_metrics(metrics):
    lines = []
    for key in SHIFT_KEYS + ROUTING_KEYS:
        if key in metrics:
            value = metrics[key]
            lines.append(f"  {key}: {value:.6g}" if isinstance(value, float) else f"  {key}: {value}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/arithmetic_olmoe.toml")
    parser.add_argument("--output", default="outputs/routing_controls.json")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    overrides = list(args.overrides)
    if not any(item.startswith("training.run_name=") for item in overrides):
        overrides.append("training.run_name=routing-controls")
    configuration = load_configuration(args.config, overrides)
    (model, tokenizer, backend, train_dataset, environment_factory,
     config, _evaluation_dataset, _run_directory) = build_components(configuration)

    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = min(backend.end_token_ids)

    from grpx.routing_diagnostics import RoutingRecorder
    recorder = RoutingRecorder(model)
    print(
        f"routing available={recorder.is_available} "
        f"gates={len(recorder.router_modules)} "
        f"experts_per_token={recorder.experts_per_token}"
    )
    if not recorder.is_available:
        raise SystemExit("模型上没找到路由模块，无法做对照")

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
    print(f"kept={len(kept)}  steps={sum(len(trajectory.steps) for trajectory in kept)}")
    if not kept:
        raise SystemExit("没有有信号的组，无法试更新")

    baseline_lr = config.learning_rate
    trials = [
        ("same_weights", baseline_lr, True),
        ("lr_1e-8", 1e-8, False),
        ("lr_baseline", baseline_lr, False),
    ]
    results = {}
    try:
        for name, learning_rate, restore_before_shift in trials:
            restore()
            trial_config = dataclasses.replace(config, learning_rate=learning_rate)
            optimizer = torch.optim.AdamW(list(trainable.values()), lr=learning_rate)
            (history, batches, old_log_probabilities, gap_metrics, merge_fraction,
             _peak, _over, routing_before) = update_policy(
                model, optimizer, kept, padding_token_id, trial_config, record_routing=True)
            if not history:
                raise SystemExit(f"{name}: 更新被跳过")
            if restore_before_shift:
                restore()
            shift = measure_policy_shift(
                model, batches, old_log_probabilities, trial_config,
                routing_before=routing_before)
            payload = {
                "learning_rate": learning_rate,
                "restore_before_shift": restore_before_shift,
                "first_step_max_ratio_deviation": history[0]["max_ratio_deviation"],
                "merge_fraction": merge_fraction,
                "minibatch_count": len(batches),
                **{key: shift[key] for key in SHIFT_KEYS},
                **{key: shift[key] for key in ROUTING_KEYS if key in shift},
                **gap_metrics,
            }
            results[name] = payload
            del optimizer
            print(f"\n=== {name} (lr={learning_rate:g}, restore={restore_before_shift}) ===")
            print(format_metrics(payload))
    finally:
        restore()
        if hasattr(backend, "close"):
            backend.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
