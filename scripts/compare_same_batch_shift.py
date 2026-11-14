"""同批更新路径 GPU 回归：merge / max_rows 变化时位移三项应一致。

单元测试（tests/test_same_batch_update_path.py）在 CPU float32 上直接比累积梯度；
本脚本在真实 vLLM 配置上跑一轮，把位移与峰值显存写到 outputs/，供改动更新路径后手工核对。

用法（WSL，已起 vLLM）：
  PYTHONPATH=. python -m scripts.compare_same_batch_shift
  PYTHONPATH=. python -m scripts.compare_same_batch_shift --config configs/guess_number_vllm.toml
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


def shift_triple(shift):
    return (
        shift["shift_median_absolute_log_ratio"],
        shift["shift_percentile_95_absolute_log_ratio"],
        shift["shift_clip_fraction"],
    )


def format_triple(values):
    median, percentile_95, clip = values
    return f"{median:.3e} / {percentile_95:.3f} / {clip:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/guess_number_vllm.toml")
    parser.add_argument("--output", default="outputs/same_batch_shift_compare.json")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()

    overrides = list(args.overrides)
    if not any(item.startswith("training.run_name=") for item in overrides):
        overrides.append("training.run_name=same-batch-shift-compare")
    configuration = load_configuration(args.config, overrides)
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
    print(f"kept={len(kept)}  steps={sum(len(trajectory.steps) for trajectory in kept)}")

    trials = [
        ("merge_on_rows8", True, 8),
        ("merge_off_rows8", False, 8),
        ("merge_on_rows4", True, 4),
        ("merge_on_rows32", True, 32),
    ]
    results = {}
    try:
        for name, merge, max_rows in trials:
            restore()
            trial_config = dataclasses.replace(
                config, merge_append_only_turns=merge, max_rows_per_minibatch=max_rows)
            optimizer = torch.optim.AdamW(list(trainable.values()), lr=trial_config.learning_rate)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            history, batches, old_log_probabilities, gap_metrics, merge_fraction, _peak, _over = update_policy(
                model, optimizer, kept, padding_token_id, trial_config)
            shift = measure_policy_shift(model, batches, old_log_probabilities, trial_config)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
            else:
                peak = 0.0
            triple = shift_triple(shift)
            results[name] = {
                "first_step_max_ratio_deviation": history[0]["max_ratio_deviation"],
                "merge_fraction": merge_fraction,
                "max_rows_per_minibatch": max_rows,
                "merge_append_only_turns": merge,
                "shift_median_absolute_log_ratio": triple[0],
                "shift_percentile_95_absolute_log_ratio": triple[1],
                "shift_clip_fraction": triple[2],
                "peak_memory_gigabytes": peak,
                "minibatch_count": len(batches),
                **gap_metrics,
            }
            del optimizer
            print(f"{name}: {format_triple(triple)}  peak={peak:.2f}GB  batches={len(batches)}")
    finally:
        restore()
        if hasattr(backend, "close"):
            backend.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")

    reference = shift_triple(results["merge_on_rows8"])
    print(f"reference merge_on_rows8: {format_triple(reference)}")
    for name in ("merge_off_rows8", "merge_on_rows4", "merge_on_rows32"):
        if name not in results:
            continue
        other = shift_triple(results[name])
        deltas = [abs(a - b) for a, b in zip(reference, other)]
        print(
            f"vs {name}: {format_triple(other)}  "
            f"abs_delta={deltas[0]:.3e}/{deltas[1]:.3e}/{deltas[2]:.3e}"
        )


if __name__ == "__main__":
    main()
