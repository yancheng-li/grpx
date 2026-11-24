"""对同一批采样 token 各算一遍不重放和重放的对数概率差。只测量，不更新权重。

本地后端不打开重放时，训练日志里没有这个差；打开之后，日志里只剩重放之后的差。
要判断某个任务上不重放的偏差有多大，用这个脚本。模型必须是 MoE，后端必须是本地后端。

用法（在仓库根目录，需要 GPU）：

    python scripts/measure_routing_replay_gap.py --config configs/arithmetic_olmoe_d2d3.toml

    python scripts/measure_routing_replay_gap.py --config configs/arithmetic_olmoe_d2d3.toml --rounds 5 --output outputs/measure_routing_replay_gap.json

配置决定模型、数据、生成长度和小批。--rounds 是采样轮数，默认 1；每轮题数用 training.groups_per_iteration。
采样会强制记下专家，与配置里的 routing_replay 无关。结果是 JSON：全体回答 token 的汇总、按回答中的位置分组、会进入更新的那一部分，以及重放一致率。
"""
from __future__ import annotations

import argparse
import json
import random
import time
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
from grpx.loop import collect_trajectories, split_into_row_chunks
from grpx.routing_replay import RoutingReplayer, merge_replay_statistics
from grpx.trainer import action_log_probabilities, build_batch

ABSOLUTE_BINS = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 512)]
QUARTILES = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def summarize_gaps(gaps: torch.Tensor) -> dict:
    if gaps.numel() == 0:
        return {"n_tokens": 0}
    gaps = gaps.float()
    return {
        "n_tokens": int(gaps.numel()),
        "median": float(gaps.median()),
        "p95": float(torch.quantile(gaps, 0.95)),
        "above_half_fraction": float((gaps > 0.5).float().mean()),
        "above_one_fraction": float((gaps > 1).float().mean()),
        "max": float(gaps.max()),
    }


def slice_summary(indices, gaps, start, end) -> dict:
    chosen = (indices >= start) & (indices < end)
    row = summarize_gaps(gaps[chosen])
    row["position_from"] = start
    row["position_to"] = end
    return row


def group_rows(indices, native, replay, selector) -> dict:
    indices = indices[selector]
    native = native[selector]
    replay = replay[selector]
    absolute = [
        {
            "no_replay": slice_summary(indices, native, start, end),
            "replay": slice_summary(indices, replay, start, end),
        }
        for start, end in ABSOLUTE_BINS
    ]
    return {
        "pooled_no_replay": summarize_gaps(native),
        "pooled_replay": summarize_gaps(replay),
        "by_absolute_position": absolute,
    }


def quartile_rows(fractions, native, replay, selector) -> list:
    fractions = fractions[selector]
    native = native[selector]
    replay = replay[selector]
    rows = []
    for start, end in QUARTILES:
        chosen = (fractions >= start) & (fractions < end)
        rows.append({
            "fraction_from": start,
            "fraction_to": min(end, 1.0),
            "no_replay": summarize_gaps(native[chosen]),
            "replay": summarize_gaps(replay[chosen]),
        })
    return rows


def length_summary(trajectories) -> dict:
    lengths = torch.tensor(
        [len(step.action_token_ids) for trajectory in trajectories for step in trajectory.steps],
        dtype=torch.float,
    )
    truncated = sum(trajectory.termination_reason == "length_limit" for trajectory in trajectories)
    if lengths.numel() == 0:
        return {"n_answers": 0, "truncated": truncated}
    return {
        "n_answers": int(lengths.numel()),
        "truncated": truncated,
        "truncated_fraction": truncated / len(trajectories) if trajectories else 0.0,
        "median_tokens": float(lengths.median()),
        "p90_tokens": float(torch.quantile(lengths, 0.9)),
        "max_tokens": int(lengths.max()),
        "mean_tokens": float(lengths.mean()),
    }


def measure_round(model, trajectories, training, pad_token_id):
    kept = compute_advantages(
        trajectories,
        normalize_std=training.normalize_std,
        step_advantage_weight=training.step_advantage_weight,
        discount_factor=training.discount,
    )
    kept_steps = {id(step) for trajectory in kept for step in trajectory.steps}
    chunks = split_into_row_chunks(
        trajectories, training.max_rows_per_minibatch, training.max_tokens_per_minibatch,
        merge_append_only=False)
    replayer = RoutingReplayer(model)
    native_gaps = []
    replay_gaps = []
    answer_indices = []
    fractions = []
    kept_flags = []
    complete_flags = []
    statistics = []
    model.eval()
    for chunk in chunks:
        batch = build_batch(chunk, pad_token_id, model.device, training.row_weighting)
        if "routed_experts" not in batch:
            raise RuntimeError("这批回答没有专家记录，采样没有走重放路径")
        with torch.no_grad():
            native, _mask = action_log_probabilities(model, batch)
            with replayer.replay(batch["routed_experts"]) as stats:
                replayed, _mask = action_log_probabilities(model, batch)
        statistics.append(stats)
        comparable = batch["sampling_log_probability_mask"][:, 1:]
        sampling = batch["sampling_log_probabilities"][:, 1:]
        native_row = (native - sampling).abs()
        replay_row = (replayed - sampling).abs()
        for row_index, unit in enumerate(chunk):
            step = unit.steps[0]
            prompt_length = len(step.prompt_token_ids)
            action_length = len(step.action_token_ids)
            positions = comparable[row_index].nonzero(as_tuple=False).flatten()
            for index in positions.tolist():
                answer_index = index + 1 - prompt_length
                if answer_index < 0 or answer_index >= action_length:
                    raise RuntimeError(
                        f"回答位置对不上：序列位置 {index + 1}，提示词 {prompt_length}，"
                        f"回答 {action_length}"
                    )
                native_gaps.append(float(native_row[row_index, index]))
                replay_gaps.append(float(replay_row[row_index, index]))
                answer_indices.append(answer_index)
                fractions.append(answer_index / max(action_length - 1, 1))
                kept_flags.append(id(step) in kept_steps)
                complete_flags.append(step.finish_reason != "length_limit")
    if not native_gaps:
        empty = torch.tensor([])
        false = torch.tensor([], dtype=torch.bool)
        native_tensor = replay_tensor = index_tensor = fraction_tensor = empty
        kept_tensor = complete_tensor = false
    else:
        native_tensor = torch.tensor(native_gaps)
        replay_tensor = torch.tensor(replay_gaps)
        index_tensor = torch.tensor(answer_indices)
        fraction_tensor = torch.tensor(fractions)
        kept_tensor = torch.tensor(kept_flags)
        complete_tensor = torch.tensor(complete_flags)
    return {
        "lengths": length_summary(trajectories),
        "kept_trajectories": len(kept),
        "scored_answers": sum(len(chunk) for chunk in chunks),
        "minibatches": len(chunks),
        "all_answers": group_rows(index_tensor, native_tensor, replay_tensor, slice(None)),
        "kept_answers": group_rows(
            index_tensor, native_tensor, replay_tensor, kept_tensor),
        "complete_answers": group_rows(
            index_tensor, native_tensor, replay_tensor, complete_tensor),
        "quartiles_all": quartile_rows(
            fraction_tensor, native_tensor, replay_tensor, slice(None)),
        "quartiles_complete": quartile_rows(
            fraction_tensor, native_tensor, replay_tensor, complete_tensor),
        "routing": merge_replay_statistics(statistics) if statistics else {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="训练配置。后端须是本地后端，模型须是 MoE")
    parser.add_argument("--rounds", type=int, default=1, help="采样轮数，不更新权重。默认 1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/measure_routing_replay_gap.json"),
        help="结果 JSON 的路径",
    )
    arguments = parser.parse_args()
    if arguments.rounds < 1:
        raise SystemExit("--rounds 至少为 1")

    configuration = load_configuration(arguments.config, [])
    training = build_training_config(configuration)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(training.memory_fraction)
    model, tokenizer, backend = build_model_and_backend(configuration, training.seed)
    if not hasattr(backend, "record_routing"):
        raise SystemExit("这个测量只支持本地后端。vLLM 后端还不能返回每一层选中的专家")
    model.grpx_output_logit_chunk_size = training.output_logit_chunk_size
    backend.record_routing = True
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
    rounds = []
    started = time.perf_counter()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(arguments.rounds):
        print(f"=== round {iteration} sample ===", flush=True)
        sample_started = time.perf_counter()
        samples = rng.sample(dataset, training.groups_per_iteration)
        trajectories = collect_trajectories(
            backend, tokenizer, environment_factory, samples, training, iteration)
        print(
            f"sampled {len(trajectories)} answers in {time.perf_counter() - sample_started:.1f}s "
            f"{json.dumps(length_summary(trajectories), ensure_ascii=False)}",
            flush=True,
        )
        print(f"=== round {iteration} rescore ===", flush=True)
        outcome = measure_round(model, trajectories, training, pad_token_id)
        outcome["iteration"] = iteration
        outcome["seconds"] = time.perf_counter() - sample_started
        rounds.append(outcome)
        arguments.output.write_text(
            json.dumps({"rounds": rounds}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "iteration": iteration,
            "lengths": outcome["lengths"],
            "kept_trajectories": outcome["kept_trajectories"],
            "all_pooled_no_replay": outcome["all_answers"]["pooled_no_replay"],
            "all_pooled_replay": outcome["all_answers"]["pooled_replay"],
            "by_absolute_position": outcome["all_answers"]["by_absolute_position"],
            "quartiles_complete": outcome["quartiles_complete"],
            "routing": outcome["routing"],
        }, ensure_ascii=False), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "config": arguments.config,
        "model": configuration["model"]["name"],
        "max_new_tokens": configuration["backend"].get("max_new_tokens"),
        "rounds": rounds,
        "seconds": time.perf_counter() - started,
    }
    arguments.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", arguments.output, flush=True)


if __name__ == "__main__":
    main()
