"""注意力-only：同一批轨迹在 4/1024 与 128/8192 分批下，action logprob 是否一致。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from grpx.command_line import (
    build_environment_factory,
    build_model_and_backend,
    build_training_config,
    call_with_options,
    import_object,
    load_configuration,
)
from grpx.loop import (
    build_batch,
    collect_trajectories,
    compute_advantages,
    prepare_update_units,
    row_length,
    segment_count,
)
from grpx.trainer import action_log_probabilities


def pack_units(units, max_rows, max_tokens):
    rows = [row for row in units if row_length(row) <= max_tokens]
    chunks, current = [], []
    current_segments = 0
    for row in rows:
        segments = segment_count(row)
        if current and (
                current_segments + segments > max_rows
                or (len(current) + 1) * row_length(current[0]) > max_tokens):
            chunks.append(current)
            current = []
            current_segments = 0
        current.append(row)
        current_segments += segments
    if current:
        chunks.append(current)
    return chunks


def run_split(model, units, padding_token_id, config, rows, tokens):
    chunks = pack_units(units, rows, tokens)
    model.eval()
    by_unit = {}
    batch_shapes = []
    with torch.no_grad():
        for chunk in chunks:
            batch = build_batch(chunk, padding_token_id, model.device, config.row_weighting)
            batch_shapes.append(list(batch["input_token_ids"].shape))
            logp, mask = action_log_probabilities(model, batch)
            logp = logp.detach().float().cpu()
            mask = mask.detach().cpu()
            for row_index, unit in enumerate(chunk):
                values = logp[row_index][mask[row_index]]
                by_unit[id(unit)] = {
                    "n_tokens": int(values.numel()),
                    "mean": float(values.mean()) if values.numel() else float("nan"),
                    "sum": float(values.sum()) if values.numel() else 0.0,
                    "values": values,
                }
    flat = torch.cat([row["values"] for row in by_unit.values() if row["n_tokens"]])
    return {
        "n_tokens": int(flat.numel()),
        "mean": float(flat.mean()),
        "sum": float(flat.sum()),
        "n_batches": len(chunks),
        "batch_shapes": batch_shapes,
        "by_unit": by_unit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/arithmetic_olmoe_d2d3.toml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/minibatch_logprob_parity.json"))
    arguments = parser.parse_args()

    configuration = load_configuration(arguments.config, [
        f"training.seed={arguments.seed}",
        "training.run_name=minibatch-logprob-parity",
        "training.iterations=1",
    ])
    training = build_training_config(configuration)
    model, tokenizer, backend = build_model_and_backend(configuration, training.seed)
    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = tokenizer.eos_token_id

    dataset = call_with_options(
        import_object(configuration["task"]["dataset"]),
        configuration["task"].get("train_dataset_options", {}),
        "dataset",
    )
    environment_factory = build_environment_factory(configuration["task"])

    random_generator = random.Random(training.seed)
    torch.manual_seed(training.seed)
    samples = random_generator.sample(dataset, training.groups_per_iteration)
    trajectories = collect_trajectories(
        backend, tokenizer, environment_factory, samples, training, iteration=0)
    kept = compute_advantages(
        trajectories,
        normalize_std=training.normalize_std,
        step_advantage_weight=training.step_advantage_weight,
        discount_factor=training.discount,
    )
    if not kept:
        raise SystemExit("没有可更新轨迹，无法比较")

    units = prepare_update_units(kept, merge_append_only=training.merge_append_only_turns)
    units.sort(key=row_length, reverse=True)

    a = run_split(model, units, padding_token_id, training, 4, 1024)
    b = run_split(model, units, padding_token_id, training, 128, 8192)

    shared = sorted(set(a["by_unit"]) & set(b["by_unit"]))
    missing = set(a["by_unit"]) ^ set(b["by_unit"])
    row_diffs = []
    token_abs = []
    for key in shared:
        left, right = a["by_unit"][key], b["by_unit"][key]
        row_diffs.append(abs(left["mean"] - right["mean"]))
        if left["n_tokens"]:
            token_abs.append((left["values"] - right["values"]).abs())
    token_abs = torch.cat(token_abs) if token_abs else torch.zeros(0)

    payload = {
        "split_4_1024": {key: a[key] for key in ("n_tokens", "mean", "sum", "n_batches", "batch_shapes")},
        "split_128_8192": {key: b[key] for key in ("n_tokens", "mean", "sum", "n_batches", "batch_shapes")},
        "abs_mean_diff": abs(a["mean"] - b["mean"]),
        "same_n_tokens": a["n_tokens"] == b["n_tokens"],
        "n_shared_units": len(shared),
        "n_mismatched_units": len(missing),
        "row_mean_abs_diff_max": max(row_diffs) if row_diffs else None,
        "row_mean_abs_diff_mean": float(sum(row_diffs) / len(row_diffs)) if row_diffs else None,
        "token_abs_diff_max": float(token_abs.max()) if token_abs.numel() else None,
        "token_abs_diff_mean": float(token_abs.mean()) if token_abs.numel() else None,
        "token_abs_diff_p95": float(torch.quantile(token_abs, 0.95)) if token_abs.numel() else None,
        "pass_mean_1e-3": abs(a["mean"] - b["mean"]) < 1e-3,
        "pass_token_max_1e-2": (float(token_abs.max()) < 1e-2) if token_abs.numel() else False,
        "pass_token_max_5e-2": (float(token_abs.max()) < 5e-2) if token_abs.numel() else False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print("wrote", arguments.output, flush=True)


if __name__ == "__main__":
    main()
