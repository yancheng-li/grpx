"""分清批组成改变 logprob：float32 复现 + bf16 路由对照。

1) float32：同一短行单独成批 vs 与更长行拼接；若 |Δ| < 1e-4 → bf16 数值问题。
2) bf16：挂 RoutingRecorder，看 |Δlogprob| 大的位置是否换了专家。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from grpx.command_line import (
    apply_lora,
    build_environment_factory,
    build_model_and_backend,
    build_training_config,
    call_with_options,
    import_object,
    load_base_model_and_tokenizer,
    load_configuration,
)
from grpx.loop import collect_trajectories, compute_advantages, prepare_update_units, row_length
from grpx.routing_diagnostics import RoutingRecorder
from grpx.trainer import action_log_probabilities, build_batch


def sample_units(configuration, training):
    model, tokenizer, backend = build_model_and_backend(configuration, training.seed)
    model.eval()
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    dataset = call_with_options(
        import_object(configuration["task"]["dataset"]),
        configuration["task"].get("train_dataset_options", {}),
        "dataset",
    )
    environment_factory = build_environment_factory(configuration["task"])
    rng = random.Random(training.seed)
    torch.manual_seed(training.seed)
    samples = rng.sample(dataset, training.groups_per_iteration)
    trajectories = collect_trajectories(
        backend, tokenizer, environment_factory, samples, training, iteration=0)
    kept = compute_advantages(
        trajectories,
        normalize_std=training.normalize_std,
        step_advantage_weight=training.step_advantage_weight,
        discount_factor=training.discount,
    )
    units = prepare_update_units(kept, merge_append_only=False)
    units.sort(key=row_length, reverse=True)
    return model, tokenizer, pad, units[0], units[-1]


def forward_logp_and_routing(model, units, pad, row_weighting, record_routing=False):
    batch = build_batch(units, pad, model.device, row_weighting)
    recorder = RoutingRecorder(model) if record_routing else None
    snapshot = None
    with torch.no_grad():
        if recorder is not None and recorder.is_available:
            with recorder.record(batch["attention_mask"]) as snapshot:
                logp, mask = action_log_probabilities(model, batch)
        else:
            logp, mask = action_log_probabilities(model, batch)
    return batch, logp.detach().float().cpu(), mask.detach().cpu(), snapshot


def short_row_action_logp(logp, mask, short_row_index):
    return logp[short_row_index][mask[short_row_index]]


def short_row_valid_slice(attention_mask, short_row_index):
    """扁平有效位置里，短行对应的切片 [start, end)。"""
    lengths = attention_mask.sum(dim=1).tolist()
    start = int(sum(lengths[:short_row_index]))
    end = start + int(lengths[short_row_index])
    return start, end, int(lengths[short_row_index])


def summarize_diff(alone_vals, paired_vals):
    diff = (alone_vals - paired_vals).abs()
    return {
        "n_tokens": int(alone_vals.numel()),
        "mean_alone": float(alone_vals.mean()) if alone_vals.numel() else None,
        "mean_paired": float(paired_vals.mean()) if paired_vals.numel() else None,
        "abs_mean_diff": float(diff.mean()) if diff.numel() else None,
        "abs_max_diff": float(diff.max()) if diff.numel() else None,
        "abs_p95_diff": float(torch.quantile(diff, 0.95)) if diff.numel() else None,
        "below_1e-4": bool(diff.numel() and float(diff.max()) < 1e-4),
        "above_0_1": bool(diff.numel() and float(diff.max()) >= 0.1),
    }


def load_float32_model(seed=0):
    model_section = {
        "name": "allenai/OLMoE-1B-7B-0924-Instruct",
        "dtype": "float32",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    model, tokenizer, dtype_name = load_base_model_and_tokenizer(model_section, device="cuda")
    model = apply_lora(model, model_section, seed=seed, dtype_name=dtype_name)
    model.eval()
    return model, tokenizer


def main():
    configuration = load_configuration("configs/arithmetic_olmoe_d2d3.toml", [
        "training.seed=0",
        "training.run_name=probe-batch-cause",
        "training.iterations=1",
        "model.dtype=bfloat16",
    ])
    training = build_training_config(configuration)

    print("=== sample units on bf16 ===", flush=True)
    bf16_model, tokenizer, pad, long_unit, short_unit = sample_units(configuration, training)
    print(
        f"long_len={row_length(long_unit)} short_len={row_length(short_unit)}",
        flush=True,
    )

    print("=== bf16 alone vs paired + routing ===", flush=True)
    alone_batch, alone_logp, alone_mask, alone_snap = forward_logp_and_routing(
        bf16_model, [short_unit], pad, training.row_weighting, record_routing=True)
    paired_batch, paired_logp, paired_mask, paired_snap = forward_logp_and_routing(
        bf16_model, [long_unit, short_unit], pad, training.row_weighting, record_routing=True)

    alone_vals = short_row_action_logp(alone_logp, alone_mask, 0)
    paired_vals = short_row_action_logp(paired_logp, paired_mask, 1)
    bf16_summary = summarize_diff(alone_vals, paired_vals)

    routing_payload = {"recorder_available": alone_snap is not None and paired_snap is not None}
    if alone_snap is not None and paired_snap is not None:
        # alone：短行是唯一行；paired：短行在 index 1，扁平有效位置接在长行之后
        _, _, a_len = short_row_valid_slice(alone_batch["attention_mask"].cpu(), 0)
        p_start, p_end, p_len = short_row_valid_slice(paired_batch["attention_mask"].cpu(), 1)
        assert a_len == p_len == row_length(short_unit)

        changed_any = None
        for layer_name in alone_snap.layer_names():
            sel_a = alone_snap.selected(layer_name)  # (short_len, E)
            sel_p = paired_snap.selected(layer_name)[p_start:p_end]
            layer_changed = (sel_a != sel_p).any(dim=-1)
            changed_any = layer_changed if changed_any is None else (changed_any | layer_changed)

        # logp 与 action_mask[:, 1:] 对齐：下标即用于预测的 hidden 位置
        action_hidden_positions = (
            alone_batch["action_mask"][0, 1:].cpu().nonzero(as_tuple=False).view(-1).tolist())
        per_action = []
        for i, hidden_pos in enumerate(action_hidden_positions):
            per_action.append({
                "action_index": i,
                "hidden_position": int(hidden_pos),
                "abs_logprob_diff": float((alone_vals[i] - paired_vals[i]).abs()),
                "routing_changed_at_hidden": bool(changed_any[hidden_pos].item()),
            })

        large = [row for row in per_action if row["abs_logprob_diff"] >= 0.05]
        changed_diffs = [
            row["abs_logprob_diff"] for row in per_action if row["routing_changed_at_hidden"]]
        unchanged_diffs = [
            row["abs_logprob_diff"] for row in per_action if not row["routing_changed_at_hidden"]]
        routing_payload.update({
            "short_len": a_len,
            "fraction_positions_routing_changed": float(changed_any.float().mean()),
            "n_positions_routing_changed": int(changed_any.sum()),
            "per_action": per_action,
            "large_diff_actions": large,
            "large_diff_also_routing_changed_fraction": (
                float(sum(row["routing_changed_at_hidden"] for row in large) / len(large))
                if large else None
            ),
            "routing_changed_mean_abs_logprob_diff": (
                float(sum(changed_diffs) / len(changed_diffs)) if changed_diffs else None),
            "routing_unchanged_mean_abs_logprob_diff": (
                float(sum(unchanged_diffs) / len(unchanged_diffs)) if unchanged_diffs else None),
        })
        print(
            "bf16 routing: changed_frac="
            f"{routing_payload['fraction_positions_routing_changed']:.4f} "
            f"large_diff_n={len(large)} "
            f"large_also_changed={routing_payload['large_diff_also_routing_changed_fraction']}",
            flush=True,
        )

    del bf16_model
    torch.cuda.empty_cache()

    print("=== float32 alone vs paired ===", flush=True)
    f32_model, _ = load_float32_model(seed=training.seed)
    # 关掉混合精度路径
    f32_model.grpx_mixed_precision = False
    alone_batch_f, alone_logp_f, alone_mask_f, _ = forward_logp_and_routing(
        f32_model, [short_unit], pad, training.row_weighting, record_routing=False)
    paired_batch_f, paired_logp_f, paired_mask_f, _ = forward_logp_and_routing(
        f32_model, [long_unit, short_unit], pad, training.row_weighting, record_routing=False)
    alone_vals_f = short_row_action_logp(alone_logp_f, alone_mask_f, 0)
    paired_vals_f = short_row_action_logp(paired_logp_f, paired_mask_f, 1)
    f32_summary = summarize_diff(alone_vals_f, paired_vals_f)
    print("float32", json.dumps(f32_summary), flush=True)

    if f32_summary["below_1e-4"]:
        verdict = "bf16_numerical_not_bug"
    elif f32_summary["above_0_1"]:
        verdict = "mask_or_padding_bug"
    else:
        verdict = "inconclusive_intermediate"

    payload = {
        "long_len": row_length(long_unit),
        "short_len": row_length(short_unit),
        "bf16": bf16_summary,
        "float32": f32_summary,
        "bf16_routing": routing_payload,
        "verdict": verdict,
    }
    out = Path("outputs/probe_batch_logprob_cause.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print("wrote", out, "verdict=", verdict, flush=True)


if __name__ == "__main__":
    main()
