"""对照：只挂注意力 LoRA vs 注意力+专家 LoRA，在同一批轨迹上试更新。

采样用注意力配置。专家配置通过 LoraConfig.layers_to_transform 只挂少量层，
避免 64 专家 × 16 层把 24GB 卡死。

用法：
  PYTHONPATH=. python -m scripts.routing_lora_compare
"""
from __future__ import annotations

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

# 专家对照：用正则限制在前 4 层（layers_to_transform 对 MoE 专家路径无效，仍会挂满 64 专家）
EXPERT_LAYERS = list(range(4))
EXPERT_TARGET_REGEX = (
    r".*\.layers\.[0-3]\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj))$"
)


def run_trial(model, kept, padding_token_id, config, learning_rate):
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    snapshot = {name: parameter.detach().clone() for name, parameter in trainable.items()}

    def restore():
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(snapshot[name])

    restore()
    trial_config = dataclasses.replace(
        config,
        learning_rate=learning_rate,
        max_rows_per_minibatch=2,
        max_tokens_per_minibatch=512,
    )
    # 专家路径上轨迹多会很慢；两边都截断到同一小子集，保证可比
    subset = kept[:16]
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=learning_rate)
    print(
        f"  trainable={sum(parameter.numel() for parameter in trainable.values()):,} "
        f"trajectories={len(subset)}",
        flush=True,
    )
    (history, batches, old_log_probabilities, gap_metrics, merge_fraction,
     _peak, _over, routing_before) = update_policy(
        model, optimizer, subset, padding_token_id, trial_config, record_routing=True)
    print("  update done, measuring shift…", flush=True)
    shift = measure_policy_shift(
        model, batches, old_log_probabilities, trial_config,
        routing_before=routing_before)
    del optimizer
    restore()
    return {
        "learning_rate": learning_rate,
        "trajectory_count": len(subset),
        "first_step_max_ratio_deviation": history[0]["max_ratio_deviation"],
        "merge_fraction": merge_fraction,
        "minibatch_count": len(batches),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable.values()),
        **{key: shift[key] for key in SHIFT_KEYS},
        **{key: shift[key] for key in ROUTING_KEYS if key in shift},
        **gap_metrics,
    }


def format_metrics(metrics):
    lines = []
    for key in ["trainable_parameter_count", "trajectory_count"] + SHIFT_KEYS + ROUTING_KEYS:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6g}")
            else:
                lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def load_attention_stack():
    configuration = load_configuration(
        "configs/arithmetic_olmoe.toml",
        ["training.run_name=routing-lora-attn-sample"],
    )
    return build_components(configuration)


def load_expert_stack():
    """注意力+专家，但只在前 EXPERT_LAYERS 层注入（含专家投影）。"""
    from peft import LoraConfig, get_peft_model

    from grpx.command_line import (
        build_training_config,
        load_base_model_and_tokenizer,
        load_configuration,
    )
    from grpx.doctor import _attach_backend

    configuration = load_configuration(
        "configs/arithmetic_olmoe.toml",
        ["training.run_name=routing-lora-experts-update"],
    )
    config = build_training_config(configuration)
    model, tokenizer, dtype_name = load_base_model_and_tokenizer(
        configuration["model"], device=config.device)
    torch.manual_seed(config.seed)
    model = get_peft_model(model, LoraConfig(
        r=configuration["model"].get("lora_rank", 16),
        lora_alpha=configuration["model"].get("lora_alpha", 32),
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=EXPERT_TARGET_REGEX,
    ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not all(parameter.dtype == torch.float32 for parameter in trainable):
        raise RuntimeError("可训练参数必须是 float32")
    model.grpx_mixed_precision = dtype_name == "bfloat16"
    model.print_trainable_parameters()
    # 绕过 apply_lora；backend 仍按配置装配
    model, tokenizer, backend = _attach_backend(model, tokenizer, configuration)
    return model, tokenizer, backend, config


def main():
    output = Path("outputs/routing_lora_compare.json")
    results = {
        "expert_layers": EXPERT_LAYERS,
        "expert_target_regex": EXPERT_TARGET_REGEX,
    }

    print("=== sample with attention-only LoRA ===", flush=True)
    (model, tokenizer, backend, train_dataset, environment_factory,
     config, _evaluation_dataset, _run_directory) = load_attention_stack()
    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = min(backend.end_token_ids)
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

    for name, learning_rate in [("attn_lr_1e-8", 1e-8), ("attn_lr_1e-5", 1e-5)]:
        print(f"\n=== {name} ===", flush=True)
        payload = run_trial(model, kept, padding_token_id, config, learning_rate)
        results[name] = payload
        print(format_metrics(payload), flush=True)
    if hasattr(backend, "close"):
        backend.close()
    del model, tokenizer, backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n=== load attention+expert LoRA (layers_to_transform) ===", flush=True)
    model, tokenizer, backend, config = load_expert_stack()
    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = min(backend.end_token_ids)
    for name, learning_rate in [("experts_lr_1e-8", 1e-8), ("experts_lr_1e-5", 1e-5)]:
        print(f"\n=== {name} ===", flush=True)
        payload = run_trial(model, kept, padding_token_id, config, learning_rate)
        results[name] = payload
        print(format_metrics(payload), flush=True)
    if hasattr(backend, "close"):
        backend.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {output}", flush=True)


if __name__ == "__main__":
    main()
