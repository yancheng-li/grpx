"""检查 OLMoE 专家包装模块上 gate_up_proj / down_proj 的 LoRA scaling。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from grpx.command_line import apply_lora, load_base_model_and_tokenizer


def collect_scaling(model):
    rows = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if scaling is None:
            continue
        rows.append({
            "name": name,
            "cls": type(module).__name__,
            "scaling": scaling if not isinstance(scaling, dict) else dict(scaling),
            "r": getattr(module, "r", None) if not isinstance(getattr(module, "r", None), dict)
            else dict(getattr(module, "r")),
            "lora_alpha": getattr(module, "lora_alpha", None)
            if not isinstance(getattr(module, "lora_alpha", None), dict)
            else dict(getattr(module, "lora_alpha")),
        })
    return rows


def summarize(rows):
    by = defaultdict(list)
    for row in rows:
        name = row["name"]
        if "gate_up" in name or name.endswith("experts.base_layer"):
            key = "gate_up_or_experts.base_layer"
        elif name.endswith("mlp.experts") or name.endswith(".experts"):
            key = "experts_wrapper"
        elif any(x in name for x in ("q_proj", "k_proj", "v_proj", "o_proj")):
            key = "attention_proj"
        else:
            key = "other"
        by[key].append(row)
    summary = {}
    for key, items in by.items():
        scalings = sorted({json.dumps(item["scaling"], sort_keys=True) for item in items})
        ranks = sorted({json.dumps(item["r"], sort_keys=True) for item in items})
        alphas = sorted({json.dumps(item["lora_alpha"], sort_keys=True) for item in items})
        summary[key] = {
            "count": len(items),
            "scaling_values": scalings,
            "r_values": ranks,
            "alpha_values": alphas,
            "example": items[0],
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fresh", "checkpoint"), default="fresh")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/olmoe-arith-d2d3-experts-s0/checkpoint-0060"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gate_up_scaling_check.json"))
    arguments = parser.parse_args()

    model_section = {
        "name": "allenai/OLMoE-1B-7B-0924-Instruct",
        "dtype": "bfloat16",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    }

    if arguments.mode == "fresh":
        model, tokenizer, dtype_name = load_base_model_and_tokenizer(
            model_section, device=arguments.device)
        del tokenizer
        model = apply_lora(model, model_section, seed=0, dtype_name=dtype_name)
        source = "fresh_apply_lora"
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_section["name"], dtype=torch.bfloat16, device_map=arguments.device)
        model = PeftModel.from_pretrained(base, str(arguments.checkpoint))
        source = str(arguments.checkpoint)

    rows = collect_scaling(model)
    summary = summarize(rows)
    # 期望：缩放都是 alpha/r = 2.0（gate_up 用 64/32，其余 32/16）
    expected_scaling = 2.0
    observed = []
    for key, info in summary.items():
        if key == "other":
            continue
        for raw in info["scaling_values"]:
            value = json.loads(raw)
            if isinstance(value, dict):
                observed.extend(value.values())
            else:
                observed.append(value)
    payload = {
        "source": source,
        "expected_scaling": expected_scaling,
        "all_scalings_are_2": all(abs(float(v) - expected_scaling) < 1e-12 for v in observed),
        "summary": summary,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print("wrote", arguments.output, flush=True)


if __name__ == "__main__":
    main()
