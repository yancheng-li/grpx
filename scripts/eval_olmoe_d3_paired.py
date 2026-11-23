"""OLMoE 检查点：500 道三位数加法，基座 vs 适配器配对（本地后端）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.arithmetic import ArithmeticEnvironment, make_dataset
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.rollout import rollout_groups

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"


def greedy_successes(backend, tokenizer, dataset):
    trajectories = rollout_groups(
        backend,
        tokenizer,
        ArithmeticEnvironment,
        [(f"evaluation-{index}", sample, index)
         for index, sample in enumerate(dataset)],
        group_size=1,
        max_steps=1,
        greedy=True,
        truncated_reward=0.0,
    )
    return [bool(trajectory.is_successful) for trajectory in trajectories]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    checkpoint = arguments.checkpoint.resolve()
    dataset = make_dataset(
        arguments.size, digits=arguments.digits, seed=arguments.seed,
        operation="addition", prompt_variant="english_strict")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print("loading base", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    backend = HuggingFaceBackend(
        base, tokenizer, max_new_tokens=arguments.max_new_tokens,
        max_rows_per_batch=8)
    before = greedy_successes(backend, tokenizer, dataset)

    print("loading adapter", checkpoint, flush=True)
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model.eval()
    backend = HuggingFaceBackend(
        model, tokenizer, max_new_tokens=arguments.max_new_tokens,
        max_rows_per_batch=8)
    after = greedy_successes(backend, tokenizer, dataset)

    gained = sum(1 for b, a in zip(before, after) if (not b) and a)
    lost = sum(1 for b, a in zip(before, after) if b and (not a))
    row = {
        "checkpoint": str(checkpoint),
        "digits": arguments.digits,
        "size": len(dataset),
        "seed": arguments.seed,
        "before_success_fraction": sum(before) / len(before),
        "after_success_fraction": sum(after) / len(after),
        "delta_success_fraction": (sum(after) - sum(before)) / len(before),
        "gained": gained,
        "lost": lost,
        "net": gained - lost,
        "before_success_flags": before,
        "after_success_flags": after,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"before={row['before_success_fraction']:.4f} after={row['after_success_fraction']:.4f} "
        f"gained={gained} lost={lost} net={gained - lost}",
        flush=True,
    )


if __name__ == "__main__":
    main()
