"""注意力 / 专家对照用：固定 500 道三位数加法，贪心重评检查点。

评测集约定（两组必须相同）：
  make_dataset(size=500, digits=3, seed=1, prompt_variant="english_strict")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.arithmetic import is_correct, make_dataset, reward_function
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.environment import SingleTurnEnvironment
from grpx.evaluation import evaluate

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
EVAL_SIZE = 500
EVAL_DIGITS = 3
EVAL_SEED = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=[
            Path("outputs/olmoe-arith-d2d3-attn-s0/checkpoint-0060"),
            Path("outputs/olmoe-arith-d2d3-attn-s1/checkpoint-0060"),
            Path("outputs/olmoe-arith-d2d3-attn-s2/checkpoint-0060"),
        ],
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-rows-per-batch", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/olmoe_d2d3_attn_d3_500_eval.json"),
    )
    arguments = parser.parse_args()

    dataset = make_dataset(
        size=EVAL_SIZE, digits=EVAL_DIGITS, seed=EVAL_SEED, prompt_variant="english_strict")
    environment_factory = lambda: SingleTurnEnvironment(
        reward_function, success_function=is_correct)

    print(f"loading base {MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")

    results = {
        "dataset": {
            "size": EVAL_SIZE,
            "digits": EVAL_DIGITS,
            "seed": EVAL_SEED,
            "builder": "examples.arithmetic:make_dataset",
        },
        "max_new_tokens": arguments.max_new_tokens,
        "checkpoints": [],
    }

    for checkpoint in arguments.checkpoints:
        checkpoint = checkpoint.resolve()
        print(f"\n=== {checkpoint} ===", flush=True)
        model = PeftModel.from_pretrained(base, str(checkpoint))
        model.eval()
        backend = HuggingFaceBackend(
            model,
            tokenizer,
            max_new_tokens=arguments.max_new_tokens,
            max_rows_per_batch=arguments.max_rows_per_batch,
        )
        metrics = evaluate(
            backend,
            tokenizer,
            environment_factory,
            dataset,
            max_steps=1,
            truncated_reward=0.0,
        )
        row = {
            "checkpoint": str(checkpoint),
            "n": len(dataset),
            **metrics,
        }
        results["checkpoints"].append(row)
        print(
            f"success={metrics['evaluation_success_fraction']:.4f}  "
            f"reward={metrics['evaluation_mean_reward']:.4f}  "
            f"trunc={metrics['evaluation_length_limit_fraction']:.4f}  "
            f"mean_tokens={metrics['evaluation_mean_answer_tokens']:.3f}",
            flush=True,
        )
        base = model.unload()
        del model
        torch.cuda.empty_cache()

    successes = [
        row["evaluation_success_fraction"] for row in results["checkpoints"]]
    if successes:
        results["mean_success_fraction"] = sum(successes) / len(successes)
        results["std_success_fraction"] = (
            sum((value - results["mean_success_fraction"]) ** 2 for value in successes)
            / len(successes)
        ) ** 0.5

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {arguments.output}", flush=True)
    if successes:
        print(
            f"mean_success={results['mean_success_fraction']:.4f}  "
            f"std={results['std_success_fraction']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
