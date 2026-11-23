"""Qwen3-30B：训练前基线，vLLM 贪心评测固定 500 道三/四位数乘法。

约定：
  make_dataset(size=500, digits=3|4, seed=1, operation="multiplication", prompt_variant="english_strict")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from examples.arithmetic import ArithmeticEnvironment, make_dataset
from grpx.backends.vllm import VllmBackend
from grpx.evaluation import evaluate

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def evaluate_digits(backend, tokenizer, digits, size, seed, max_steps=1):
    dataset = make_dataset(
        size=size, digits=digits, seed=seed, operation="multiplication",
        prompt_variant="english_strict")
    metrics = evaluate(
        backend,
        tokenizer,
        ArithmeticEnvironment,
        dataset,
        max_steps=max_steps,
        truncated_reward=0.0,
    )
    return {
        "digits": digits,
        "size": size,
        "seed": seed,
        "operation": "multiplication",
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--digits", type=int, nargs="+", default=[3, 4],
        help="要评测的乘数位数列表")
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/qwen3_mul_pretrain_baseline_500.json"),
    )
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    end_token_id = tokenizer.eos_token_id
    if end_token_id is None:
        raise SystemExit("tokenizer 没有 eos_token_id")

    scratch = Path("outputs/vllm_eval_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    backend = VllmBackend(
        tokenizer,
        {end_token_id},
        adapter_directory=str(scratch.resolve()),
        base_url=arguments.base_url,
        adapter_name="baseline",
        expected_base_model=MODEL,
        max_new_tokens=arguments.max_new_tokens,
        max_concurrent_requests=64,
    )
    backend.verify_server_base_model()
    # 不挂 LoRA，直接用服务端基座名
    backend.adapter_name = MODEL

    results = {
        "model": MODEL,
        "backend": "vllm",
        "base_url": arguments.base_url,
        "max_new_tokens": arguments.max_new_tokens,
        "evaluations": [],
    }
    for digits in arguments.digits:
        print(f"=== baseline digits={digits} n={arguments.size} ===", flush=True)
        row = evaluate_digits(
            backend, tokenizer, digits, arguments.size, arguments.seed)
        results["evaluations"].append(row)
        print(
            f"digits={digits}  "
            f"success={row['evaluation_success_fraction']:.4f}  "
            f"reward={row['evaluation_mean_reward']:.4f}  "
            f"trunc={row['evaluation_length_limit_fraction']:.4f}  "
            f"mean_tokens={row['evaluation_mean_answer_tokens']:.3f}",
            flush=True,
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
