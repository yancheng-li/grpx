"""GSM8K 计算器任务：对训练运行的 checkpoint-0060 做完整测试集贪心评测。

从运行目录的 resolved_configuration.json 读取停止串、环境选项等，保证与训练时一致。
需要已启动的 vLLM 服务。

用法（项目根目录）：
  PYTHONPATH=. python scripts/eval_calculator_fulltest.py
  PYTHONPATH=. python scripts/eval_calculator_fulltest.py outputs/gsm8k-calc-worked-stop-s0
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.gsm8k import make_dataset
from examples.gsm8k_calculator import CalculatorEnvironment
from grpx.backends.common import resolve_end_token_ids
from grpx.backends.vllm import VllmBackend
from grpx.evaluation import evaluate

DEFAULT_RUNS = [
    ("worked_example_stop", "outputs/gsm8k-calc-worked-stop-s0"),
]


def evaluate_run(variant: str, run_dir: str, dataset) -> dict:
    run = Path(run_dir)
    cfg = json.loads((run / "resolved_configuration.json").read_text(encoding="utf-8"))
    ckpt = run / "checkpoint-0060"
    model_name = cfg["model"]["name"]
    max_new_tokens = cfg["backend"]["max_new_tokens"]
    max_steps = cfg["training"]["max_steps"]
    truncated_reward = cfg["training"].get("truncated_reward", 0.0)
    opts = dict(cfg["task"]["environment_options"])
    stop_strings = list(cfg["backend"].get("stop_strings") or [])

    print(f"\n=== {variant}  {ckpt}  stop_strings={stop_strings} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ckpt)
    end_ids = resolve_end_token_ids(model, tokenizer)

    adapter_dir = run / "vllm_adapter_fulltest"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    backend = VllmBackend(
        tokenizer, end_ids,
        adapter_directory=adapter_dir,
        base_url=cfg["backend"].get("base_url", "http://localhost:8000"),
        max_new_tokens=max_new_tokens,
        max_concurrent_requests=cfg["backend"].get("max_concurrent_requests", 64),
        adapter_name=f"fulltest-{run.name}",
        expected_base_model=model_name,
        stop_strings=stop_strings,
    )
    backend.synchronize_policy(model)
    started = time.perf_counter()
    try:
        metrics = evaluate(
            backend, tokenizer,
            lambda: CalculatorEnvironment(**opts),
            dataset,
            max_steps=max_steps,
            truncated_reward=truncated_reward,
        )
    finally:
        getattr(backend, "close", lambda: None)()
        del model, base
        torch.cuda.empty_cache()

    metrics.update(
        prompt_variant=variant,
        n=len(dataset),
        seconds=time.perf_counter() - started,
        run_directory=str(run),
        checkpoint=str(ckpt),
        max_new_tokens=max_new_tokens,
        max_steps=max_steps,
        stop_strings=stop_strings,
    )
    out = Path(f"outputs/gsm8k_calculator_fulltest_{variant}_post.json")
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        variant,
        "success=", metrics["evaluation_success_fraction"],
        "used_calc=", metrics.get("evaluation_environment_used_calculator_fraction"),
        "wo_wait=", metrics.get("evaluation_environment_answered_without_waiting_fraction"),
        "call_after=", metrics.get("evaluation_environment_call_after_answer_marker_fraction"),
        "budget=", metrics.get("evaluation_environment_exceeded_response_budget_fraction"),
        "trunc=", metrics.get("evaluation_length_limit_fraction"),
        flush=True,
    )
    print("wrote", out, flush=True)
    return metrics


def main():
    if len(sys.argv) > 1:
        runs = [("custom", sys.argv[1])]
        # 用目录名当输出标签更清晰
        runs = [(Path(sys.argv[1]).name, sys.argv[1])]
    else:
        runs = DEFAULT_RUNS

    dataset = make_dataset(split="test")
    print(f"full test n={len(dataset)}", flush=True)
    for variant, run_dir in runs:
        evaluate_run(variant, run_dir, dataset)


if __name__ == "__main__":
    main()
