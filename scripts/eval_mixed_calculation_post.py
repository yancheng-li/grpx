"""混合计算任务检查点的训后评测：GSM8K 全测、计算题 test 1000、未见题型 500。

对指定运行目录的最新检查点，用与训练相同的停止串和环境选项做贪心评测。
需要已启动的 vLLM 服务。

用法（项目根目录）：
  PYTHONPATH=. python scripts/eval_mixed_calculation_post.py
  PYTHONPATH=. python scripts/eval_mixed_calculation_post.py outputs/mixed-calc-optional-stop-s0
  PYTHONPATH=. python scripts/eval_mixed_calculation_post.py outputs/mixed-calc-optional-stop-s0 checkpoint-0060
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples import calculation_problems, gsm8k
from examples.gsm8k_calculator import CalculatorEnvironment
from grpx.backends.common import resolve_end_token_ids
from grpx.backends.vllm import VllmBackend
from grpx.evaluation import evaluate


def print_metrics(label, metrics):
    print(
        label,
        "success=", metrics["evaluation_success_fraction"],
        "used_calc=", metrics.get("evaluation_environment_used_calculator_fraction"),
        "wo_wait=", metrics.get("evaluation_environment_answered_without_waiting_fraction"),
        "mean_steps=", metrics.get("evaluation_mean_step_count"),
        "trunc=", metrics.get("evaluation_length_limit_fraction"),
        "call_after=", metrics.get("evaluation_environment_call_after_answer_marker_fraction"),
        flush=True,
    )


def latest_checkpoint(run_dir: Path) -> Path:
    if len(sys.argv) > 2:
        return run_dir / sys.argv[2]
    candidates = [
        path for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and not path.name.endswith(".partial")
        and (path / "adapter_model.safetensors").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"{run_dir} 里没有可用检查点")
    return max(candidates, key=lambda path: int(path.name.split("-")[1]))


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/mixed-calc-optional-stop-s0")
    cfg = json.loads((run_dir / "resolved_configuration.json").read_text(encoding="utf-8"))
    ckpt = latest_checkpoint(run_dir)
    opts = dict(cfg["task"]["environment_options"])
    # 单来源评测不需要按 source 拆分；去掉也不影响，缺字段时本来就不写。
    max_steps = cfg["training"]["max_steps"]
    truncated_reward = cfg["training"].get("truncated_reward", 0.0)
    model_name = cfg["model"]["name"]
    seed = cfg["training"].get("seed", 0)
    stop_strings = list(cfg["backend"].get("stop_strings") or [])
    max_new_tokens = cfg["backend"]["max_new_tokens"]

    suites = [
        ("gsm8k_test", gsm8k.make_dataset(split="test")),
        ("calculation_test1000", calculation_problems.make_dataset(split="test", size=1000, seed=seed)),
        ("calculation_unseen500", calculation_problems.make_dataset(split="unseen", size=500, seed=seed)),
    ]
    print(
        f"run={run_dir} ckpt={ckpt.name} stop_strings={stop_strings} "
        f"max_new_tokens={max_new_tokens} max_steps={max_steps}",
        flush=True,
    )
    for name, dataset in suites:
        print(f"  suite {name}: n={len(dataset)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ckpt)
    end_ids = resolve_end_token_ids(model, tokenizer)

    adapter_dir = run_dir / "vllm_adapter_post_evals"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    backend = VllmBackend(
        tokenizer, end_ids,
        adapter_directory=adapter_dir,
        base_url=cfg["backend"].get("base_url", "http://localhost:8000"),
        max_new_tokens=max_new_tokens,
        max_concurrent_requests=cfg["backend"].get("max_concurrent_requests", 64),
        adapter_name=f"post-evals-{run_dir.name}",
        expected_base_model=model_name,
        stop_strings=stop_strings,
    )
    backend.synchronize_policy(model)
    env_factory = lambda: CalculatorEnvironment(**opts)

    summary = {"run_directory": str(run_dir), "checkpoint": str(ckpt)}
    try:
        for name, dataset in suites:
            started = time.perf_counter()
            metrics = evaluate(
                backend, tokenizer, env_factory, dataset,
                max_steps=max_steps, truncated_reward=truncated_reward,
            )
            metrics.update(
                label=name,
                n=len(dataset),
                seconds=time.perf_counter() - started,
                run_directory=str(run_dir),
                checkpoint=str(ckpt),
                stop_strings=stop_strings,
                max_new_tokens=max_new_tokens,
                max_steps=max_steps,
            )
            out = Path(f"outputs/mixed_post_{run_dir.name}_{name}.json")
            out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print_metrics(name, metrics)
            print("wrote", out, flush=True)
            summary[name] = {
                "success": metrics["evaluation_success_fraction"],
                "used_calc": metrics.get("evaluation_environment_used_calculator_fraction"),
                "wo_wait": metrics.get("evaluation_environment_answered_without_waiting_fraction"),
                "mean_steps": metrics.get("evaluation_mean_step_count"),
                "n": len(dataset),
            }
    finally:
        getattr(backend, "close", lambda: None)()
        del model, base
        torch.cuda.empty_cache()

    path = Path(f"outputs/mixed_post_{run_dir.name}_summary.json")
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
