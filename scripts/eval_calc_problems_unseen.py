"""计算题未见题型（split=unseen）评测：未训练 LoRA 与最终检查点各贪心评测一次。

用于检查模型学到的是否只对训练见过的题型模板有效。需要已启动的 vLLM 服务
（地址从运行目录的 resolved_configuration.json 读取，默认 localhost:8000）。

用法（项目根目录）：
  PYTHONPATH=. python scripts/eval_calc_problems_unseen.py
  PYTHONPATH=. python scripts/eval_calc_problems_unseen.py outputs/calc-problems-worked-stop-s0
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.calculation_problems import make_dataset
from examples.gsm8k_calculator import CalculatorEnvironment
from grpx.backends.common import resolve_end_token_ids
from grpx.backends.vllm import VllmBackend
from grpx.evaluation import evaluate


def run_one(label, model, tokenizer, backend, env_factory, dataset, max_steps,
            truncated_reward, extra):
    backend.synchronize_policy(model)
    started = time.perf_counter()
    metrics = evaluate(
        backend, tokenizer, env_factory, dataset,
        max_steps=max_steps, truncated_reward=truncated_reward,
    )
    metrics.update(
        label=label, n=len(dataset), seconds=time.perf_counter() - started, **extra)
    tag = extra.get("run_tag", "run")
    out = Path(f"outputs/calc_problems_unseen_{tag}_{label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        label,
        "success=", metrics["evaluation_success_fraction"],
        "used_calc=", metrics.get("evaluation_environment_used_calculator_fraction"),
        "answered=", metrics.get("evaluation_environment_answered_fraction"),
        "call_after=", metrics.get("evaluation_environment_call_after_answer_marker_fraction"),
        "mean_steps=", metrics.get("evaluation_mean_step_count"),
        "wrote", out,
        flush=True,
    )
    return metrics


def make_backend(cfg, tokenizer, end_ids, adapter_directory, adapter_name, stop_strings):
    return VllmBackend(
        tokenizer, end_ids,
        adapter_directory=adapter_directory,
        base_url=cfg["backend"].get("base_url", "http://localhost:8000"),
        max_new_tokens=cfg["backend"]["max_new_tokens"],
        max_concurrent_requests=cfg["backend"].get("max_concurrent_requests", 64),
        adapter_name=adapter_name,
        expected_base_model=cfg["model"]["name"],
        stop_strings=stop_strings,
    )


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/calc-problems-worked-stop-s0")
    cfg = json.loads((run_dir / "resolved_configuration.json").read_text(encoding="utf-8"))
    ckpt = run_dir / "checkpoint-0060"
    opts = dict(cfg["task"]["environment_options"])
    max_steps = cfg["training"]["max_steps"]
    truncated_reward = cfg["training"].get("truncated_reward", 0.0)
    model_name = cfg["model"]["name"]
    seed = cfg["training"].get("seed", 0)
    stop_strings = list(cfg["backend"].get("stop_strings") or [])

    dataset = make_dataset(split="unseen", size=500, seed=seed)
    print(f"unseen n={len(dataset)}  run={run_dir}  ckpt={ckpt.name}  "
          f"stop_strings={stop_strings}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    env_factory = lambda: CalculatorEnvironment(**opts)
    run_tag = run_dir.name
    extra = {
        "split": "unseen",
        "run_directory": str(run_dir),
        "run_tag": run_tag,
        "stop_strings": stop_strings,
        "max_steps": max_steps,
        "max_new_tokens": cfg["backend"]["max_new_tokens"],
    }

    backend = None
    try:
        # 未训练：与训练相同的 LoRA 初始化（同 seed）
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        torch.manual_seed(seed)
        untrained = get_peft_model(base, LoraConfig(
            r=cfg["model"].get("lora_rank", 16),
            lora_alpha=cfg["model"].get("lora_alpha", 32),
            lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=cfg["model"].get("lora_target_modules", "all-linear"),
        ))
        end_ids = resolve_end_token_ids(untrained, tokenizer)
        adapter_pre = run_dir / "vllm_adapter_unseen_pre"
        adapter_pre.mkdir(parents=True, exist_ok=True)
        backend = make_backend(
            cfg, tokenizer, end_ids, adapter_pre, f"unseen-pre-{run_dir.name}", stop_strings)
        pre = run_one(
            "untrained", untrained, tokenizer, backend, env_factory, dataset,
            max_steps, truncated_reward, extra)

        backend.close()
        del untrained, base
        torch.cuda.empty_cache()

        # 训练后
        base2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        trained = PeftModel.from_pretrained(base2, ckpt)
        end_ids = resolve_end_token_ids(trained, tokenizer)
        adapter_post = run_dir / "vllm_adapter_unseen_post"
        adapter_post.mkdir(parents=True, exist_ok=True)
        backend = make_backend(
            cfg, tokenizer, end_ids, adapter_post, f"unseen-post-{run_dir.name}", stop_strings)
        post = run_one(
            "trained", trained, tokenizer, backend, env_factory, dataset,
            max_steps, truncated_reward, {**extra, "checkpoint": str(ckpt)})
    finally:
        if backend is not None:
            getattr(backend, "close", lambda: None)()

    summary = {
        "run_directory": str(run_dir),
        "untrained_success": pre["evaluation_success_fraction"],
        "trained_success": post["evaluation_success_fraction"],
        "untrained_used_calc": pre.get("evaluation_environment_used_calculator_fraction"),
        "trained_used_calc": post.get("evaluation_environment_used_calculator_fraction"),
        "delta_success": post["evaluation_success_fraction"] - pre["evaluation_success_fraction"],
    }
    path = Path(f"outputs/calc_problems_unseen_compare_{run_tag}.json")
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
