"""对 GRPO / Dr. GRPO 六个最终检查点做完整 GSM8K test 贪心重评。

用法：
  python -m examples.reevaluate_gsm8k_comparison
  python -m examples.reevaluate_gsm8k_comparison outputs
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer, GenerationConfig

from examples.diagnose_gsm8k_splits import evaluate_split
from examples.gsm8k import make_dataset, reward_function, success_function
from grpx.backends.vllm import VllmBackend, sanitize_adapter_name_component
from grpx.environment import SingleTurnEnvironment

RUN_NAMES = [
    "gsm8k-grpo-truncfix-s0", "gsm8k-grpo-truncfix-s1", "gsm8k-grpo-truncfix-s2",
    "gsm8k-dr_grpo-truncfix-s0", "gsm8k-dr_grpo-truncfix-s1", "gsm8k-dr_grpo-truncfix-s2",
]


def load_adapter(backend, name, directory):
    backend._load_adapter(name, directory)
    backend.adapter_name = name
    backend._active_adapter_directory = directory


def unload_adapter(backend, name):
    if name is None:
        return
    backend._unload_adapter(name, raise_on_error=False)
    if backend.adapter_name == name:
        backend.adapter_name = None
        backend._active_adapter_directory = None


def main():
    outputs_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs").resolve()
    runs = []
    for name in RUN_NAMES:
        run_directory = outputs_root / name
        checkpoint = run_directory / "checkpoint-0060"
        if not (checkpoint / "adapter_model.safetensors").exists():
            raise SystemExit(f"缺少检查点：{checkpoint}")
        configuration = json.loads(
            (run_directory / "resolved_configuration.json").read_text(encoding="utf-8"))
        runs.append((name, run_directory, checkpoint, configuration))

    configuration = runs[0][3]
    model_name = configuration["model"]["name"]
    base_url = configuration["backend"].get("base_url", "http://localhost:8000")
    max_new_tokens = configuration["backend"].get("max_new_tokens", 768)

    dataset = make_dataset(split="test")
    print(f"完整测试集：{len(dataset)} 题")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    end_token_ids = generation_config.eos_token_id
    if end_token_ids is None:
        end_token_ids = tokenizer.eos_token_id
    if not isinstance(end_token_ids, list):
        end_token_ids = [end_token_ids]

    adapter_directory = outputs_root / "vllm_adapter_reeval_comparison"
    adapter_directory.mkdir(parents=True, exist_ok=True)
    backend = VllmBackend(
        tokenizer, set(end_token_ids),
        adapter_directory=adapter_directory,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_concurrent_requests=64,
        adapter_name="reeval-gsm8k",
    )
    environment_factory = lambda: SingleTurnEnvironment(
        reward_function, success_function=success_function)

    # 探测服务是否可达
    requests.get(f"{base_url}/v1/models", timeout=30).raise_for_status()

    results = {}
    previous_name = None
    try:
        for run_name, run_directory, checkpoint, _ in runs:
            unload_adapter(backend, previous_name)
            adapter_name = sanitize_adapter_name_component(f"reeval-{run_name}")
            print(f"\n=== {run_name}  checkpoint={checkpoint.name}  adapter={adapter_name} ===")
            load_adapter(backend, adapter_name, checkpoint)
            previous_name = adapter_name
            started = time.perf_counter()
            metrics = evaluate_split(backend, tokenizer, environment_factory, dataset)
            metrics["seconds"] = time.perf_counter() - started
            results[run_name] = metrics
            answer = metrics["answer_tokens"]
            mean_all = (
                (answer["correct"]["count"] * answer["correct"]["mean"]
                 + answer["incorrect"]["count"] * answer["incorrect"]["mean"])
                / metrics["n"]
            )
            print(
                f"success={metrics['success_fraction']:.4f}  "
                f"reward={metrics['mean_reward']:.4f}  "
                f"trunc={metrics['length_limit_fraction']:.4f}  "
                f"len_all={mean_all:.1f}  "
                f"len_ok={answer['correct']['mean']:.1f}  "
                f"len_bad={answer['incorrect']['mean']:.1f}  "
                f"sec={metrics['seconds']:.1f}"
            )
    finally:
        unload_adapter(backend, previous_name)
        backend.close()

    output_path = outputs_root / "gsm8k_comparison_full_test.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {output_path}")

    print("\n对照表：")
    print(f"{'run':<22} {'success':>8} {'len_ok':>8} {'len_bad':>8} {'trunc':>8}")
    for run_name in RUN_NAMES:
        metrics = results[run_name]
        answer = metrics["answer_tokens"]
        print(
            f"{run_name:<22} {metrics['success_fraction']:8.4f} "
            f"{answer['correct']['mean']:8.1f} {answer['incorrect']['mean']:8.1f} "
            f"{metrics['length_limit_fraction']:8.4f}"
        )


if __name__ == "__main__":
    main()
