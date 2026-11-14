"""基线 vs 最终检查点，在三份 GSM8K 题目上做贪心评测。

用法：
  python -m examples.diagnose_gsm8k_splits outputs/20261112-023205
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig

from examples.gsm8k import make_dataset, reward_function, success_function
from grpx.backends.vllm import VllmBackend
from grpx.environment import SingleTurnEnvironment
from grpx.rollout import rollout_groups


def summarize_lengths(trajectories):
    correct, incorrect = [], []
    for trajectory in trajectories:
        length = sum(len(step.action_token_ids) for step in trajectory.steps)
        if trajectory.is_successful:
            correct.append(length)
        else:
            incorrect.append(length)

    def stats(values):
        if not values:
            return {"count": 0, "mean": None, "median": None}
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    return {"correct": stats(correct), "incorrect": stats(incorrect)}


def evaluate_split(backend, tokenizer, environment_factory, dataset, max_steps=1):
    trajectories = rollout_groups(
        backend, tokenizer, environment_factory,
        [(f"evaluation-{index}", sample, index) for index, sample in enumerate(dataset)],
        group_size=1, max_steps=max_steps, greedy=True,
    )
    count = len(trajectories)
    successes = sum(1 for trajectory in trajectories if trajectory.is_successful)
    length_limits = sum(trajectory.termination_reason == "length_limit"
                        for trajectory in trajectories)
    return {
        "n": count,
        "success_fraction": successes / count,
        "mean_reward": sum(trajectory.total_reward for trajectory in trajectories) / count,
        "length_limit_fraction": length_limits / count,
        "answer_tokens": summarize_lengths(trajectories),
    }


def main():
    run_directory = Path(sys.argv[1] if len(sys.argv) > 1 else
                         "outputs/20261112-023205").resolve()
    configuration = json.loads(
        (run_directory / "resolved_configuration.json").read_text(encoding="utf-8"))
    checkpoint = run_directory / "checkpoint-0100"
    if not (checkpoint / "adapter_model.safetensors").exists():
        raise SystemExit(f"找不到最终检查点：{checkpoint}")

    model_name = configuration["model"]["name"]
    base_url = configuration["backend"].get("base_url", "http://localhost:8000")
    max_new_tokens = configuration["backend"].get("max_new_tokens", 768)
    # 训练第 0 轮评测用的就是未更新的随机 LoRA；这里用底模对照，并与已同步的最终适配器对比
    final_adapter = "policy-20261112-023205-101"

    splits = {
        "test_full": make_dataset(split="test"),
        "train_500": make_dataset(split="train", size=500, seed=0),
        "test_500": make_dataset(split="test", size=500, seed=0),
    }
    print("题目规模：")
    for name, dataset in splits.items():
        print(f"  {name}: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    end_token_ids = generation_config.eos_token_id
    if end_token_ids is None:
        end_token_ids = tokenizer.eos_token_id
    if not isinstance(end_token_ids, list):
        end_token_ids = [end_token_ids]
    end_token_ids = set(end_token_ids)

    adapter_directory = run_directory / "vllm_adapter_diagnose"
    adapter_directory.mkdir(parents=True, exist_ok=True)
    backend = VllmBackend(
        tokenizer, end_token_ids,
        adapter_directory=adapter_directory,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_concurrent_requests=64,
        adapter_name="diagnose",
    )
    environment_factory = lambda: SingleTurnEnvironment(
        reward_function, success_function=success_function)

    policies = [
        ("baseline_base_model", model_name),
        ("final_checkpoint_adapter", final_adapter),
    ]
    # 若服务上没有最终适配器名，则从检查点重新同步
    import requests
    model_ids = {item["id"] for item in requests.get(f"{base_url}/v1/models", timeout=30).json()["data"]}
    if final_adapter not in model_ids:
        print(f"服务上没有 {final_adapter}，从 {checkpoint.name} 同步 ……")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model = PeftModel.from_pretrained(base, checkpoint)
        backend.synchronize_policy(model)
        policies[1] = ("final_checkpoint_adapter", backend.adapter_name)
        del model, base
        torch.cuda.empty_cache()

    results = {}
    for policy_label, model_id in policies:
        backend.adapter_name = model_id
        print(f"\n=== {policy_label}  model={model_id} ===")
        results[policy_label] = {}
        for split_name, dataset in splits.items():
            started = time.perf_counter()
            metrics = evaluate_split(backend, tokenizer, environment_factory, dataset)
            metrics["seconds"] = time.perf_counter() - started
            results[policy_label][split_name] = metrics
            answer = metrics["answer_tokens"]
            print(
                f"{split_name}: success={metrics['success_fraction']:.4f}  "
                f"reward={metrics['mean_reward']:.4f}  "
                f"trunc={metrics['length_limit_fraction']:.4f}  "
                f"sec={metrics['seconds']:.1f}  "
                f"len_ok={answer['correct']['mean']}  "
                f"len_bad={answer['incorrect']['mean']}"
            )

    output_path = run_directory / "diagnose_splits.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {output_path}")

    print("\n对照表（success_fraction）：")
    header = f"{'split':<12} {'baseline':>10} {'final':>10} {'delta':>10}"
    print(header)
    for split_name in splits:
        left = results["baseline_base_model"][split_name]["success_fraction"]
        right = results["final_checkpoint_adapter"][split_name]["success_fraction"]
        print(f"{split_name:<12} {left:10.4f} {right:10.4f} {right - left:+10.4f}")

    backend.close()


if __name__ == "__main__":
    main()
