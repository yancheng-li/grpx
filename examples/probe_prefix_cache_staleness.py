"""验证：热 prefix cache 下 load_inplace 后贪心是否仍停在旧策略。

1) 同步随机初始化 LoRA，评测（应接近基线 0.110）→ 填满 cache
2) 同步 checkpoint-0005，再评测（若仍是 0.110，说明 cache 未失效）
"""
import json
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.guess_number import GuessNumberEnvironment, make_dataset
from grpx.backends.common import resolve_end_token_ids
from grpx.backends.vllm import VllmBackend
from grpx.evaluation import evaluate


def main():
    run_directory = Path(sys.argv[1])
    configuration = json.loads((run_directory / "resolved_configuration.json").read_text(
        encoding="utf-8"))
    checkpoint = run_directory / "checkpoint-0005"
    model_name = configuration["model"]["name"]
    evaluation_options = configuration["task"].get(
        "evaluation_dataset_options", {"size": 200, "seed": 1})
    environment_options = configuration["task"].get("environment_options", {})
    max_steps = configuration["training"].get("max_steps", 6)
    max_new_tokens = configuration["backend"].get("max_new_tokens", 32)
    dataset = make_dataset(**evaluation_options)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    adapter_directory = run_directory / "vllm_adapter_cache_probe"
    adapter_directory.mkdir(parents=True, exist_ok=True)

    def make_backend(model):
        return VllmBackend(
            tokenizer, resolve_end_token_ids(model, tokenizer),
            adapter_directory=adapter_directory,
            base_url=configuration["backend"].get("base_url", "http://localhost:8000"),
            max_new_tokens=max_new_tokens,
            max_concurrent_requests=64,
        )

    def run(backend, label):
        started = time.perf_counter()
        metrics = evaluate(
            backend, tokenizer,
            lambda: GuessNumberEnvironment(**environment_options),
            dataset, max_steps=max_steps,
        )
        print(f"{label}: success={metrics['evaluation_success_fraction']:.4f}  "
              f"reward={metrics['evaluation_mean_reward']:.4f}  "
              f"seconds={time.perf_counter() - started:.1f}")
        return metrics

    print("同步随机 LoRA（seed=0）并评测，预热 cache……")
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    torch.manual_seed(0)
    random_model = get_peft_model(base, LoraConfig(
        r=configuration["model"].get("lora_rank", 16),
        lora_alpha=configuration["model"].get("lora_alpha", 32),
        lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=configuration["model"].get("lora_target_modules", "all-linear"),
    ))
    backend = make_backend(random_model)
    backend.synchronize_policy(random_model)
    run(backend, "after-random-lora")

    print("同步 checkpoint-0005（不手动清 cache）……")
    trained = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda(),
        checkpoint,
    )
    backend.synchronize_policy(trained)
    metrics = run(backend, "after-trained-lora-no-cache-reset")
    if abs(metrics["evaluation_success_fraction"] - 0.11) < 0.01:
        print("结论：同步后仍是基线 → prefix cache 很可能未失效")
    elif abs(metrics["evaluation_success_fraction"] - 0.185) < 0.02:
        print("结论：同步后已是训练后水平 → 当前服务上 cache 不是问题")
    else:
        print("结论：落在中间，需继续看")
    backend.close()


if __name__ == "__main__":
    main()
