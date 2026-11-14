"""把检查点 LoRA 按倍数放大后同步到 vLLM，看贪心是否离开基线。

用法：python -m examples.probe_vllm_lora_scale outputs/20261111-232923
"""
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.guess_number import GuessNumberEnvironment, make_dataset
from grpx.backends.common import resolve_end_token_ids
from grpx.backends.vllm import VllmBackend
from grpx.evaluation import evaluate

SCALES = [1.0, 10.0, 100.0]


def lora_stats(model):
    values = [parameter.detach().float().abs()
              for name, parameter in model.named_parameters()
              if "lora_" in name]
    flat = torch.cat([value.reshape(-1) for value in values])
    return {
        "max": flat.max().item(),
        "mean": flat.mean().item(),
        "median": flat.median().item(),
        "p99": flat.quantile(0.99).item(),
    }


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
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    model = PeftModel.from_pretrained(base, checkpoint)
    # 保留一份原始 LoRA，便于按倍数重放
    originals = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    print("checkpoint LoRA stats:", lora_stats(model))

    adapter_directory = run_directory / "vllm_adapter_scale_probe"
    adapter_directory.mkdir(parents=True, exist_ok=True)
    backend = VllmBackend(
        tokenizer, resolve_end_token_ids(model, tokenizer),
        adapter_directory=adapter_directory,
        base_url=configuration["backend"].get("base_url", "http://localhost:8000"),
        max_new_tokens=max_new_tokens,
        max_concurrent_requests=64,
    )

    for scale in SCALES:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in originals:
                    parameter.copy_(originals[name] * scale)
        print(f"\nscale={scale:g}  LoRA stats={lora_stats(model)}")
        backend.synchronize_policy(model)
        started = time.perf_counter()
        metrics = evaluate(
            backend, tokenizer,
            lambda: GuessNumberEnvironment(**environment_options),
            dataset, max_steps=max_steps,
        )
        print(f"  vLLM success={metrics['evaluation_success_fraction']:.4f}  "
              f"mean_reward={metrics['evaluation_mean_reward']:.4f}  "
              f"seconds={time.perf_counter() - started:.1f}")

    backend.close()


if __name__ == "__main__":
    main()
