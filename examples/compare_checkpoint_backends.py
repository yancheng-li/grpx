"""用两个后端评测同一检查点，区分「权重没训出来」和「服务端没体现」。

用法：python -m examples.compare_checkpoint_backends outputs/20261111-232923
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.guess_number import GuessNumberEnvironment, make_dataset
from grpx.backends.common import resolve_end_token_ids
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.backends.vllm import VllmBackend
from grpx.evaluation import evaluate

EXPECTED_TRAINED = {"evaluation_success_fraction": 0.185, "evaluation_mean_reward": 0.368}
BASELINE = {"evaluation_success_fraction": 0.110, "evaluation_mean_reward": 0.1545}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(label, metrics, seconds):
    success = metrics["evaluation_success_fraction"]
    reward = metrics["evaluation_mean_reward"]
    print(f"{label}: success={success:.4f}  mean_reward={reward:.4f}  "
          f"split_quality={metrics.get('evaluation_environment_split_quality')}  "
          f"informative={metrics.get('evaluation_environment_informative_step_fraction')}  "
          f"seconds={seconds:.1f}")
    return success, reward


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法：python examples/compare_checkpoint_backends.py <运行目录>")
    run_directory = Path(sys.argv[1])
    configuration = json.loads((run_directory / "resolved_configuration.json").read_text(
        encoding="utf-8"))
    checkpoint = run_directory / "checkpoint-0005"
    if not (checkpoint / "training_state.pt").exists():
        raise SystemExit(f"找不到 {checkpoint}/training_state.pt")

    evaluation_options = configuration["task"].get(
        "evaluation_dataset_options", {"size": 200, "seed": 1})
    environment_options = configuration["task"].get("environment_options", {})
    max_steps = configuration["training"].get("max_steps", 6)
    max_new_tokens = configuration["backend"].get("max_new_tokens", 32)
    model_name = configuration["model"]["name"]
    dataset = make_dataset(**evaluation_options)

    adapter_weights = run_directory / "vllm_adapter" / "adapter_model.safetensors"
    checkpoint_weights = checkpoint / "adapter_model.safetensors"
    if adapter_weights.exists() and checkpoint_weights.exists():
        left, right = file_sha256(adapter_weights), file_sha256(checkpoint_weights)
        print(f"adapter sha256 相同={left == right}")
        print(f"  vllm_adapter: {left}")
        print(f"  checkpoint:   {right}")

    print(f"加载检查点 {checkpoint} ……")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    model = PeftModel.from_pretrained(base, checkpoint)
    # 与训练结束校验对齐
    checksum = sum(parameter.detach().double().abs().sum().item()
                   for parameter in model.parameters() if parameter.requires_grad)
    print(f"trainable_parameter_checksum={checksum:.6f}")

    hf_backend = HuggingFaceBackend(model, tokenizer, max_new_tokens=max_new_tokens)
    started = time.perf_counter()
    hf_metrics = evaluate(
        hf_backend, tokenizer,
        lambda: GuessNumberEnvironment(**environment_options),
        dataset, max_steps=max_steps,
    )
    summarize("HuggingFace(float32)", hf_metrics, time.perf_counter() - started)

    adapter_directory = run_directory / "vllm_adapter_reeval"
    adapter_directory.mkdir(parents=True, exist_ok=True)
    vllm_backend = VllmBackend(
        tokenizer, resolve_end_token_ids(model, tokenizer),
        adapter_directory=adapter_directory,
        base_url=configuration["backend"].get("base_url", "http://localhost:8000"),
        max_new_tokens=max_new_tokens,
        max_concurrent_requests=64,
    )
    print(f"同步检查点 LoRA 到 vLLM（{adapter_directory.resolve()}）……")
    vllm_backend.synchronize_policy(model)
    started = time.perf_counter()
    vllm_metrics = evaluate(
        vllm_backend, tokenizer,
        lambda: GuessNumberEnvironment(**environment_options),
        dataset, max_steps=max_steps,
    )
    summarize("vLLM", vllm_metrics, time.perf_counter() - started)
    vllm_backend.close()

    hf_success = hf_metrics["evaluation_success_fraction"]
    vllm_success = vllm_metrics["evaluation_success_fraction"]
    print("解读：")
    print(f"  对照训练后期望 ≈ {EXPECTED_TRAINED['evaluation_success_fraction']:.3f} / "
          f"{EXPECTED_TRAINED['evaluation_mean_reward']:.3f}")
    print(f"  对照基线 ≈ {BASELINE['evaluation_success_fraction']:.3f} / "
          f"{BASELINE['evaluation_mean_reward']}")
    if abs(hf_success - EXPECTED_TRAINED["evaluation_success_fraction"]) < 0.02 and abs(
            vllm_success - BASELINE["evaluation_success_fraction"]) < 0.01:
        print("  → 权重是对的，服务端没有体现出来（同步/缓存/精度问题）")
    elif abs(hf_success - BASELINE["evaluation_success_fraction"]) < 0.01 and abs(
            vllm_success - BASELINE["evaluation_success_fraction"]) < 0.01:
        print("  → 两边都是基线：训练后的权重本身还没让贪心输出翻转")
    elif abs(hf_success - EXPECTED_TRAINED["evaluation_success_fraction"]) < 0.02 and abs(
            vllm_success - EXPECTED_TRAINED["evaluation_success_fraction"]) < 0.02:
        print("  → 两边都离开基线且一致：检查点权重与服务端同步都正常")
    else:
        print("  → 结果不落在预设分型，需结合具体数值继续看")


if __name__ == "__main__":
    main()
