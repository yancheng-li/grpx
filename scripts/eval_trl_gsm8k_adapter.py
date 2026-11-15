"""用 grpx 评测口径，对 TRL 保存的 LoRA 适配器做 GSM8K 完整测试集贪心评测。

用法（项目根目录，grpx 环境或装了 peft/transformers 的环境均可）：
  PYTHONPATH=. python scripts/eval_trl_gsm8k_adapter.py outputs/trl-gsm8k-s0
  PYTHONPATH=. python scripts/eval_trl_gsm8k_adapter.py outputs/trl-gsm8k-s0 --size 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.gsm8k import make_dataset, reward_function, success_function
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.environment import SingleTurnEnvironment
from grpx.evaluation import evaluate

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_directory", type=Path)
    parser.add_argument("--size", type=int, default=None,
                        help="测试集题数；默认完整测试集")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    arguments = parser.parse_args()

    adapter_directory = arguments.adapter_directory
    if not (adapter_directory / "adapter_config.json").exists():
        raise SystemExit(f"找不到适配器：{adapter_directory}/adapter_config.json")

    dtype = torch.bfloat16 if arguments.dtype == "bfloat16" else torch.float32
    print(f"加载基座 {MODEL_NAME}（{arguments.dtype}）……", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype, device_map="cuda")
    print(f"加载适配器 {adapter_directory}……", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_directory))
    model.eval()

    backend = HuggingFaceBackend(model, tokenizer, max_new_tokens=arguments.max_new_tokens)
    dataset = make_dataset(split="test", size=arguments.size, seed=arguments.seed)
    print(f"评测 {len(dataset)} 题（贪心）……", flush=True)

    metrics = evaluate(
        backend, tokenizer,
        lambda: SingleTurnEnvironment(reward_function, success_function=success_function),
        dataset,
        max_steps=1,
        truncated_reward=0.0,
    )
    out = adapter_directory / "grpx_fulltest_eval.json"
    payload = {
        "adapter_directory": str(adapter_directory.resolve()),
        "n": len(dataset),
        "max_new_tokens": arguments.max_new_tokens,
        "dtype": arguments.dtype,
        **metrics,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
