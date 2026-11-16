"""把专家组 LoRA 检查点加载到本地 vLLM，评 digits=2 n=200。

对照：同设定下注意力 LoRA 曾到约 0.97；基座约 0.38。
若专家层 LoRA 在 vLLM 上生效，成绩应明显高于基座。

用法（先起服务，--max-lora-rank 至少 32）：
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
  vllm serve allenai/OLMoE-1B-7B-0924-Instruct \\
    --enable-lora --max-lora-rank 32 \\
    --generation-config vllm --gpu-memory-utilization 0.45 \\
    --max-model-len 1024 --port 8000
  PYTHONPATH=. python scripts/eval_olmoe_experts_vllm.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.arithmetic import is_correct, make_dataset, reward_function
from grpx.backends.common import resolve_end_token_ids
from grpx.backends.vllm import VllmBackend
from grpx.environment import SingleTurnEnvironment
from grpx.evaluation import evaluate

BASE = "allenai/OLMoE-1B-7B-0924-Instruct"
DEFAULT_CHECKPOINTS = [
    "outputs/rent_pull/runs/olmoe-arith-d2d3-experts-s0/checkpoint-0060",
    "outputs/rent_pull/runs/olmoe-arith-d2d3-experts-s1/checkpoint-0060",
    "outputs/rent_pull/runs/olmoe-arith-d2d3-experts-s2/checkpoint-0060",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--output",
        default="outputs/olmoe_experts_vllm_adapter_eval.json",
    )
    return parser.parse_args()


def unload_adapter(base_url: str, name: str) -> None:
    try:
        requests.post(
            f"{base_url}/v1/unload_lora_adapter",
            json={"lora_name": name},
            timeout=60,
        )
    except requests.RequestException:
        pass


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    probe = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cpu")
    end_ids = resolve_end_token_ids(probe, tokenizer)
    del probe

    dataset = make_dataset(size=args.size, seed=args.seed, digits=args.digits)
    scratch = Path("outputs/vllm_experts_eval_scratch").resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    backend = VllmBackend(
        tokenizer,
        end_ids,
        adapter_directory=str(scratch),
        base_url=args.base_url,
        adapter_name=BASE,
        expected_base_model=BASE,
        max_new_tokens=args.max_new_tokens,
    )
    backend.verify_server_base_model()
    environment_factory = lambda: SingleTurnEnvironment(
        reward_function, success_function=is_correct)

    results = {"base_url": args.base_url, "digits": args.digits, "size": args.size}

    print("=== base (no adapter) ===", flush=True)
    base_metrics = evaluate(
        backend, tokenizer, environment_factory, dataset, max_steps=1)
    print(json.dumps(base_metrics, ensure_ascii=False, indent=2), flush=True)
    results["base"] = base_metrics

    adapters = []
    for index, raw in enumerate(args.checkpoints):
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        name = f"experts-s{index}"
        print(f"=== load {name} from {path} ===", flush=True)
        unload_adapter(args.base_url, name)
        response = requests.post(
            f"{args.base_url}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": str(path)},
            timeout=180,
        )
        print(response.status_code, response.text[:800], flush=True)
        entry = {
            "checkpoint": str(path),
            "lora_name": name,
            "load_status": response.status_code,
            "load_body": response.text[:2000],
        }
        if response.status_code >= 400:
            entry["error"] = "load_failed"
            adapters.append(entry)
            continue
        backend.adapter_name = name
        metrics = evaluate(
            backend, tokenizer, environment_factory, dataset, max_steps=1)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        entry["metrics"] = metrics
        adapters.append(entry)
        unload_adapter(args.base_url, name)
        backend.adapter_name = BASE

    results["adapters"] = adapters
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
