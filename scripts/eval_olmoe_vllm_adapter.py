"""OLMoE：vLLM 上 base vs LoRA 贪心评测（digits=2, n=200）。"""
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer

from examples.arithmetic import is_correct, make_dataset, reward_function
from grpx.backends.common import resolve_end_token_ids
from grpx.backends.vllm import VllmBackend
from grpx.environment import SingleTurnEnvironment
from grpx.evaluation import evaluate

BASE = "allenai/OLMoE-1B-7B-0924-Instruct"
URL = "http://localhost:8000"
ADAPTER = Path("outputs/olmoe-arith-d2-s0/checkpoint-0030").resolve()
LORA_NAME = "olmoe-s0"

tokenizer = AutoTokenizer.from_pretrained(BASE)
# 用基座构造一次只为拿 end_token_ids；不占 GPU
from transformers import AutoModelForCausalLM
import torch
probe = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cpu")
end_ids = resolve_end_token_ids(probe, tokenizer)
del probe

dataset = make_dataset(size=200, seed=1, digits=2, prompt_variant="english_strict")

backend = VllmBackend(
    tokenizer, end_ids,
    adapter_directory=str(ADAPTER.parent / "vllm_eval_scratch"),
    base_url=URL,
    adapter_name=BASE,  # 先评基座
    expected_base_model=BASE,
    max_new_tokens=128,
)
backend.verify_server_base_model()

environment_factory = lambda: SingleTurnEnvironment(
    reward_function, success_function=is_correct)

print("=== base (no adapter) ===", flush=True)
base_metrics = evaluate(backend, tokenizer, environment_factory, dataset, max_steps=1)
print(json.dumps(base_metrics, ensure_ascii=False, indent=2), flush=True)

print("=== load adapter ===", flush=True)
resp = requests.post(
    f"{URL}/v1/load_lora_adapter",
    json={"lora_name": LORA_NAME, "lora_path": str(ADAPTER)},
    timeout=120,
)
print(resp.status_code, resp.text[:500], flush=True)
resp.raise_for_status()

backend.adapter_name = LORA_NAME
print("=== with adapter ===", flush=True)
lora_metrics = evaluate(backend, tokenizer, environment_factory, dataset, max_steps=1)
print(json.dumps(lora_metrics, ensure_ascii=False, indent=2), flush=True)

out = Path("outputs/olmoe_vllm_adapter_smoke.json")
out.write_text(json.dumps({"base": base_metrics, "lora": lora_metrics}, indent=2), encoding="utf-8")
print("wrote", out)