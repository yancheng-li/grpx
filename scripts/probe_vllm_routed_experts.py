"""检查已启动的 vLLM 是否在 /v1/completions 里返回专家路由。

不安装、不升级任何环境。不要在 verl-vllm 里改 0.18。0.34 请另建环境再启动，例如：

    conda create -n vllm-0.34 python=3.12 -y
    conda activate vllm-0.34
    pip install vllm==0.34.0
    vllm serve allenai/OLMoE-1B-7B-0924-Instruct \
        --enable-return-routed-experts --enable-prefix-caching \
        --enable-lora --max-lora-rank 16 --generation-config vllm \
        --dtype bfloat16 --gpu-memory-utilization 0.80 \
        --max-model-len 2048 --port 8002

然后在原来的环境里：

    python scripts/probe_vllm_routed_experts.py --base-url http://127.0.0.1:8002

通过标准（对照 0.34.0 发布包里的约定，还没有在 OLMoE 上跑过）：

- 每个 choice 一段路由，长度是「提示词长度 + 回答长度 - 1」。
  前 prompt_len 行是提示词，其后是除最后一个生成 token 以外的回答。
- n=2 时两条的前 prompt_len 行相同，各自长度按自己的回答长度计算。
- 同一提示词再请求一次，长度仍然完整；并且服务端的前缀缓存命中计数增加。
- 加载现成的注意力适配器之后，字段还在，长度规则不变。

不启动服务，也不改 grpx 的后端。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
# 0.34 的前缀缓存按 64 个 token 一块。短于一块时命中计数不会增加，看不出缓存命中后记录还在不在。
PROMPT = list(range(10, 138))
ADAPTER_NAME = "olmoe-attn"
DEFAULT_ADAPTER = (
    "outputs/olmoe-d2d3-replay-off-s0/checkpoint-0060"
)
OLMOE_LAYERS = 16
OLMOE_TOPK = 8
OLMOE_EXPERTS = 64


def expected_rows(prompt_length, generated_length):
    return prompt_length + generated_length - 1


def decode_routed_experts(value):
    """返回 (array 或 None, 格式说明)。数组形状是 (位置, 层, k)。"""
    if value is None:
        return None, "missing"
    if isinstance(value, str):
        raw = base64.b64decode(value)
        array = np.load(io.BytesIO(raw))
        return np.asarray(array), "base64-npy"
    if isinstance(value, list):
        return np.asarray(value), "json-list"
    return None, type(value).__name__


def request(method, url, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    call = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(call, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "")
            return response.status, content_type, body
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def server_version(base_url):
    status, _, body = request("GET", f"{base_url}/version", timeout=10)
    if status != 200:
        return None
    try:
        return json.loads(body.decode()).get("version")
    except json.JSONDecodeError:
        return None


def complete(base_url, model, samples, temperature):
    status, _, body = request("POST", f"{base_url}/v1/completions", {
        "model": model,
        "prompt": PROMPT,
        "n": samples,
        "max_tokens": 8,
        "temperature": temperature,
        "top_p": 1.0,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
    })
    text = body.decode(errors="replace")
    if status != 200:
        raise RuntimeError(f"completions 返回 {status}：{text[:500]}")
    return json.loads(text)


def generated_length(choice):
    token_ids = choice.get("token_ids")
    if isinstance(token_ids, list) and token_ids:
        return len(token_ids)
    logprobs = (choice.get("logprobs") or {}).get("token_logprobs")
    if isinstance(logprobs, list) and logprobs:
        return len(logprobs)
    raise RuntimeError("choice 里没有 token_ids，无法知道回答长度")


def routing_of(choice):
    if "routed_experts" in choice:
        return decode_routed_experts(choice.get("routed_experts"))
    return None, "field-absent"


def check_shape(array, prompt_length, generated_length, label):
    failures = []
    rows = expected_rows(prompt_length, generated_length)
    if array.ndim != 3:
        return [f"{label} 不是三维数组，形状 {array.shape}"]
    if array.shape[0] != rows:
        failures.append(
            f"{label} 长度 {array.shape[0]}，期望 {rows}"
            f"（提示词 {prompt_length} + 回答 {generated_length} - 1）"
        )
    if array.shape[1] != OLMOE_LAYERS or array.shape[2] != OLMOE_TOPK:
        failures.append(
            f"{label} 层数或每层专家数是 {array.shape[1:]}，"
            f"期望 ({OLMOE_LAYERS}, {OLMOE_TOPK})"
        )
    if array.size and (array.min() < 0 or array.max() >= OLMOE_EXPERTS):
        failures.append(
            f"{label} 专家编号超出 0 到 {OLMOE_EXPERTS - 1}："
            f"{int(array.min())} 到 {int(array.max())}"
        )
    return failures


def prefix_hit_count(metrics_text):
    """只加命中次数这种计数。命中率是比例，重复请求时它不一定变大。"""
    total = 0.0
    names = []
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        lowered = name.lower()
        if "prefix" not in lowered or "hit" not in lowered:
            continue
        if "rate" in lowered or lowered.endswith((
            "_bucket", "_sum", "_count", "_created",
        )):
            continue
        try:
            total += float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        names.append(name)
    return bool(names), total, names


def read_metrics(base_url):
    status, content_type, body = request("GET", f"{base_url}/metrics", timeout=10)
    if status != 200 or "json" in content_type.lower():
        return None
    return body.decode(errors="replace")


def load_adapter(base_url, adapter_path):
    status, _, body = request("POST", f"{base_url}/v1/load_lora_adapter", {
        "lora_name": ADAPTER_NAME,
        "lora_path": adapter_path,
    })
    text = body.decode(errors="replace")
    if status == 200:
        return
    if status == 400 and "already been loaded" in text:
        return
    raise RuntimeError(f"加载适配器返回 {status}：{text[:500]}")


def report(name, failures):
    if failures:
        print(f"FAIL  {name}")
        for item in failures:
            print(f"      {item}")
        return False
    print(f"PASS  {name}")
    return True


def run_self_check():
    assert expected_rows(32, 8) == 39
    array = np.zeros((39, OLMOE_LAYERS, OLMOE_TOPK), dtype=np.uint8)
    array[-1] = 3
    assert check_shape(array, 32, 8, "self") == []
    short = np.zeros((8, OLMOE_LAYERS, OLMOE_TOPK), dtype=np.uint8)
    assert check_shape(short, 32, 8, "self")
    encoded = io.BytesIO()
    np.save(encoded, array)
    decoded, kind = decode_routed_experts(base64.b64encode(encoded.getvalue()).decode())
    assert kind == "base64-npy"
    assert decoded.shape == array.shape
    print("PASS  self-check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        run_self_check()
        return

    base_url = args.base_url.rstrip("/")
    version = server_version(base_url)
    print(f"server_version {version}")
    if version is not None and version.startswith("0.18"):
        raise SystemExit(
            "这是 vLLM 0.18，HTTP 响应里没有路由。不要升级 verl-vllm。"
        )

    passed = True
    first = complete(base_url, args.model, 1, 0.0)
    choice = first["choices"][0]
    generated = generated_length(choice)
    array, kind = routing_of(choice)
    print(
        f"n=1 format={kind} prompt_len={len(PROMPT)} gen_len={generated} "
        f"rows={None if array is None else int(array.shape[0])} "
        f"shape={None if array is None else tuple(int(n) for n in array.shape)}"
    )
    if array is None:
        passed = report("单请求长度", [f"没有 routed_experts（{kind}）"])
    else:
        passed = report(
            "单请求长度", check_shape(array, len(PROMPT), generated, "n=1")
        ) and passed

    several = complete(base_url, args.model, 2, 1.0)
    shared_failures = []
    prefixes = []
    if len(several.get("choices") or []) != 2:
        shared_failures.append(f"期望 2 条回答，得到 {len(several.get('choices') or [])}")
    for choice in several.get("choices") or []:
        generated = generated_length(choice)
        array, kind = routing_of(choice)
        print(
            f"n=2 choice={choice.get('index')} format={kind} gen_len={generated} "
            f"rows={None if array is None else int(array.shape[0])}"
        )
        if array is None:
            shared_failures.append(f"choice {choice.get('index')} 没有路由（{kind}）")
            continue
        shared_failures.extend(check_shape(
            array, len(PROMPT), generated, f"choice {choice.get('index')}"
        ))
        if array.shape[0] >= len(PROMPT):
            prefixes.append(array[:len(PROMPT)])
    if len(prefixes) == 2 and not np.array_equal(prefixes[0], prefixes[1]):
        shared_failures.append("两条回答的前 prompt_len 行不一样")
    passed = report("n>1 提示词各带一份", shared_failures) and passed

    before = read_metrics(base_url)
    before_found, before_hits, before_names = (
        (False, 0, []) if before is None else prefix_hit_count(before)
    )
    repeated = complete(base_url, args.model, 1, 0.0)
    after = read_metrics(base_url)
    after_found, after_hits, after_names = (
        (False, 0, []) if after is None else prefix_hit_count(after)
    )
    choice = repeated["choices"][0]
    generated = generated_length(choice)
    array, kind = routing_of(choice)
    print(
        f"prefix-repeat format={kind} gen_len={generated} "
        f"rows={None if array is None else int(array.shape[0])} "
        f"hit_count {before_hits} -> {after_hits}"
    )
    prefix_failures = []
    if array is None:
        prefix_failures.append(f"再次请求没有路由（{kind}）")
    else:
        prefix_failures.extend(check_shape(
            array, len(PROMPT), generated, "再次请求"
        ))
    if not before_found or not after_found:
        prefix_failures.append(
            "没有读到前缀缓存命中计数，不能确认这次是命中。"
            f"看到的计数：{before_names or after_names or '无'}"
        )
    elif after_hits <= before_hits:
        prefix_failures.append("前缀缓存命中计数没有增加")
    passed = report("前缀缓存命中后提示词还在", prefix_failures) and passed

    if args.skip_lora:
        print("SKIP  加载适配器")
    else:
        adapter_path = str(Path(args.adapter).resolve())
        if not Path(adapter_path).is_dir():
            passed = report("加载适配器后仍返回路由", [f"适配器目录不存在：{adapter_path}"]) and passed
        else:
            load_adapter(base_url, adapter_path)
            loaded = complete(base_url, ADAPTER_NAME, 1, 0.0)
            choice = loaded["choices"][0]
            generated = generated_length(choice)
            array, kind = routing_of(choice)
            print(
                f"lora format={kind} model={loaded.get('model')} gen_len={generated} "
                f"rows={None if array is None else int(array.shape[0])}"
            )
            lora_failures = []
            if array is None:
                lora_failures.append(f"加载适配器后没有路由（{kind}）")
            else:
                lora_failures.extend(check_shape(
                    array, len(PROMPT), generated, "适配器"
                ))
            passed = report("加载适配器后仍返回路由", lora_failures) and passed

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
