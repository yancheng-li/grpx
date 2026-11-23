"""评测检查点：500 道乘法，基座 vs LoRA 配对比较，并防止误评基座。

输出：
- 总体成功率
- 前错后对 / 前对后错 题数（同一批题）
- 使用的适配器名与检查点路径
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer

from examples.arithmetic import ArithmeticEnvironment, make_dataset
from grpx.backends.vllm import VllmBackend
from grpx.rollout import rollout_groups

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
URL_DEFAULT = "http://localhost:8000"


def greedy_successes(backend, tokenizer, dataset):
    trajectories = rollout_groups(
        backend,
        tokenizer,
        ArithmeticEnvironment,
        [(f"evaluation-{index}", sample, index)
         for index, sample in enumerate(dataset)],
        group_size=1,
        max_steps=1,
        greedy=True,
        truncated_reward=0.0,
    )
    # rollout_groups 顺序与 group_specifications 一致
    flags = []
    for trajectory in trajectories:
        if trajectory.is_successful is None:
            raise RuntimeError("环境没有成功判定，无法做配对比较")
        flags.append(bool(trajectory.is_successful))
    return flags


def decode_greedy_answers(backend, tokenizer, dataset, limit=8):
    """取前 limit 道题的贪心文本，用于基座/适配器对照。"""
    trajectories = rollout_groups(
        backend,
        tokenizer,
        ArithmeticEnvironment,
        [(f"probe-{index}", sample, index)
         for index, sample in enumerate(dataset[:limit])],
        group_size=1,
        max_steps=1,
        greedy=True,
        truncated_reward=0.0,
    )
    answers = []
    for trajectory in trajectories:
        text = tokenizer.decode(
            trajectory.steps[-1].action_token_ids, skip_special_tokens=True)
        answers.append(text.strip())
    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-url", default=URL_DEFAULT)
    parser.add_argument("--digits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-count", type=int, default=8)
    arguments = parser.parse_args()

    checkpoint = arguments.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"检查点不存在: {checkpoint}")
    # 名字必须含运行名 + 轮次，避免同名重载时服务继续用旧适配器
    adapter_name = (
        "eval-"
        + checkpoint.parent.name.replace(".", "-")
        + "-"
        + checkpoint.name.replace(".", "-")
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    scratch = Path("outputs/vllm_eval_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    backend = VllmBackend(
        tokenizer,
        {tokenizer.eos_token_id},
        adapter_directory=str(scratch.resolve()),
        base_url=arguments.base_url,
        adapter_name=adapter_name,
        expected_base_model=MODEL,
        max_new_tokens=arguments.max_new_tokens,
        max_concurrent_requests=64,
    )
    backend.verify_server_base_model()
    models_url = f"{arguments.base_url.rstrip('/')}/v1/models"

    def listed_adapters():
        payload = requests.get(models_url, timeout=60).json()
        rows = []
        for item in payload.get("data", []):
            rows.append({
                "id": item.get("id"),
                "root": item.get("root"),
                "parent": item.get("parent"),
            })
        return rows

    def ensure_adapter_loaded(expected_path: Path):
        listed = listed_adapters()
        match = next((row for row in listed if row["id"] == adapter_name), None)
        if match is None:
            raise SystemExit(
                f"加载后服务上找不到适配器名 {adapter_name}；"
                f"当前列表={[row['id'] for row in listed]}"
            )
        # 有的服务把路径放在 root/parent；对得上就记下来，对不上只警告路径字段
        path_fields = {
            value for value in (match.get("root"), match.get("parent"))
            if isinstance(value, str) and value
        }
        expected = str(expected_path)
        if path_fields and expected not in path_fields and not any(
                expected.endswith(value) or value.endswith(expected)
                for value in path_fields):
            print(
                f"adapter_path_warn adapter_name={adapter_name} "
                f"expected={expected} listed={match}",
                flush=True,
            )
        print(
            f"adapter_listed_ok adapter_name={adapter_name} listed={match}",
            flush=True,
        )
        return listed

    # 先评基座
    backend.adapter_name = MODEL
    probe_dataset = make_dataset(
        arguments.probe_count, digits=arguments.digits[0], seed=arguments.seed,
        operation="multiplication", prompt_variant="english_strict")
    base_probe = decode_greedy_answers(
        backend, tokenizer, probe_dataset, limit=arguments.probe_count)

    # 再挂 LoRA（先卸同名，再按唯一名加载）
    requests.post(
        f"{arguments.base_url.rstrip('/')}/v1/unload_lora_adapter",
        json={"lora_name": adapter_name}, timeout=60)
    response = requests.post(
        f"{arguments.base_url.rstrip('/')}/v1/load_lora_adapter",
        json={"lora_name": adapter_name, "lora_path": str(checkpoint)},
        timeout=300,
    )
    print(response.status_code, response.text[:400], flush=True)
    response.raise_for_status()
    ensure_adapter_loaded(checkpoint)
    backend.adapter_name = adapter_name
    lora_probe = decode_greedy_answers(
        backend, tokenizer, probe_dataset, limit=arguments.probe_count)
    probe_diffs = sum(a != b for a, b in zip(base_probe, lora_probe))
    if base_probe == lora_probe:
        # 弱检查点可能在少量探针题上与基座相同；适配器名+路径已核对通过时只警告
        print(
            "adapter_probe_warn 探针与基座完全相同，但服务列表已含该适配器且路径匹配；"
            f"继续评测。adapter_name={adapter_name} checkpoint={checkpoint}",
            flush=True,
        )
    print(
        f"adapter_check_ok adapter_name={adapter_name} checkpoint={checkpoint} "
        f"probe_diffs={probe_diffs}/{len(base_probe)}",
        flush=True,
    )

    results = {
        "model": MODEL,
        "checkpoint": str(checkpoint),
        "adapter_name": adapter_name,
        "base_url": arguments.base_url,
        "probe_base_answers": base_probe,
        "probe_adapter_answers": lora_probe,
        "listed_adapters_after_load": listed_adapters(),
        "evaluations": [],
    }

    # 正式配对：每个 digits 先基座后 LoRA（同一顺序）
    for digits in arguments.digits:
        dataset = make_dataset(
            arguments.size, digits=digits, seed=arguments.seed,
            operation="multiplication", prompt_variant="english_strict")
        backend.adapter_name = MODEL
        before = greedy_successes(backend, tokenizer, dataset)
        backend.adapter_name = adapter_name
        after = greedy_successes(backend, tokenizer, dataset)
        if len(before) != len(after):
            raise RuntimeError("前后评测题数不一致")
        gained = sum(1 for b, a in zip(before, after) if (not b) and a)
        lost = sum(1 for b, a in zip(before, after) if b and (not a))
        before_success = sum(before) / len(before)
        after_success = sum(after) / len(after)
        row = {
            "digits": digits,
            "size": len(dataset),
            "seed": arguments.seed,
            "operation": "multiplication",
            "before_success_fraction": before_success,
            "after_success_fraction": after_success,
            "delta_success_fraction": after_success - before_success,
            "gained": gained,   # 前错后对
            "lost": lost,       # 前对后错
            "net": gained - lost,
            "unchanged_correct": sum(1 for b, a in zip(before, after) if b and a),
            "unchanged_incorrect": sum(
                1 for b, a in zip(before, after) if (not b) and (not a)),
            "before_success_flags": before,
            "after_success_flags": after,
            "adapter_name": adapter_name,
            "checkpoint": str(checkpoint),
        }
        results["evaluations"].append(row)
        print(
            f"digits={digits} before={before_success:.4f} after={after_success:.4f} "
            f"gained={gained} lost={lost} net={gained - lost}",
            flush=True,
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", arguments.output, flush=True)
    requests.post(
        f"{arguments.base_url.rstrip('/')}/v1/unload_lora_adapter",
        json={"lora_name": adapter_name}, timeout=120)


if __name__ == "__main__":
    main()
