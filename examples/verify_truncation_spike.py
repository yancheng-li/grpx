"""从 Dr. GRPO s2 的 checkpoint-0050 在运行副本上恢复，验证截断尖峰。

用法（在副本目录上）：
  # 可选：旧规则，只跑到能看到截断样本
  python -m examples.verify_truncation_spike prepare

  # 对照：旧规则，从 50 再跑 10 轮，重复 3 次
  python -m examples.verify_truncation_spike control --trials 3

  # 新规则：truncated_reward=0，同样重复 3 次
  python -m examples.verify_truncation_spike fixed --trials 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests

from grpx.backends.vllm import sanitize_adapter_name_component

SOURCE = Path("outputs/gsm8k-dr_grpo-s2")
WORK_ROOT = Path("outputs/spike-verify")


def copy_run(tag: str, source: Path) -> Path:
    destination = WORK_ROOT / tag
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        "vllm_adapter", "vllm_adapter_*", "trajectory_samples.jsonl"))
    # 删掉 0050 之后的检查点，强制从 50 恢复
    for path in destination.glob("checkpoint-*"):
        if path.name.endswith(".partial"):
            shutil.rmtree(path)
            continue
        completed = int(path.name.split("-")[1])
        if completed > 50:
            shutil.rmtree(path)
    return destination


def base_url_from(run_directory: Path) -> str:
    configuration = json.loads(
        (run_directory / "resolved_configuration.json").read_text(encoding="utf-8"))
    return configuration.get("backend", {}).get("base_url", "http://localhost:8000").rstrip("/")


def unload_run_adapters(run_directory: Path):
    """清掉同名前缀残留，避免副本目录名撞上服务端已加载的适配器。"""
    prefix = f"policy-{sanitize_adapter_name_component(run_directory.name)}-"
    url = base_url_from(run_directory)
    models = requests.get(f"{url}/v1/models", timeout=30).json()["data"]
    for model in models:
        name = model["id"]
        if not name.startswith(prefix):
            continue
        response = requests.post(
            f"{url}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=60)
        print(f"卸载残留适配器 {name}：HTTP {response.status_code}", flush=True)


def resume(run_directory: Path, iterations: int, truncated_reward):
    unload_run_adapters(run_directory)
    command = [
        "grpx", "train",
        "--resume", str(run_directory),
        "--set", f"training.iterations={iterations}",
        "--set", "training.checkpoint_interval=10",
        "--set", "training.evaluation_interval=10",
    ]
    if truncated_reward is None:
        command.extend(["--set", "training.truncated_reward=null"])
    else:
        command.extend(["--set", f"training.truncated_reward={truncated_reward}"])
    print("运行：", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def summarize(run_directory: Path, start_iteration=50):
    rows = [json.loads(line) for line in (run_directory / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    print(f"\n=== {run_directory.name} ===")
    print(f"{'iter':>4} {'trunc':>7} {'success':>8} {'reward':>8}")
    for row in rows:
        if row.get("iteration", -1) < start_iteration:
            continue
        if "length_limit_fraction" not in row:
            continue
        mark = " <<" if row["length_limit_fraction"] >= 0.05 else ""
        print(
            f"{row['iteration']:4d} {row['length_limit_fraction']:7.3f} "
            f"{row.get('success_fraction', float('nan')):8.3f} "
            f"{row.get('mean_reward', float('nan')):8.3f}{mark}"
        )


def show_truncated_samples(run_directory: Path):
    path = run_directory / "trajectory_samples.jsonl"
    if not path.exists():
        print("没有 trajectory_samples.jsonl")
        return
    print(f"\n截断样本摘录（{run_directory.name}）：")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["iteration"] < 50:
            continue
        for sample in row["samples"]:
            if sample.get("termination_reason") != "length_limit":
                continue
            completion = sample.get("completion", "")
            tail = completion[-400:].replace("\n", "\\n")
            print(f"  iter={row['iteration']} tokens={sample.get('answer_token_count')} "
                  f"success={sample.get('is_successful')} reward={sample.get('total_reward')}")
            print(f"  ...{tail}\n")


def unique_tag(prefix: str) -> str:
    stamp = datetime.now().strftime("%H%M%S")
    return f"{prefix}-{stamp}-{int(time.time() * 1000) % 1000:03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "control", "fixed"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    if args.mode == "prepare":
        # 旧规则，多跑几轮跨过原尖峰窗口，并落盘轨迹
        run_directory = copy_run(unique_tag("prepare-legacy"), args.source)
        resume(run_directory, iterations=58, truncated_reward=None)
        summarize(run_directory)
        show_truncated_samples(run_directory)
        return

    for trial in range(args.trials):
        tag = unique_tag(f"{args.mode}-{trial}")
        run_directory = copy_run(tag, args.source)
        reward = None if args.mode == "control" else 0.0
        resume(run_directory, iterations=60, truncated_reward=reward)
        summarize(run_directory)


if __name__ == "__main__":
    main()
