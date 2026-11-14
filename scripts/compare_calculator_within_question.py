"""未训练模型：同题内比较「没调用 / 真调用并等待 / 调用但没等」的成功率。

对 worked_example、温度采样。每题默认 8 条；只在同一题里同时出现多类的组内比较。

用法（需 vLLM，在项目根目录）：
  PYTHONPATH=. python -m scripts.compare_calculator_within_question
  PYTHONPATH=. python -m scripts.compare_calculator_within_question --n 200 --group-size 8
  PYTHONPATH=. python -m scripts.compare_calculator_within_question --stop-strings ''
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from examples.gsm8k import make_dataset
from grpx.command_line import build_components, load_configuration
from grpx.rollout import rollout_groups
from grpx.types import success_fraction

CATEGORIES = ("no_call", "waited", "without_waiting")


def classify(trajectory):
    metrics = trajectory.environment_metrics or {}
    if metrics.get("answered_without_waiting_fraction", 0.0) >= 0.5:
        return "without_waiting"
    if metrics.get("used_calculator_fraction", 0.0) >= 0.5:
        return "waited"
    return "no_call"


def paired_question_diffs(by_question, left, right):
    """同题内两类都有样本时，返回每题的 (left_rate, right_rate, diff)。"""
    pairs = []
    for rates in by_question.values():
        if left not in rates or right not in rates:
            continue
        pairs.append((rates[left], rates[right], rates[left] - rates[right]))
    return pairs


def bootstrap_mean_ci(values, repetitions=10000, alpha=0.05, seed=0):
    """对一列数值做均值的百分位自助法区间。"""
    if not values:
        return None, None, None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(repetitions):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lower = means[int(alpha / 2 * repetitions)]
    upper = means[min(repetitions - 1, int((1 - alpha / 2) * repetitions))]
    return sum(values) / n, lower, upper


def summarize_pair(by_question, left, right, bootstrap_seed=0):
    pairs = paired_question_diffs(by_question, left, right)
    if not pairs:
        return {
            "n_questions": 0,
            f"mean_{left}": None,
            f"mean_{right}": None,
            "mean_diff": None,
            "mean_diff_ci95": None,
        }
    lefts = [item[0] for item in pairs]
    rights = [item[1] for item in pairs]
    diffs = [item[2] for item in pairs]
    mean_diff, lower, upper = bootstrap_mean_ci(diffs, seed=bootstrap_seed)
    return {
        "n_questions": len(pairs),
        f"mean_{left}": sum(lefts) / len(lefts),
        f"mean_{right}": sum(rights) / len(rights),
        "mean_diff": mean_diff,
        "mean_diff_ci95": [lower, upper],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--stop-strings", default="</calc>",
        help="逗号分隔的停止串；默认 </calc>。传空字符串关闭。")
    args = parser.parse_args()

    opts = (
        '{ success_function = "examples.gsm8k:success_function", '
        'reward_function = "examples.gsm8k:reward_function", '
        'question_field = "question", '
        'prompt_variant = "worked_example", '
        "maximum_response_characters = 5000 }"
    )
    stop_strings = [part for part in args.stop_strings.split(",") if part]
    overrides = [
        "training.run_name=gsm8k-calc-within-question-probe",
        "model.dtype=bfloat16",
        f"backend.max_new_tokens={args.max_new_tokens}",
        f"training.max_steps={args.max_steps}",
        "training.memory_fraction=0.65",
        "training.merge_append_only_turns=false",
        f"training.seed={args.seed}",
        f"task.environment_options={opts}",
    ]
    if stop_strings:
        listed = ", ".join(f'"{item}"' for item in stop_strings)
        overrides.append(f"backend.stop_strings=[{listed}]")
    else:
        overrides.append("backend.stop_strings=[]")
    configuration = load_configuration("configs/gsm8k_calculator_vllm.toml", overrides)
    model, tokenizer, backend, _train, env_factory, config, _eval, _rd = build_components(
        configuration)

    dataset = make_dataset(split="test", size=args.n, seed=args.seed)
    backend.synchronize_policy(model)
    print(
        f"n={len(dataset)} group_size={args.group_size} "
        f"max_steps={config.max_steps} max_new_tokens={backend.max_new_tokens} "
        f"temperature={backend.temperature} "
        f"stop_strings={list(getattr(backend, 'stop_strings', ()))}",
        flush=True,
    )

    started = time.perf_counter()
    trajectories = rollout_groups(
        backend, tokenizer, env_factory,
        [(f"q{index}", sample, index) for index, sample in enumerate(dataset)],
        group_size=args.group_size,
        max_steps=config.max_steps,
        greedy=False,
        truncated_reward=config.truncated_reward,
    )
    getattr(backend, "close", lambda: None)()
    seconds = time.perf_counter() - started

    by_group = defaultdict(list)
    for trajectory in trajectories:
        category = classify(trajectory)
        successful = bool(trajectory.is_successful) if trajectory.is_successful is not None else (
            trajectory.total_reward > 0.5)
        by_group[trajectory.group_id].append((category, successful))

    overall = {name: {"n": 0, "correct": 0} for name in CATEGORIES}
    by_question_rates = {}
    multi_category_questions = 0

    for group_id, items in by_group.items():
        per_cat = defaultdict(list)
        for category, successful in items:
            per_cat[category].append(successful)
            overall[category]["n"] += 1
            overall[category]["correct"] += int(successful)
        rates = {cat: sum(vals) / len(vals) for cat, vals in per_cat.items()}
        by_question_rates[group_id] = rates
        if len(rates) >= 2:
            multi_category_questions += 1

    summary = {
        "n_questions": len(dataset),
        "group_size": args.group_size,
        "seconds": seconds,
        "prompt_variant": "worked_example",
        "stop_strings": list(getattr(backend, "stop_strings", ())),
        "max_steps": config.max_steps,
        "max_new_tokens": backend.max_new_tokens,
        "overall": {
            name: {
                "n": overall[name]["n"],
                "success": (overall[name]["correct"] / overall[name]["n"]
                            if overall[name]["n"] else None),
            }
            for name in CATEGORIES
        },
        "multi_category_questions": multi_category_questions,
        "by_question_rates": by_question_rates,
        "paired": {},
    }
    for index, (left, right) in enumerate((
        ("waited", "no_call"),
        ("waited", "without_waiting"),
        ("without_waiting", "no_call"),
    )):
        summary["paired"][f"{left}_vs_{right}"] = summarize_pair(
            by_question_rates, left, right, bootstrap_seed=args.seed + index)

    multi_ids = {gid for gid, rates in by_question_rates.items() if len(rates) >= 2}
    multi_overall = {name: {"n": 0, "correct": 0} for name in CATEGORIES}
    for group_id, items in by_group.items():
        if group_id not in multi_ids:
            continue
        for category, successful in items:
            multi_overall[category]["n"] += 1
            multi_overall[category]["correct"] += int(successful)
    summary["overall_in_multi_category_questions"] = {
        name: {
            "n": multi_overall[name]["n"],
            "success": (multi_overall[name]["correct"] / multi_overall[name]["n"]
                        if multi_overall[name]["n"] else None),
        }
        for name in CATEGORIES
    }

    out = Path("outputs/gsm8k_calculator_within_question_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {out}", flush=True)
    print(
        f"global success_fraction={success_fraction(trajectories)}  "
        f"n_traj={len(trajectories)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
