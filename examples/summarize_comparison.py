"""汇总 examples.compare_algorithms 的结果。

用法：python -m examples.summarize_comparison [outputs/comparison] [基准配置名，默认 grpo]
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/comparison")
reference_name = sys.argv[2] if len(sys.argv) > 2 else "grpo"
LATE_ITERATIONS = 5    # 采样指标取最后几轮的均值，单轮波动太大


def load_runs():
    runs = defaultdict(dict)    # 配置名 -> 种子 -> 记录列表
    for run_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"(.+)-seed(\d+)", run_directory.name)
        metrics_path = run_directory / "metrics.jsonl"
        if not match or not metrics_path.exists():
            continue
        config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        if not any(record["iteration"] == config["iterations"] for record in records):
            print(f"跳过未完成的运行：{run_directory.name}")
            continue
        runs[match.group(1)][int(match.group(2))] = records
    return runs


def mean_of(records, name):
    values = [record[name] for record in records if name in record]
    return sum(values) / len(values) if values else None


def per_seed_values(records_by_seed):
    """每个种子一组汇总值"""
    values = {}
    for seed, records in records_by_seed.items():
        final = records[-1]
        training = [record for record in records if "mean_reward" in record]
        late = training[-LATE_ITERATIONS:]
        baseline = next((record for record in records if record["iteration"] == 0), {})
        values[seed] = {
            "最终评测成功率": final.get("evaluation_success_fraction"),
            "最终评测平均回报": final.get("evaluation_mean_reward"),
            "最终评测平均步数": final.get("evaluation_mean_step_count"),
            "最终评测有效步比例": final.get("evaluation_environment_informative_step_fraction"),
            "最终评测切分质量": final.get("evaluation_environment_split_quality"),
            "第0轮采样有效步比例": baseline.get("environment_informative_step_fraction"),
            "第0轮采样切分质量": baseline.get("environment_split_quality"),
            f"采样有效步比例（最后{LATE_ITERATIONS}轮）":
                mean_of(late, "environment_informative_step_fraction"),
            f"采样切分质量（最后{LATE_ITERATIONS}轮）":
                mean_of(late, "environment_split_quality"),
            f"采样成功率（最后{LATE_ITERATIONS}轮）": mean_of(late, "success_fraction"),
            f"采样平均回报（最后{LATE_ITERATIONS}轮）": mean_of(late, "mean_reward"),
            "loss 均值": mean_of(training, "loss"),
            "步级信号覆盖率": mean_of(training, "step_signal_coverage"),
            "位移中位数": mean_of(training, "shift_median_absolute_log_ratio"),
            "截断比例": mean_of(training, "length_limit_fraction"),
        }
    return values


def describe(values):
    values = [value for value in values if value is not None]
    if not values:
        return "无"
    if len(values) == 1:
        return f"{values[0]:.4g}"
    return f"均值 {statistics.fmean(values):.4g}  范围 [{min(values):.4g}, {max(values):.4g}]"


runs = load_runs()
summaries = {name: per_seed_values(records_by_seed) for name, records_by_seed in runs.items()}

for name, records_by_seed in runs.items():
    seeds = sorted(records_by_seed)
    print(f"\n===== {name}（种子 {seeds}）=====")
    for metric in summaries[name][seeds[0]]:
        seed_values = [summaries[name][seed][metric] for seed in seeds]
        if all(value is None for value in seed_values):
            continue
        listed = "  ".join("无" if value is None else f"{value:.4g}" for value in seed_values)
        print(f"  {metric}: {listed}  |  {describe(seed_values)}")

    evaluation_iterations = sorted({record["iteration"] for records in records_by_seed.values()
                                    for record in records if "evaluation_success_fraction" in record})
    print("  评测曲线（种子均值）:")
    for iteration in evaluation_iterations:
        at_iteration = [record for records in records_by_seed.values() for record in records
                        if record["iteration"] == iteration and "evaluation_success_fraction" in record]
        informative = mean_of(at_iteration, "evaluation_environment_informative_step_fraction")
        split_quality = mean_of(at_iteration, "evaluation_environment_split_quality")
        extras = ""
        if informative is not None:
            extras += f"  有效步 {informative:.3f}"
        if split_quality is not None:
            extras += f"  切分质量 {split_quality:.3f}"
        print(f"    第 {iteration:>2} 轮  成功率 {mean_of(at_iteration, 'evaluation_success_fraction'):.3f}"
              f"  平均回报 {mean_of(at_iteration, 'evaluation_mean_reward'):.4f}{extras}")
    print("  可训练参数校验值: " + "  ".join(
        f"种子{seed}={records_by_seed[seed][-1].get('trainable_parameter_checksum')!r}" for seed in seeds))

if reference_name in summaries:
    print(f"\n===== 按种子配对，相对 {reference_name} 的差值 =====")
    paired_metrics = ["最终评测成功率", "最终评测平均回报", "最终评测有效步比例", "最终评测切分质量",
                      f"采样有效步比例（最后{LATE_ITERATIONS}轮）", f"采样切分质量（最后{LATE_ITERATIONS}轮）",
                      f"采样成功率（最后{LATE_ITERATIONS}轮）", f"采样平均回报（最后{LATE_ITERATIONS}轮）"]
    for name, summary in summaries.items():
        if name == reference_name:
            continue
        seeds = sorted(set(summary) & set(summaries[reference_name]))
        print(f"  {name}（共同种子 {seeds}）")
        for metric in paired_metrics:
            differences = [summary[seed][metric] - summaries[reference_name][seed][metric] for seed in seeds
                           if summary[seed][metric] is not None and summaries[reference_name][seed][metric] is not None]
            listed = "  ".join(f"{difference:+.4f}" for difference in differences)
            print(f"    {metric}: {listed}  |  {describe(differences)}")

all_checksums = [records[-1].get("trainable_parameter_checksum")
                 for records_by_seed in runs.values() for records in records_by_seed.values()]
known_checksums = [checksum for checksum in all_checksums if checksum is not None]
if len(set(known_checksums)) < len(known_checksums):
    print("\n注意：有运行的可训练参数校验值完全相同，它们最后评测的可能是同一份权重")
