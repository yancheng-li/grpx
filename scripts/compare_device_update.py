"""同一批轨迹上，交替比较单卡更新和多副本更新。

采样只做一次（本地后端），轨迹写盘后每次更新都从同一份权重和一份新的优化器开始。
单卡、双卡交替，避免顺序把第一次的编译时间算进某一种设置。计时只包住 update_policy，
与训练日志里的 update_seconds 相同；位移前向在计时之外。

两张同型号卡：

    python scripts/compare_device_update.py \\
        --config configs/multi_device_qwen05_arithmetic.toml \\
        --devices cuda:0 cuda:1

    python scripts/compare_device_update.py \\
        --config configs/multi_device_qwen15_calculator.toml \\
        --devices cuda:0 cuda:1

    python scripts/compare_device_update.py \\
        --config configs/arithmetic_olmoe_d2d3.toml \\
        --devices cuda:0 cuda:1

一张卡上放两份副本只用来把脚本跑通（需测试开关，脚本在设备字符串重复时自己打开）。
这样测出来的耗时不是加速比：两份权重挤在同一张卡上，单卡那一侧也带着另一份模型。

    python scripts/compare_device_update.py \\
        --config configs/multi_device_qwen05_arithmetic.toml \\
        --devices cuda:0 cuda:0 \\
        --repeats 2

整轮耗时不在这里。采样不随卡数变快，完整训练日志里的 iteration_seconds 才是整轮。
本脚本另给一个估算：这一次采样耗时 + 更新中位数。评测轮还要再加上 evaluation_seconds。
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--devices", nargs="+", required=True, metavar="DEVICE")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectories", default=None,
                        help="轨迹文件。已存在则直接读，否则采样后写入。")
    parser.add_argument("--resample", action="store_true", help="忽略已有轨迹，重新采样并覆盖。")
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats 至少为 1")
    if args.warmup < 0:
        raise SystemExit("--warmup 不能为负")
    if len(args.devices) < 2:
        raise SystemExit("至少两份副本。同一张卡：--devices cuda:0 cuda:0；两张卡：cuda:0 cuda:1")

    from grpx.config import SAME_DEVICE_REPLICAS_ENV
    same_card = len(set(args.devices)) != len(args.devices)
    if same_card:
        os.environ[SAME_DEVICE_REPLICAS_ENV] = "1"

    import torch

    from grpx.advantage import compute_advantages
    from grpx.command_line import ConfigurationError, build_components, load_configuration
    from grpx.gradient_checkpointing import gradient_checkpointing
    from grpx.iteration_state import _trajectory_from_dict, _trajectory_to_dict
    from grpx.loop import _apply_memory_fraction, _record_peak_memory, _reset_peak_memory
    from grpx.loop import collect_trajectories, measure_policy_shift, update_policy

    try:
        configuration = load_configuration(args.config, list(args.overrides))
    except ConfigurationError as error:
        raise SystemExit(str(error)) from None
    backend_type = configuration.get("backend", {}).get("type", "huggingface")
    if backend_type != "huggingface":
        raise SystemExit(
            f"这个对比用本地后端采样，当前是 {backend_type!r}。"
            "计算器任务用 configs/multi_device_qwen15_calculator.toml。")
    configuration.setdefault("training", {})["devices"] = list(args.devices)
    fraction = configuration["training"].get("memory_fraction", 1.0)
    _apply_memory_fraction(args.devices, fraction)

    (model, tokenizer, backend, train_dataset, environment_factory,
     config, _evaluation_dataset, _run_directory) = build_components(configuration)
    replicas = list(model.grpx_replicas)
    if len(replicas) < 2:
        raise SystemExit(f"只加载了 {len(replicas)} 份副本")
    if config.gradient_checkpointing:
        for replica in replicas:
            gradient_checkpointing(replica, True)

    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = min(backend.end_token_ids)

    output_path = Path(args.output)
    trajectory_path = Path(args.trajectories) if args.trajectories else output_path.with_suffix(".trajectories.pt")
    kept, sampling_seconds = load_or_sample(
        trajectory_path, args, model, tokenizer, backend, train_dataset,
        environment_factory, config, torch, _trajectory_to_dict, _trajectory_from_dict,
        collect_trajectories, compute_advantages)
    if not kept:
        raise SystemExit("这批轨迹里没有可更新的样本")

    initial = capture(replicas[0])
    initial_gap = max(parameter_gap(initial, capture(replica)) for replica in replicas[1:])
    print(f"loaded_replicas={len(replicas)} initial_max_abs_diff={initial_gap:.3e} kept={len(kept)}")
    if initial_gap > 1e-6:
        raise SystemExit("刚加载的副本权重不一致，不能拿来对比")

    def restore():
        for replica in replicas:
            restore_parameters(replica, initial)

    def run_once(mode):
        restore()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in replicas[0].parameters() if parameter.requires_grad],
            lr=config.learning_rate)
        torch.manual_seed(config.seed)
        active = replicas if mode == "dual" else None
        _reset_peak_memory(config.devices)
        synchronize(config.devices)
        started = time.perf_counter()
        (_history, batches, old_log_probabilities, _gap, _merge, _peak, _oversized,
         routing_before) = update_policy(
            replicas[0], optimizer, kept, padding_token_id, config, replicas=active)
        synchronize(config.devices)
        elapsed = time.perf_counter() - started
        memory = {}
        _record_peak_memory(memory, config.devices)
        shift = measure_policy_shift(
            replicas[0], batches, old_log_probabilities, config,
            routing_before=routing_before, replicas=active)
        weights = capture(replicas[0])
        return elapsed, memory, shift, weights

    for _ in range(args.warmup):
        run_once("single")
        run_once("dual")
    restore()

    trials = []
    for index in range(args.repeats):
        order = ("single", "dual") if index % 2 == 0 else ("dual", "single")
        by_mode = {}
        for mode in order:
            elapsed, memory, shift, weights = run_once(mode)
            by_mode[mode] = {
                "seconds": elapsed,
                "peak_memory_gigabytes": memory.get("peak_memory_gigabytes"),
                "peak_memory_gigabytes_by_device": memory.get("peak_memory_gigabytes_by_device"),
                "shift_median": shift["shift_median_absolute_log_ratio"],
                "shift_p95": shift["shift_percentile_95_absolute_log_ratio"],
                "weights": weights,
            }
            print(f"repeat={index} order={','.join(order)} mode={mode} seconds={elapsed:.3f}")
        diff = parameter_gap(by_mode["single"]["weights"], by_mode["dual"]["weights"])
        for mode in by_mode:
            by_mode[mode].pop("weights")
        trials.append({"order": list(order), "max_abs_param_diff": diff, "modes": by_mode})
        print(f"repeat={index} max_abs_param_diff={diff:.3e}")

    summary = summarize(
        trials, sampling_seconds, same_card, args, config, trajectory_path,
        configuration.get("model", {}).get("dtype"))
    summary["hardware"] = hardware_report(torch)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"wrote {output_path}")


def load_or_sample(trajectory_path, args, model, tokenizer, backend, train_dataset,
                   environment_factory, config, torch, to_dict, from_dict,
                   collect_trajectories, compute_advantages):
    config_key = str(Path(args.config).resolve())
    if trajectory_path.is_file() and not args.resample:
        payload = torch.load(trajectory_path, map_location="cpu", weights_only=False)
        if payload.get("config") != config_key:
            raise SystemExit(
                f"{trajectory_path} 来自 {payload.get('config')!r}，与当前配置不一致。"
                "换一批轨迹请加 --resample。")
        kept = [from_dict(item) for item in payload["kept"]]
        print(f"loaded {len(kept)} trajectories from {trajectory_path}")
        return kept, None

    torch.manual_seed(config.seed)
    samples = random.Random(config.seed).sample(train_dataset, config.groups_per_iteration)
    started = time.perf_counter()
    trajectories = collect_trajectories(
        backend, tokenizer, environment_factory, samples, config, iteration=0)
    sampling_seconds = time.perf_counter() - started
    kept = compute_advantages(
        trajectories, normalize_std=config.normalize_std,
        step_advantage_weight=config.step_advantage_weight,
        discount_factor=config.discount)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config_key, "kept": [to_dict(item) for item in kept]}, trajectory_path)
    print(f"sampled kept={len(kept)} sampling_seconds={sampling_seconds:.3f} -> {trajectory_path}")
    return kept, sampling_seconds


def capture(model):
    import torch
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_parameters(model, snapshot):
    import torch
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in snapshot:
                parameter.copy_(snapshot[name].to(device=parameter.device, dtype=parameter.dtype))


def parameter_gap(left, right):
    if set(left) != set(right):
        raise SystemExit("两份权重的可训练参数名字不一致")
    worst = 0.0
    for name in left:
        worst = max(worst, (left[name] - right[name]).abs().max().item())
    return worst


def synchronize(devices):
    import torch
    from grpx.device import indexed_cuda_device
    seen = []
    for name in devices:
        cuda_device = indexed_cuda_device(name)
        if cuda_device is None or cuda_device in seen:
            continue
        seen.append(cuda_device)
        torch.cuda.synchronize(cuda_device)


def summarize(trials, sampling_seconds, same_card, args, config, trajectory_path, dtype):
    def column(mode, key):
        return [trial["modes"][mode][key] for trial in trials]

    single = column("single", "seconds")
    dual = column("dual", "seconds")
    diffs = [trial["max_abs_param_diff"] for trial in trials]
    single_median = statistics.median(single)
    dual_median = statistics.median(dual)

    def peaks(mode):
        collected = {}
        for trial in trials:
            record = trial["modes"][mode]
            by_device = record["peak_memory_gigabytes_by_device"]
            if by_device:
                for device, value in by_device.items():
                    collected.setdefault(device, []).append(value)
            elif record["peak_memory_gigabytes"] is not None:
                collected.setdefault(args.devices[0], []).append(record["peak_memory_gigabytes"])
        return {device: max(values) for device, values in collected.items()}

    estimated = None
    if sampling_seconds is not None:
        estimated = {
            "sampling_seconds": sampling_seconds,
            "single_round_seconds": sampling_seconds + single_median,
            "dual_round_seconds": sampling_seconds + dual_median,
        }
    return {
        "config": args.config,
        "devices": list(args.devices),
        "same_card": same_card,
        "repeats": args.repeats,
        "warmup_each": args.warmup,
        "dtype": dtype,
        "learning_rate": config.learning_rate,
        "trajectories": str(trajectory_path),
        "update_seconds": {
            "single": pack_times(single),
            "dual": pack_times(dual),
            "speedup_median": single_median / dual_median if dual_median else None,
        },
        "max_abs_param_diff": {"median": statistics.median(diffs), "max": max(diffs)},
        "shift": {
            "single_median": statistics.median(column("single", "shift_median")),
            "single_p95": statistics.median(column("single", "shift_p95")),
            "dual_median": statistics.median(column("dual", "shift_median")),
            "dual_p95": statistics.median(column("dual", "shift_p95")),
        },
        "peak_memory_gigabytes": {"single": peaks("single"), "dual": peaks("dual")},
        "estimated_round_seconds": estimated,
        "trials": trials,
    }


def pack_times(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def hardware_report(torch):
    names = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            names.append(torch.cuda.get_device_name(index))
    driver = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=30)
        if completed.returncode == 0:
            driver = completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        driver = None
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": names,
        "nvidia_smi": driver,
    }


def print_summary(summary):
    update = summary["update_seconds"]
    print(
        f"single update median={update['single']['median']:.3f}s "
        f"range={update['single']['min']:.3f}..{update['single']['max']:.3f}")
    print(
        f"dual   update median={update['dual']['median']:.3f}s "
        f"range={update['dual']['min']:.3f}..{update['dual']['max']:.3f}")
    speedup = update["speedup_median"]
    if speedup is not None:
        label = "（同一张卡，不是加速比）" if summary["same_card"] else ""
        print(f"median speedup={speedup:.3f}{label}")
    diff = summary["max_abs_param_diff"]
    print(f"max_abs_param_diff median={diff['median']:.3e} max={diff['max']:.3e}")
    shift = summary["shift"]
    print(
        f"shift single median/p95={shift['single_median']:.6e} / {shift['single_p95']:.6e}")
    print(
        f"shift dual   median/p95={shift['dual_median']:.6e} / {shift['dual_p95']:.6e}")
    print(f"peak_memory_gigabytes single={summary['peak_memory_gigabytes']['single']}")
    print(f"peak_memory_gigabytes dual={summary['peak_memory_gigabytes']['dual']}")
    estimated = summary["estimated_round_seconds"]
    if estimated is None:
        print("sampling_seconds=（本次读的是已保存轨迹，没有重新采样）")
    else:
        print(
            f"sampling_seconds={estimated['sampling_seconds']:.3f} "
            f"estimated_round single={estimated['single_round_seconds']:.3f} "
            f"dual={estimated['dual_round_seconds']:.3f}")


if __name__ == "__main__":
    main()
