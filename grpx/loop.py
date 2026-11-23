import dataclasses
import json
import pathlib
import random
import time
import warnings

import torch

from grpx.rollout import rollout_groups
from grpx.trainer import (
    action_log_probabilities,
    build_batch,
    grpo_loss,
    merged_trajectory_fraction,
    prepare_update_units,
)
from grpx.advantage import compute_advantages
from grpx.logging import format_record, create_run_directory
from grpx.evaluation import evaluate
from grpx.checkpoint import (
    find_latest_checkpoint, load_checkpoint, prune_rolling_checkpoints, save_checkpoint,
    truncate_metrics,
)
from grpx.gradient_checkpointing import (
    gradient_bearing_forward,
    gradient_checkpointing,
    suspend_kv_cache,
)
from grpx.multi_device_update import (
    ReplicaGroup,
    assign_minibatches,
)
from grpx.iteration_state import (
    accumulation_checkpoint_path,
    capture_gradients,
    load_accumulation_checkpoint,
    load_iteration_trajectories,
    make_fingerprint,
    remove_accumulation_checkpoint,
    restore_gradients,
    save_accumulation_checkpoint,
    save_iteration_trajectories,
)
from grpx.types import mean_environment_metrics, success_fraction
from grpx.trajectory_logging import (
    append_trajectory_samples,
    select_trajectory_samples,
    summarize_answer_lengths,
    truncate_trajectory_samples,
)
from grpx.config import ALGORITHM_PRESETS, TrainingConfig    # noqa: F401  兼容旧的导入路径
from grpx.device import duplicate_physical_devices, indexed_cuda_device, is_cuda_device
from grpx.learning_rate import make_learning_rate_lambda

TRAJECTORY_SAMPLES_PER_CLASS = 2

# 兼容旧导入；真正上限以 TrainingConfig.memory_fraction 为准（默认同此值）
TRAINING_MEMORY_FRACTION = 0.6
SHARED_GPU_MEMORY_FRACTION_HINT = 0.9


def should_measure_shift(iteration, interval):
    """与评测间隔同一写法：interval>0 且 iteration 整除时测；第 0 轮总会测。"""
    return interval > 0 and iteration % interval == 0


def loss_options(config):
    return {
        "clip_range": config.clip_range,
        "length_normalization": config.length_normalization,
        "constant_length": config.constant_length,
    }

def collect_trajectories(
    backend, tokenizer, environment_factory, samples, config, iteration
):
    return rollout_groups(
        backend, tokenizer, environment_factory,
        [(f"iteration-{iteration}-group-{group_index}", sample, iteration * 100003 + group_index)
         for group_index, sample in enumerate(samples)],
        group_size=config.group_size,
        policy_version=iteration,
        max_steps=config.max_steps,
        truncated_reward=config.truncated_reward,
    )

# 第 0 轮 LoRA 增量近零，中位数应只差精度（千分之几）；超过 0.05 视为训推不是同一模型。
# 尾部用「差超过 1」的比例。普通模型上，配置正确时这个比例在 0 到约 0.6%，
# 最大差到过 1.6；配置错位（例如 RoPE 用了默认 10000）时比例是 2%–4%，最大差 5–11。
# 警告线 0.3%、硬失败 1%，两边各留出余量。限度按普通模型定，MoE 上还没有校准。
BACKEND_LOGPROB_GAP_ONE = 1.0
BACKEND_LOGPROB_GAP_ABOVE_ONE_WARN = 0.003    # 0.3%，只提示
BACKEND_LOGPROB_GAP_ABOVE_ONE_LIMIT = 0.01    # 1%
BACKEND_LOGPROB_GAP_MAX_LIMIT = 5.0
BACKEND_LOGPROB_GAP_MEDIAN_LIMIT = 0.05


def measure_backend_log_probability_gap(batches, old_log_probabilities):
    """用更新前算出的旧策略对数概率，与同位置的服务端采样对数概率对比。

    不另做前向：old_log_probabilities 已是采样时刻训练端权重下的值。
    只覆盖有信号且带服务端概率的位置（本地后端、补上的结束符会被 mask 掉）。
    多推理地址时按地址分别统计，主指标取各地址最大值。
    """
    gaps_by_url = {}
    for batch, old in zip(batches, old_log_probabilities):
        comparable = batch["sampling_log_probability_mask"][:, 1:]
        if not comparable.any():
            continue
        difference = (old - batch["sampling_log_probabilities"][:, 1:]).abs()
        urls = batch.get("backend_urls")
        row_count = comparable.shape[0]
        for row_index in range(row_count):
            row_mask = comparable[row_index]
            if not row_mask.any():
                continue
            url = ""
            if urls is not None and row_index < len(urls) and urls[row_index]:
                url = urls[row_index]
            gaps_by_url.setdefault(url, []).append(difference[row_index][row_mask])
    if not gaps_by_url:
        return {}
    medians = {}
    percentiles = {}
    above_half = {}
    above_one = {}
    maxima = {}
    for url, pieces in gaps_by_url.items():
        gap = torch.cat(pieces).float()
        medians[url] = gap.median().item()
        percentiles[url] = torch.quantile(gap, 0.95).item()
        above_half[url] = (gap > 0.5).float().mean().item()
        above_one[url] = (gap > BACKEND_LOGPROB_GAP_ONE).float().mean().item()
        maxima[url] = gap.max().item()
    # 最差地址按「差超过 1」的比例排；没有这种 token 时退回最大差
    worst_url = max(
        percentiles,
        key=lambda url: (above_one[url], maxima[url], percentiles[url]))
    metrics = {
        "backend_log_probability_gap_median": medians[worst_url],
        "backend_log_probability_gap_p95": percentiles[worst_url],
        "backend_log_probability_gap_above_half_fraction": max(above_half.values()),
        "backend_log_probability_gap_above_one_fraction": max(above_one.values()),
        "backend_log_probability_gap_max": max(maxima.values()),
        "gap_by_url": percentiles,
    }
    if len(percentiles) > 1:
        metrics["backend_log_probability_gap_worst_url"] = worst_url or "(unknown)"
    return metrics


def _gap_mismatch_message(gap_metrics):
    above = gap_metrics.get("backend_log_probability_gap_above_one_fraction")
    gap_max = gap_metrics.get("backend_log_probability_gap_max")
    p95 = gap_metrics.get("backend_log_probability_gap_p95")
    median = gap_metrics.get("backend_log_probability_gap_median")
    parts = []
    if above is not None:
        parts.append(f"above_one_fraction={above:.2e}")
    if gap_max is not None:
        parts.append(f"max={gap_max:.2e}")
    if p95 is not None:
        parts.append(f"p95={p95:.2e}")
    if median is not None:
        parts.append(f"median={median:.2e}")
    message = (
        "训推对数概率差过大（" + "，".join(parts) + "）；"
        "训练端和推理服务不是同一个模型（或分词器 / 对话模板 / RoPE 配置不一致）。"
        "请确认 vLLM 加载的基座与 [model].name 相同，"
        "合并模型的 config.json 含有顶层 rope_theta，"
        "或用 [backend].server_model_name 声明服务端名字。"
    )
    by_url = gap_metrics.get("gap_by_url") or {}
    if len(by_url) > 1:
        detail = "、".join(
            f"{(url or '(unknown)')}={value:.2e}" for url, value in sorted(by_url.items()))
        message += f" 各地址 p95：{detail}。"
        worst = gap_metrics.get("backend_log_probability_gap_worst_url")
        if worst:
            message += f" 最差地址：{worst}。"
    return message


def backend_log_probability_gap_is_hard_failure(gap_metrics):
    """差超过 1 的比例 > 1%，或最大差 > 5，或中位数 > 0.05。"""
    above = gap_metrics.get("backend_log_probability_gap_above_one_fraction")
    gap_max = gap_metrics.get("backend_log_probability_gap_max")
    median = gap_metrics.get("backend_log_probability_gap_median")
    return (
        (above is not None and above > BACKEND_LOGPROB_GAP_ABOVE_ONE_LIMIT)
        or (gap_max is not None and gap_max > BACKEND_LOGPROB_GAP_MAX_LIMIT)
        or (median is not None and median > BACKEND_LOGPROB_GAP_MEDIAN_LIMIT)
    )


def backend_log_probability_gap_is_warning(gap_metrics):
    """差超过 1 的比例 > 0.3%。硬失败时不算警告。"""
    if backend_log_probability_gap_is_hard_failure(gap_metrics):
        return False
    above = gap_metrics.get("backend_log_probability_gap_above_one_fraction")
    return above is not None and above > BACKEND_LOGPROB_GAP_ABOVE_ONE_WARN


def enforce_backend_log_probability_gap(gap_metrics, *, hard=False):
    """硬失败看差超过 1 的比例、最大差和中位数；0.3%–1% 只警告。

    hard=True 用于本进程首次核对（含 --resume），与 iteration 编号无关。
    之后的轮次即使越过硬限度也只警告，避免一次采样噪声中断已经在跑的训练。
    """
    if backend_log_probability_gap_is_hard_failure(gap_metrics):
        message = _gap_mismatch_message(gap_metrics)
        if hard:
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)
        return
    if backend_log_probability_gap_is_warning(gap_metrics):
        above = gap_metrics["backend_log_probability_gap_above_one_fraction"]
        warnings.warn(
            f"backend_log_probability_gap_above_one_fraction={above:.2e} "
            f"超过 {BACKEND_LOGPROB_GAP_ABOVE_ONE_WARN:g}；"
            "普通模型上偶发到这个范围，先不中止。"
            "比例到百分之几，或最大差超过 5，才按配置错位处理。",
            stacklevel=2,
        )


def public_gap_metrics(gap_metrics):
    """写入 metrics.jsonl / 终端的字段；去掉仅供报错用的 gap_by_url。"""
    return {key: value for key, value in gap_metrics.items() if key != "gap_by_url"}


def summarize_rollout(trajectories, kept):
    total_rewards = [trajectory.total_reward for trajectory in trajectories]
    successes = success_fraction(trajectories)
    return {
        "mean_reward": sum(total_rewards) / len(total_rewards),
        **({"success_fraction": successes} if successes is not None else {}),
        "signal_fraction": len(kept) / len(trajectories),
        "length_limit_fraction": sum(
            trajectory.termination_reason == "length_limit" 
            for trajectory in trajectories) / len(trajectories),
        "mean_step_count": sum(len(trajectory.steps) for trajectory in trajectories) / len(trajectories),
        "step_limit_fraction": sum(
            trajectory.termination_reason == "step_limit"
            for trajectory in trajectories) / len(trajectories),
        **summarize_answer_lengths(trajectories),
        **mean_environment_metrics(trajectories),
    }

def row_length(row):
    last = row.steps[-1]
    return len(last.prompt_token_ids) + len(last.action_token_ids)

def segment_count(row):
    """小批预算按段（步）计，合并后一行多段仍占相应步数。"""
    return len(row.steps)

def batch_segment_weight(batch):
    weights = batch.get("segment_weights", batch["row_weights"])
    return weights.sum().item()

def _warn_dropped_rows(dropped_lengths, total, max_tokens):
    if not dropped_lengths:
        return
    fraction = len(dropped_lengths) / total
    percent = 100.0 * fraction
    base = (
        f"本轮有 {len(dropped_lengths)} 行超过 max_tokens_per_minibatch={max_tokens}"
        f"（最长 {max(dropped_lengths)}，占 {percent:.0f}%），已跳过不参与更新；"
        f"对话历史可能过长"
    )
    # 比例高时升级措辞，避免当成普通噪声略过（与预检 5%/20% 阈值一致）
    if fraction > 0.20:
        warnings.warn(
            f"严重：{base}。超过两成更新单元被丢掉，本轮几乎没在学；"
            f"请增大 max_tokens_per_minibatch",
            UserWarning,
        )
    elif fraction > 0.05:
        warnings.warn(
            f"注意：{base}。超过 5% 的更新单元被丢掉；建议增大 max_tokens_per_minibatch",
            UserWarning,
        )
    else:
        warnings.warn(base)


def _minibatch_index_plan(units, max_rows, max_tokens):
    """按长度降序后的下标划分。下标指向传入的 units，恢复时必须用同一份 units，不能重排。"""
    order = sorted(range(len(units)), key=lambda index: row_length(units[index]), reverse=True)
    dropped_lengths = [row_length(units[index]) for index in order
                       if row_length(units[index]) > max_tokens]
    _warn_dropped_rows(dropped_lengths, len(units), max_tokens)
    kept_order = [index for index in order if row_length(units[index]) <= max_tokens]
    chunks, current = [], []
    current_segments = 0
    for index in kept_order:
        segments = segment_count(units[index])
        if current and (
                current_segments + segments > max_rows
                or (len(current) + 1) * row_length(units[current[0]]) > max_tokens):
            chunks.append(current)
            current = []
            current_segments = 0
        current.append(index)
        current_segments += segments
    if current:
        chunks.append(current)
    return chunks


def build_minibatch_plan(trajectories, config):
    """开始累积前定死的小批划分。恢复时原样使用，不再调用切分。"""
    units = prepare_update_units(trajectories, merge_append_only=config.merge_append_only_turns)
    return _minibatch_index_plan(
        units, config.max_rows_per_minibatch, config.max_tokens_per_minibatch)


def split_into_row_chunks(trajectories, max_rows, max_tokens, merge_append_only=False):
    """按更新单元切小批：默认每步一行；merge_append_only 时追加式轨迹合并为一行。
    max_rows 限制的是段数（步数），不是合并后的物理行数；另限制 行数 × 补齐后长度。

    单行已超过 max_tokens 的对话无法安全更新（单独成批也会顶满甚至 OOM），
    在组批前丢掉并每轮汇总一条警告，避免训练整体崩溃。
    """
    units = prepare_update_units(trajectories, merge_append_only=merge_append_only)
    plan = _minibatch_index_plan(units, max_rows, max_tokens)
    return [[units[index] for index in group] for group in plan]


def _chunks_from_plan(units, minibatch_plan):
    if any(index < 0 or index >= len(units) for group in minibatch_plan for index in group):
        raise ValueError("小批划分方案与当前轨迹不一致，不能按这份方案累积")
    return [[units[index] for index in group] for group in minibatch_plan]

def _batch_shape(batch):
    return {
        "rows": int(batch["input_token_ids"].shape[0]),
        "length": int(batch["input_token_ids"].shape[1]),
        "segments": int(batch["segment_advantages"].shape[0]) if "segment_advantages" in batch
        else int(batch["advantages"].shape[0]),
    }

def _average_weighted_metrics(weighted_metrics):
    # max_ 开头的取各批最大值，加权平均会把单个小批的大偏差稀释掉；
    # 中位数和分位数按批加权平均并不是真正的全局分位数，单次更新时它们恒为零，只看 measure_policy_shift 的
    return {
        name: (max(metrics[name] for _, metrics in weighted_metrics) if name.startswith("max_")
               else sum(weight * metrics[name] for weight, metrics in weighted_metrics))
        for name in weighted_metrics[0][1]
    }


def accumulate_gradients(model, batches, old_log_probabilities, config, peak_tracker=None,
                         accumulation=None):
    """各小批的损失按段权重之和的占比加权后累积梯度，结果等于全体段一次算出的梯度；返回按同样权重平均的指标。

    total_weight 在循环前算完。accumulation 不为空时，每 K 个小批把梯度写盘；
    指纹对不上就丢弃，从第一个小批重算。全部累积完立刻删掉中途文件，裁剪和更新由调用方在这之后做。
    """
    total_weight = sum(batch_segment_weight(batch) for batch in batches)
    start_index = 0
    weighted_metrics = []
    if accumulation is not None:
        trainable_names = [name for name, parameter in model.named_parameters()
                           if parameter.requires_grad]
        restored = load_accumulation_checkpoint(
            accumulation["path"], accumulation["fingerprint"], total_weight, len(batches),
            trainable_names)
        if restored is not None:
            start_index = restored["next_index"]
            weighted_metrics = [(item[0], item[1]) for item in restored["weighted_metrics"]]
            restore_gradients(model, restored["gradients"])
            print(f"累积梯度从第 {start_index} 个小批继续（本轮共 {len(batches)} 个小批）")
    for index, (batch, old_log_probability) in enumerate(zip(batches, old_log_probabilities)):
        if index < start_index:
            continue
        # 反向会重算前向，模式必须保持到 backward 结束；无梯度的前向不进这里
        with gradient_bearing_forward(model):
            new_log_probability, mask = action_log_probabilities(model, batch)
            loss, metrics = grpo_loss(
                new_log_probability, old_log_probability, mask, batch["advantages"],
                row_weights=batch["row_weights"],
                segment_ids=batch.get("segment_ids"),
                segment_advantages=batch.get("segment_advantages"),
                segment_weights=batch.get("segment_weights"),
                **loss_options(config)
            )
            weight = batch_segment_weight(batch) / total_weight
            (loss * weight).backward()
        if peak_tracker is not None:
            peak_tracker.note(batch, "backward")
        weighted_metrics.append((weight, metrics))
        completed = index + 1
        if (accumulation is not None and completed % accumulation["interval"] == 0
                and completed < len(batches)):
            save_accumulation_checkpoint(
                accumulation["path"], accumulation["fingerprint"], completed, total_weight,
                len(batches), weighted_metrics, capture_gradients(model))
    if accumulation is not None:
        # 更新还没做。这里删掉，崩溃在裁剪或 step 时会整段重算，而不会在已经 step 过的梯度上再 step 一次
        remove_accumulation_checkpoint(accumulation["path"])
    return _average_weighted_metrics(weighted_metrics)

class _PeakBatchTracker:
    """记录全局峰值上扬时所在小批的形状；不 reset，以免干扰外层峰值统计。"""

    def __init__(self, device=None):
        self.best = None
        self._peak_bytes = -1
        self.device = device

    def note(self, batch, phase):
        # 先看 CUDA 在不在：配置里的 device 可能是 "cuda"，但单元测试跑在没有 CUDA 的环境上
        if self.device is None or not torch.cuda.is_available():
            return
        device = indexed_cuda_device(self.device)
        if device is None:
            return
        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        if peak_bytes > self._peak_bytes:
            self._peak_bytes = peak_bytes
            self.best = {**_batch_shape(batch), "phase": phase,
                         "peak_gigabytes": peak_bytes / 2 ** 30}

def _batch_on_device(batch, device):
    moved = {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    return moved


def _minibatch_token_counts(batches):
    return [int(batch["attention_mask"].sum().item()) for batch in batches]


def _add_captured_gradients(model, captured):
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    for name, value in captured.items():
        if value is None:
            continue
        parameter = parameters[name]
        contribution = value.detach().to(device=parameter.device, dtype=parameter.dtype)
        if parameter.grad is None:
            parameter.grad = contribution.clone()
        else:
            parameter.grad.add_(contribution)


def _accumulation_waves(batch_count, start_index, interval):
    """按全局小批序号切段。interval 为 0 时整段一次做完。保存点落在 completed % interval == 0。"""
    pending = list(range(start_index, batch_count))
    if not pending:
        return []
    if not interval:
        return [pending]
    waves = []
    current = []
    for index in pending:
        current.append(index)
        completed = index + 1
        if completed % interval == 0 and completed < batch_count:
            waves.append(current)
            current = []
    if current:
        waves.append(current)
    return waves


def _routing_recorders(models, record_routing):
    if not record_routing:
        return {}
    from grpx.routing_diagnostics import RoutingRecorder

    recorders = {}
    for model in models:
        recorder = RoutingRecorder(model)
        if not recorder.is_available:
            return {}
        recorders[id(model)] = recorder
    return recorders


def _merge_routing_snapshots(snapshots_by_index):
    from grpx.routing_diagnostics import RoutingSnapshot

    merged = RoutingSnapshot()
    for index in sorted(snapshots_by_index):
        snapshot = snapshots_by_index[index]
        if snapshot is None:
            continue
        for layer_name, tensors in snapshot.selected_by_layer.items():
            merged.selected_by_layer.setdefault(layer_name, []).extend(tensors)
    if not merged.layer_names():
        return None
    return merged


def _accumulate_on_replicas(group, models, batches, old_log_probabilities, assignment, config,
                            accumulation=None):
    """各副本按事先分好的小批累积。损失除以全局分母。梯度加总后留在主副本上。"""
    total_weight = sum(batch_segment_weight(batch) for batch in batches)
    start_index = 0
    weighted_metrics = []
    held_gradients = None
    primary = models[0]
    if accumulation is not None:
        trainable_names = [name for name, parameter in primary.named_parameters() if parameter.requires_grad]
        restored = load_accumulation_checkpoint(
            accumulation["path"], accumulation["fingerprint"], total_weight, len(batches),
            trainable_names)
        if restored is not None:
            start_index = restored["next_index"]
            weighted_metrics = [(item[0], item[1]) for item in restored["weighted_metrics"]]
            restore_gradients(primary, restored["gradients"])
            held_gradients = capture_gradients(primary)
            print(f"累积梯度从第 {start_index} 个小批继续（本轮共 {len(batches)} 个小批）")

    def compute_minibatch_loss(model, index, device):
        batch = _batch_on_device(batches[index], device)
        old_log_probability = old_log_probabilities[index].to(device)
        new_log_probability, mask = action_log_probabilities(model, batch)
        loss, metrics = grpo_loss(
            new_log_probability, old_log_probability, mask, batch["advantages"],
            row_weights=batch["row_weights"],
            segment_ids=batch.get("segment_ids"),
            segment_advantages=batch.get("segment_advantages"),
            segment_weights=batch.get("segment_weights"),
            **loss_options(config),
        )
        weight = batch_segment_weight(batch) / total_weight
        return loss * weight, (weight, metrics)

    interval = accumulation["interval"] if accumulation is not None else 0
    waves = _accumulation_waves(len(batches), start_index, interval)
    for wave_number, wave in enumerate(waves):
        wave_set = set(wave)
        wave_assignment = [[index for index in share if index in wave_set] for share in assignment]
        collected = group.accumulate_gradients(wave_assignment, compute_minibatch_loss)
        group.reduce_gradients()
        if held_gradients is not None:
            _add_captured_gradients(primary, held_gradients)
        for index in wave:
            weighted_metrics.append(collected[index])
        completed = wave[-1] + 1
        if (accumulation is not None and completed % interval == 0 and completed < len(batches)):
            save_accumulation_checkpoint(
                accumulation["path"], accumulation["fingerprint"], completed, total_weight,
                len(batches), weighted_metrics, capture_gradients(primary))
        if wave_number + 1 < len(waves):
            held_gradients = capture_gradients(primary)
    if accumulation is not None:
        remove_accumulation_checkpoint(accumulation["path"])
    return _average_weighted_metrics(weighted_metrics)


def _update_on_replicas(model, replicas, optimizer, batches, config, record_routing,
                        run_directory, iteration, trajectory_hash_value):
    """replicas[0] 是主副本，优化器只挂在它上面。一张卡时不要走这里。"""
    if replicas[0] is not model:
        raise ValueError("replicas 的第一项必须是主副本")
    devices = [replica.device for replica in replicas]
    group = ReplicaGroup(replicas, devices)
    try:
        group.assert_synchronized()
        token_counts = _minibatch_token_counts(batches)
        assignment = assign_minibatches(token_counts, len(replicas))
        recorders = _routing_recorders(replicas, record_routing)

        def compute_old_log_probability(replica, index, device):
            from grpx.routing_diagnostics import RoutingSnapshot

            batch = _batch_on_device(batches[index], device)
            recorder = recorders.get(id(replica))
            if recorder is None:
                return action_log_probabilities(replica, batch)[0], None
            snapshot = RoutingSnapshot()
            with recorder.record(batch["attention_mask"], snapshot=snapshot):
                log_probability = action_log_probabilities(replica, batch)[0]
            return log_probability, snapshot

        for replica in replicas:
            replica.eval()
        old_parts = group.map_without_gradient(assignment, compute_old_log_probability)
        primary_device = batches[0]["input_token_ids"].device
        old_log_probabilities = [
            old_parts[index][0].to(primary_device) for index in range(len(batches))]
        routing_before = _merge_routing_snapshots(
            {index: old_parts[index][1] for index in range(len(batches))})
        gap_metrics = measure_backend_log_probability_gap(batches, old_log_probabilities)

        restore_caches = [suspend_kv_cache(replica) for replica in replicas]
        history = []
        try:
            for _ in range(config.updates_per_iteration):
                group.assert_synchronized()
                optimizer.zero_grad()
                accumulation = None
                if config.accumulation_checkpoint_interval and run_directory is not None:
                    if iteration is None or trajectory_hash_value is None:
                        raise ValueError("保存累积梯度需要本轮轮次和轨迹哈希")
                    accumulation = {
                        "path": accumulation_checkpoint_path(run_directory),
                        "interval": config.accumulation_checkpoint_interval,
                        "fingerprint": make_fingerprint(
                            config, iteration, trajectory_hash_value, model),
                    }
                averaged = _accumulate_on_replicas(
                    group, replicas, batches, old_log_probabilities, assignment, config,
                    accumulation=accumulation)
                history.append(averaged)
                if (config.max_approximate_kl is not None
                        and averaged["approximate_kl"] > config.max_approximate_kl):
                    break
                trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_gradient_norm)
                optimizer.step()
                group.broadcast_parameters()
        finally:
            for restore in restore_caches:
                restore()
        return history, old_log_probabilities, gap_metrics, routing_before
    finally:
        group.close()


def _shift_on_replicas(replicas, batches, old_log_probabilities, config, routing_before):
    devices = [replica.device for replica in replicas]
    group = ReplicaGroup(replicas, devices)
    try:
        group.assert_synchronized()
        assignment = assign_minibatches(_minibatch_token_counts(batches), len(replicas))
        recorders = _routing_recorders(replicas, routing_before is not None)

        def compute_shift(replica, index, device):
            from grpx.routing_diagnostics import RoutingSnapshot

            batch = _batch_on_device(batches[index], device)
            old_log_probability = old_log_probabilities[index].to(device)
            recorder = recorders.get(id(replica))
            if recorder is None:
                new_log_probability, mask = action_log_probabilities(replica, batch)
                return (new_log_probability - old_log_probability)[mask], None
            snapshot = RoutingSnapshot()
            with recorder.record(batch["attention_mask"], snapshot=snapshot):
                new_log_probability, mask = action_log_probabilities(replica, batch)
            return (new_log_probability - old_log_probability)[mask], snapshot

        for replica in replicas:
            replica.eval()
        parts = group.map_without_gradient(assignment, compute_shift)
    finally:
        group.close()
    log_ratio = torch.cat([parts[index][0].float().cpu() for index in range(len(batches))])
    absolute_log_ratio = log_ratio.abs()
    metrics = {
        "shift_clip_fraction": ((log_ratio.exp() - 1).abs() > config.clip_range).float().mean().item(),
        "shift_median_absolute_log_ratio": absolute_log_ratio.median().item(),
        "shift_percentile_95_absolute_log_ratio": torch.quantile(absolute_log_ratio, 0.95).item(),
    }
    routing_after = _merge_routing_snapshots({index: parts[index][1] for index in range(len(batches))})
    if routing_after is not None and routing_before is not None:
        from grpx.routing_diagnostics import summarize_routing
        metrics.update(summarize_routing(routing_before, routing_after))
    return metrics


def _cuda_devices(devices):
    seen = []
    for name in devices:
        cuda_device = indexed_cuda_device(name)
        if cuda_device is None or cuda_device in seen:
            continue
        seen.append(cuda_device)
    return seen


def _apply_memory_fraction(devices, fraction):
    for cuda_device in _cuda_devices(devices):
        torch.cuda.set_per_process_memory_fraction(fraction, device=cuda_device)


def _reset_peak_memory(devices):
    for cuda_device in _cuda_devices(devices):
        torch.cuda.reset_peak_memory_stats(cuda_device)


def _record_peak_memory(record, devices):
    allocated = {}
    reserved = {}
    for cuda_device in _cuda_devices(devices):
        key = str(cuda_device)
        allocated[key] = torch.cuda.max_memory_allocated(cuda_device) / 2 ** 30
        reserved[key] = torch.cuda.max_memory_reserved(cuda_device) / 2 ** 30
    if not allocated:
        return
    primary = _cuda_devices(devices[:1])
    primary_key = str(primary[0]) if primary else next(iter(allocated))
    record["peak_memory_gigabytes"] = allocated[primary_key]
    record["peak_reserved_memory_gigabytes"] = reserved[primary_key]
    if len(allocated) > 1:
        record["peak_memory_gigabytes_by_device"] = allocated
        record["peak_reserved_memory_gigabytes_by_device"] = reserved


def update_policy(model, optimizer, kept, padding_token_id, config, record_routing=False,
                  minibatch_plan=None, run_directory=None, iteration=None,
                  trajectory_hash_value=None, replicas=None):
    units = prepare_update_units(kept, merge_append_only=config.merge_append_only_turns)
    merge_fraction = merged_trajectory_fraction(kept, units)
    oversized_lengths = [
        row_length(unit) for unit in units
        if row_length(unit) > config.max_tokens_per_minibatch
    ]
    row_total = len(units)
    oversized_stats = {
        "oversized_row_count": len(oversized_lengths),
        "oversized_row_total": row_total,
        "oversized_row_fraction": (
            len(oversized_lengths) / row_total if row_total else 0.0),
        "oversized_row_max_length": max(oversized_lengths) if oversized_lengths else 0,
    }
    if minibatch_plan is None:
        minibatch_plan = _minibatch_index_plan(
            units, config.max_rows_per_minibatch, config.max_tokens_per_minibatch)
    chunks = _chunks_from_plan(units, minibatch_plan)
    if not chunks:
        # 全部超长被筛掉：跳过更新，让训练继续
        return [], [], [], {}, merge_fraction, None, oversized_stats, None
    batches = [
        build_batch(chunk, padding_token_id, model.device, config.row_weighting)
        for chunk in chunks
    ]
    if replicas is not None and len(replicas) > 1:
        history, old_log_probabilities, gap_metrics, routing_before = _update_on_replicas(
            model, replicas, optimizer, batches, config, record_routing,
            run_directory, iteration, trajectory_hash_value)
        return (history, batches, old_log_probabilities, gap_metrics, merge_fraction,
                None, oversized_stats, routing_before)

    peak_tracker = _PeakBatchTracker(device=getattr(config, "device", None) or model.device)
    routing_before = None
    routing_recorder = None
    if record_routing:
        from grpx.routing_diagnostics import RoutingRecorder, RoutingSnapshot
        routing_recorder = RoutingRecorder(model)
        if routing_recorder.is_available:
            routing_before = RoutingSnapshot()
        else:
            routing_recorder = None
    model.eval()
    with torch.no_grad():         # 更新前，把所有小批的旧概率算出来
        old_log_probabilities = []
        for batch in batches:
            if routing_recorder is not None:
                with routing_recorder.record(batch["attention_mask"], snapshot=routing_before):
                    old_log_probabilities.append(action_log_probabilities(model, batch)[0])
            else:
                old_log_probabilities.append(action_log_probabilities(model, batch)[0])
            peak_tracker.note(batch, "old_log_probs")
    gap_metrics = measure_backend_log_probability_gap(batches, old_log_probabilities)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]

    history = []
    for _ in range(config.updates_per_iteration):
        model.train()
        optimizer.zero_grad()
        accumulation = None
        if config.accumulation_checkpoint_interval and run_directory is not None:
            if iteration is None or trajectory_hash_value is None:
                raise ValueError("保存累积梯度需要本轮轮次和轨迹哈希")
            # 每次更新前重新取权重哈希：上一次 step 之后权重变了，不能接着用上一份中途梯度
            accumulation = {
                "path": accumulation_checkpoint_path(run_directory),
                "interval": config.accumulation_checkpoint_interval,
                "fingerprint": make_fingerprint(
                    config, iteration, trajectory_hash_value, model),
            }
        averaged = accumulate_gradients(
            model, batches, old_log_probabilities, config, peak_tracker=peak_tracker,
            accumulation=accumulation)
        history.append(averaged)
        if (config.max_approximate_kl is not None
                and averaged["approximate_kl"] > config.max_approximate_kl):
            break
        torch.nn.utils.clip_grad_norm_(trainable_params, config.max_gradient_norm)
        optimizer.step()
    return (history, batches, old_log_probabilities, gap_metrics, merge_fraction,
            peak_tracker.best, oversized_stats, routing_before)

def measure_policy_shift(model, batches, old_log_probabilities, config, routing_before=None,
                         replicas=None):
    """更新之后对所有小批再做一遍无梯度前向，把回答位置的对数比值收集到一起，统一算分位数。

    必须复用 update_policy 里算旧对数概率时的同一批 tensors（切分、顺序、补齐都相同）。
    另建小批会在 bf16 下引入与学习率无关的路径差，把位移指标打成底噪。
    若传入 routing_before（来自同一次 update_policy 的旧概率前向），顺带汇总路由变化。"""
    if replicas is not None and len(replicas) > 1:
        return _shift_on_replicas(replicas, batches, old_log_probabilities, config, routing_before)
    model.eval()
    log_ratios = []
    routing_after = None
    routing_recorder = None
    if routing_before is not None:
        from grpx.routing_diagnostics import RoutingRecorder, RoutingSnapshot
        routing_recorder = RoutingRecorder(model)
        if routing_recorder.is_available:
            routing_after = RoutingSnapshot()
        else:
            routing_recorder = None
    with torch.no_grad():
        for batch, old_log_probability in zip(batches, old_log_probabilities, strict=True):
            if routing_recorder is not None:
                with routing_recorder.record(batch["attention_mask"], snapshot=routing_after):
                    new_log_probability, mask = action_log_probabilities(model, batch)
            else:
                new_log_probability, mask = action_log_probabilities(model, batch)
            log_ratios.append((new_log_probability - old_log_probability)[mask])
    log_ratio = torch.cat(log_ratios).float()
    absolute_log_ratio = log_ratio.abs()
    metrics = {
        "shift_clip_fraction": ((log_ratio.exp() - 1).abs() > config.clip_range).float().mean().item(),
        "shift_median_absolute_log_ratio": absolute_log_ratio.median().item(),
        "shift_percentile_95_absolute_log_ratio": torch.quantile(absolute_log_ratio, 0.95).item(),
    }
    if routing_after is not None:
        from grpx.routing_diagnostics import summarize_routing
        metrics.update(summarize_routing(routing_before, routing_after))
    return metrics

def train(model, tokenizer, backend, dataset, environment_factory, config, evaluation_dataset=None,
          resolved_configuration=None, resume_from=None, run_directory=None):
    """resolved_configuration：命令行读入并合并覆盖项后的完整配置，原样存进运行目录以便复现。
    resume_from：已有运行目录；从其中最新检查点恢复，配置应由调用方从该目录读入。
    run_directory：调用方已创建好的运行目录（例如 vLLM 需要先建目录再写适配器）。"""
    if config.length_normalization == "constant" and config.constant_length is None:
        config = dataclasses.replace(config, constant_length=backend.max_new_tokens)
    replicas = list(getattr(model, "grpx_replicas", None) or [model])
    if replicas[0] is not model:
        raise ValueError("主副本必须是训练入口拿到的那个模型")
    if len(replicas) == 1:
        replicas_for_update = None
    else:
        replicas_for_update = replicas
        duplicates = duplicate_physical_devices(config.devices)
        if duplicates:
            warnings.warn(
                f"training.devices 里有重复的设备 {duplicates}，多份副本会挤在同一张卡上。"
                "正式训练请让每张卡只出现一次",
                UserWarning,
            )
    for replica in replicas:
        if config.gradient_checkpointing:
            gradient_checkpointing(replica, True)
        replica.grpx_output_logit_chunk_size = config.output_logit_chunk_size
    cuda_devices = _cuda_devices(config.devices)
    if cuda_devices:
        _apply_memory_fraction(config.devices, config.memory_fraction)

    if resume_from is not None:
        run_directory = pathlib.Path(resume_from)
        checkpoint_directory = find_latest_checkpoint(run_directory)
        if checkpoint_directory is None:
            raise ValueError(f"{run_directory} 里没有可用的检查点")
        (run_directory / "config.json").write_text(
            json.dumps(dataclasses.asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
    elif run_directory is None:
        run_directory = create_run_directory(config)
        checkpoint_directory = None
    else:
        run_directory = pathlib.Path(run_directory)
        checkpoint_directory = None

    if hasattr(backend, "adapter_directory"):
        backend.adapter_directory = run_directory / "vllm_adapter"

    if resolved_configuration is not None:
        (run_directory / "resolved_configuration.json").write_text(
            json.dumps(resolved_configuration, indent=2, ensure_ascii=False), encoding="utf-8")

    random_generator = random.Random(config.seed)
    torch.manual_seed(config.seed)    # 恢复时随后被检查点里的随机状态覆盖
    trainable_params = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    # 衰减长度开跑时钉死；恢复时若检查点里有记录则沿用，延长 iterations 不会把学习率抬回去
    schedule_horizon = config.iterations
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_learning_rate_lambda(config, schedule_horizon)
    )
    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = min(backend.end_token_ids)

    start_iteration = 0
    if checkpoint_directory is not None:
        start_iteration, saved_horizon = load_checkpoint(
            checkpoint_directory, model, optimizer, scheduler, random_generator,
            device=config.device, devices=config.devices)
        if replicas_for_update is not None:
            sync_group = ReplicaGroup(replicas, [replica.device for replica in replicas])
            try:
                sync_group.broadcast_parameters()
            finally:
                sync_group.close()
        if saved_horizon is not None:
            schedule_horizon = saved_horizon
            # 用钉死的 horizon 重建 lambda；last_epoch / _last_lr 已由 load_state_dict 恢复
            scheduler.lr_lambdas = [make_learning_rate_lambda(config, schedule_horizon)]
        if start_iteration >= config.iterations:
            # 已完成且未延长：不动日志，保留最终评测那一条
            print(f"从 {checkpoint_directory.name} 恢复，已完成 {start_iteration} 轮；"
                  "这次运行已经完成。要继续训练，请用 --set training.iterations=<更大的值>")
            return run_directory
        dropped = truncate_metrics(run_directory / "metrics.jsonl", start_iteration)
        dropped_samples = truncate_trajectory_samples(
            run_directory / "trajectory_samples.jsonl", start_iteration)
        print(f"从 {checkpoint_directory.name} 恢复，已完成 {start_iteration} 轮，"
              f"丢弃其后的 {dropped} 条日志"
              + (f"、{dropped_samples} 条轨迹抽样" if dropped_samples else ""))

    # 恢复后与首次采样前合并为一次：评测/采样都看到当前权重
    backend.synchronize_policy(model)
    # 本进程第一次拿到 gap 指标时硬检查（resume 的「第一轮」也算）
    gap_hard_check_pending = True

    with open(run_directory / "metrics.jsonl", "a", encoding="utf-8", buffering=1) as log_file:
        for iteration in range(start_iteration, config.iterations):
            iteration_start_time = time.perf_counter()
            if cuda_devices:
                _reset_peak_memory(config.devices)
            record = {"iteration": iteration, "learning_rate": scheduler.get_last_lr()[0]}
            # 评测放在本轮更新之前：第 0 轮得到的就是未训练的基线
            evaluation_seconds = 0.0
            if evaluation_dataset and iteration % config.evaluation_interval == 0:
                phase_start = time.perf_counter()
                record.update(evaluate(backend, tokenizer, environment_factory,
                                       evaluation_dataset, config.max_steps,
                                       truncated_reward=config.truncated_reward))
                evaluation_seconds = time.perf_counter() - phase_start
            loaded_iteration = None
            if config.save_iteration_trajectories:
                loaded_iteration = load_iteration_trajectories(
                    run_directory, iteration, model, config, random_generator,
                    device=config.device)
            if loaded_iteration is None:
                samples = random_generator.sample(dataset, config.groups_per_iteration)
                phase_start = time.perf_counter()
                trajectories = collect_trajectories(
                    backend, tokenizer, environment_factory, samples, config, iteration
                    )
                sampling_seconds = time.perf_counter() - phase_start
                phase_start = time.perf_counter()
                diagnostics = {}
                kept = compute_advantages(trajectories, normalize_std=config.normalize_std,
                                          step_advantage_weight=config.step_advantage_weight,
                                          discount_factor=config.discount, diagnostics=diagnostics)
                minibatch_plan = None
                trajectory_hash_value = None
                if config.save_iteration_trajectories:
                    minibatch_plan = build_minibatch_plan(kept, config)
                    trajectory_hash_value = save_iteration_trajectories(
                        run_directory, iteration, model, config, trajectories, kept,
                        diagnostics, minibatch_plan, random_generator, device=config.device)
                metrics_seconds = time.perf_counter() - phase_start
            else:
                trajectories, kept, diagnostics, minibatch_plan, trajectory_hash_value = loaded_iteration
                sampling_seconds = 0.0
                metrics_seconds = 0.0
                print(f"第 {iteration} 轮轨迹已在盘上且校验通过，跳过采样")
            record.update(summarize_rollout(trajectories, kept))
            record.update(diagnostics)
            append_trajectory_samples(
                run_directory / "trajectory_samples.jsonl",
                iteration,
                select_trajectory_samples(
                    trajectories, tokenizer, TRAJECTORY_SAMPLES_PER_CLASS),
            )
            update_seconds = 0.0
            shift_seconds = 0.0
            synchronize_seconds = 0.0
            if kept:
                phase_start = time.perf_counter()
                measure_shift = should_measure_shift(
                    iteration, config.shift_measurement_interval)
                (history, batches, old_log_probabilities, gap_metrics, merge_fraction,
                 _peak_batch, oversized_stats, routing_before) = update_policy(
                    model, optimizer, kept, padding_token_id, config,
                    record_routing=measure_shift,
                    minibatch_plan=minibatch_plan,
                    run_directory=run_directory,
                    iteration=iteration,
                    trajectory_hash_value=trajectory_hash_value,
                    replicas=replicas_for_update,
                )
                record.update(oversized_stats)
                record["merged_trajectory_fraction"] = merge_fraction
                if history:
                    record["loss"] = history[-1]["loss"]
                    record.update(public_gap_metrics(gap_metrics))
                    enforce_backend_log_probability_gap(
                        gap_metrics, hard=gap_hard_check_pending)
                    gap_hard_check_pending = False
                    if measure_shift:
                        shift_start = time.perf_counter()
                        record.update(measure_policy_shift(
                            model, batches, old_log_probabilities, config,
                            routing_before=routing_before,
                            replicas=replicas_for_update))
                        shift_seconds = time.perf_counter() - shift_start
                    scheduler.step()
                    # 位移前向从 update_seconds 拆出，便于不同间隔之间比较更新耗时
                    update_seconds = time.perf_counter() - phase_start - shift_seconds
                    phase_start = time.perf_counter()
                    backend.synchronize_policy(model)
                    synchronize_seconds = time.perf_counter() - phase_start
                else:
                    update_seconds = time.perf_counter() - phase_start
            record["evaluation_seconds"] = evaluation_seconds
            record["sampling_seconds"] = sampling_seconds
            record["metrics_seconds"] = metrics_seconds
            record["update_seconds"] = update_seconds
            record["shift_seconds"] = shift_seconds
            record["synchronize_seconds"] = synchronize_seconds
            if cuda_devices:
                _record_peak_memory(record, config.devices)
            record["iteration_seconds"] = time.perf_counter() - iteration_start_time
            print(format_record(record))
            # 先写日志再保存：若在两者之间被杀，恢复时截断会丢掉这轮日志并重跑，不会永久缺一轮
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed = iteration + 1
            # 轨迹按这一轮开头的权重校验。间隔大于 1 时，最近的间隔检查点往往是好几轮之前的权重，
            # 恢复会丢掉已经采样的轨迹。打开轨迹保存就每轮再写一份，并删掉更早的非间隔检查点。
            if (config.save_iteration_trajectories or completed % config.checkpoint_interval == 0
                    or completed == config.iterations):
                save_checkpoint(run_directory, completed, model, optimizer, scheduler,
                                random_generator, device=config.device,
                                devices=config.devices,
                                schedule_horizon=schedule_horizon)
            if config.save_iteration_trajectories:
                prune_rolling_checkpoints(
                    run_directory, completed, config.checkpoint_interval, config.iterations)

        final_record = {"iteration": config.iterations}
        if evaluation_dataset:      # 训练结束后的最终评测
            final_record.update(evaluate(backend, tokenizer, environment_factory,
                                         evaluation_dataset, config.max_steps,
                                         truncated_reward=config.truncated_reward))
        # 两次运行结果一模一样时，用它区分"权重相同"和"权重不同但贪心输出相同"
        final_record["trainable_parameter_checksum"] = sum(
            parameter.detach().double().abs().sum().item() for parameter in trainable_params)
        print(format_record(final_record))
        log_file.write(json.dumps(final_record, ensure_ascii=False) + "\n")
    return run_directory
