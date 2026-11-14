"""学习率倍数：预热 + 恒定 / 余弦衰减。

LambdaLR 的 step 与训练轮次对齐（每轮更新后 step 一次；构造时会先 step 到 0）。
第 i 轮开始时 last_epoch == i，日志里的 learning_rate 就是此时的值。
"""
from __future__ import annotations

import math


def learning_rate_factor(
    step,
    warmup_iterations,
    schedule="constant",
    schedule_horizon=None,
    final_fraction=0.1,
):
    """返回相对峰值学习率的倍数。

    constant：与改动前相同，min(1, (step + 1) / warmup_iterations)。
    cosine：预热相同；预热结束后按余弦从 1 降到 final_fraction，
        在 step == schedule_horizon - 1（最后一轮开始时）到达终点；
        step 更大时保持终点值（延长训练不会把学习率抬回去）。
    """
    if warmup_iterations < 1:
        raise ValueError(f"warmup_iterations 必须是正整数，收到 {warmup_iterations!r}")
    if schedule not in ("constant", "cosine"):
        raise ValueError(f"learning_rate_schedule 只能是 'constant' 或 'cosine'，收到 {schedule!r}")
    if isinstance(final_fraction, bool) or not isinstance(final_fraction, (int, float)):
        raise ValueError(f"final_learning_rate_fraction 必须是数字，收到 {final_fraction!r}")
    final_fraction = float(final_fraction)
    if not 0 <= final_fraction <= 1:
        raise ValueError(
            f"final_learning_rate_fraction 必须在 [0, 1] 内，收到 {final_fraction!r}")

    warmup = min(1.0, (step + 1) / warmup_iterations)
    if schedule == "constant":
        return warmup

    if schedule_horizon is None:
        raise ValueError("cosine 调度需要 schedule_horizon（最初设定的训练轮数）")
    if isinstance(schedule_horizon, bool) or not isinstance(schedule_horizon, int) or schedule_horizon < 1:
        raise ValueError(f"schedule_horizon 必须是正整数，收到 {schedule_horizon!r}")

    # 预热未结束：与 constant 完全一致
    if step < warmup_iterations - 1:
        return warmup

    decay_start = warmup_iterations - 1
    decay_end = schedule_horizon - 1
    if decay_end <= decay_start:
        # 总轮数不足以在预热后再衰减：预热结束后直接停在终点
        return final_fraction if step >= max(decay_end, decay_start) else 1.0

    if step >= decay_end:
        return final_fraction

    progress = (step - decay_start) / (decay_end - decay_start)
    return final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_learning_rate_lambda(config, schedule_horizon):
    """供 LambdaLR 使用的闭包；schedule_horizon 为最初设定的衰减总长度。"""
    schedule = getattr(config, "learning_rate_schedule", "constant")
    final_fraction = getattr(config, "final_learning_rate_fraction", 0.1)
    warmup = config.warmup_iterations

    def lr_lambda(step):
        return learning_rate_factor(
            step,
            warmup_iterations=warmup,
            schedule=schedule,
            schedule_horizon=schedule_horizon,
            final_fraction=final_fraction,
        )

    return lr_lambda
