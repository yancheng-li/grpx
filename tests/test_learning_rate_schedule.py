"""学习率调度：constant 与改动前一致；cosine 首尾中点正确；恢复后曲线连续。"""
from __future__ import annotations

import math

import torch

from grpx.checkpoint import load_checkpoint, save_checkpoint
from grpx.config import TrainingConfig
from grpx.learning_rate import learning_rate_factor, make_learning_rate_lambda


def old_constant_factor(step, warmup_iterations):
    return min(1.0, (step + 1) / warmup_iterations)


def test_constant_matches_previous_formula_bit_for_bit():
    warmup = 5
    for step in range(40):
        assert learning_rate_factor(step, warmup, schedule="constant") == old_constant_factor(
            step, warmup)


def test_constant_lambda_lr_matches_previous_scheduler():
    peak = 5e-5
    warmup = 5
    model = torch.nn.Linear(2, 2)
    old_optimizer = torch.optim.AdamW(model.parameters(), lr=peak)
    old_scheduler = torch.optim.lr_scheduler.LambdaLR(
        old_optimizer, lambda step: min(1.0, (step + 1) / warmup))

    new_model = torch.nn.Linear(2, 2)
    config = TrainingConfig(
        iterations=20, groups_per_iteration=2, group_size=2,
        learning_rate=peak, warmup_iterations=warmup,
        learning_rate_schedule="constant",
    )
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=peak)
    new_scheduler = torch.optim.lr_scheduler.LambdaLR(
        new_optimizer, make_learning_rate_lambda(config, schedule_horizon=20))

    for _ in range(25):
        assert new_scheduler.get_last_lr()[0] == old_scheduler.get_last_lr()[0]
        old_scheduler.step()
        new_scheduler.step()


def test_cosine_warmup_peak_midpoint_and_final():
    warmup = 5
    horizon = 100
    final_fraction = 0.1
    peak = 5e-5

    # 预热段与 constant 相同
    for step in range(warmup - 1):
        assert learning_rate_factor(
            step, warmup, "cosine", horizon, final_fraction
        ) == old_constant_factor(step, warmup)

    # 预热刚结束（第 warmup-1 轮开始）：峰值
    assert learning_rate_factor(
        warmup - 1, warmup, "cosine", horizon, final_fraction) == 1.0

    # 衰减中点
    decay_start = warmup - 1
    decay_end = horizon - 1
    mid_step = decay_start + (decay_end - decay_start) // 2
    mid = learning_rate_factor(mid_step, warmup, "cosine", horizon, final_fraction)
    progress = (mid_step - decay_start) / (decay_end - decay_start)
    expected_mid = final_fraction + (1.0 - final_fraction) * 0.5 * (
        1.0 + math.cos(math.pi * progress))
    assert mid == expected_mid
    # 整数中点时 progress 恰为 0.5，倍数为 (1 + final) / 2
    if (decay_end - decay_start) % 2 == 0:
        assert mid == (1.0 + final_fraction) / 2

    # 最后一轮开始：终点
    assert learning_rate_factor(
        horizon - 1, warmup, "cosine", horizon, final_fraction) == final_fraction

    # 延长轮次：保持终点
    for step in range(horizon, horizon + 30):
        assert learning_rate_factor(
            step, warmup, "cosine", horizon, final_fraction) == final_fraction

    # 绝对值：最后一轮学习率 = 峰值 × 终点比例
    config = TrainingConfig(
        iterations=horizon, groups_per_iteration=2, group_size=2,
        learning_rate=peak, warmup_iterations=warmup,
        learning_rate_schedule="cosine",
        final_learning_rate_fraction=final_fraction,
    )
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_learning_rate_lambda(config, schedule_horizon=horizon))
    for _ in range(horizon - 1):
        scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - peak * final_fraction) < 1e-15


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)

    def save_pretrained(self, directory):
        pass


def test_resume_keeps_schedule_horizon_and_lr_continuous(tmp_path):
    peak = 1e-3
    warmup = 2
    original_horizon = 10
    extended_iterations = 20
    final_fraction = 0.1
    config = TrainingConfig(
        iterations=original_horizon, groups_per_iteration=2, group_size=2,
        learning_rate=peak, warmup_iterations=warmup,
        learning_rate_schedule="cosine",
        final_learning_rate_fraction=final_fraction,
    )

    model = _Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_learning_rate_lambda(config, schedule_horizon=original_horizon))
    rng = __import__("random").Random(0)

    # 跑到第 7 轮结束后保存（completed_iterations=7）；此时 last_epoch=7，
    # 下一轮开始时的学习率应等于 factor(7)
    for _ in range(7):
        scheduler.step()
    lr_before_save = scheduler.get_last_lr()[0]
    expected_at_step_7 = peak * learning_rate_factor(
        7, warmup, "cosine", original_horizon, final_fraction)
    assert lr_before_save == expected_at_step_7

    checkpoint = save_checkpoint(
        tmp_path, 7, model, optimizer, scheduler, rng,
        schedule_horizon=original_horizon)

    # 用延长后的 iterations 恢复：horizon 必须仍是 10
    resume_config = TrainingConfig(
        iterations=extended_iterations, groups_per_iteration=2, group_size=2,
        learning_rate=peak, warmup_iterations=warmup,
        learning_rate_schedule="cosine",
        final_learning_rate_fraction=final_fraction,
    )
    other = _Tiny()
    other_optimizer = torch.optim.AdamW(other.parameters(), lr=peak)
    # 故意先用错误的 horizon 建调度器（模拟 train() 里先按 config.iterations 再建）
    other_scheduler = torch.optim.lr_scheduler.LambdaLR(
        other_optimizer, make_learning_rate_lambda(resume_config, schedule_horizon=extended_iterations))
    other_rng = __import__("random").Random(99)
    completed, saved_horizon = load_checkpoint(
        checkpoint, other, other_optimizer, other_scheduler, other_rng)
    assert completed == 7
    assert saved_horizon == original_horizon
    other_scheduler.lr_lambdas = [
        make_learning_rate_lambda(resume_config, schedule_horizon=saved_horizon)]

    assert other_scheduler.get_last_lr()[0] == lr_before_save

    # 继续 step 到超过原 horizon：学习率停在终点，不会按 20 轮重新抬高
    for _ in range(extended_iterations - 7):
        other_scheduler.step()
    assert abs(other_scheduler.get_last_lr()[0] - peak * final_fraction) < 1e-15
    # 对照：若错误地用 extended 做 horizon，最后一轮会刚好到终点，但中段会更高；
    # 这里已过原终点多轮，仍钉在终点即可说明没有重拉长。
    wrong = peak * learning_rate_factor(
        extended_iterations - 1, warmup, "cosine", extended_iterations, final_fraction)
    assert abs(wrong - peak * final_fraction) < 1e-15  # 两者终点相同
    # 关键差异在「原终点之后、延长终点之前」：错误 horizon 尚未到终点
    mid_wrong = peak * learning_rate_factor(
        original_horizon + 2, warmup, "cosine", extended_iterations, final_fraction)
    assert mid_wrong > peak * final_fraction
