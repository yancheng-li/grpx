import pytest

from examples.guess_number import GuessNumberEnvironment, make_dataset
from grpx.config import TrainingConfig
from grpx.doctor import (FAIL, PASS, WARNING, CheckResult, Report, build_worst_case_batch,
                         enumerate_worst_case_specs, estimate_duration, is_out_of_memory,
                         judge_backend_log_probability_gap, judge_dataset,
                         judge_first_step_ratio, judge_log_probabilities,
                         judge_gradient_checkpointing_effect, judge_memory,
                         judge_package_version, judge_sampling, judge_shift,
                         judge_step_signal_coverage,                          judge_oversized_row_fraction, judge_truncated_beats_normal,
                         judge_truncated_high_reward, select_worst_case_probe_targets,
                         walk_through_environment)
from grpx.environment import SingleTurnEnvironment
from grpx.types import Step, Trajectory


def make_trajectory(reward, action_length=3, termination_reason="done", is_successful=None,
                    group_id="g", environment_reward=None):
    steps = [Step([1], [2] * action_length, reward=reward, environment_reward=environment_reward)]
    return Trajectory(group_id, 0, 1.0, steps, termination_reason, is_successful)

def statuses(results):
    return {result.name: result.status for result in results}


def test_package_versions():
    assert judge_package_version("transformers", "4.57.6", "4.56").status == PASS
    assert judge_package_version("transformers", "4.55.2", "4.56").status == FAIL
    assert judge_package_version("transformers", "4.100.0", "4.56").status == PASS    # 按数字比较，不按字符串
    assert judge_package_version("torch", "2.6.0+cu124", "2.6").status == PASS
    assert judge_package_version("peft", None, None).status == FAIL


def test_out_of_memory_detection():
    assert is_out_of_memory(RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasSgemm"))
    assert is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not is_out_of_memory(RuntimeError("CUDA error: device-side assert triggered"))
    assert not is_out_of_memory(ValueError("out of memory"))
    try:
        import torch
        assert is_out_of_memory(torch.cuda.OutOfMemoryError("cuda out of memory"))
    except Exception:    # noqa: BLE001  CPU 版 torch 没有这个类型
        pass


def test_dataset_checks():
    assert statuses(judge_dataset(list(range(20)), [1], 16)) == {"训练集": PASS, "评测集": PASS}
    assert statuses(judge_dataset(list(range(10)), [1], 16))["训练集"] == FAIL
    assert statuses(judge_dataset(list(range(20)), None, 16))["评测集"] == WARNING
    assert statuses(judge_dataset(list(range(20)), [], 16))["评测集"] == FAIL
    assert statuses(judge_dataset(iter(range(20)), None, 16))["训练集"] == FAIL    # 不能按下标取样


def test_walk_through_guess_number():
    result = walk_through_environment(lambda: GuessNumberEnvironment(reveal_interval=True),
                                      make_dataset(1)[0], max_steps=6)
    assert result.status == PASS and "6 步内未结束" in result.detail

def test_walk_through_single_turn():
    result = walk_through_environment(lambda: SingleTurnEnvironment(lambda prompt, completion, sample: 0.0),
                                      {"prompt": "1+1=?"}, max_steps=1)
    assert result.status == PASS and "已结束" in result.detail

def test_walk_through_shared_instance():
    environment = GuessNumberEnvironment()
    assert walk_through_environment(lambda: environment, make_dataset(1)[0], 6).status == FAIL

class BrokenEnvironment:
    def __init__(self, reset_result=None, step_result=None, raises=False, raise_on_factory=False,
                 raise_on_reset=False, raise_on_state_key=False):
        self.reset_result = reset_result or [{"role": "user", "content": "hi"}]
        self.step_result = step_result
        self.raises, self.raise_on_factory = raises, raise_on_factory
        self.raise_on_reset, self.raise_on_state_key = raise_on_reset, raise_on_state_key

    def reset(self, sample, seed):
        if self.raise_on_reset:
            raise RuntimeError("reset 失败")
        return self.reset_result

    def step(self, action):
        if self.raises and action == "":
            raise ValueError("空回答解析失败")
        return self.step_result if self.step_result is not None else ([], 1.0, False)

    def state_key(self):
        if self.raise_on_state_key:
            raise RuntimeError("state_key 失败")
        return ""

@pytest.mark.parametrize("environment_options, expected_text", [
    ({"reset_result": "hi"}, "reset 返回"),
    ({"step_result": ([], 1.0)}, "step 返回"),
    ({"step_result": ("oops", 1.0, True)}, "新消息"),
    ({"step_result": ([], "1.0", True)}, "奖励是"),
    ({"step_result": ([], float("nan"), True)}, "奖励是"),
    ({"step_result": ([], True, True)}, "奖励是"),
    ({"raises": True}, "step('') 引发 ValueError: 空回答解析失败"),
    ({"raise_on_reset": True}, "reset(sample, seed=0) 引发 RuntimeError: reset 失败"),
    ({"raise_on_state_key": True}, "state_key() 引发 RuntimeError: state_key 失败"),
])
def test_walk_through_catches_interface_errors(environment_options, expected_text):
    result = walk_through_environment(lambda: BrokenEnvironment(**environment_options), {}, max_steps=3)
    assert result.status == FAIL and expected_text in result.detail
    if "奖励是" in expected_text:
        assert "float(...)" in result.suggestion

def test_walk_through_factory_error_names_the_call():
    def broken_factory():
        raise RuntimeError("构造失败")
    result = walk_through_environment(broken_factory, {}, max_steps=1)
    assert result.status == FAIL and "environment_factory() 引发 RuntimeError: 构造失败" in result.detail

class MetricsEnvironment(BrokenEnvironment):
    def __init__(self, metrics):
        super().__init__(step_result=([], 1.0, True))
        self._metrics = metrics

    def metrics(self):
        return self._metrics

def test_walk_through_rejects_bad_metrics():
    assert walk_through_environment(lambda: MetricsEnvironment("oops"), {}, 1).status == FAIL
    assert walk_through_environment(lambda: MetricsEnvironment({1: 0.5}), {}, 1).status == FAIL
    assert walk_through_environment(lambda: MetricsEnvironment({"a": float("nan")}), {}, 1).status == FAIL
    assert walk_through_environment(lambda: MetricsEnvironment({"informative_step_fraction": 0.5}), {}, 1).status == PASS


def test_sampling_healthy():
    trajectories = [make_trajectory(reward, is_successful=bool(reward)) for reward in (1, 0, 0, 1)]
    results = judge_sampling(trajectories, trajectories, max_new_tokens=32)
    assert set(statuses(results).values()) == {PASS}

def test_sampling_never_successful_and_no_signal():
    trajectories = [make_trajectory(0.0, is_successful=False) for _ in range(4)]
    result_statuses = statuses(judge_sampling(trajectories, [], max_new_tokens=32))
    assert result_statuses["成功率"] == WARNING and result_statuses["有信号的比例"] == FAIL

def test_sampling_too_easy_and_low_signal():
    trajectories = [make_trajectory(1.0, is_successful=True, group_id=f"g{index}") for index in range(10)]
    result_statuses = statuses(judge_sampling(trajectories, trajectories[:2], max_new_tokens=32))
    assert result_statuses["成功率"] == WARNING and result_statuses["有信号的比例"] == WARNING

def test_sampling_truncation_and_length():
    trajectories = ([make_trajectory(0.0, action_length=32, termination_reason="length_limit") for _ in range(3)]
                    + [make_trajectory(1.0) for _ in range(7)])
    result_statuses = statuses(judge_sampling(trajectories, trajectories, max_new_tokens=32))
    assert result_statuses["结束原因"] == WARNING and result_statuses["回答长度"] == WARNING
    assert result_statuses["成功率"] == PASS    # 环境没有成功判定
    assert result_statuses["截断却高回报"] == PASS


def test_truncated_high_environment_reward_warns():
    # 训练回报已被覆盖为 0，但环境原始回报仍是 1 → 应警告
    trajectories = (
        [make_trajectory(0.0, termination_reason="length_limit", environment_reward=1.0)]
        + [make_trajectory(1.0) for _ in range(99)]
    )
    result = judge_truncated_high_reward(trajectories)
    assert result.status == WARNING
    assert "0.010" in result.detail

    clean = [make_trajectory(0.0, termination_reason="length_limit", environment_reward=0.0)
             for _ in range(10)]
    assert judge_truncated_high_reward(clean).status == PASS


def test_truncated_beats_normal_endings_warns():
    # 正常结局里 2/4 整局回报 < truncated_reward=0 → 截断（得 0）比它们更高
    trajectories = (
        [make_trajectory(-0.5), make_trajectory(-0.2), make_trajectory(0.1), make_trajectory(1.0)]
        + [make_trajectory(0.0, termination_reason="length_limit", environment_reward=-0.5)]
    )
    result = judge_truncated_beats_normal(trajectories, truncated_reward=0.0)
    assert result.status == WARNING
    assert "50%" in result.detail
    assert "2/4" in result.detail

    # 预检这轮没有截断轨迹，但正常结局仍有负回报 → 仍应警告
    no_truncation = [make_trajectory(-0.3), make_trajectory(0.5)]
    assert judge_truncated_beats_normal(no_truncation, truncated_reward=0.0).status == WARNING

    # 正常结局都 ≥ truncated_reward
    clean = [make_trajectory(0.0), make_trajectory(0.5), make_trajectory(1.0)]
    assert judge_truncated_beats_normal(clean, truncated_reward=0.0).status == PASS

    # 关掉覆盖则不比
    assert judge_truncated_beats_normal(no_truncation, truncated_reward=None).status == PASS


def test_sampling_includes_truncated_beats_normal_check():
    trajectories = [make_trajectory(-0.5), make_trajectory(1.0)]
    names = {result.name for result in judge_sampling(
        trajectories, trajectories, max_new_tokens=32, truncated_reward=0.0)}
    assert "截断优于正常结局" in names


def test_oversized_row_fraction_warn_and_fail():
    # 每条轨迹一步；prompt+action 长度 = action_length + 1（prompt 为 [1]）
    long = [make_trajectory(0.0, action_length=99) for _ in range(3)]
    short = [make_trajectory(0.0, action_length=4) for _ in range(7)]
    # 3/10 = 30% > 20% → 失败
    fail = judge_oversized_row_fraction(long + short, max_tokens=50)
    assert fail.status == FAIL and "30%" in fail.detail

    # 1/10 = 10% → 警告
    warn = judge_oversized_row_fraction(
        [make_trajectory(0.0, action_length=99)]
        + [make_trajectory(0.0, action_length=4) for _ in range(9)],
        max_tokens=50)
    assert warn.status == WARNING and "10%" in warn.detail

    ok = judge_oversized_row_fraction(short, max_tokens=50)
    assert ok.status == PASS and "0%" in ok.detail


def test_step_signal_coverage_warns_when_gigpo_has_no_anchors():
    assert judge_step_signal_coverage(0.0, 1.0).status == WARNING
    assert judge_step_signal_coverage(0.04, 1.0).status == WARNING
    assert judge_step_signal_coverage(0.5, 1.0).status == PASS


def test_log_probability_judgement():
    assert judge_log_probabilities([-0.5] * 1000).status == PASS
    assert judge_log_probabilities([-0.5] * 999 + [-30.0]).status == PASS        # 千分之一，偶发
    assert judge_log_probabilities([-0.5] * 90 + [-30.0] * 10).status == FAIL
    assert judge_log_probabilities([-4.0] * 100).status == WARNING

def test_first_step_ratio_judgement():
    assert judge_first_step_ratio(0.0).status == PASS
    assert judge_first_step_ratio(0.05).status == WARNING


def test_backend_log_probability_gap_uses_tail():
    assert judge_backend_log_probability_gap({}).status == WARNING
    assert judge_backend_log_probability_gap({
        "backend_log_probability_gap_median": 1e-7,
        "backend_log_probability_gap_p95": 5e-4,
        "backend_log_probability_gap_above_half_fraction": 0.0,
        "backend_log_probability_gap_max": 0.12,
    }).status == PASS
    # 高熵任务：p95 偏高但无尾部大偏差 → 只警告
    warned = judge_backend_log_probability_gap({
        "backend_log_probability_gap_median": 1e-3,
        "backend_log_probability_gap_p95": 0.05,
        "backend_log_probability_gap_above_half_fraction": 0.0,
        "backend_log_probability_gap_max": 0.24,
    })
    assert warned.status == WARNING
    assert "高熵" in warned.suggestion
    # 孤立大偏差：max 略高于旧阈值 1，但远低于新阈值 5，且比例正常 → 通过
    assert judge_backend_log_probability_gap({
        "backend_log_probability_gap_median": 1e-7,
        "backend_log_probability_gap_p95": 1e-5,
        "backend_log_probability_gap_above_half_fraction": 1.1e-4,
        "backend_log_probability_gap_max": 1.03,
    }).status == PASS
    # 配置错位：尾部大偏差（比例超标）
    failed = judge_backend_log_probability_gap({
        "backend_log_probability_gap_median": 1e-7,
        "backend_log_probability_gap_p95": 0.05,
        "backend_log_probability_gap_above_half_fraction": 0.02,
        "backend_log_probability_gap_max": 8.0,
    })
    assert failed.status == FAIL
    assert "rope_theta" in failed.suggestion
    # 配置错位：单点 max 也到了错位量级
    assert judge_backend_log_probability_gap({
        "backend_log_probability_gap_median": 1e-7,
        "backend_log_probability_gap_p95": 1e-4,
        "backend_log_probability_gap_above_half_fraction": 0.0,
        "backend_log_probability_gap_max": 5.1,
    }).status == FAIL

def shift(median, percentile_95, clip_fraction):
    return {"shift_median_absolute_log_ratio": median, "shift_percentile_95_absolute_log_ratio": percentile_95,
            "shift_clip_fraction": clip_fraction}

def test_shift_judgement():
    assert judge_shift(shift(4e-3, 0.15, 0.04), 1e-5).status == PASS
    assert judge_shift(shift(0.0, 0.0, 0.0), 1e-5).status == FAIL
    assert judge_shift(shift(5e-2, 0.9, 0.35), 1e-3).status == WARNING          # 裁剪比例过高
    assert judge_shift(shift(1e-3, 5.0, 0.12), 1e-4).status == WARNING          # 95 分位过高，裁剪比例未超标
    assert judge_shift(shift(1e-7, 1e-5, 0.0), 1e-9).status == WARNING          # 95 分位过小
    assert judge_shift(shift(1e-7, 0.1, 0.0), 1e-5).status == PASS               # 中位数小但 95 分位正常
    # bf16：底噪与满学习率同量级，不作过大/过小判断；裁剪爆了仍警告
    bf16_floor = judge_shift(shift(5.6e-3, 0.087, 0.005), 1e-8, mixed_precision=True)
    assert bf16_floor.status == PASS and "bf16" in bf16_floor.detail
    assert judge_shift(shift(4e-3, 0.09, 0.0), 1e-5, mixed_precision=True).status == PASS
    assert judge_shift(shift(5e-2, 0.9, 0.35), 1e-3, mixed_precision=True).status == WARNING
    assert judge_shift(shift(0.0, 0.0, 0.0), 1e-5, mixed_precision=True).status == FAIL

def test_gradient_checkpointing_effect_warns_when_peak_does_not_drop():
    gigabyte = 2 ** 30
    assert judge_gradient_checkpointing_effect(8 * gigabyte, 5 * gigabyte).status == PASS
    assert judge_gradient_checkpointing_effect(None, 5 * gigabyte).status == PASS
    unchanged = judge_gradient_checkpointing_effect(8 * gigabyte, 8 * gigabyte)
    assert unchanged.status == WARNING
    assert "没有下降" in unchanged.detail
    higher = judge_gradient_checkpointing_effect(8 * gigabyte, 8 * gigabyte + 4096)
    assert higher.status == WARNING


def test_memory_judgement():
    from grpx.loop import TRAINING_MEMORY_FRACTION

    gigabyte = 2 ** 30
    total = 24 * gigabyte
    limit = total * TRAINING_MEMORY_FRACTION  # 0.6 → 14.4 GB
    # 远低于上限
    assert judge_memory({"采样": 4 * gigabyte, "更新": 8 * gigabyte}, total).status == PASS
    # 超过上限 80%、仍低于上限 → 警告
    soft = judge_memory({"采样": 4 * gigabyte, "最坏小批": int(0.85 * limit)}, total)
    assert soft.status == WARNING and "max_tokens_per_minibatch" in soft.suggestion
    # 超过训练显存上限 → 失败（与训练 set_per_process_memory_fraction 对齐）
    hard = judge_memory({"采样": 4 * gigabyte, "最坏小批": int(0.95 * total)}, total)
    assert hard.status == FAIL and "max_tokens_per_minibatch" in hard.suggestion
    sampling = judge_memory({"采样": int(0.95 * total), "更新": 4 * gigabyte}, total)
    assert sampling.status == FAIL and "max_tokens_per_batch" in sampling.suggestion


def test_worst_case_batch_fills_token_budget():
    batch, row_count, row_length, steps_per_unit = build_worst_case_batch(
        prompt_token_ids=[1, 2, 3, 4],
        max_new_tokens=6,
        max_rows=8,
        max_tokens=40,
        padding_token_id=0,
        device="cpu",
    )
    assert row_length == 10 and steps_per_unit == 1
    assert row_count == 4    # 40 // 10，且不超过 max_rows=8
    assert batch["input_token_ids"].shape == (4, 10)
    assert batch["action_mask"].sum().item() == 4 * 6

    alone, alone_count, alone_length, alone_steps = build_worst_case_batch(
        [1] * 20, max_new_tokens=20, max_rows=8, max_tokens=30,
        padding_token_id=0, device="cpu",
    )
    assert alone_count == 1 and alone_length == 30 and alone_steps == 1
    assert alone["input_token_ids"].shape[0] == 1


def test_worst_case_batch_merged_uses_steps_and_segment_budget():
    # shape=long 在均匀行枚举里取总面积最大：steps=2×4 行面积 44 最大
    batch, unit_count, row_length, steps_per_unit = build_worst_case_batch(
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=4,
        max_rows=8,
        max_tokens=10**6,
        padding_token_id=0,
        device="cpu",
        max_steps=6,
        merge_append_only=True,
        shape="long",
    )
    assert steps_per_unit == 2 and unit_count == 4
    assert row_length == 3 + 1 * 4 + 4
    assert batch["segment_advantages"].shape[0] == 8
    assert batch["action_mask"].sum().item() == 8 * 4


def test_worst_case_batch_long_uses_observed_prompt_growth():
    from grpx.doctor import max_prompt_growth_tokens
    from grpx.types import Step, Trajectory

    trajectory = Trajectory("g", 0, 1.0, [
        Step([1, 2], [3], advantage=1.0),
        Step([1, 2, 3, 8, 9, 10], [4], advantage=1.0),  # 增长 4
        Step([1, 2, 3, 8, 9, 10, 4, 11, 12, 13, 14, 15], [5], advantage=1.0),  # 增长 6
    ])
    assert max_prompt_growth_tokens([trajectory], minimum=1) == 6

    batch, unit_count, row_length, steps_per_unit = build_worst_case_batch(
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=2,
        max_rows=4,
        max_tokens=10**6,
        padding_token_id=0,
        device="cpu",
        max_steps=4,
        merge_append_only=True,
        shape="long",
        tokens_per_step=6,
        row_count=1,
        steps_per_unit=4,
    )
    assert steps_per_unit == 4 and unit_count == 1
    assert row_length == 3 + 3 * 6 + 2
    assert batch["action_mask"].sum().item() == 4 * 2


def test_worst_case_batch_short_shape_is_one_step_rows():
    batch, unit_count, row_length, steps_per_unit = build_worst_case_batch(
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=4,
        max_rows=8,
        max_tokens=10**6,
        padding_token_id=0,
        device="cpu",
        max_steps=6,
        merge_append_only=True,
        shape="short",
    )
    assert steps_per_unit == 1
    assert row_length == 7
    assert unit_count == 8
    assert batch["segment_advantages"].shape[0] == 8
    assert batch["input_token_ids"].shape == (8, 7)


def test_enumerate_worst_case_respects_both_budgets_and_picks_max_area():
    # 完整行长 10；token 预算 40 → 面积最大 40（例如 4×10）；也允许更短行用满预算（如 8×5）
    specs = enumerate_worst_case_specs(
        prompt_length=4, max_new_tokens=6, max_rows=8, max_tokens=40,
        max_steps=1, merge_append_only=False,
    )
    assert all(spec.segments <= 8 for spec in specs)
    assert all(spec.row_count == 1 or spec.area <= 40 for spec in specs)
    assert max(spec.area for spec in specs) == 40
    assert any(spec.row_count == 4 and spec.row_length == 10 for spec in specs)

    targets = select_worst_case_probe_targets(specs)
    # 面积并列时优先更长的行 → 4×10
    assert any(spec.area == 40 and spec.row_length == 10 for spec in targets)


def test_select_worst_case_probes_area_and_answer_extrema():
    # 合并开启：短行多回答位 vs 长行大面积，可能是两个目标
    specs = enumerate_worst_case_specs(
        prompt_length=3, max_new_tokens=4, max_rows=8, max_tokens=10**6,
        max_steps=6, merge_append_only=True, tokens_per_step=4,
    )
    by_area = max(specs, key=lambda spec: spec.area)
    by_answers = max(specs, key=lambda spec: spec.answer_positions)
    targets = select_worst_case_probe_targets(specs)
    keys = {(spec.steps_per_unit, spec.row_count) for spec in targets}
    assert (by_area.steps_per_unit, by_area.row_count) in keys
    assert (by_answers.steps_per_unit, by_answers.row_count) in keys
    if (by_area.steps_per_unit, by_area.row_count) == (by_answers.steps_per_unit, by_answers.row_count):
        assert len(targets) == 1
    else:
        assert len(targets) == 2


def test_oversized_row_is_capped_at_token_budget_not_probed_uncapped():
    # 完整最坏 40 > 预算 30：只摸封顶后的形状，不再摸未截断的 40
    specs = enumerate_worst_case_specs(
        prompt_length=20, max_new_tokens=20, max_rows=8, max_tokens=30,
        max_steps=1, merge_append_only=False,
    )
    assert all(spec.row_length <= 30 for spec in specs)
    assert not any(spec.row_length == 40 for spec in specs)
    assert any(spec.row_count == 1 and spec.row_length == 30 for spec in specs)
    # 用满预算时也可能有短行多行形状（行长 = min(40, 30//n)）
    assert any(spec.row_count > 1 and spec.area <= 30 for spec in specs)


def test_merge_steps_beyond_row_budget_still_gets_capped_single_row():
    # max_rows=8、max_steps=10 → 10 步时 8//10=0，仍应有封顶单行
    specs = enumerate_worst_case_specs(
        prompt_length=199, max_new_tokens=288, max_rows=8, max_tokens=3072,
        max_steps=10, merge_append_only=True, tokens_per_step=288,
    )
    ten = [spec for spec in specs if spec.steps_per_unit == 10]
    assert ten and ten[0].row_count == 1 and ten[0].row_length == 3072
    assert all(spec.row_length <= 3072 for spec in specs)


def test_worst_case_exceeds_budget_warns_with_drop_stats():
    from grpx.doctor import judge_worst_case_exceeds_token_budget
    ok = judge_worst_case_exceeds_token_budget(3000, 3072)
    assert ok.status == PASS
    warn = judge_worst_case_exceeds_token_budget(
        3400, 3072, oversized_fraction=0.0, oversized_count=0, oversized_total=173)
    assert warn.status == WARNING
    assert "满长的局会被丢弃" in warn.detail
    assert "0/173" in warn.detail


def test_enumerate_caps_row_length_by_token_budget_per_row():
    # 完整最坏行长 20；预算 50 → 3 行时应取 min(20, 50//3)=16，面积 48
    specs = enumerate_worst_case_specs(
        prompt_length=10, max_new_tokens=10, max_rows=8, max_tokens=50,
        max_steps=1, merge_append_only=False,
    )
    three = [spec for spec in specs if spec.row_count == 3]
    assert three and three[0].row_length == 16 and three[0].area == 48
    assert max(spec.area for spec in specs if spec.row_length * spec.row_count <= 50) >= 48


def test_duration_estimate():
    config = TrainingConfig(iterations=30, evaluation_interval=5, groups_per_iteration=16, group_size=8)
    result = estimate_duration(config, sampling_seconds=20, update_seconds=20, evaluation_size=128)
    # 每轮 40 秒 × 30 = 1200 秒；评测 7 次（第 0、5、…、25 轮和最终），每次按 128/128 × 20 秒
    assert "每轮约 40 秒" in result.detail and "7 次评测" in result.detail
    assert f"约 {(1200 + 7 * 20) / 60:.0f} 分钟" in result.detail

def test_evaluation_interval_must_be_positive():
    with pytest.raises(ValueError, match="evaluation_interval 必须是正整数"):
        TrainingConfig(evaluation_interval=0)


def test_report_counts_and_failure():
    lines = []
    report = Report(output=lines.append)
    report.add(CheckResult("a", "x", PASS, "ok"), CheckResult("a", "y", WARNING, "hmm", "do this"))
    assert not report.failed and "建议：do this" in lines[1]
    report.add(CheckResult("a", "z", FAIL, "bad"))
    assert report.failed and report.summary() == "汇总：通过 1，警告 1，失败 1"


def _linear_stack(module_names):
    """按 dotted 名挂若干 Linear，便于测 LoRA 目标匹配。"""
    import torch

    root = torch.nn.Module()
    for name in module_names:
        parent, _, leaf = name.rpartition(".")
        holder = root
        if parent:
            for part in parent.split("."):
                if not hasattr(holder, part):
                    holder.add_module(part, torch.nn.Module())
                holder = getattr(holder, part)
        holder.add_module(leaf, torch.nn.Linear(4, 4, bias=False))
    return root


def test_summarize_lora_targets_matches_suffix_and_counts_experts_routers():
    from grpx.doctor import summarize_lora_targets

    model = _linear_stack([
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
        "model.layers.0.mlp.gate",
    ])
    attention_only = summarize_lora_targets(
        model, ["q_proj", "k_proj", "v_proj", "o_proj"])
    assert attention_only["matched_module_count"] == 2
    assert attention_only["expert_module_count"] == 0
    assert attention_only["router_module_count"] == 0

    with_mlp = summarize_lora_targets(
        model, ["q_proj", "gate_proj", "up_proj", "gate"])
    assert with_mlp["matched_module_count"] == 4
    assert with_mlp["expert_module_count"] == 2
    assert with_mlp["router_module_count"] == 1


def test_summarize_lora_targets_all_linear_includes_experts():
    from grpx.doctor import summarize_lora_targets

    model = _linear_stack([
        "layers.0.q_proj",
        "layers.0.experts.3.down_proj",
        "layers.0.gate",
    ])
    summary = summarize_lora_targets(model, "all-linear")
    assert summary["matched_module_count"] == 3
    assert summary["expert_module_count"] == 1
    assert summary["router_module_count"] == 1


def test_judge_lora_targets_fail_warn_pass():
    from grpx.doctor import judge_lora_targets

    failed = statuses(judge_lora_targets({
        "matched_module_count": 0, "expert_module_count": 0, "router_module_count": 0,
        "matched_parameter_count": 0,
    }))
    assert failed["LoRA 目标模块"] == FAIL

    warned = statuses(judge_lora_targets({
        "matched_module_count": 10, "expert_module_count": 8, "router_module_count": 1,
        "expert_parameter_count": 1_000_000, "matched_parameter_count": 0,
        "router_parameter_names": [], "expert_parameter_names": [],
    }))
    assert warned["LoRA 目标模块"] == PASS
    assert warned["LoRA 与专家层"] == WARNING
    assert warned["LoRA 与路由"] == WARNING

    ok = statuses(judge_lora_targets({
        "matched_module_count": 32, "expert_module_count": 0, "router_module_count": 0,
        "matched_parameter_count": 0,
    }))
    assert ok == {"LoRA 目标模块": PASS}

    param_warned = statuses(judge_lora_targets({
        "matched_module_count": 4, "expert_module_count": 0, "router_module_count": 16,
        "matched_parameter_count": 16, "matched_parameter_numel": 2_097_152,
        "matched_parameter_names": ["model.layers.0.mlp.gate.weight"],
        "router_parameter_names": ["model.layers.0.mlp.gate.weight"],
        "expert_parameter_names": [], "expert_parameter_count": 0,
    }))
    assert param_warned["LoRA 目标参数"] == PASS
    assert param_warned["LoRA 与路由"] == WARNING


def test_judge_chat_template():
    from grpx.doctor import judge_chat_template

    class Tokenizer:
        def __init__(self, template):
            self.chat_template = template

    assert judge_chat_template(Tokenizer("{{ messages }}")).status == PASS
    assert judge_chat_template(Tokenizer(None)).status == FAIL
    assert judge_chat_template(Tokenizer("")).status == FAIL
    assert judge_chat_template(Tokenizer("   ")).status == FAIL
