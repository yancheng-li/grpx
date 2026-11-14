import copy

import pytest
import torch

from grpx.advantage import compute_advantages, discounted_step_returns, episode_advantages
from grpx.loop import ALGORITHM_PRESETS, TrainingConfig, split_into_row_chunks
from grpx.trainer import build_batch, grpo_loss
from grpx.types import Step, Trajectory


def make_trajectory(states, rewards, group_id="g"):
    steps = [Step(prompt_token_ids=[1], action_token_ids=[2], reward=reward, state_key=state)
             for state, reward in zip(states, rewards)]
    return Trajectory(group_id, 0, 1.0, steps)

def advantages_of(trajectories):
    return [[step.advantage for step in trajectory.steps] for trajectory in trajectories]


def test_gigpo_hand_computed_example():
    first = make_trajectory(["A", "B"], [0, 1])
    second = make_trajectory(["A", "A"], [0, 0])
    kept = compute_advantages([first, second], normalize_std=False,
                              step_advantage_weight=1.0, discount_factor=1.0)
    expected = [[0.5 + 2 / 3, 0.5], [-0.5 - 1 / 3, -0.5 - 1 / 3]]
    for actual_row, expected_row in zip(advantages_of(kept), expected):
        assert actual_row == pytest.approx(expected_row)

def test_step_advantage_weight_scales_step_term():
    first = make_trajectory(["A", "B"], [0, 1])
    second = make_trajectory(["A", "A"], [0, 0])
    kept = compute_advantages([first, second], normalize_std=False,
                              step_advantage_weight=0.5, discount_factor=1.0)
    assert advantages_of(kept)[0] == pytest.approx([0.5 + 0.5 * 2 / 3, 0.5])

def test_zero_step_weight_matches_episode_advantages_exactly():
    trajectories = [make_trajectory(["A", "B", "A"], [0, 0.3, 1]), make_trajectory(["A", "A"], [0.2, 0]),
                    make_trajectory(["B"], [0.7]), make_trajectory(["A", "C"], [0, 0], group_id="h"),
                    make_trajectory(["C"], [1], group_id="h")]
    for normalize_std in (True, False):
        reference = episode_advantages(copy.deepcopy(trajectories), normalize_standard_deviation=normalize_std)
        combined = compute_advantages(copy.deepcopy(trajectories), normalize_std=normalize_std,
                                      step_advantage_weight=0.0, discount_factor=0.95)
        assert advantages_of(combined) == advantages_of(reference)

def test_discounted_step_returns():
    assert discounted_step_returns(make_trajectory(["A", "B", "C"], [0, 0, 1]), 0.5) == [0.25, 0.5, 1]

def test_group_without_episode_difference_but_with_step_difference_is_kept():
    # 总回报都是 1；锚状态 X 的三个成员步级回报为 1、1、0
    first = make_trajectory(["X", "Y"], [0, 1])
    second = make_trajectory(["X", "X"], [1, 0])
    assert compute_advantages(copy.deepcopy([first, second]), step_advantage_weight=0.0) == []
    kept = compute_advantages([first, second], normalize_std=False, step_advantage_weight=1.0, discount_factor=1.0)
    assert advantages_of(kept)[0] == pytest.approx([1 / 3, 0.0])
    assert advantages_of(kept)[1] == pytest.approx([1 / 3, -2 / 3])

def test_empty_state_keys_reduce_gigpo_to_grpo():
    trajectories = [make_trajectory(["", ""], [0, 1]), make_trajectory([""], [0]), make_trajectory([""], [0.4])]
    grpo = compute_advantages(copy.deepcopy(trajectories), step_advantage_weight=0.0)
    gigpo = compute_advantages(copy.deepcopy(trajectories), step_advantage_weight=1.0, discount_factor=0.95)
    assert advantages_of(gigpo) == advantages_of(grpo)


def test_step_signal_coverage():
    # 锚状态 A 有三个成员且回报不同；B 只有一个成员，没有步级信号
    trajectories = [make_trajectory(["A", "B"], [0, 1]), make_trajectory(["A", "A"], [0, 0]),
                    make_trajectory(["C"], [1], group_id="h"), make_trajectory(["C"], [1], group_id="h")]
    for weight in (1.0, 0.0):
        diagnostics = {}
        compute_advantages(copy.deepcopy(trajectories), step_advantage_weight=weight, diagnostics=diagnostics)
        assert diagnostics["step_signal_coverage"] == pytest.approx(3 / 6)

def test_diagnostics_do_not_change_baseline_behaviour():
    # 轨迹级无差异、步级有差异的组：基线照样丢弃，即使为了统计算了步级优势
    trajectories = [make_trajectory(["X", "Y"], [0, 1]), make_trajectory(["X", "X"], [1, 0])]
    diagnostics = {}
    assert compute_advantages(trajectories, step_advantage_weight=0.0, diagnostics=diagnostics) == []
    assert diagnostics["step_signal_coverage"] == pytest.approx(3 / 4)

def test_row_weights_in_batch():
    from grpx.trainer import prepare_update_units
    one_step = make_trajectory(["A"], [1])
    three_steps = make_trajectory(["A", "B", "C"], [0, 0, 0])
    units = prepare_update_units([one_step, three_steps], merge_append_only=False)
    assert build_batch(units, 0, "cpu")["row_weights"].tolist() == [1, 1, 1, 1]
    assert build_batch(units, 0, "cpu", "per_trajectory")["row_weights"].tolist() \
        == pytest.approx([1, 1 / 3, 1 / 3, 1 / 3])
    with pytest.raises(ValueError):
        build_batch([one_step], 0, "cpu", "per_token")

@pytest.mark.parametrize("length_normalization", ["per_sequence", "constant"])
def test_per_trajectory_weighting_gives_each_trajectory_equal_total_contribution(length_normalization):
    one_step = make_trajectory(["A"], [1])
    three_steps = make_trajectory(["A", "B", "C"], [0, 0, 0])
    for step in one_step.steps:
        step.advantage = 1.0
    for step in three_steps.steps:
        step.advantage = -1.0
    # 先拆成单步行再组批，和训练时的路径一致：行权重要从原轨迹的步数来
    rows = [row for chunk in split_into_row_chunks([one_step, three_steps], 1, 10**6) for row in chunk]
    options = {"length_normalization": length_normalization, "constant_length": 1}

    def loss_for(row_weighting):
        batch = build_batch(rows, 0, "cpu", row_weighting)
        log_probabilities = torch.zeros(batch["action_mask"][:, 1:].shape)
        loss, _ = grpo_loss(log_probabilities, log_probabilities, batch["action_mask"][:, 1:],
                            batch["advantages"], row_weights=batch["row_weights"], **options)
        return loss.item()

    # 比值为 1 时每行的值就是它的优势：+1 的一条与 -1 的一条贡献相互抵消
    assert loss_for("per_trajectory") == pytest.approx(0.0, abs=1e-6)
    assert loss_for("per_step") == pytest.approx(-(1 - 3) / 4)

def test_explicit_unit_weights_equal_default():
    torch.manual_seed(0)
    new, old = torch.randn(3, 5), torch.randn(3, 5)
    mask = torch.rand(3, 5) > 0.3
    advantages = torch.randn(3)
    for options in ({"length_normalization": "per_sequence"},
                    {"length_normalization": "constant", "constant_length": 7}):
        default, _ = grpo_loss(new, old, mask, advantages, **options)
        weighted, _ = grpo_loss(new, old, mask, advantages, row_weights=torch.ones(3), **options)
        assert weighted.item() == default.item()


@pytest.mark.parametrize("algorithm, expected", [
    ("grpo", (True, "per_sequence", 0.0, 1.0)),
    ("dr_grpo", (False, "constant", 0.0, 1.0)),
    ("gigpo", (True, "per_sequence", 1.0, 0.95)),
])
def test_algorithm_presets(algorithm, expected):
    config = TrainingConfig.from_algorithm(algorithm)
    assert (config.normalize_std, config.length_normalization,
            config.step_advantage_weight, config.discount) == expected
    assert config.row_weighting == "per_step"

def test_overrides_win_over_presets():
    config = TrainingConfig.from_algorithm("gigpo", discount=0.9, row_weighting="per_trajectory", seed=3)
    assert (config.discount, config.row_weighting, config.seed, config.step_advantage_weight) == (0.9, "per_trajectory", 3, 1.0)
    assert set(ALGORITHM_PRESETS) == {"grpo", "dr_grpo", "gigpo"}
