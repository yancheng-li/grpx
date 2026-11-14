from grpx.types import Trajectory, Step
from grpx.advantage import episode_advantages

def make_trajectory(group_id, reward):
    return Trajectory(group_id, 0, 1.0, [Step(prompt_token_ids=[1], action_token_ids=[2], reward=reward)])

def test_basic():
    out = episode_advantages([make_trajectory("a", reward) for reward in (1, 0, 0, 1)])
    advantages = [trajectory.steps[0].advantage for trajectory in out]
    expected = [1, -1, -1, 1]
    assert all(abs(advantage - expected_advantage) < 1e-3 for advantage, expected_advantage in zip(advantages, expected))

def test_zero_variance_group_dropped():
    assert episode_advantages([make_trajectory("b", 1), make_trajectory("b", 1)]) == []

def test_nearly_equal_rewards_treated_as_zero_variance():
    assert 0.1 + 0.2 != 0.3
    assert episode_advantages([make_trajectory("b", 0.1 + 0.2), make_trajectory("b", 0.3)]) == []

def test_single_trajectory_group_dropped():
    assert episode_advantages([make_trajectory("b", 1)]) == []

def test_empty_input():
    assert episode_advantages([]) == []

def test_without_standard_deviation_normalization():
    out = episode_advantages([make_trajectory("c", reward) for reward in (1, 0)], normalize_standard_deviation=False)
    assert [trajectory.steps[0].advantage for trajectory in out] == [0.5, -0.5]

def test_groups_are_independent():
    trajectories = [
        make_trajectory("a", 1), make_trajectory("b", 5), make_trajectory("a", 0), make_trajectory("b", 5),
        make_trajectory("c", 3), make_trajectory("c", 1),
    ]
    out = episode_advantages(trajectories, normalize_standard_deviation=False)
    assert [(trajectory.group_id, trajectory.steps[0].advantage) for trajectory in out] == [
        ("a", 0.5), ("a", -0.5), ("c", 1.0), ("c", -1.0),
    ]

def test_all_steps_share_trajectory_advantage():
    steps = [Step(prompt_token_ids=[1], action_token_ids=[2], reward=reward) for reward in (1, 2)]
    out = episode_advantages([Trajectory("d", 0, 1.0, steps), make_trajectory("d", 1)], normalize_standard_deviation=False)
    assert [step.advantage for step in out[0].steps] == [1.0, 1.0]
