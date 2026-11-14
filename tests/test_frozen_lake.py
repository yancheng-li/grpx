import pytest

from examples.frozen_lake import FrozenLakeEnvironment


def test_hole_penalty_rejects_negative():
    with pytest.raises(ValueError, match="不能为负数"):
        FrozenLakeEnvironment(hole_penalty=-0.1)
    with pytest.raises(ValueError, match="不能为负数"):
        FrozenLakeEnvironment(step_cost=-0.01)


def test_step_cost_applies_to_invalid_and_wall():
    grid = ["....", "....", "....", "...."]
    env = FrozenLakeEnvironment(reward_mode="progress", step_cost=0.04)
    env.reset({"grid": grid, "start": [0, 0]}, seed=0)
    _, reward, done = env.step("无效")
    assert not done and reward == -0.04
    _, reward, done = env.step("上")  # 撞岸
    assert not done and reward == -0.04
    assert env.progress_reward_total == pytest.approx(-0.08)


def test_success_and_hole_settle_without_extra_step_cost():
    # 掉洞一步：整局恰好 -hole_penalty（该步不另扣 step_cost）
    grid = [".H..", "....", "....", "...."]
    env = FrozenLakeEnvironment(reward_mode="progress", hole_penalty=0.5, step_cost=0.04)
    env.reset({"grid": grid, "start": [0, 0]}, seed=0)
    _, reward, done = env.step("右")
    assert done and reward == -0.5

    # 无洞直达：整局恰好 1
    clear = ["....", "....", "....", "...."]
    env = FrozenLakeEnvironment(reward_mode="progress", hole_penalty=0.5, step_cost=0.04)
    env.reset({"grid": clear, "start": [2, 3]}, seed=0)
    total = 0.0
    for action in ("下",):
        _, reward, done = env.step(action)
        total += reward
    assert done and env.is_successful()
    assert total == pytest.approx(1.0)


def test_hole_penalty_makes_fall_exactly_minus_penalty():
    # 起点右侧是洞：一步掉洞，进度分为 0，整局回报 = -hole_penalty
    grid = [".H..", "....", "....", "...."]
    env = FrozenLakeEnvironment(reward_mode="progress", hole_penalty=0.5)
    env.reset({"grid": grid, "start": [0, 0]}, seed=0)
    _, reward, done = env.step("右")
    assert done and env.fell_into_hole
    assert reward == -0.5


def test_hole_penalty_default_zero_keeps_old_fall_return():
    grid = [".H..", "....", "....", "...."]
    env = FrozenLakeEnvironment(reward_mode="progress")
    env.reset({"grid": grid, "start": [0, 0]}, seed=0)
    _, reward, done = env.step("右")
    assert done and reward == 0.0


def test_describe_neighbors_default_off():
    env = FrozenLakeEnvironment(reward_mode="progress")
    messages = env.reset({"grid": ["..H.", ".H..", "....", "...H"]}, seed=0)
    assert "边是" not in messages[0]["content"]
    feedback, _, done = env.step("右")
    assert not done
    assert "边是" not in feedback[0]["content"]


def test_describe_neighbors_lists_shore_hole_ice_goal():
    # 起点 (0,0) 可达终点；右是洞，下是冰，上/左湖岸
    grid = [
        ".H..",
        "....",
        ".H..",
        "....",
    ]
    env = FrozenLakeEnvironment(reward_mode="sparse", describe_neighbors=True)
    messages = env.reset({"grid": grid}, seed=0)
    text = messages[0]["content"]
    assert "上边是湖岸" in text
    assert "左边是湖岸" in text
    assert "右边是洞" in text
    assert "下边是冰" in text

    env.position = (2, 3)
    near_goal = env._describe_position()
    assert "下边是终点" in near_goal
    assert "你现在在第 3 行第 4 列。" in near_goal
    assert "左边是冰" in near_goal
