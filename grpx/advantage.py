from collections import defaultdict
import statistics

def _relative_advantages(values, normalize_std, epsilon=1e-6):
    """组内相对优势，组内无差异时返回None"""
    mean = sum(values) / len(values)
    std = statistics.pstdev(values)
    if std < epsilon:
        return None
    scale = std + epsilon if normalize_std else 1.0
    return [(value - mean) / scale for value in values]

def discounted_step_returns(trajectory, discount_factor):
    returns, running_return = [], 0.0
    for step in reversed(trajectory.steps):
        running_return = step.reward + discount_factor * running_return
        returns.append(running_return)
    return list(reversed(returns))

def _anchor_state_advantages(group, normalize_std, discount_factor, epsilon):
    """返回 {(轨迹序号, 步序号): 步级优势}"""
    members_by_state = defaultdict(list)
    for trajectory_index, trajectory in enumerate(group):
        step_returns = discounted_step_returns(trajectory, discount_factor)
        for step_index, (step, step_return) in enumerate(zip(trajectory.steps, step_returns)):
            if step.state_key:              # 空状态键表示环境不提供状态
                members_by_state[step.state_key].append((trajectory_index, step_index, step_return))
    advantages = {}
    for members in members_by_state.values():
        if len(members) < 2:
            continue
        relative = _relative_advantages(
            [step_return for _, _, step_return in members], normalize_std, epsilon
        )
        if relative is None:
            continue
        for (trajectory_index, step_index, _), advantage in zip(members, relative):
            advantages[(trajectory_index, step_index)] = advantage
    return advantages

def compute_advantages(trajectories, normalize_std=True, step_advantage_weight=0.0, discount_factor=1.0, epsilon=1e-6,
                       diagnostics=None):
    """diagnostics 传入字典时写入 step_signal_coverage：拿到步级优势的步占全部步的比例。
    权重为 0 时也统计，表示开启 GiGPO 的话会有多少步得到步级信号"""
    groups = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.group_id].append(trajectory)

    kept = []
    covered_step_count = 0
    for group in groups.values():
        episode = _relative_advantages(
            [trajectory.total_reward for trajectory in group], normalize_std, epsilon
        )
        step_level = (
            _anchor_state_advantages(group, normalize_std, discount_factor, epsilon)
            if step_advantage_weight or diagnostics is not None else {}
        )
        covered_step_count += len(step_level)
        if not step_advantage_weight:
            step_level = {}
        if episode is None and not step_level:
            continue
        for trajectory_index, trajectory in enumerate(group):
            episode_advantage = episode[trajectory_index] if episode else 0.0
            for step_index, step in enumerate(trajectory.steps):
                step.advantage = (episode_advantage
                                  + step_advantage_weight * step_level.get((trajectory_index, step_index), 0.0))
            kept.append(trajectory)
    if diagnostics is not None:
        total_step_count = sum(len(trajectory.steps) for trajectory in trajectories)
        diagnostics["step_signal_coverage"] = covered_step_count / total_step_count if total_step_count else 0.0
    return kept

def episode_advantages(trajectories, normalize_standard_deviation=True, epsilon=1e-6):
    return compute_advantages(trajectories, normalize_std=normalize_standard_deviation, epsilon=epsilon)
