from dataclasses import dataclass, field
from typing import Literal, Optional

# done：环境判定结束；length_limit：某步生成达到长度上限，回答不完整；step_limit：每步回答完整，但用完了步数
TerminationReason = Literal["done", "length_limit", "step_limit"]
# 单步生成如何结束。stop_string 算正常停，不触发 truncated_reward；只有 length_limit 算截断。
GenerationFinishReason = Literal["end_token", "stop_string", "length_limit"]

@dataclass
class Step:
    prompt_token_ids: list[int]        # 该步模型真实看到的上下文
    action_token_ids: list[int]        # 模型生成的 token
    reward: float = 0.0
    state_key: str = ""          # GiGPO 占位
    advantage: float = 0.0
    # 采样后端返回的逐 token 对数概率；本地后端为 None，只用于与训练端对账
    backend_log_probabilities: list[float] | None = None
    # 采样该步的推理服务地址；多 base_url 时用于按地址统计 gap
    backend_base_url: str | None = None
    # 环境 step 返回的原始回报；截断覆盖后可能与 reward 不同，供预检识别
    environment_reward: float | None = None
    # 这一步生成的结束原因；旧轨迹 / 未填时为 None
    finish_reason: Optional[GenerationFinishReason] = None
    # 采样时每层选中的专家，int8，形状 (提示+回答, 层数, 每 token 专家数)。
    # 没有记录的位置是 -1。不重放时为 None
    routed_experts: object | None = None

@dataclass
class Trajectory:
    group_id: str
    policy_version: int
    sampling_temperature: float    # 行为策略是 softmax(logits / T)，训练端算 logprob 时须按同一个 T 缩放
    steps: list[Step] = field(default_factory=list)
    termination_reason: Optional[TerminationReason] = None
    is_successful: Optional[bool] = None    # 由环境判定，独立于回报；环境没有成功概念时为 None
    source_step_count: Optional[int] = None    # 拆成单步行后，原轨迹的步数；按轨迹等权时要用
    environment_metrics: dict = field(default_factory=dict)  # 环境可选的过程指标，如有效步比例

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)


def success_fraction(trajectories):
    """有轨迹缺少成功标志时返回 None，避免把"未知"当成"失败"计入"""
    if not trajectories or any(trajectory.is_successful is None for trajectory in trajectories):
        return None
    return sum(trajectory.is_successful for trajectory in trajectories) / len(trajectories)


def mean_environment_metrics(trajectories, prefix="environment_"):
    """对轨迹里出现过的环境指标键取平均；某条轨迹缺某个键时该键跳过该条"""
    totals, counts = {}, {}
    for trajectory in trajectories:
        for name, value in trajectory.environment_metrics.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    return {f"{prefix}{name}": totals[name] / counts[name] for name in totals}
