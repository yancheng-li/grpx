import os
from dataclasses import dataclass

# 不导入 torch：命令行检查配置时要能立刻报错，不必等重的依赖加载

# 仅供测试。正式配置和预检都不读它。设为 1 时才允许 devices 里把同一张卡写两次，
# 用来在一张卡上把多副本路径跑通。预检仍然会拒绝。
SAME_DEVICE_REPLICAS_ENV = "GRPX_ALLOW_SAME_DEVICE_REPLICAS"

ALGORITHM_PRESETS = {
    "grpo":    {"normalize_std": True,  "length_normalization": "per_sequence",
                "step_advantage_weight": 0.0, "discount": 1.0},
    "dr_grpo": {"normalize_std": False, "length_normalization": "constant",
                "step_advantage_weight": 0.0, "discount": 1.0},
    "gigpo":   {"normalize_std": True,  "length_normalization": "per_sequence",
                "step_advantage_weight": 1.0, "discount": 0.95},
}

@dataclass
class TrainingConfig:
    iterations: int = 20
    groups_per_iteration: int = 16
    group_size: int = 8
    max_rows_per_minibatch: int = 8          # 按段（步）计；合并开启时一行多段仍占相应步数
    max_tokens_per_minibatch: int = 4096     # 物理行数 × 补齐后长度 的上限；用 peak_memory_gigabytes 实测后再调
    learning_rate: float = 1e-5
    warmup_iterations: int = 5
    # constant：预热后保持峰值；cosine：预热后余弦降到峰值 × final_learning_rate_fraction
    learning_rate_schedule: str = "constant"
    final_learning_rate_fraction: float = 0.1
    updates_per_iteration: int = 1
    max_approximate_kl: float | None = None
    max_gradient_norm: float = 1.0
    clip_range: float = 0.2
    normalize_std: bool = True
    length_normalization: str = "per_sequence"
    constant_length: int | None = None       # 不填时取后端的 max_new_tokens
    step_advantage_weight: float = 0.0       # GiGPO 步级优势的权重 ω；0 即 GRPO
    discount: float = 1.0                    # 步级回报的折扣 γ
    row_weighting: str = "per_step"          # "per_trajectory"：每条轨迹的各行权重之和为 1
    # 追加式多轮轨迹合并成一条序列只前向一次；默认关，在猜数字上验证后再开
    merge_append_only_turns: bool = False
    seed: int = 0
    output_root: str = "outputs"
    run_name: str | None = None
    evaluation_interval: int = 10
    checkpoint_interval: int = 5
    # 带梯度的前向重算激活，降低显存；评测模式不会生效。默认关
    gradient_checkpointing: bool = False
    # 采样后把本轮轨迹和小批划分写盘；恢复且校验通过时跳过采样
    save_iteration_trajectories: bool = False
    # 每累积多少个小批保存一次梯度；0 表示不保存。大于 0 时必须同时保存轨迹
    accumulation_checkpoint_interval: int = 0
    # 每隔多少轮测一次位移；0 表示不测。第 0 轮在 interval>0 时总会测（与评测间隔同一写法）
    shift_measurement_interval: int = 1
    max_steps: int = 6
    # 生成被长度截断时覆盖该步回报（非整局），并一律标为不成功；None 表示不覆盖（旧行为）
    truncated_reward: float | None = 0.0
    device: str = "cuda"                         # 训练设备，如 "cuda"、"cuda:1"、"cpu"；多卡时是 devices[0] 的简写
    devices: tuple[str, ...] | None = None       # 训练用的显卡。不写时等于 (device,)，一张卡走原来的单卡路径
    memory_fraction: float = 0.6                 # 训练进程显存上限，对 devices 里的每张卡分别生效
    # 输出层按回答 token 分块做线性+CE，降低 (N, vocab) 峰值；None/0 表示不分块
    output_logit_chunk_size: int | None = 512
    # 本地后端采样时记下每层选中的专家，训练前向强制使用同一选择。默认关。vLLM 还不支持
    routing_replay: bool = False

    def __post_init__(self):
        for name in ("iterations", "groups_per_iteration", "max_rows_per_minibatch", "max_tokens_per_minibatch",
                     "warmup_iterations", "updates_per_iteration", "evaluation_interval", "checkpoint_interval",
                     "max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数，收到 {value!r}")
        if self.output_logit_chunk_size is not None:
            if (isinstance(self.output_logit_chunk_size, bool)
                    or not isinstance(self.output_logit_chunk_size, int)
                    or self.output_logit_chunk_size < 0):
                raise ValueError(
                    f"output_logit_chunk_size 必须是非负整数或 null，收到 {self.output_logit_chunk_size!r}")
            if self.output_logit_chunk_size == 0:
                self.output_logit_chunk_size = None
        if (isinstance(self.shift_measurement_interval, bool)
                or not isinstance(self.shift_measurement_interval, int)
                or self.shift_measurement_interval < 0):
            raise ValueError(
                f"shift_measurement_interval 必须是非负整数，收到 {self.shift_measurement_interval!r}")
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int) or self.group_size < 2:
            raise ValueError(f"group_size 至少为 2（组内要有比较才有优势），收到 {self.group_size!r}")
        if self.length_normalization not in ("per_sequence", "constant"):
            raise ValueError(f"length_normalization 只能是 'per_sequence' 或 'constant'，收到 {self.length_normalization!r}")
        if self.row_weighting not in ("per_step", "per_trajectory"):
            raise ValueError(f"row_weighting 只能是 'per_step' 或 'per_trajectory'，收到 {self.row_weighting!r}")
        if not isinstance(self.merge_append_only_turns, bool):
            raise ValueError(f"merge_append_only_turns 必须是布尔值，收到 {self.merge_append_only_turns!r}")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError(
                f"gradient_checkpointing 必须是布尔值，收到 {self.gradient_checkpointing!r}")
        if not isinstance(self.routing_replay, bool):
            raise ValueError(f"routing_replay 必须是布尔值，收到 {self.routing_replay!r}")
        if self.routing_replay and self.merge_append_only_turns:
            raise ValueError(
                "routing_replay 还不支持把多轮合并成一行（merge_append_only_turns）")
        if not isinstance(self.save_iteration_trajectories, bool):
            raise ValueError(
                f"save_iteration_trajectories 必须是布尔值，收到 {self.save_iteration_trajectories!r}")
        if (isinstance(self.accumulation_checkpoint_interval, bool)
                or not isinstance(self.accumulation_checkpoint_interval, int)
                or self.accumulation_checkpoint_interval < 0):
            raise ValueError(
                "accumulation_checkpoint_interval 必须是非负整数，"
                f"收到 {self.accumulation_checkpoint_interval!r}")
        if self.accumulation_checkpoint_interval > 0 and not self.save_iteration_trajectories:
            raise ValueError(
                "accumulation_checkpoint_interval > 0 时必须打开 save_iteration_trajectories，"
                "否则恢复后会重新采样，累积到一半的梯度对不上")
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate 必须大于 0，收到 {self.learning_rate!r}")
        if self.learning_rate_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"learning_rate_schedule 只能是 'constant' 或 'cosine'，"
                f"收到 {self.learning_rate_schedule!r}")
        if (isinstance(self.final_learning_rate_fraction, bool)
                or not isinstance(self.final_learning_rate_fraction, (int, float))
                or not 0 <= float(self.final_learning_rate_fraction) <= 1):
            raise ValueError(
                f"final_learning_rate_fraction 必须在 [0, 1] 内，"
                f"收到 {self.final_learning_rate_fraction!r}")
        self.final_learning_rate_fraction = float(self.final_learning_rate_fraction)
        if self.truncated_reward is not None and isinstance(self.truncated_reward, bool):
            raise ValueError(f"truncated_reward 应是数字或 null，收到 {self.truncated_reward!r}")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError(f"device 必须是非空字符串，收到 {self.device!r}")
        self.device, self.devices = resolve_training_devices(self.device, self.devices)
        if (isinstance(self.memory_fraction, bool)
                or not isinstance(self.memory_fraction, (int, float))
                or not 0 < float(self.memory_fraction) <= 1):
            raise ValueError(
                f"memory_fraction 必须在 (0, 1] 内，收到 {self.memory_fraction!r}")
        self.memory_fraction = float(self.memory_fraction)

    @classmethod
    def from_algorithm(cls, algorithm, **overrides):
        return cls(**{**ALGORITHM_PRESETS[algorithm], **overrides})


def same_device_replicas_allowed():
    """测试开关。预检不调用它，所以打开开关也不会让预检放过同一张卡。"""
    return os.environ.get(SAME_DEVICE_REPLICAS_ENV) == "1"


def resolve_training_devices(device, devices):
    """device 是只写一张卡时的简写。返回 (主设备, 设备元组)。

    字符串重复时默认拒绝。测试要在一张卡上放两份副本时，设置
    GRPX_ALLOW_SAME_DEVICE_REPLICAS=1。cuda 和 cuda:0 是不是同一张卡仍交给预检，
    预检不看这个开关。
    """
    if not isinstance(device, str) or not device.strip():
        raise ValueError(f"device 必须是非空字符串，收到 {device!r}")
    device = device.strip()
    if devices is None:
        return device, (device,)
    if isinstance(devices, str) or not isinstance(devices, (list, tuple)) or len(devices) == 0:
        raise ValueError(f"devices 必须是非空的设备列表，收到 {devices!r}")
    normalized = []
    for item in devices:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"devices 里的每一项都必须是非空字符串，收到 {item!r}")
        normalized.append(item.strip())
    if len(set(normalized)) != len(normalized) and not same_device_replicas_allowed():
        raise ValueError(f"devices 里有重复的设备：{normalized}")
    if device != "cuda" and device != normalized[0]:
        raise ValueError(
            f"device 是 devices 的简写，必须等于列表的第一项 {normalized[0]!r}，收到 {device!r}")
    return normalized[0], tuple(normalized)
