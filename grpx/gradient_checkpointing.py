"""梯度检查点的启用、查询，以及带梯度前向的模式切换。

transformers 6 会把 gradient_checkpointing_* 改名为 activation_checkpointing_*。
启用和查询都走 gradient_checkpointing()，以后只改这里。

两个不报错但结果是错的坑：
- 模型处于评测模式时，transformers 不会真正检查点。带梯度的前向必须先进训练模式，
  反向重算也还在这次前向的上下文里，结束再恢复原来的模式。
- 基座冻结时，检查点包住的输入往往不带梯度。use_reentrant=True 会把整段反向跳过，
  LoRA 梯度是空的。启用时固定 use_reentrant=False。
无梯度的前向（旧策略概率、测位移、路由）不要进 gradient_bearing_forward。
"""
from contextlib import contextmanager


def gradient_checkpointing(model, enabled=None):
    """enabled 为 None 时查询是否已启用，否则按该值开关。返回开关后的状态。"""
    if enabled is None:
        if hasattr(model, "is_activation_checkpointing"):
            return bool(model.is_activation_checkpointing)
        if hasattr(model, "is_gradient_checkpointing"):
            return bool(model.is_gradient_checkpointing)
        return False

    if enabled:
        kwargs = {"use_reentrant": False}
        if hasattr(model, "activation_checkpointing_enable"):
            model.activation_checkpointing_enable(activation_checkpointing_kwargs=kwargs)
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        else:
            raise ValueError(
                "当前模型没有梯度检查点接口（activation_checkpointing_enable / "
                "gradient_checkpointing_enable），无法启用 training.gradient_checkpointing")
    elif hasattr(model, "activation_checkpointing_disable"):
        model.activation_checkpointing_disable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return bool(enabled)


def _configs(model):
    found = []
    config = getattr(model, "config", None)
    if config is not None:
        found.append(config)
    getter = getattr(model, "get_base_model", None)
    if getter is not None:
        base_config = getattr(getter(), "config", None)
        if base_config is not None and all(base_config is not item for item in found):
            found.append(base_config)
    return found


def suspend_kv_cache(model):
    """进入训练模式并关掉 use_cache，直到调用返回的恢复函数。

    多卡的反向发生在另一层调用栈里，包不住同一次 with。恢复时只把 use_cache 改回去，
    模型留在训练模式，和单卡更新结束时的状态一致。
    """
    previous_use_cache = []
    for config in _configs(model):
        use_cache = getattr(config, "use_cache", None)
        previous_use_cache.append((config, use_cache))
        if use_cache:
            config.use_cache = False
    model.train()

    def restore():
        for config, use_cache in previous_use_cache:
            if use_cache:
                config.use_cache = use_cache
        model.train()

    return restore


@contextmanager
def gradient_bearing_forward(model):
    """包住带梯度的前向和紧接着的反向。

    检查点只在训练模式生效，反向重算走的还是这一次前向，所以反向结束前不能切回评测模式。
    结束后恢复进入时的模式，避免把 dropout 留在开着的状态。dropout 是否真的关上，
    由首步 ratio 是否为 1 来检查。
    """
    was_training = model.training
    previous_use_cache = []
    for config in _configs(model):
        use_cache = getattr(config, "use_cache", None)
        previous_use_cache.append((config, use_cache))
        if use_cache:
            config.use_cache = False
    model.train()
    try:
        yield
    finally:
        for config, use_cache in previous_use_cache:
            if use_cache:
                config.use_cache = use_cache
        model.train(was_training)
