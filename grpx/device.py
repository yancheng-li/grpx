"""训练设备辅助：统一解析配置里的 device 字符串。"""


def resolve_device(device_name):
    """把配置字符串收成 torch.device；cuda 不可用时立刻报错。"""
    import torch

    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"无法解析 device={device_name!r}：{error}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device={device_name!r} 需要 CUDA，但当前环境看不到 GPU")
    if device.type not in ("cuda", "cpu"):
        raise ValueError(f"device 目前只支持 cuda / cpu，收到 {device_name!r}")
    return device


def is_cuda_device(device):
    import torch
    return torch.device(device).type == "cuda"


def cuda_device_ordinal(device):
    """用于按设备读写 RNG / 显存统计；'cuda' 落到 current_device。"""
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"cuda_device_ordinal 只接受 cuda 设备，收到 {device!r}")
    if resolved.index is not None:
        return resolved.index
    return torch.cuda.current_device()


def indexed_cuda_device(device):
    """torch.cuda.* 部分接口不接受无 index 的 'cuda'，收成 cuda:N。"""
    import torch

    if not is_cuda_device(device):
        return None
    resolved = torch.device(device)
    if resolved.index is not None:
        return resolved
    return torch.device("cuda", torch.cuda.current_device())
