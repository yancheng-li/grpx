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


def physical_device_key(device):
    """用来判断两份配置是不是同一张卡。没有 CUDA 时，cuda 设备退回字符串本身。"""
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return str(resolved)
    if not torch.cuda.is_available():
        return str(resolved)
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    return f"cuda:{index}"


def duplicate_physical_devices(devices):
    """返回物理上重复的设备名。cuda 和 cuda:0 在当前设备是 0 时算同一张卡。"""
    seen = {}
    duplicates = []
    for name in devices:
        key = physical_device_key(name)
        if key in seen:
            duplicates.append(name)
        else:
            seen[key] = name
    return duplicates


def indexed_cuda_device(device):
    """torch.cuda.* 部分接口不接受无 index 的 'cuda'，收成 cuda:N。没有 CUDA 时返回 None。"""
    import torch

    if not torch.cuda.is_available() or not is_cuda_device(device):
        return None
    resolved = torch.device(device)
    if resolved.index is not None:
        return resolved
    return torch.device("cuda", torch.cuda.current_device())
