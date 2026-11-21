import json
import os
import pathlib
import shutil

import torch

from grpx.device import cuda_device_ordinal, indexed_cuda_device


def _capture_cuda_rng_state(device):
    cuda_device = indexed_cuda_device(device) if device is not None else None
    if cuda_device is None or not torch.cuda.is_available():
        return None
    return torch.cuda.get_rng_state(cuda_device)


def _restore_cuda_rng_state(states, device):
    """兼容旧检查点：cuda_random_states 曾是 get_rng_state_all() 的列表。"""
    if states is None:
        return
    cuda_device = indexed_cuda_device(device) if device is not None else None
    if cuda_device is None or not torch.cuda.is_available():
        return
    if isinstance(states, list):
        index = cuda_device_ordinal(cuda_device)
        if index >= len(states):
            raise ValueError(
                f"旧格式检查点只有 {len(states)} 个 CUDA RNG 状态，无法恢复 device={device!r}")
        torch.cuda.set_rng_state(states[index], cuda_device)
        return
    torch.cuda.set_rng_state(states, cuda_device)


def save_checkpoint(run_directory, completed_iterations, model, optimizer, scheduler,
                    random_generator, device=None, schedule_horizon=None):
    final_directory = run_directory / f"checkpoint-{completed_iterations:04d}"
    partial_directory = run_directory / f"checkpoint-{completed_iterations:04d}.partial"
    if partial_directory.exists():
        shutil.rmtree(partial_directory)
    partial_directory.mkdir(parents=True)

    model.save_pretrained(partial_directory)         # 适配器格式，供推理和外部工具使用
    state = {
        "completed_iterations": completed_iterations,
        "trainable_parameters": {name: parameter.detach().cpu()
                                 for name, parameter in model.named_parameters()
                                 if parameter.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_random_state": random_generator.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": _capture_cuda_rng_state(device),
    }
    # 学习率衰减的总长度按开跑时的 iterations 固定；延长训练时超出部分停在终点学习率
    if schedule_horizon is not None:
        state["schedule_horizon"] = schedule_horizon
    torch.save(state, partial_directory / "training_state.pt")

    if final_directory.exists():
        shutil.rmtree(final_directory)
    os.replace(partial_directory, final_directory)   # 写完整了才改成正式名字
    return final_directory


def prune_rolling_checkpoints(run_directory, keep_completed, interval, total_iterations):
    """轨迹保存打开时，间隔之外只留最新一份检查点。

    间隔边界和最后一轮仍全部保留。这样中断时加载到的权重就是正在进行的那一轮开头，
    轨迹和累积梯度的校验才对得上。
    """
    for directory in pathlib.Path(run_directory).glob("checkpoint-*"):
        if not directory.is_dir() or directory.name.endswith(".partial"):
            continue
        if not (directory / "training_state.pt").is_file():
            continue
        suffix = directory.name[len("checkpoint-"):]
        if not suffix.isdigit():
            continue
        completed = int(suffix)
        if completed == keep_completed:
            continue
        if completed % interval == 0 or completed == total_iterations:
            continue
        shutil.rmtree(directory)


def find_latest_checkpoint(run_directory):
    candidates = [directory for directory in run_directory.glob("checkpoint-*")
                  if directory.is_dir() and not directory.name.endswith(".partial")
                  and (directory / "training_state.pt").exists()]
    return max(candidates, key=lambda directory: int(directory.name.split("-")[1]),
               default=None)


def load_checkpoint(checkpoint_directory, model, optimizer, scheduler, random_generator,
                    device=None):
    state = torch.load(checkpoint_directory / "training_state.pt",
                       map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    trainable_names = {name for name, parameter in parameters.items()
                       if parameter.requires_grad}
    saved_parameters = state["trainable_parameters"]
    if set(saved_parameters) != trainable_names:
        raise ValueError("检查点里的可训练参数与当前模型不一致，"
                         "请确认模型和 LoRA 配置与原运行相同")
    with torch.no_grad():
        for name, value in saved_parameters.items():
            parameters[name].copy_(value)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random_generator.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    _restore_cuda_rng_state(state["cuda_random_states"], device)
    return state["completed_iterations"], state.get("schedule_horizon")


def truncate_metrics(metrics_path, start_iteration):
    """丢掉检查点之后写入的记录，返回丢掉的条数"""
    if not metrics_path.exists():
        metrics_path.write_text("", encoding="utf-8")
        return 0
    lines = [line for line in metrics_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    kept = [line for line in lines if json.loads(line)["iteration"] < start_iteration]
    metrics_path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    return len(lines) - len(kept)
