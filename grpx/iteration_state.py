"""一轮中间的可恢复状态。

第一层：采样结束后立刻把轨迹、优势和小批划分写盘。恢复时校验通过就跳过采样。
第二层：每累积 K 个小批，把可训练参数的梯度、下一个小批序号和已累加的指标写盘。

结果不变靠四件事，都在调用方定死后再写盘：小批划分不再重算、归一化分母事先算好、
优势在整轮上一次算完、裁剪和优化器更新只在全部累积之后。优化器状态不在一轮中间变化，
沿用轮次边界的检查点。打开轨迹保存时，训练循环每轮都会留下最新检查点，所以恢复用的权重
就是这一轮开头，不必把 checkpoint_interval 改成 1。

中途梯度带一份指纹（配置哈希、轮次、轨迹哈希、适配器权重哈希）。任何一项对不上就丢弃，
并明确报出来，从这一轮的累积开头重做。写盘都是先写临时文件再改名。
"""
import dataclasses
import hashlib
import json
import os
import pathlib
import warnings
from typing import NamedTuple

import torch

from grpx.checkpoint import _capture_cuda_rng_state, _restore_cuda_rng_state
from grpx.types import Step, Trajectory


class LoadedIteration(NamedTuple):
    trajectories: list
    kept: list
    diagnostics: dict
    minibatch_plan: list
    content_hash: str


def config_hash(config):
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def adapter_weight_hash(model):
    hasher = hashlib.sha256()
    found = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        found = True
        tensor = parameter.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(str(tensor.dtype).encode())
        hasher.update(tensor.numpy().tobytes())
    if not found:
        raise ValueError("没有可训练参数，无法计算适配器权重哈希")
    return hasher.hexdigest()


def make_fingerprint(config, iteration, content_hash, model):
    return {
        "config_hash": config_hash(config),
        "iteration": iteration,
        "trajectory_hash": content_hash,
        "adapter_weight_hash": adapter_weight_hash(model),
    }


def _routing_to_payload(routed_experts):
    """int8 张量写成 JSON 能哈希的列表。没有记录时是 None。"""
    if routed_experts is None:
        return None
    tensor = routed_experts.detach().cpu().to(torch.int8).contiguous()
    return {"shape": list(tensor.shape), "values": tensor.view(-1).tolist()}


def _routing_from_payload(payload):
    if payload is None:
        return None
    if torch.is_tensor(payload):
        return payload.detach().cpu().to(torch.int8).contiguous()
    values = torch.tensor(payload["values"], dtype=torch.int8)
    return values.view(payload["shape"])


def _step_to_dict(step):
    payload = {
        "prompt_token_ids": list(step.prompt_token_ids),
        "action_token_ids": list(step.action_token_ids),
        "reward": step.reward,
        "state_key": step.state_key,
        "advantage": step.advantage,
        "backend_log_probabilities": step.backend_log_probabilities,
        "backend_base_url": step.backend_base_url,
        "environment_reward": step.environment_reward,
        "finish_reason": step.finish_reason,
    }
    # 没有记录时不写这个键，这样关掉重放时轨迹哈希和原来一致
    routed_experts = getattr(step, "routed_experts", None)
    if routed_experts is not None:
        payload["routed_experts"] = _routing_to_payload(routed_experts)
    return payload


def _step_from_dict(payload):
    return Step(
        prompt_token_ids=list(payload["prompt_token_ids"]),
        action_token_ids=list(payload["action_token_ids"]),
        reward=payload["reward"],
        state_key=payload["state_key"],
        advantage=payload["advantage"],
        backend_log_probabilities=payload["backend_log_probabilities"],
        backend_base_url=payload["backend_base_url"],
        environment_reward=payload["environment_reward"],
        finish_reason=payload["finish_reason"],
        routed_experts=_routing_from_payload(payload.get("routed_experts")),
    )


def _trajectory_to_dict(trajectory):
    return {
        "group_id": trajectory.group_id,
        "policy_version": trajectory.policy_version,
        "sampling_temperature": trajectory.sampling_temperature,
        "steps": [_step_to_dict(step) for step in trajectory.steps],
        "termination_reason": trajectory.termination_reason,
        "is_successful": trajectory.is_successful,
        "source_step_count": trajectory.source_step_count,
        "environment_metrics": dict(trajectory.environment_metrics),
    }


def _trajectory_from_dict(payload):
    return Trajectory(
        payload["group_id"], payload["policy_version"], payload["sampling_temperature"],
        steps=[_step_from_dict(step) for step in payload["steps"]],
        termination_reason=payload["termination_reason"],
        is_successful=payload["is_successful"],
        source_step_count=payload["source_step_count"],
        environment_metrics=dict(payload["environment_metrics"]),
    )


def iteration_content_hash(trajectories, kept, diagnostics, minibatch_plan):
    payload = {
        "trajectories": [_trajectory_to_dict(item) for item in trajectories],
        "kept": [_trajectory_to_dict(item) for item in kept],
        "diagnostics": diagnostics,
        "minibatch_plan": minibatch_plan,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def iteration_trajectory_path(run_directory, iteration):
    return pathlib.Path(run_directory) / "iteration_trajectories" / f"iteration-{iteration:04d}.pt"


def accumulation_checkpoint_path(run_directory):
    return pathlib.Path(run_directory) / "accumulation.pt"


def atomic_torch_save(path, payload):
    """先写临时文件再改名。中断只可能留下 .partial，正式路径要么是上一份完整文件，要么不存在。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def _load_torch(path):
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001  写到一半或损坏的文件不能接着用
        warnings.warn(f"{path.name} 无法读取，已丢弃：{error}")
        path.unlink(missing_ok=True)
        return None


def _fingerprint_mismatches(saved, expected):
    reasons = []
    for key, label in (
            ("config_hash", "配置哈希"),
            ("iteration", "轮次"),
            ("trajectory_hash", "轨迹哈希"),
            ("adapter_weight_hash", "适配器权重哈希"),
    ):
        if saved.get(key) != expected.get(key):
            reasons.append(f"{label}不一致（文件 {saved.get(key)!r}，当前 {expected.get(key)!r}）")
    return reasons


def save_iteration_trajectories(run_directory, iteration, model, config, trajectories, kept,
                                diagnostics, minibatch_plan, random_generator, device=None):
    content_hash = iteration_content_hash(trajectories, kept, diagnostics, minibatch_plan)
    payload = {
        "fingerprint": make_fingerprint(config, iteration, content_hash, model),
        "trajectories": [_trajectory_to_dict(item) for item in trajectories],
        "kept": [_trajectory_to_dict(item) for item in kept],
        "diagnostics": dict(diagnostics),
        "minibatch_plan": minibatch_plan,
        "python_random_state": random_generator.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": _capture_cuda_rng_state(device),
    }
    atomic_torch_save(iteration_trajectory_path(run_directory, iteration), payload)
    return content_hash


def load_iteration_trajectories(run_directory, iteration, model, config, random_generator,
                                device=None):
    """校验通过则恢复采样之后的随机数状态并返回轨迹。对不上就丢弃，调用方重新采样。"""
    path = iteration_trajectory_path(run_directory, iteration)
    payload = _load_torch(path)
    if payload is None:
        return None
    trajectories = [_trajectory_from_dict(item) for item in payload["trajectories"]]
    kept = [_trajectory_from_dict(item) for item in payload["kept"]]
    diagnostics = dict(payload["diagnostics"])
    minibatch_plan = payload["minibatch_plan"]
    content_hash = iteration_content_hash(trajectories, kept, diagnostics, minibatch_plan)
    expected = make_fingerprint(config, iteration, content_hash, model)
    reasons = _fingerprint_mismatches(payload.get("fingerprint", {}), expected)
    if reasons:
        warnings.warn(
            f"第 {iteration} 轮轨迹校验未通过，丢弃并重新采样：" + "；".join(reasons))
        path.unlink(missing_ok=True)
        return None
    random_generator.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    _restore_cuda_rng_state(payload.get("cuda_random_states"), device)
    return LoadedIteration(trajectories, kept, diagnostics, minibatch_plan, content_hash)


def capture_gradients(model):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_gradients(model, gradients):
    parameters = dict(model.named_parameters())
    for name, value in gradients.items():
        parameter = parameters[name]
        if value is None:
            parameter.grad = None
        else:
            parameter.grad = value.detach().to(
                device=parameter.device, dtype=parameter.dtype).clone()


def save_accumulation_checkpoint(path, fingerprint, next_index, total_weight, batch_count,
                                 weighted_metrics, gradients):
    atomic_torch_save(path, {
        "fingerprint": fingerprint,
        "next_index": next_index,
        "total_weight": total_weight,
        "batch_count": batch_count,
        "weighted_metrics": [(weight, dict(metrics)) for weight, metrics in weighted_metrics],
        "gradients": gradients,
    })


def load_accumulation_checkpoint(path, fingerprint, total_weight, batch_count, trainable_names):
    """对不上就删除并返回 None。调用方从第一个小批重新累积。"""
    path = pathlib.Path(path)
    payload = _load_torch(path)
    if payload is None:
        return None
    reasons = _fingerprint_mismatches(payload.get("fingerprint", {}), fingerprint)
    if payload.get("batch_count") != batch_count:
        reasons.append(
            f"小批数量不一致（文件 {payload.get('batch_count')!r}，当前 {batch_count}）")
    saved_weight = payload.get("total_weight")
    if saved_weight is None or abs(saved_weight - total_weight) > 1e-9:
        reasons.append(f"归一化分母不一致（文件 {saved_weight!r}，当前 {total_weight!r}）")
    next_index = payload.get("next_index")
    if isinstance(next_index, bool) or not isinstance(next_index, int) or not 0 <= next_index <= batch_count:
        reasons.append(f"下一个小批序号无效：{next_index!r}")
    saved_names = set(payload.get("gradients", {}))
    if saved_names != set(trainable_names):
        reasons.append("可训练参数与中途梯度不一致")
    if reasons:
        warnings.warn("累积中途状态已丢弃，从本轮开头重新累积：" + "；".join(reasons))
        path.unlink(missing_ok=True)
        return None
    return payload


def remove_accumulation_checkpoint(path):
    pathlib.Path(path).unlink(missing_ok=True)
