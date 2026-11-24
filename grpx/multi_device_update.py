"""多卡更新：每张卡一份模型副本，各自在分到的小批上累积梯度，加总后只更新一步。

单进程、每个副本一个线程。只负责三件事：把小批分给各副本、并行地跑、把梯度加到主副本上
并把更新后的可训练参数同步回去。优化器、梯度裁剪、日志都留在调用方，只作用于主副本。

用法（每一轮）：

    group = ReplicaGroup(models, devices)        # models[0] 是主副本，优化器只管它的参数
    group.assert_synchronized()                   # 各副本的可训练参数必须相同
    assignment = assign_minibatches(token_counts, len(devices))

    old = group.map_without_gradient(assignment, compute_old_log_probabilities)
    metrics = group.accumulate_gradients(assignment, compute_minibatch_loss)
    group.reduce_gradients()                      # 各副本的梯度加到主副本上
    clip_grad_norm_(primary_trainable_parameters, max_gradient_norm)
    optimizer.step(); optimizer.zero_grad()
    group.broadcast_parameters()                  # 主副本 -> 其他副本

约定：

- compute_minibatch_loss(model, minibatch_index, device) 返回 (loss, metrics)。loss 必须已经
  除以**全局**的归一化分母（在累积开始前定好），不能除以这个副本自己那部分数据的数量。
- 传给 map_without_gradient 的函数返回任意结果，这里按小批序号收集。
- 梯度开关和 autocast 是线程局部的：主线程里的 torch.no_grad() 管不到工作线程，所以
  不带梯度的计算一律走 map_without_gradient，由它在工作线程里关梯度。
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence

import torch


class ReplicaSynchronizationError(RuntimeError):
    """各副本的可训练参数不一致。"""


class ReplicaWorkerError(RuntimeError):
    """某个副本在计算时抛了异常。"""


def assign_minibatches(minibatch_token_counts: Sequence[int], replica_count: int) -> list[list[int]]:
    """把小批分给各副本，尽量让每个副本的 token 总数接近。结果是确定的。

    从大到小依次放进当前负载最小的副本；负载相同取序号小的副本。
    每个副本内部按小批序号升序排列。
    """
    if replica_count < 1:
        raise ValueError("replica_count 必须至少为 1")
    assignment: list[list[int]] = [[] for _ in range(replica_count)]
    loads = [0] * replica_count
    order = sorted(range(len(minibatch_token_counts)), key=lambda index: (-minibatch_token_counts[index], index))
    for minibatch_index in order:
        replica_index = min(range(replica_count), key=lambda index: (loads[index], index))
        assignment[replica_index].append(minibatch_index)
        loads[replica_index] += minibatch_token_counts[minibatch_index]
    for indices in assignment:
        indices.sort()
    return assignment


def trainable_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def trainable_parameter_fingerprint(model: torch.nn.Module) -> str:
    """可训练参数的哈希。名字、形状、数值任何一项不同，结果都不同。"""
    digest = hashlib.sha256()
    for name, parameter in trainable_parameters(model):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("utf-8"))
        digest.update(parameter.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


class ReplicaGroup:
    def __init__(self, models: Sequence[torch.nn.Module], devices: Sequence[torch.device | str]) -> None:
        if len(models) == 0:
            raise ValueError("至少需要一个模型副本")
        if len(models) != len(devices):
            raise ValueError("模型副本的数量和设备的数量不一致")
        self.models = list(models)
        self.devices = [torch.device(device) for device in devices]
        primary_names = [name for name, _ in trainable_parameters(self.models[0])]
        if len(primary_names) == 0:
            raise ValueError("主副本没有可训练参数")
        for replica_index, model in enumerate(self.models[1:], start=1):
            names = [name for name, _ in trainable_parameters(model)]
            if names != primary_names:
                raise ValueError(f"第 {replica_index} 个副本的可训练参数名和主副本不同")
        self._executor = ThreadPoolExecutor(max_workers=len(self.models), thread_name_prefix="replica")

    @property
    def primary(self) -> torch.nn.Module:
        return self.models[0]

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    # ---- 同步 ----------------------------------------------------------------

    def fingerprints(self) -> list[str]:
        return [trainable_parameter_fingerprint(model) for model in self.models]

    def assert_synchronized(self) -> None:
        fingerprints = self.fingerprints()
        mismatched = [index for index, value in enumerate(fingerprints) if value != fingerprints[0]]
        if mismatched:
            devices = ", ".join(str(self.devices[index]) for index in mismatched)
            raise ReplicaSynchronizationError(
                f"这些设备上副本的可训练参数和主副本不一致：{devices}。上一轮更新后的同步没有完成"
            )

    def broadcast_parameters(self) -> None:
        """把主副本的可训练参数复制到其他副本。"""
        primary = dict(trainable_parameters(self.primary))
        with torch.no_grad():
            for model, device in zip(self.models[1:], self.devices[1:]):
                for name, parameter in trainable_parameters(model):
                    parameter.copy_(primary[name].detach().to(device=device, dtype=parameter.dtype))

    # ---- 并行执行 --------------------------------------------------------------

    def _run(self, assignment: Sequence[Sequence[int]], worker: Callable[[int, Sequence[int]], Any]) -> list[Any]:
        if len(assignment) != len(self.models):
            raise ValueError("小批分配的份数和副本的数量不一致")
        futures = [
            self._executor.submit(worker, replica_index, list(minibatch_indices))
            for replica_index, minibatch_indices in enumerate(assignment)
        ]
        results: list[Any] = []
        first_error: tuple[int, BaseException] | None = None
        for replica_index, future in enumerate(futures):
            try:
                results.append(future.result())
            except BaseException as error:  # 等所有线程结束后再统一抛出
                results.append(None)
                if first_error is None:
                    first_error = (replica_index, error)
        if first_error is not None:
            replica_index, error = first_error
            raise ReplicaWorkerError(f"设备 {self.devices[replica_index]} 上的副本计算失败：{error!r}") from error
        return results

    def map_without_gradient(
        self,
        assignment: Sequence[Sequence[int]],
        function: Callable[[torch.nn.Module, int, torch.device], Any],
    ) -> dict[int, Any]:
        """在各副本上不带梯度地计算，按小批序号返回结果。"""

        def worker(replica_index: int, minibatch_indices: Sequence[int]) -> dict[int, Any]:
            model, device = self.models[replica_index], self.devices[replica_index]
            with torch.no_grad():
                return {index: function(model, index, device) for index in minibatch_indices}

        merged: dict[int, Any] = {}
        for partial in self._run(assignment, worker):
            merged.update(partial)
        return merged

    def accumulate_gradients(
        self,
        assignment: Sequence[Sequence[int]],
        compute_minibatch_loss: Callable[[torch.nn.Module, int, torch.device], tuple[torch.Tensor, Any]],
    ) -> dict[int, Any]:
        """各副本在分到的小批上累积梯度。返回按小批序号收集的指标。

        开始前会清掉所有副本上已有的梯度。梯度留在各自的副本上，随后调用 reduce_gradients。
        """
        self.zero_gradients()

        def worker(replica_index: int, minibatch_indices: Sequence[int]) -> dict[int, Any]:
            model, device = self.models[replica_index], self.devices[replica_index]
            collected: dict[int, Any] = {}
            with torch.enable_grad():
                for index in minibatch_indices:
                    result = compute_minibatch_loss(model, index, device)
                    close_replay = None
                    if len(result) == 3:
                        loss, metrics, close_replay = result
                    else:
                        loss, metrics = result
                    try:
                        loss.backward()
                    finally:
                        if close_replay is not None:
                            close_replay()
                    collected[index] = metrics
            return collected

        merged: dict[int, Any] = {}
        for partial in self._run(assignment, worker):
            merged.update(partial)
        return merged

    # ---- 梯度 ----------------------------------------------------------------

    def zero_gradients(self) -> None:
        for model in self.models:
            for _, parameter in trainable_parameters(model):
                parameter.grad = None

    def reduce_gradients(self) -> None:
        """把其他副本的梯度加到主副本上，然后清掉其他副本的梯度。"""
        primary = dict(trainable_parameters(self.primary))
        primary_device = self.devices[0]
        for model in self.models[1:]:
            for name, parameter in trainable_parameters(model):
                if parameter.grad is None:
                    continue
                target = primary[name]
                contribution = parameter.grad.detach().to(device=primary_device, dtype=target.dtype)
                if target.grad is None:
                    target.grad = contribution.clone()
                else:
                    target.grad.add_(contribution)
                parameter.grad = None
