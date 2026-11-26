"""MoE 路由重放：训练端的前向不自己选专家，而是使用采样时记录下来的选择（R3）。

    replayer = RoutingReplayer(model)
    with replayer.replay(routed_experts) as statistics:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss.backward()          # 打开梯度检查点时，反向重算也要留在这个上下文里
    statistics.summary()         # 一致率、退回模型自己路由的比例

routed_experts 的形状是 (批大小, 序列长度, MoE 层数, 每个 token 选几个专家)，整数。
某个位置没有记录（补齐位置、最后一个采样出的 token）时，把这个位置填成 -1，
它会退回模型自己的路由。层的顺序是 model.named_modules() 里路由模块出现的顺序。

实现方式：在每层的路由模块上挂前向钩子，改写它的输出。transformers 5.22 里
OlmoeTopKRouter 和 Qwen3MoeTopKRouter 的输出都是 (logits, 选中专家的权重, 选中专家的编号)。
编号换成记录的；权重从这次前向自己的 logits 里按新编号重新取出并归一化，写法和路由类
的源码一致，所以路由的 logits 仍然有梯度。

也提供采样时的记录：RoutingCapture 把每次前向选中的编号按调用顺序存下来，
assemble_captured_routing 把它们拼成上面的形状。
"""

from __future__ import annotations

import base64
import contextlib
import io
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import torch


def find_router_modules(model: torch.nn.Module, router_module_name: str = "gate") -> dict[str, torch.nn.Module]:
    """按名字找路由模块，顺序就是层的顺序。"""
    return {
        name: module
        for name, module in model.named_modules()
        if name == router_module_name or name.endswith("." + router_module_name)
    }


def _router_attribute(module: torch.nn.Module, name: str):
    """路由被 PEFT 包了一层时，属性在 base_layer 上。"""
    current = module
    for _ in range(4):
        if hasattr(current, name):
            return getattr(current, name)
        if not hasattr(current, "base_layer"):
            break
        current = current.base_layer
    raise AttributeError(f"路由模块 {type(module).__name__} 上找不到属性 {name}")


@dataclass
class ReplayStatistics:
    """一次 replay 上下文里的统计。同一层被调用多次（比如反向重算）时只记第一次。"""

    positions_with_record: dict[str, int] = field(default_factory=dict)
    positions_without_record: dict[str, int] = field(default_factory=dict)
    positions_agreeing: dict[str, int] = field(default_factory=dict)
    experts_agreeing: dict[str, int] = field(default_factory=dict)
    experts_per_token: int = 0

    def summary(self) -> dict[str, float]:
        with_record = sum(self.positions_with_record.values())
        without_record = sum(self.positions_without_record.values())
        total = with_record + without_record
        agreement_by_layer = [
            self.positions_agreeing[name] / self.positions_with_record[name]
            for name in self.positions_with_record
            if self.positions_with_record[name] > 0
        ]
        return {
            # 重放之前，模型自己选的专家集合和记录完全相同的（位置，层）比例
            "routing_replay_agreement_fraction": (
                sum(self.positions_agreeing.values()) / with_record if with_record else float("nan")
            ),
            "routing_replay_agreement_fraction_minimum": min(agreement_by_layer, default=float("nan")),
            # 按专家算：记录里的专家有多大比例也在模型自己的选择里
            "routing_replay_expert_agreement_fraction": (
                sum(self.experts_agreeing.values()) / (with_record * self.experts_per_token)
                if with_record and self.experts_per_token
                else float("nan")
            ),
            # 没有记录、退回模型自己路由的（位置，层）比例
            "routing_replay_fallback_fraction": without_record / total if total else float("nan"),
        }


class RoutingReplayer:
    def __init__(self, model: torch.nn.Module, router_module_name: str = "gate") -> None:
        self.router_modules = find_router_modules(model, router_module_name)

    @property
    def is_available(self) -> bool:
        return len(self.router_modules) > 0

    @property
    def layer_count(self) -> int:
        return len(self.router_modules)

    @contextlib.contextmanager
    def replay(self, routed_experts: torch.Tensor) -> Iterator[ReplayStatistics]:
        if not self.is_available:
            raise RuntimeError("模型里没有找到路由模块")
        if routed_experts.dim() != 4:
            raise ValueError("routed_experts 的形状应为 (批大小, 序列长度, MoE 层数, 每个 token 选几个专家)")
        if routed_experts.dtype.is_floating_point or routed_experts.dtype == torch.bool:
            raise ValueError("routed_experts 必须是整数")
        if routed_experts.shape[2] != self.layer_count:
            raise ValueError(
                f"记录里有 {routed_experts.shape[2]} 层，模型有 {self.layer_count} 个 MoE 层"
            )
        position_count = routed_experts.shape[0] * routed_experts.shape[1]
        statistics = ReplayStatistics()
        handles = []

        def make_hook(layer_index: int, layer_name: str, module: torch.nn.Module):
            experts_per_token = int(_router_attribute(module, "top_k"))
            expert_count = int(_router_attribute(module, "num_experts"))
            normalize = bool(_router_attribute(module, "norm_topk_prob"))
            if routed_experts.shape[3] != experts_per_token:
                raise ValueError(
                    f"记录里每个 token 有 {routed_experts.shape[3]} 个专家，"
                    f"模型每个 token 选 {experts_per_token} 个（层 {layer_name}）"
                )
            recorded = routed_experts[:, :, layer_index, :].reshape(position_count, experts_per_token).long()
            has_record_cpu = (recorded >= 0).all(dim=-1)
            partial = ((recorded >= 0).any(dim=-1)) & ~has_record_cpu
            if partial.any():
                raise ValueError(f"有位置只填了一部分专家（层 {layer_name}）。没有记录的位置要整行填 -1")
            if has_record_cpu.any():
                valid_rows = recorded[has_record_cpu]
                if int(valid_rows.max()) >= expert_count:
                    raise ValueError(f"记录里的专家编号超出范围 [0, {expert_count})（层 {layer_name}）")
                sorted_rows = valid_rows.sort(dim=-1).values
                if (sorted_rows[:, 1:] == sorted_rows[:, :-1]).any():
                    raise ValueError(f"记录里同一个位置有重复的专家（层 {layer_name}）")
            statistics.experts_per_token = experts_per_token

            def hook(_module, _inputs, output):
                router_logits, native_scores, native_indices = output
                if router_logits.shape[0] != position_count:
                    raise RuntimeError(
                        f"路由输出有 {router_logits.shape[0]} 个位置，记录有 {position_count} 个（层 {layer_name}）"
                    )
                device = router_logits.device
                recorded_on_device = recorded.to(device)
                has_record = has_record_cpu.to(device)

                if layer_name not in statistics.positions_with_record:
                    native_mask = torch.zeros(position_count, expert_count, dtype=torch.bool, device=device)
                    native_mask.scatter_(1, native_indices, True)
                    recorded_mask = torch.zeros_like(native_mask)
                    recorded_mask.scatter_(1, recorded_on_device.clamp(min=0), True)
                    shared = (native_mask & recorded_mask).sum(dim=1)[has_record]
                    statistics.positions_with_record[layer_name] = int(has_record.sum())
                    statistics.positions_without_record[layer_name] = int((~has_record).sum())
                    statistics.positions_agreeing[layer_name] = int((shared == experts_per_token).sum())
                    statistics.experts_agreeing[layer_name] = int(shared.sum())

                forced_indices = torch.where(has_record.unsqueeze(-1), recorded_on_device, native_indices)
                # 和路由类的源码一致：在 float 上做 softmax，取出选中专家的概率，按需归一化
                router_probabilities = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
                forced_scores = router_probabilities.gather(1, forced_indices)
                if normalize:
                    forced_scores = forced_scores / forced_scores.sum(dim=-1, keepdim=True)
                forced_scores = forced_scores.to(router_logits.dtype)
                return router_logits, forced_scores, forced_indices

            return hook

        try:
            for layer_index, (layer_name, module) in enumerate(self.router_modules.items()):
                handles.append(module.register_forward_hook(make_hook(layer_index, layer_name, module)))
            yield statistics
        finally:
            for handle in handles:
                handle.remove()


class RoutingCapture:
    """记录每次前向里每层选中的专家编号。用于采样时（本地后端）。"""

    def __init__(self, model: torch.nn.Module, router_module_name: str = "gate") -> None:
        self.router_modules = find_router_modules(model, router_module_name)

    @contextlib.contextmanager
    def capture(self) -> Iterator[dict[str, list[torch.Tensor]]]:
        if len(self.router_modules) == 0:
            raise RuntimeError("模型里没有找到路由模块")
        calls: dict[str, list[torch.Tensor]] = {name: [] for name in self.router_modules}
        handles = []

        def make_hook(layer_name: str):
            def hook(_module, _inputs, output):
                calls[layer_name].append(output[2].detach().to("cpu"))

            return hook

        try:
            for layer_name, module in self.router_modules.items():
                handles.append(module.register_forward_hook(make_hook(layer_name)))
            yield calls
        finally:
            for handle in handles:
                handle.remove()


def decode_vllm_routed_experts(
    encoded,
    prompt_length: int,
    action_length: int,
    layer_count: int,
    experts_per_token: int,
    extra_unrecorded: int = 0,
) -> torch.Tensor:
    """把 /v1/completions 里的 base64 .npy 填进训练用的 int8 记录。

    服务端的行数是提示词加回答再减 1，和 RoutingCapture 记的范围一样：
    所有输入过模型的 token。最后一个采样出的 token 没有记录，末尾补一行 -1。
    本地另外补上的结束符也没有记录，用 extra_unrecorded 再补几行 -1。
    """
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError(
            "推理服务没有返回 routed_experts。"
            "vLLM 0.18 的 HTTP 响应里没有这个字段；"
            "需要 0.34.0，并用 --enable-return-routed-experts 启动。"
        )
    try:
        array = np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)
    except Exception as error:
        raise RuntimeError(f"routed_experts 不是关掉 pickle 也能读的 .npy：{error}") from error
    routing = _routing_from_vllm_array(
        np.asarray(array), prompt_length, action_length, layer_count, experts_per_token)
    if extra_unrecorded < 0:
        raise ValueError(f"extra_unrecorded 不能为负，收到 {extra_unrecorded}")
    if extra_unrecorded:
        padding = torch.full(
            (extra_unrecorded, routing.shape[1], routing.shape[2]), -1, dtype=torch.int8)
        routing = torch.cat((routing, padding), dim=0)
    return routing


def _routing_from_vllm_array(
    array: np.ndarray,
    prompt_length: int,
    action_length: int,
    layer_count: int,
    experts_per_token: int,
) -> torch.Tensor:
    if array.ndim != 3:
        raise ValueError(f"路由记录应为三维数组，收到形状 {array.shape}")
    expected_rows = prompt_length + action_length - 1
    if array.shape[0] != expected_rows:
        raise ValueError(
            f"路由记录有 {array.shape[0]} 行，"
            f"提示词 {prompt_length} 加回答 {action_length} 应是 {expected_rows} 行"
        )
    if array.shape[1] != layer_count:
        raise ValueError(f"路由记录有 {array.shape[1]} 层，训练端找到 {layer_count} 个路由模块")
    if array.shape[2] != experts_per_token:
        raise ValueError(
            f"路由记录里每个 token 有 {array.shape[2]} 个专家，"
            f"模型每个 token 选 {experts_per_token} 个"
        )
    if array.size and int(array.min()) < 0:
        raise ValueError("路由记录里出现了负数编号")
    if array.size and int(array.max()) > 127:
        raise ValueError(
            f"专家编号最大是 {int(array.max())}，超过 127，有符号单字节会把它当成没有记录"
        )
    routing = torch.full(
        (prompt_length + action_length, layer_count, experts_per_token), -1, dtype=torch.int8)
    if expected_rows:
        routing[:expected_rows] = torch.as_tensor(array).to(torch.int8)
    return routing


def align_generated_routing(
    assembled_row: torch.Tensor,
    prompt_length: int,
    action_length: int,
    padded_prompt_length: int,
) -> torch.Tensor:
    """把一次左填充生成的记录对齐到训练时的那一行。

    assembled_row 的形状是 (填充后的提示词长度 + 生成步数 - 1, 层数, 每个 token 的专家数)。
    前一段是左填充的提示词，后面每个位置是一个被送回模型的生成 token。
    返回 int8，形状 (提示词 + 回答, 层数, 专家数)。最后一个回答 token 没有被送回模型，填 -1。
    """
    if assembled_row.dim() != 3:
        raise ValueError("一层的路由记录应为 (位置数, 层数, 每个 token 的专家数)")
    token_length = prompt_length + action_length
    layers, experts = assembled_row.shape[1], assembled_row.shape[2]
    routing = torch.full((token_length, layers, experts), -1, dtype=torch.long)
    left_pad = padded_prompt_length - prompt_length
    if left_pad < 0 or left_pad + prompt_length > assembled_row.shape[0]:
        raise ValueError("提示词长度和生成时的填充宽度对不上")
    routing[:prompt_length] = assembled_row[left_pad:left_pad + prompt_length]
    fed = max(action_length - 1, 0)
    if fed:
        end = padded_prompt_length + fed
        if end > assembled_row.shape[0]:
            raise ValueError("生成 token 的路由记录比回答短")
        routing[prompt_length:prompt_length + fed] = assembled_row[padded_prompt_length:end]
    if int(routing.max()) > 127:
        raise ValueError("专家编号超过 127，单字节存不下")
    return routing.to(torch.int8)


def merge_replay_statistics(parts: Sequence[ReplayStatistics]) -> dict[str, float]:
    """把多个小批的重放统计合成一轮的四个数。"""
    if not parts:
        return {}
    with_record = sum(sum(part.positions_with_record.values()) for part in parts)
    without_record = sum(sum(part.positions_without_record.values()) for part in parts)
    agreeing = sum(sum(part.positions_agreeing.values()) for part in parts)
    experts_agreeing = sum(sum(part.experts_agreeing.values()) for part in parts)
    experts_per_token = parts[0].experts_per_token
    with_record_by_layer: dict[str, int] = {}
    agreeing_by_layer: dict[str, int] = {}
    for part in parts:
        for name, count in part.positions_with_record.items():
            with_record_by_layer[name] = with_record_by_layer.get(name, 0) + count
            agreeing_by_layer[name] = agreeing_by_layer.get(name, 0) + part.positions_agreeing.get(name, 0)
    minimum = min(
        (agreeing_by_layer[name] / count for name, count in with_record_by_layer.items() if count),
        default=float("nan"),
    )
    total = with_record + without_record
    return {
        "routing_replay_agreement_fraction": agreeing / with_record if with_record else float("nan"),
        "routing_replay_agreement_fraction_minimum": minimum,
        "routing_replay_expert_agreement_fraction": (
            experts_agreeing / (with_record * experts_per_token)
            if with_record and experts_per_token else float("nan")
        ),
        "routing_replay_fallback_fraction": without_record / total if total else float("nan"),
    }


def capture_training_routing(calls: dict[str, Sequence[torch.Tensor]], batch_size: int, sequence_length: int) -> torch.Tensor:
    """一次完整训练前向里各层实际选中的专家。int8，形状 (批, 序列, 层, 每个 token 的专家数)。"""
    assembled = assemble_captured_routing(calls, batch_size)
    if assembled.shape[1] != sequence_length:
        raise RuntimeError(
            f"训练前向的路由记录有 {assembled.shape[1]} 个位置，序列长度是 {sequence_length}"
        )
    if assembled.numel() and int(assembled.max()) > 127:
        raise ValueError("专家编号超过 127，单字节存不下")
    return assembled.to(torch.int8)


def assemble_captured_routing(calls: dict[str, Sequence[torch.Tensor]], batch_size: int) -> torch.Tensor:
    """把按调用顺序记录的编号拼成 (批大小, 位置数, 层数, 每个 token 选几个专家)。

    每次调用的输出是压平的 (批大小 * 这次的位置数, k)。带缓存的逐步生成里，第一次调用是整个提示，
    之后每次一个 token，所以拼出来的位置是「所有输入过模型的 token」：提示加上除最后一个之外的
    生成 token。最后一个采样出的 token 没有被输入过，没有记录。
    """
    per_layer = []
    for layer_name, layer_calls in calls.items():
        if len(layer_calls) == 0:
            raise ValueError(f"层 {layer_name} 没有记录到任何调用")
        pieces = []
        for indices in layer_calls:
            if indices.shape[0] % batch_size != 0:
                raise ValueError(f"层 {layer_name} 有一次调用的位置数不是批大小的整数倍")
            pieces.append(indices.reshape(batch_size, indices.shape[0] // batch_size, indices.shape[1]))
        per_layer.append(torch.cat(pieces, dim=1))
    position_counts = {piece.shape[1] for piece in per_layer}
    if len(position_counts) != 1:
        raise ValueError("各层记录到的位置数不一致")
    return torch.stack(per_layer, dim=2)
