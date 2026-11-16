"""MoE 路由诊断：记录每层路由选中的专家，比较两次前向之间的变化。

用法（两次前向必须是同一批输入、同样的分批和顺序）：

    recorder = RoutingRecorder(model)
    if recorder.is_available:
        with recorder.record(attention_mask) as before:
            model(input_ids=input_ids, attention_mask=attention_mask)
        ...  # 更新权重
        with recorder.record(attention_mask) as after:
            model(input_ids=input_ids, attention_mask=attention_mask)
        metrics = summarize_routing(before, after)

路由模块按名字识别：名字等于 router_module_name 或以 "." + router_module_name 结尾。
OLMoE 里是每层的 `mlp.gate`（旧版是 Linear，新版是 OlmoeTopKRouter）；钩子取路由
logits，形状是 (批大小 * 序列长度, 专家数)。若 forward 返回元组，取第一个元素。
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import torch


def _resolve_experts_per_token(model: torch.nn.Module) -> int | None:
    """从 model.config 取 num_experts_per_tok；PEFT 包装后回退到底座配置。"""
    candidates = [model]
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        try:
            candidates.append(get_base())
        except Exception:  # noqa: BLE001
            pass
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        nested = getattr(base_model, "model", None)
        if nested is not None:
            candidates.append(nested)
    for candidate in candidates:
        configuration = getattr(candidate, "config", None)
        value = getattr(configuration, "num_experts_per_tok", None)
        if value is not None:
            return value
    return None


@dataclass
class RoutingSnapshot:
    """一次或多次前向里，每层在有效位置上选中的专家。"""

    # 每层一个列表，列表里每个元素是一次前向的布尔张量 (有效位置数, 专家数)
    selected_by_layer: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def layer_names(self) -> list[str]:
        return list(self.selected_by_layer.keys())

    def selected(self, layer_name: str) -> torch.Tensor:
        return torch.cat(self.selected_by_layer[layer_name], dim=0)


class RoutingRecorder:
    def __init__(
        self,
        model: torch.nn.Module,
        router_module_name: str = "gate",
        experts_per_token: int | None = None,
    ) -> None:
        self.router_modules: dict[str, torch.nn.Module] = {
            name: module
            for name, module in model.named_modules()
            if name == router_module_name or name.endswith("." + router_module_name)
        }
        if experts_per_token is None:
            experts_per_token = _resolve_experts_per_token(model)
        self.experts_per_token = experts_per_token

    @property
    def is_available(self) -> bool:
        return len(self.router_modules) > 0 and self.experts_per_token is not None

    @contextlib.contextmanager
    def record(self, attention_mask: torch.Tensor, snapshot: RoutingSnapshot | None = None):
        """在这个上下文里做的每一次前向都会被记录。

        attention_mask 的形状是 (批大小, 序列长度)，只记录值为 1 的位置。
        传入已有的 snapshot 可以把多个小批累积到同一个快照里。
        """
        if not self.is_available:
            raise RuntimeError("模型里没有找到路由模块，或不知道每个 token 选几个专家")
        if snapshot is None:
            snapshot = RoutingSnapshot()
        valid_positions = attention_mask.reshape(-1).bool()
        experts_per_token = self.experts_per_token
        handles = []

        def make_hook(layer_name: str):
            def hook(module, inputs, output):
                # Linear 直接返回 logits；OlmoeTopKRouter 返回 (logits, scores, indices)
                router_logits = output[0] if isinstance(output, tuple) else output
                router_logits = router_logits.detach()
                expert_count = router_logits.shape[-1]
                router_logits = router_logits.reshape(-1, expert_count)
                if router_logits.shape[0] != valid_positions.shape[0]:
                    raise RuntimeError(
                        f"路由输出的位置数 {router_logits.shape[0]} 和 attention_mask 的 "
                        f"{valid_positions.shape[0]} 对不上（层 {layer_name}）"
                    )
                router_logits = router_logits[valid_positions.to(router_logits.device)]
                top_indices = router_logits.float().topk(experts_per_token, dim=-1).indices
                selected = torch.zeros(router_logits.shape, dtype=torch.bool, device=router_logits.device)
                selected.scatter_(1, top_indices, True)
                snapshot.selected_by_layer.setdefault(layer_name, []).append(selected.cpu())

            return hook

        try:
            for layer_name, module in self.router_modules.items():
                handles.append(module.register_forward_hook(make_hook(layer_name)))
            yield snapshot
        finally:
            for handle in handles:
                handle.remove()


def summarize_load(snapshot: RoutingSnapshot) -> dict[str, float]:
    """专家负载：是否集中在少数专家上。各层分别计算，再汇总。"""
    normalized_entropies = []
    maximum_shares = []
    unused_fractions = []
    for layer_name in snapshot.layer_names():
        selected = snapshot.selected(layer_name)
        expert_count = selected.shape[1]
        counts = selected.sum(dim=0).double()
        shares = counts / counts.sum().clamp(min=1.0)
        nonzero_shares = shares[shares > 0]
        entropy = -(nonzero_shares * nonzero_shares.log()).sum().item()
        normalized_entropies.append(entropy / math.log(expert_count))
        maximum_shares.append(shares.max().item() * expert_count)
        unused_fractions.append((counts == 0).double().mean().item())
    layer_count = max(len(normalized_entropies), 1)
    return {
        # 1.0 表示完全均匀；越低越集中
        "routing_load_entropy_mean": sum(normalized_entropies) / layer_count,
        "routing_load_entropy_minimum": min(normalized_entropies, default=float("nan")),
        # 负载最高的专家是均匀负载的多少倍（各层里的最大值）
        "routing_load_maximum_ratio": max(maximum_shares, default=float("nan")),
        # 这批数据里一次都没被选中的专家比例（各层平均）
        "routing_unused_expert_fraction": sum(unused_fractions) / layer_count,
    }


def summarize_change(before: RoutingSnapshot, after: RoutingSnapshot) -> dict[str, float]:
    """同一批输入两次前向之间，路由变了多少。"""
    changed_position_fractions = []
    replaced_expert_fractions = []
    any_layer_changed = None
    for layer_name in before.layer_names():
        selected_before = before.selected(layer_name)
        selected_after = after.selected(layer_name)
        if selected_before.shape != selected_after.shape:
            raise RuntimeError(f"两次前向的位置数不同（层 {layer_name}），无法比较")
        experts_per_token = selected_before[0].sum().item() if selected_before.shape[0] else 1
        kept = (selected_before & selected_after).sum(dim=1)
        position_changed = kept < experts_per_token
        changed_position_fractions.append(position_changed.double().mean().item())
        replaced_expert_fractions.append((1.0 - kept.double() / experts_per_token).mean().item())
        any_layer_changed = position_changed if any_layer_changed is None else (any_layer_changed | position_changed)
    layer_count = max(len(changed_position_fractions), 1)
    return {
        # 每层里，选中的专家集合发生变化的位置比例（各层平均 / 各层最大）
        "routing_changed_position_fraction_mean": sum(changed_position_fractions) / layer_count,
        "routing_changed_position_fraction_maximum": max(changed_position_fractions, default=float("nan")),
        # 被换掉的专家占每个位置所选专家的比例（各层平均）
        "routing_replaced_expert_fraction_mean": sum(replaced_expert_fractions) / layer_count,
        # 至少有一层的路由发生变化的位置比例
        "routing_changed_in_any_layer_fraction": (
            any_layer_changed.double().mean().item() if any_layer_changed is not None else float("nan")
        ),
    }


def changed_in_any_layer(before: RoutingSnapshot, after: RoutingSnapshot) -> torch.Tensor:
    """每个有效位置是否至少有一层的路由变了。布尔张量，长度是有效位置数。"""
    result = None
    for layer_name in before.layer_names():
        selected_before = before.selected(layer_name)
        selected_after = after.selected(layer_name)
        experts_per_token = selected_before[0].sum().item() if selected_before.shape[0] else 1
        position_changed = (selected_before & selected_after).sum(dim=1) < experts_per_token
        result = position_changed if result is None else (result | position_changed)
    return result


def summarize_routing(before: RoutingSnapshot, after: RoutingSnapshot) -> dict[str, float]:
    metrics = summarize_load(after)
    metrics.update(summarize_change(before, after))
    return metrics
