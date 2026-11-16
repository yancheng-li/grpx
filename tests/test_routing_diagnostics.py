import types

import pytest
import torch

from grpx.routing_diagnostics import (
    RoutingRecorder,
    changed_in_any_layer,
    summarize_change,
    summarize_load,
    summarize_routing,
)

HIDDEN_SIZE = 16
EXPERT_COUNT = 8
EXPERTS_PER_TOKEN = 2


class ToyMoeBlock(torch.nn.Module):
    """和 OLMoE 一样：先把隐藏状态压平成 (批大小 * 序列长度, 隐藏维)，再过 gate。"""

    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(HIDDEN_SIZE, EXPERT_COUNT, bias=False)
        self.gate_proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_size)
        self.gate(flat)
        return hidden_states + self.gate_proj(hidden_states)


class ToyModel(torch.nn.Module):
    def __init__(self, layer_count=3):
        super().__init__()
        self.config = types.SimpleNamespace(num_experts_per_tok=EXPERTS_PER_TOKEN)
        self.embedding = torch.nn.Embedding(50, HIDDEN_SIZE)
        self.layers = torch.nn.ModuleList([ToyMoeBlock() for _ in range(layer_count)])

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def make_inputs():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (4, 10))
    attention_mask = torch.ones(4, 10, dtype=torch.long)
    attention_mask[0, 7:] = 0
    attention_mask[2, 4:] = 0
    return input_ids, attention_mask


def test_finds_only_router_modules():
    recorder = RoutingRecorder(ToyModel())
    assert recorder.is_available
    assert sorted(recorder.router_modules) == ["layers.0.gate", "layers.1.gate", "layers.2.gate"]


def test_not_available_on_dense_model():
    dense = torch.nn.Sequential(torch.nn.Linear(4, 4))
    assert not RoutingRecorder(dense).is_available


class TupleRouter(torch.nn.Module):
    """模拟新版 OLMoE：gate forward 返回 (logits, scores, indices)。"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(HIDDEN_SIZE, EXPERT_COUNT, bias=False)

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = self.linear(flat)
        probs = torch.softmax(logits.float(), dim=-1)
        values, indices = probs.topk(EXPERTS_PER_TOKEN, dim=-1)
        return logits, values, indices


class TupleRouterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(num_experts_per_tok=EXPERTS_PER_TOKEN)
        self.embedding = torch.nn.Embedding(50, HIDDEN_SIZE)
        self.mlp = torch.nn.Module()
        self.mlp.gate = TupleRouter()

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        self.mlp.gate(hidden)
        return hidden


def test_tuple_router_output_and_non_linear_gate():
    torch.manual_seed(4)
    model = TupleRouterModel()
    input_ids, attention_mask = make_inputs()
    recorder = RoutingRecorder(model)
    assert recorder.is_available
    assert list(recorder.router_modules) == ["mlp.gate"]
    with recorder.record(attention_mask) as before:
        model(input_ids, attention_mask)
    with recorder.record(attention_mask) as after:
        model(input_ids, attention_mask)
    assert summarize_change(before, after)["routing_changed_position_fraction_mean"] == 0.0
    assert before.selected("mlp.gate").shape == (int(attention_mask.sum()), EXPERT_COUNT)


def test_same_weights_twice_no_change_and_padding_excluded():
    torch.manual_seed(1)
    model = ToyModel()
    input_ids, attention_mask = make_inputs()
    recorder = RoutingRecorder(model)
    with recorder.record(attention_mask) as before:
        model(input_ids, attention_mask)
    with recorder.record(attention_mask) as after:
        model(input_ids, attention_mask)
    valid_position_count = int(attention_mask.sum())
    selected = before.selected("layers.0.gate")
    assert selected.shape == (valid_position_count, EXPERT_COUNT)
    assert (selected.sum(dim=1) == EXPERTS_PER_TOKEN).all()
    change = summarize_change(before, after)
    assert change["routing_changed_position_fraction_mean"] == 0.0
    assert change["routing_changed_in_any_layer_fraction"] == 0.0
    assert not changed_in_any_layer(before, after).any()


def test_hooks_removed_after_context():
    model = ToyModel()
    input_ids, attention_mask = make_inputs()
    recorder = RoutingRecorder(model)
    with recorder.record(attention_mask) as snapshot:
        model(input_ids, attention_mask)
    model(input_ids, attention_mask)
    assert len(snapshot.selected_by_layer["layers.0.gate"]) == 1
    assert all(len(module._forward_hooks) == 0 for module in recorder.router_modules.values())


def test_change_detected_only_downstream_of_perturbation():
    torch.manual_seed(2)
    model = ToyModel()
    input_ids, attention_mask = make_inputs()
    recorder = RoutingRecorder(model)
    with recorder.record(attention_mask) as before:
        model(input_ids, attention_mask)
    with torch.no_grad():
        model.layers[0].gate_proj.weight.add_(torch.randn_like(model.layers[0].gate_proj.weight))
    with recorder.record(attention_mask) as after:
        model(input_ids, attention_mask)
    # 第 0 层的路由输入没变；后面的层变了
    first = (before.selected("layers.0.gate") != after.selected("layers.0.gate")).any()
    later = (before.selected("layers.2.gate") != after.selected("layers.2.gate")).any()
    assert not first
    assert later
    metrics = summarize_routing(before, after)
    assert 0.0 < metrics["routing_changed_position_fraction_mean"] <= 1.0
    assert metrics["routing_changed_in_any_layer_fraction"] >= metrics["routing_changed_position_fraction_maximum"] - 1e-12
    assert 0.0 <= metrics["routing_replaced_expert_fraction_mean"] <= metrics["routing_changed_position_fraction_mean"]


def test_load_metrics_uniform_and_collapsed():
    model = ToyModel(layer_count=1)
    recorder = RoutingRecorder(model)
    input_ids, attention_mask = make_inputs()
    # 塌到固定两个专家
    with torch.no_grad():
        model.layers[0].gate.weight.zero_()
    with recorder.record(attention_mask) as collapsed:
        model(input_ids, attention_mask)
    load = summarize_load(collapsed)
    assert load["routing_unused_expert_fraction"] == pytest.approx(1 - EXPERTS_PER_TOKEN / EXPERT_COUNT)
    assert load["routing_load_maximum_ratio"] == pytest.approx(EXPERT_COUNT / EXPERTS_PER_TOKEN)
    assert load["routing_load_entropy_mean"] == pytest.approx(1 / 3)  # log(2) / log(8)


def test_accumulates_minibatches_and_bfloat16():
    torch.manual_seed(3)
    model = ToyModel().to(torch.bfloat16)
    input_ids, attention_mask = make_inputs()
    recorder = RoutingRecorder(model)
    with recorder.record(attention_mask[:2]) as snapshot:
        model(input_ids[:2], attention_mask[:2])
    with recorder.record(attention_mask[2:], snapshot=snapshot):
        model(input_ids[2:], attention_mask[2:])
    assert snapshot.selected("layers.1.gate").shape[0] == int(attention_mask.sum())


def test_mask_mismatch_raises():
    model = ToyModel()
    input_ids, attention_mask = make_inputs()
    recorder = RoutingRecorder(model)
    with pytest.raises(RuntimeError):
        with recorder.record(attention_mask[:, :5]):
            model(input_ids, attention_mask)
