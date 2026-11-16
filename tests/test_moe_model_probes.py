"""真实 MoE 模型类的小号路径探针：OLMoE 与 Qwen3 MoE 共用同一组断言。

不下载权重；对话模板检查见 test_qwen3_moe_chat_template_online（默认跳过）。
"""
from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, OlmoeConfig, Qwen3MoeConfig

from grpx.doctor import summarize_lora_targets
from grpx.routing_diagnostics import RoutingRecorder, summarize_routing

pytest.importorskip("transformers")


def _build_olmoe_tiny():
    config = OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
    )
    return AutoModelForCausalLM.from_config(config)


def _build_qwen3_moe_tiny():
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
    )
    return AutoModelForCausalLM.from_config(config)


MOE_BUILDERS = [
    pytest.param((_build_olmoe_tiny, "OlmoeTopKRouter"), id="olmoe"),
    pytest.param((_build_qwen3_moe_tiny, "Qwen3MoeTopKRouter"), id="qwen3_moe"),
]


@pytest.fixture(params=MOE_BUILDERS)
def moe_case(request):
    builder, expected_router_class = request.param
    model = builder()
    model.eval()
    return model, expected_router_class


def test_lora_targets_match_attention_projections(moe_case):
    model, _ = moe_case
    summary = summarize_lora_targets(
        model, ["q_proj", "k_proj", "v_proj", "o_proj"])
    assert summary["matched_module_count"] > 0
    assert summary["expert_module_count"] == 0
    assert summary["router_module_count"] == 0


def test_routing_recorder_finds_gate_and_experts_per_tok(moe_case):
    model, expected_router_class = moe_case
    recorder = RoutingRecorder(model)
    assert recorder.is_available
    assert recorder.experts_per_token == 2
    assert recorder.router_modules
    assert all(name.endswith(".gate") for name in recorder.router_modules)
    for module in recorder.router_modules.values():
        assert type(module).__name__ == expected_router_class


def test_router_output_format_is_tuple_of_flat_logits(moe_case):
    model, _ = moe_case
    recorder = RoutingRecorder(model)
    layer_name, module = next(iter(recorder.router_modules.items()))
    captured = {}

    def hook(_module, _inputs, output):
        captured["output"] = output

    handle = module.register_forward_hook(hook)
    input_ids = torch.randint(0, 128, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()
    output = captured["output"]
    assert isinstance(output, tuple), f"{layer_name} output type={type(output)}"
    assert len(output) >= 1
    logits = output[0]
    assert torch.is_tensor(logits)
    # 压平：(批大小 × 序列长度, 专家数)
    assert logits.ndim == 2
    assert logits.shape == (2 * 6, model.config.num_experts)


def test_experts_per_tok_survives_peft_wrap(moe_case):
    model, _ = moe_case
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    recorder = RoutingRecorder(peft_model)
    assert recorder.is_available
    assert recorder.experts_per_token == 2


def test_identical_forwards_show_no_routing_change(moe_case):
    model, _ = moe_case
    recorder = RoutingRecorder(model)
    input_ids = torch.randint(0, 128, (3, 8))
    attention_mask = torch.ones(3, 8, dtype=torch.long)
    attention_mask[0, 5:] = 0
    attention_mask[2, 3:] = 0
    with torch.no_grad():
        with recorder.record(attention_mask) as before:
            model(input_ids=input_ids, attention_mask=attention_mask)
        with recorder.record(attention_mask) as after:
            model(input_ids=input_ids, attention_mask=attention_mask)
    metrics = summarize_routing(before, after)
    assert metrics["routing_changed_position_fraction_mean"] == 0.0
    assert metrics["routing_changed_position_fraction_maximum"] == 0.0
    assert metrics["routing_replaced_expert_fraction_mean"] == 0.0
    assert metrics["routing_changed_in_any_layer_fraction"] == 0.0
    # 有效位置：3*8 - 3 - 5 = 16
    valid = int(attention_mask.sum().item())
    for layer_name in before.layer_names():
        assert before.selected(layer_name).shape[0] == valid


def test_weight_update_can_change_routing(moe_case):
    """专家数 > 每 token 选中数时，改 gate 权重后路由应能变化。"""
    model, _ = moe_case
    assert model.config.num_experts > model.config.num_experts_per_tok
    recorder = RoutingRecorder(model)
    input_ids = torch.randint(0, 128, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    with torch.no_grad():
        with recorder.record(attention_mask) as before:
            model(input_ids=input_ids, attention_mask=attention_mask)
        # 扰动所有 gate 权重
        for module in recorder.router_modules.values():
            for parameter in module.parameters():
                parameter.add_(0.5 * torch.randn_like(parameter))
        with recorder.record(attention_mask) as after:
            model(input_ids=input_ids, attention_mask=attention_mask)
    metrics = summarize_routing(before, after)
    assert metrics["routing_changed_in_any_layer_fraction"] > 0.0


def test_indices_and_logits_paths_agree_on_unwrapped_model(moe_case):
    """未包装模型上，第三项 indices 与 logits top-k 选出的专家一致。"""
    from grpx.routing_diagnostics import selected_experts_from_router_output

    model, _ = moe_case
    recorder = RoutingRecorder(model)
    layer_name, module = next(iter(recorder.router_modules.items()))
    captured = {}

    def hook(_module, _inputs, output):
        captured["output"] = output

    handle = module.register_forward_hook(hook)
    input_ids = torch.randint(0, 128, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    handle.remove()
    output = captured["output"]
    assert isinstance(output, tuple) and len(output) >= 3
    valid = attention_mask.reshape(-1).bool()
    from_indices = selected_experts_from_router_output(
        output, valid, recorder.experts_per_token,
        expert_count=model.config.num_experts)
    # 强制走 logits 路径：只传 (logits,)
    from_logits = selected_experts_from_router_output(
        (output[0],), valid, recorder.experts_per_token,
        expert_count=model.config.num_experts)
    assert torch.equal(from_indices, from_logits), layer_name


def test_router_lora_perturbation_changes_routing(moe_case):
    """PEFT 按参数注入 gate.weight 后，扰动路由 LoRA 必须让记录器看到变化。"""
    model, _ = moe_case
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_parameters=["gate.weight"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    recorder = RoutingRecorder(peft_model)
    assert recorder.is_available
    # 钩子挂在 ParamWrapper（名字仍以 .gate 结尾），不是 base_layer
    for name, module in recorder.router_modules.items():
        assert name.endswith(".gate")
        assert type(module).__name__ == "ParamWrapper"
    input_ids = torch.randint(0, 128, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    with torch.no_grad():
        with recorder.record(attention_mask) as before:
            peft_model(input_ids=input_ids, attention_mask=attention_mask)
        for name, parameter in peft_model.named_parameters():
            if parameter.requires_grad and "gate" in name and "lora_" in name:
                parameter.add_(1.0 * torch.randn_like(parameter))
        with recorder.record(attention_mask) as after:
            peft_model(input_ids=input_ids, attention_mask=attention_mask)
    metrics = summarize_routing(before, after)
    assert metrics["routing_changed_in_any_layer_fraction"] > 0.0


def test_summarize_lora_targets_sees_gate_weight_parameters(moe_case):
    model, _ = moe_case
    summary = summarize_lora_targets(
        model,
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_target_parameters=["gate.weight"],
    )
    assert summary["matched_parameter_count"] > 0
    assert all(name.endswith("gate.weight") for name in summary["matched_parameter_names"])
    assert summary["router_module_count"] > 0


@pytest.mark.network
def test_qwen3_moe_chat_template_online():
    """需要下载官方分词器；默认跳过，上卡前手动：pytest -m network。"""
    from transformers import AutoTokenizer

    from grpx.doctor import judge_chat_template

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-30B-A3B-Instruct-2507")
    result = judge_chat_template(tokenizer)
    assert result.status == "通过"
