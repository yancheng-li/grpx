"""上机预检：同权两次 / 只扰动注意力 / 只扰动路由，核对 RoutingRecorder。

用法：
  PYTHONPATH=. python scripts/check_routing_recorder.py --config configs/arithmetic_olmoe_d2d3_router.toml
"""
from __future__ import annotations

import argparse

import torch

from grpx.command_line import apply_lora, load_base_model_and_tokenizer, load_configuration
from grpx.routing_diagnostics import RoutingRecorder, summarize_routing


def _forward(model, input_ids, attention_mask, recorder):
    with torch.no_grad():
        with recorder.record(attention_mask) as snapshot:
            model(input_ids=input_ids, attention_mask=attention_mask)
    return snapshot


def _perturb(model, predicate, scale=1.0):
    count = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and predicate(name):
                parameter.add_(scale * torch.randn_like(parameter))
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/arithmetic_olmoe_d2d3_router.toml")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    configuration = load_configuration(arguments.config, [])
    model_section = configuration["model"]

    input_ids = torch.randint(0, 1000, (4, 24), device="cuda")
    attention_mask = torch.ones(4, 24, dtype=torch.long, device="cuda")

    def fresh_model():
        base, _tokenizer, dtype_name = load_base_model_and_tokenizer(
            model_section, device="cuda")
        return apply_lora(base, model_section, arguments.seed, dtype_name)

    # 1) 同权两次
    model = fresh_model()
    recorder = RoutingRecorder(model)
    assert recorder.is_available
    before = _forward(model, input_ids, attention_mask, recorder)
    after = _forward(model, input_ids, attention_mask, recorder)
    identical = summarize_routing(before, after)
    print("identical_forwards", identical, flush=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={trainable}", flush=True)
    del model
    torch.cuda.empty_cache()

    # 2) 只扰动注意力 LoRA
    model = fresh_model()
    recorder = RoutingRecorder(model)
    before = _forward(model, input_ids, attention_mask, recorder)
    n_attn = _perturb(
        model,
        lambda name: "lora_" in name and "self_attn" in name,
        scale=0.5,
    )
    after = _forward(model, input_ids, attention_mask, recorder)
    attn_only = summarize_routing(before, after)
    print(f"perturb_attention n={n_attn}", attn_only, flush=True)
    del model
    torch.cuda.empty_cache()

    # 3) 只扰动路由 LoRA（若无路由 LoRA 则扰动 gate 权重本身）
    model = fresh_model()
    recorder = RoutingRecorder(model)
    before = _forward(model, input_ids, attention_mask, recorder)
    n_router = _perturb(
        model,
        lambda name: "lora_" in name and "gate" in name and "gate_up" not in name,
        scale=0.5,
    )
    if n_router == 0:
        n_router = _perturb(
            model,
            lambda name: ".gate." in name and "gate_up" not in name and name.endswith("weight"),
            scale=0.5,
        )
    after = _forward(model, input_ids, attention_mask, recorder)
    router_only = summarize_routing(before, after)
    print(f"perturb_router n={n_router}", router_only, flush=True)

    assert identical["routing_changed_in_any_layer_fraction"] == 0.0
    assert router_only["routing_changed_in_any_layer_fraction"] > 0.0
    # 放开路由后，只扰动路由应明显高于只扰动注意力
    print(
        "attn_changed_any="
        f"{attn_only['routing_changed_in_any_layer_fraction']:.4f} "
        "router_changed_any="
        f"{router_only['routing_changed_in_any_layer_fraction']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
