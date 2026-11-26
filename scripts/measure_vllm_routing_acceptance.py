"""vLLM 0.34 先后验收：先只采样并存盘，停掉服务后再在训练端重算。

一张 24 GB 的卡放不下两份 OLMoE。采样阶段不要加载训练权重。

    # 服务已用 --enable-return-routed-experts 启动之后
    python scripts/measure_vllm_routing_acceptance.py sample \
        --config configs/arithmetic_olmoe_d2d3.toml \
        --base-url http://127.0.0.1:8002 \
        --output outputs/vllm034-routing-acceptance

    # 停掉服务之后
    python scripts/measure_vllm_routing_acceptance.py rescore \
        --config configs/arithmetic_olmoe_d2d3.toml \
        --output outputs/vllm034-routing-acceptance
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from grpx.backends.vllm import VllmBackend
from grpx.command_line import (
    build_environment_factory,
    build_training_config,
    call_with_options,
    import_object,
    load_configuration,
    load_model,
)
from grpx.iteration_state import _trajectory_from_dict, _trajectory_to_dict
from grpx.loop import collect_trajectories
from grpx.routing_replay import find_router_modules, _router_attribute


def gap_module():
    path = Path(__file__).resolve().parent / "measure_routing_replay_gap.py"
    spec = importlib.util.spec_from_file_location("measure_routing_replay_gap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def routing_shape(model_name):
    config = AutoConfig.from_pretrained(model_name)
    experts = getattr(config, "num_experts_per_tok", None)
    layers = getattr(config, "num_hidden_layers", None)
    if experts is None or layers is None:
        raise SystemExit(f"{model_name} 的配置里没有 num_experts_per_tok / num_hidden_layers")
    return int(layers), int(experts)


def end_token_ids(model_name, tokenizer):
    generation = GenerationConfig.from_pretrained(model_name)
    token_ids = generation.eos_token_id
    if token_ids is None:
        token_ids = tokenizer.eos_token_id
    if token_ids is None:
        raise SystemExit("找不到 eos_token_id")
    if not isinstance(token_ids, list):
        token_ids = [token_ids]
    return set(token_ids)


def sample(arguments):
    configuration = load_configuration(arguments.config, [])
    training = build_training_config(configuration)
    model_name = configuration["model"]["name"]
    backend_section = configuration["backend"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    layer_count, experts_per_token = routing_shape(model_name)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    backend = VllmBackend(
        tokenizer,
        end_token_ids(model_name, tokenizer),
        adapter_directory=output / "vllm_adapter",
        base_url=arguments.base_url,
        adapter_name="base",
        max_new_tokens=backend_section.get("max_new_tokens", 16),
        temperature=backend_section.get("temperature", 1.0),
        max_concurrent_requests=backend_section.get("max_concurrent_requests", 64),
        expected_base_model=model_name,
        training_model_name=model_name,
    )
    backend.adapter_name = model_name
    backend.verify_server_base_model()
    versions = backend.server_versions()
    print("server", json.dumps(versions, ensure_ascii=False), flush=True)
    if not any(item["version"].startswith("0.34") for item in versions):
        raise SystemExit(f"这不是 0.34 的服务：{versions}")
    backend.record_routing = True
    backend.routing_layer_count = layer_count
    backend.routing_experts_per_token = experts_per_token
    dataset = call_with_options(
        import_object(configuration["task"]["dataset"]),
        configuration["task"].get("train_dataset_options", {}),
        "dataset",
    )
    environment_factory = build_environment_factory(configuration["task"])
    rng = random.Random(training.seed)
    samples = rng.sample(dataset, training.groups_per_iteration)
    trajectories = collect_trajectories(
        backend, tokenizer, environment_factory, samples, training, 0)
    missing = [
        step for trajectory in trajectories for step in trajectory.steps
        if step.routed_experts is None
    ]
    if missing:
        raise SystemExit(f"有 {len(missing)} 步没有路由记录")
    payload = {
        "config": arguments.config,
        "model": model_name,
        "base_url": arguments.base_url,
        "server_versions": versions,
        "routing_layer_count": layer_count,
        "routing_experts_per_token": experts_per_token,
        "trajectories": [_trajectory_to_dict(item) for item in trajectories],
    }
    destination = output / "trajectories.pt"
    torch.save(payload, destination)
    print(f"wrote {destination} trajectories={len(trajectories)}", flush=True)
    backend.close()


def rescore(arguments):
    configuration = load_configuration(arguments.config, [])
    training = build_training_config(configuration)
    source = Path(arguments.output) / "trajectories.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    trajectories = [_trajectory_from_dict(item) for item in payload["trajectories"]]
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(training.memory_fraction)
    model, tokenizer = load_model(configuration["model"], training.seed)
    modules = find_router_modules(model)
    if len(modules) != payload["routing_layer_count"]:
        raise SystemExit(
            f"训练端有 {len(modules)} 个路由模块，采样时按 {payload['routing_layer_count']} 层解码"
        )
    experts = int(_router_attribute(next(iter(modules.values())), "top_k"))
    if experts != payload["routing_experts_per_token"]:
        raise SystemExit(
            f"训练端每个 token 选 {experts} 个专家，采样时按 {payload['routing_experts_per_token']} 解码"
        )
    model.grpx_output_logit_chunk_size = training.output_logit_chunk_size
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    outcome = gap_module().measure_round(model, trajectories, training, pad_token_id)
    result = {
        "server_versions": payload["server_versions"],
        "lengths": outcome["lengths"],
        "routing": outcome["routing"],
        "all_pooled_no_replay": outcome["all_answers"]["pooled_no_replay"],
        "all_pooled_replay": outcome["all_answers"]["pooled_replay"],
        "kept_pooled_no_replay": outcome["kept_answers"]["pooled_no_replay"],
        "kept_pooled_replay": outcome["kept_answers"]["pooled_replay"],
    }
    destination = Path(arguments.output) / "rescore.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("wrote", destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("sample", "rescore"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", default="outputs/vllm034-routing-acceptance")
    arguments = parser.parse_args()
    if arguments.phase == "sample":
        sample(arguments)
    else:
        rescore(arguments)


if __name__ == "__main__":
    main()
