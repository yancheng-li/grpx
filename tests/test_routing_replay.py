import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from grpx.routing_replay import (
    RoutingCapture,
    RoutingReplayer,
    align_generated_routing,
    assemble_captured_routing,
)

LAYER_COUNT = 3
EXPERT_COUNT = 8
EXPERTS_PER_TOKEN = 2
COMMON = dict(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=LAYER_COUNT,
    num_attention_heads=4,
    num_experts=EXPERT_COUNT,
    num_experts_per_tok=EXPERTS_PER_TOKEN,
    max_position_embeddings=64,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)


def make_model(kind, normalize=False, seed=0):
    torch.manual_seed(seed)
    if kind == "olmoe":
        config = OlmoeConfig(intermediate_size=16, num_key_value_heads=4, norm_topk_prob=normalize, **COMMON)
        model = OlmoeForCausalLM(config)
    else:
        config = Qwen3MoeConfig(
            intermediate_size=64, moe_intermediate_size=16, num_key_value_heads=2, norm_topk_prob=normalize, **COMMON
        )
        model = Qwen3MoeForCausalLM(config)
    # 随机初始化的路由权重很小，选择几乎是并列的；放大一些，让路由有明确的偏好
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("gate.weight"):
                parameter.normal_(0.0, 1.0)
    return model.eval()


def make_batch(seed=0, batch_size=2, length=9):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(3, 100, (batch_size, length), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def native_routing(model, input_ids, attention_mask):
    capture = RoutingCapture(model)
    with capture.capture() as calls, torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return assemble_captured_routing(calls, input_ids.shape[0]), logits


def different_routing(routing):
    """每个位置换成另一组不重复的专家。"""
    shifted = (routing + 3) % EXPERT_COUNT
    assert (shifted.sort(dim=-1).values != routing.sort(dim=-1).values).any()
    return shifted


KINDS = ["olmoe", "qwen3_moe"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("normalize", [False, True])
def test_replaying_native_routing_changes_nothing(kind, normalize):
    model = make_model(kind, normalize)
    input_ids, attention_mask = make_batch()
    routing, native_logits = native_routing(model, input_ids, attention_mask)
    assert routing.shape == (2, 9, LAYER_COUNT, EXPERTS_PER_TOKEN)

    replayer = RoutingReplayer(model)
    with replayer.replay(routing) as statistics, torch.no_grad():
        replayed_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.equal(replayed_logits, native_logits)
    summary = statistics.summary()
    assert summary["routing_replay_agreement_fraction"] == 1.0
    assert summary["routing_replay_expert_agreement_fraction"] == 1.0
    assert summary["routing_replay_fallback_fraction"] == 0.0


@pytest.mark.parametrize("kind", KINDS)
def test_expert_order_within_a_position_does_not_matter(kind):
    model = make_model(kind)
    input_ids, attention_mask = make_batch()
    routing, native_logits = native_routing(model, input_ids, attention_mask)
    with RoutingReplayer(model).replay(routing.flip(-1)), torch.no_grad():
        replayed_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    torch.testing.assert_close(replayed_logits, native_logits, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_forced_routing_is_used_and_changes_output(kind):
    model = make_model(kind)
    input_ids, attention_mask = make_batch()
    routing, native_logits = native_routing(model, input_ids, attention_mask)
    forced = different_routing(routing)

    replayer, capture = RoutingReplayer(model), RoutingCapture(model)
    # 先挂重放，再挂记录：记录读到的应该是重放之后的选择
    with replayer.replay(forced) as statistics, capture.capture() as calls, torch.no_grad():
        replayed_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    effective = assemble_captured_routing(calls, input_ids.shape[0])
    assert torch.equal(effective.sort(dim=-1).values, forced.sort(dim=-1).values)
    assert not torch.allclose(replayed_logits, native_logits)
    assert statistics.summary()["routing_replay_agreement_fraction"] < 1.0


@pytest.mark.parametrize("kind", KINDS)
def test_positions_without_record_fall_back_to_native(kind):
    model = make_model(kind)
    input_ids, attention_mask = make_batch()
    routing, native_logits = native_routing(model, input_ids, attention_mask)

    all_missing = torch.full_like(routing, -1)
    with RoutingReplayer(model).replay(all_missing) as statistics, torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.equal(logits, native_logits)
    assert statistics.summary()["routing_replay_fallback_fraction"] == 1.0

    partly_missing = routing.clone()
    partly_missing[:, -1] = -1  # 最后一个位置没有记录
    with RoutingReplayer(model).replay(partly_missing) as statistics, torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.equal(logits, native_logits)
    assert statistics.summary()["routing_replay_fallback_fraction"] == pytest.approx(1 / 9)


@pytest.mark.parametrize("kind", KINDS)
def test_replay_restores_original_routing_after_weights_drift(kind):
    """权重有了小的变化、模型自己的路由变了之后，重放仍然用回记录的选择。"""
    model = make_model(kind)
    input_ids, attention_mask = make_batch(length=24)
    routing, _ = native_routing(model, input_ids, attention_mask)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "gate.weight" not in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)
    drifted, _ = native_routing(model, input_ids, attention_mask)
    assert not torch.equal(drifted.sort(dim=-1).values, routing.sort(dim=-1).values)

    replayer, capture = RoutingReplayer(model), RoutingCapture(model)
    with replayer.replay(routing) as statistics, capture.capture() as calls, torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    effective = assemble_captured_routing(calls, input_ids.shape[0])
    assert torch.equal(effective.sort(dim=-1).values, routing.sort(dim=-1).values)
    assert 0.0 < statistics.summary()["routing_replay_agreement_fraction"] < 1.0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("normalize", [False, True])
def test_gradient_reaches_router_and_other_parameters(kind, normalize):
    model = make_model(kind, normalize)
    input_ids, attention_mask = make_batch()
    routing, _ = native_routing(model, input_ids, attention_mask)
    forced = different_routing(routing)
    with RoutingReplayer(model).replay(forced):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits.log_softmax(-1)[..., 5].sum().backward()
    gradients = dict(model.named_parameters())
    router_gradient = gradients["model.layers.0.mlp.gate.weight"].grad
    assert router_gradient is not None and router_gradient.abs().sum() > 0
    attention_gradient = gradients["model.layers.0.self_attn.q_proj.weight"].grad
    assert attention_gradient is not None and attention_gradient.abs().sum() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_gradients_match_native_when_replaying_native_routing(kind):
    model = make_model(kind)
    input_ids, attention_mask = make_batch()
    routing, _ = native_routing(model, input_ids, attention_mask)

    def gradient(replay):
        model.zero_grad()
        context = RoutingReplayer(model).replay(routing) if replay else torch.enable_grad()
        with context:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits.log_softmax(-1)[..., 5].sum().backward()
        return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    native, replayed = gradient(False), gradient(True)
    assert native.keys() == replayed.keys()
    for name in native:
        torch.testing.assert_close(replayed[name], native[name], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("kind", KINDS)
def test_replay_with_gradient_checkpointing(kind):
    model = make_model(kind)
    input_ids, attention_mask = make_batch()
    routing, _ = native_routing(model, input_ids, attention_mask)
    forced = different_routing(routing)

    def gradient(checkpointing):
        model.zero_grad()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
        else:
            model.gradient_checkpointing_disable()
            model.eval()
        with RoutingReplayer(model).replay(forced):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            logits.log_softmax(-1)[..., 5].sum().backward()  # 反向重算也在上下文里
        model.eval()
        return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    plain, checkpointed = gradient(False), gradient(True)
    for name in plain:
        torch.testing.assert_close(checkpointed[name], plain[name], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("kind", KINDS)
def test_padded_batch_with_missing_records_on_padding(kind):
    model = make_model(kind)
    input_ids, attention_mask = make_batch(batch_size=3, length=8)
    attention_mask[1, 5:] = 0
    input_ids[1, 5:] = 0
    routing, native_logits = native_routing(model, input_ids, attention_mask)
    routing_with_gaps = routing.clone()
    routing_with_gaps[1, 5:] = -1
    with RoutingReplayer(model).replay(routing_with_gaps) as statistics, torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.equal(logits, native_logits)
    assert statistics.summary()["routing_replay_fallback_fraction"] == pytest.approx(3 / 24)


@pytest.mark.parametrize("kind", KINDS)
def test_capture_during_cached_generation_matches_full_forward(kind):
    model = make_model(kind)
    input_ids, attention_mask = make_batch(batch_size=2, length=6)
    capture = RoutingCapture(model)
    with capture.capture() as calls, torch.no_grad():
        generated = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=5, min_new_tokens=5,
            do_sample=False, use_cache=True, pad_token_id=0,
        )
    captured = assemble_captured_routing(calls, batch_size=2)
    # 记录到的是所有输入过模型的 token：提示加上除最后一个之外的生成 token
    assert captured.shape == (2, generated.shape[1] - 1, LAYER_COUNT, EXPERTS_PER_TOKEN)
    full, _ = native_routing(model, generated[:, :-1], torch.ones_like(generated[:, :-1]))
    agreement = (captured.sort(dim=-1).values == full.sort(dim=-1).values).all(dim=-1).float().mean()
    assert agreement > 0.98  # 带缓存和不带缓存的数值有极小的差别，并列的专家可能互换

    # 把记录补上最后一个位置（没有记录），在完整序列上重放
    padded = torch.cat([captured, torch.full_like(captured[:, :1], -1)], dim=1)
    with RoutingReplayer(model).replay(padded) as statistics, torch.no_grad():
        model(input_ids=generated, attention_mask=torch.ones_like(generated))
    assert statistics.summary()["routing_replay_fallback_fraction"] == pytest.approx(1 / generated.shape[1])


def test_hooks_are_removed_and_errors_leave_no_hooks():
    model = make_model("olmoe")
    input_ids, attention_mask = make_batch()
    routing, native_logits = native_routing(model, input_ids, attention_mask)
    replayer = RoutingReplayer(model)
    with replayer.replay(different_routing(routing)), torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    with torch.no_grad():
        after = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.equal(after, native_logits)
    with pytest.raises(ValueError):
        with replayer.replay(routing[:, :, :2]):
            pass
    assert all(len(module._forward_hooks) == 0 for module in replayer.router_modules.values())


def test_invalid_records_are_rejected():
    model = make_model("olmoe")
    input_ids, attention_mask = make_batch()
    routing, _ = native_routing(model, input_ids, attention_mask)
    replayer = RoutingReplayer(model)

    def enter(bad):
        with replayer.replay(bad):
            pass

    with pytest.raises(ValueError, match="层"):
        enter(routing[:, :, :2])
    with pytest.raises(ValueError, match="专家"):
        enter(torch.cat([routing, routing], dim=-1))
    out_of_range = routing.clone(); out_of_range[0, 0, 0, 0] = EXPERT_COUNT
    with pytest.raises(ValueError, match="超出范围"):
        enter(out_of_range)
    duplicated = routing.clone(); duplicated[0, 0, 0, 1] = duplicated[0, 0, 0, 0]
    with pytest.raises(ValueError, match="重复"):
        enter(duplicated)
    half_filled = routing.clone(); half_filled[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="整行"):
        enter(half_filled)
    with pytest.raises(ValueError, match="整数"):
        enter(routing.float())
    with pytest.raises(RuntimeError, match="位置"):
        with replayer.replay(routing[:, :5]), torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)


def test_dense_model_is_reported_unavailable():
    assert not RoutingReplayer(torch.nn.Linear(4, 4)).is_available


def _training_batch(input_ids, attention_mask):
    return {
        "input_token_ids": input_ids,
        "attention_mask": attention_mask,
        "action_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "temperatures": torch.ones(input_ids.shape[0]),
    }


def test_routing_freeze_keeps_the_first_choice_after_the_gate_moves():
    from grpx.config import TrainingConfig
    from grpx.loop import consume_old_policy_forward

    model = make_model("olmoe")
    input_ids, attention_mask = make_batch()
    batch = _training_batch(input_ids, attention_mask)
    config = TrainingConfig(routing_freeze=True, updates_per_iteration=2)
    _log_probability, statistics, frozen = consume_old_policy_forward(
        model, batch, None, None, config)
    assert statistics is None
    assert frozen is not None and frozen.dtype == torch.int8
    assert torch.equal(batch["routed_experts"], frozen)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("gate.weight"):
                # 每个专家加不同的偏置。全体加同一个常数不会改变排序
                parameter.zero_()
                parameter[:, 0] = torch.arange(parameter.shape[0], dtype=parameter.dtype)
    native_after, _logits = native_routing(model, input_ids, attention_mask)
    assert not torch.equal(native_after.sort(dim=-1).values, frozen.long().sort(dim=-1).values)
    capture = RoutingCapture(model)
    with RoutingReplayer(model).replay(frozen), capture.capture() as calls, torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    effective = assemble_captured_routing(calls, input_ids.shape[0])
    assert torch.equal(effective.sort(dim=-1).values, frozen.long().sort(dim=-1).values)


def test_routing_freeze_leaves_a_sampler_record_untouched():
    from grpx.config import TrainingConfig
    from grpx.loop import consume_old_policy_forward

    model = make_model("olmoe")
    input_ids, attention_mask = make_batch()
    batch = _training_batch(input_ids, attention_mask)
    sampler_record = torch.tensor([0, 1], dtype=torch.int8).repeat(
        input_ids.shape[0], input_ids.shape[1], LAYER_COUNT, 1)
    batch["routed_experts"] = sampler_record
    config = TrainingConfig(routing_freeze=True, routing_replay=True, updates_per_iteration=2)
    _log_probability, statistics, frozen = consume_old_policy_forward(
        model, batch, None, None, config)
    assert frozen is None
    assert batch["routed_experts"] is sampler_record
    assert statistics is not None


def test_routing_freeze_rejects_a_dense_model():
    from grpx.config import TrainingConfig
    from grpx.loop import consume_old_policy_forward

    batch = {"input_token_ids": torch.zeros(1, 2, dtype=torch.long)}
    config = TrainingConfig(routing_freeze=True, updates_per_iteration=2)
    with pytest.raises(RuntimeError, match="routing_freeze"):
        consume_old_policy_forward(torch.nn.Linear(4, 4), batch, None, None, config)


def test_routing_freeze_must_be_boolean():
    from grpx.config import TrainingConfig
    with pytest.raises(ValueError, match="routing_freeze"):
        TrainingConfig(routing_freeze="yes")


def test_routing_freeze_is_idle_for_a_single_update():
    from grpx.config import TrainingConfig
    from grpx.loop import consume_old_policy_forward

    model = make_model("olmoe")
    input_ids, attention_mask = make_batch()
    batch = _training_batch(input_ids, attention_mask)
    config = TrainingConfig(routing_freeze=True, updates_per_iteration=1)
    _log_probability, statistics, frozen = consume_old_policy_forward(
        model, batch, None, None, config)
    assert frozen is None
    assert "routed_experts" not in batch
    assert statistics is None


def test_unreplayed_shift_ignores_a_forced_expert_record():
    from grpx.config import TrainingConfig
    from grpx.loop import measure_policy_shift
    from grpx.trainer import action_log_probabilities, build_batch
    from grpx.types import Step, Trajectory

    model = make_model("olmoe")
    trajectory = Trajectory("g", 0, 1.0, steps=[Step([1, 2, 3], [4, 5, 6], advantage=1.0)])
    batch = build_batch([trajectory], padding_token_id=0, device=model.device)
    with torch.no_grad():
        old_log_probability, _mask = action_log_probabilities(model, batch)
    routing, _logits = native_routing(model, batch["input_token_ids"], batch["attention_mask"])
    batch["routed_experts"] = different_routing(routing).to(torch.int8)
    config = TrainingConfig()
    replayed = measure_policy_shift(model, [batch], [old_log_probability], config, replay=True)
    native = measure_policy_shift(model, [batch], [old_log_probability], config, replay=False)
    assert native["shift_percentile_95_absolute_log_ratio"] < 1e-5
    assert replayed["shift_percentile_95_absolute_log_ratio"] > 1e-3


def test_freeze_measurement_script_runs_on_cpu():
    import importlib.util
    from pathlib import Path

    from grpx.config import TrainingConfig
    from grpx.types import Step, Trajectory

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "measure_routing_freeze.py"
    spec = importlib.util.spec_from_file_location("measure_routing_freeze", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = make_model("olmoe")
    kept = [
        Trajectory("g", 0, 1.0, steps=[Step([1, 2, 3], [4, 5, 6], advantage=1.0)]),
        Trajectory("g", 1, 1.0, steps=[Step([7, 8], [9, 10, 11], advantage=-1.0)]),
    ]
    config = TrainingConfig(learning_rate=1e-4, updates_per_iteration=2)
    for freeze in (False, True):
        result = module.measure_condition(model, kept, 0, config, freeze)
        assert result["n_tokens"] > 0
        assert "clip_fraction" in result
        assert "p95_absolute_log_ratio" in result


def test_align_generated_routing_keeps_prompt_and_drops_the_last_token():
    # 左填充 1 个位置，提示词 2 个，再喂回 2 个生成 token。最后一个回答 token 没有记录
    assembled = torch.arange(5 * 2 * 2).reshape(5, 2, 2)
    aligned = align_generated_routing(
        assembled, prompt_length=2, action_length=3, padded_prompt_length=3)
    assert aligned.dtype == torch.int8
    assert aligned.shape == (5, 2, 2)
    assert torch.equal(aligned[:2].long(), assembled[1:3])
    assert torch.equal(aligned[2:4].long(), assembled[3:5])
    assert torch.all(aligned[4] == -1)
