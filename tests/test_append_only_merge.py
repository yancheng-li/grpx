"""对话只前向一次：前缀检查、合并组批与按步组批在真因果模型上等价。"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from grpx.trainer import (
    action_log_probabilities,
    build_batch,
    grpo_loss,
    is_append_only,
    prepare_update_units,
)
from grpx.types import Step, Trajectory


def make_step(prompt, action, advantage=1.0):
    return Step(list(prompt), list(action), advantage=advantage)


def test_is_append_only_accepts_growing_context():
    trajectory = Trajectory("g", 0, 1.0, steps=[
        make_step([1, 2], [3, 4], 0.5),
        make_step([1, 2, 3, 4], [5], -0.5),
        make_step([1, 2, 3, 4, 5], [6, 7], 1.0),
    ])
    assert is_append_only(trajectory)


def test_is_append_only_rejects_rewritten_history():
    trajectory = Trajectory("g", 0, 1.0, steps=[
        make_step([1, 2], [3], 1.0),
        make_step([1, 9, 3], [4], -1.0),  # 中间被改写
    ])
    assert not is_append_only(trajectory)


def test_is_append_only_rejects_reencoded_action_mismatch():
    trajectory = Trajectory("g", 0, 1.0, steps=[
        make_step([1], [2, 3], 1.0),
        make_step([1, 2, 99], [4], -1.0),  # 回答重编码后不一致
    ])
    assert not is_append_only(trajectory)


def test_segment_loss_matches_per_row_when_one_segment_per_row():
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=64, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        eos_token_id=0, bos_token_id=0))
    trajectories = [
        Trajectory("a", 0, 1.0, [make_step([1, 2], [3, 4], 1.0)]),
        Trajectory("b", 0, 0.7, [make_step([5], [6, 7, 8], -1.0)]),
    ]
    batch = build_batch(trajectories, padding_token_id=0, device="cpu")
    with torch.no_grad():
        log_probs, mask = action_log_probabilities(model, batch)
    loss_new, _ = grpo_loss(
        log_probs, log_probs.detach(), mask, batch["advantages"],
        length_normalization="per_sequence",
        row_weights=batch["row_weights"],
        segment_ids=batch["segment_ids"],
        segment_weights=batch["segment_weights"],
        segment_advantages=batch["segment_advantages"],
    )
    loss_old, _ = grpo_loss(
        log_probs, log_probs.detach(), mask, batch["advantages"],
        length_normalization="per_sequence",
        row_weights=batch["row_weights"],
    )
    assert torch.allclose(loss_new, loss_old)


def test_merged_and_per_step_batches_match_on_causal_model():
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=48, n_positions=64, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        eos_token_id=0, bos_token_id=0))
    trajectory = Trajectory("g", 0, 1.0, steps=[
        make_step([1, 2], [3, 4], 0.5),
        make_step([1, 2, 3, 4], [5, 6], -0.5),
        make_step([1, 2, 3, 4, 5, 6], [7], 1.0),
    ])
    assert is_append_only(trajectory)

    per_step_units = prepare_update_units([trajectory], merge_append_only=False)
    merged_units = prepare_update_units([trajectory], merge_append_only=True)
    assert len(per_step_units) == 3
    assert len(merged_units) == 1

    per_step_batch = build_batch(per_step_units, padding_token_id=0, device="cpu")
    merged_batch = build_batch(merged_units, padding_token_id=0, device="cpu")

    with torch.no_grad():
        per_step_log, per_step_mask = action_log_probabilities(model, per_step_batch)
        merged_log, merged_mask = action_log_probabilities(model, merged_batch)

    # 合并后三段回答的对数概率应与分步一致（按段拼接）
    per_step_tokens = per_step_log[per_step_mask]
    merged_tokens = merged_log[merged_mask]
    assert torch.allclose(per_step_tokens, merged_tokens, atol=1e-5)

    loss_options = dict(length_normalization="per_sequence", clip_range=0.2)
    loss_per_step, _ = grpo_loss(
        per_step_log, per_step_log.detach(), per_step_mask,
        per_step_batch["advantages"], row_weights=per_step_batch["row_weights"],
        segment_ids=per_step_batch["segment_ids"],
        segment_weights=per_step_batch["segment_weights"],
        segment_advantages=per_step_batch["segment_advantages"],
        **loss_options,
    )
    loss_merged, _ = grpo_loss(
        merged_log, merged_log.detach(), merged_mask,
        merged_batch["advantages"], row_weights=merged_batch["row_weights"],
        segment_ids=merged_batch["segment_ids"],
        segment_weights=merged_batch["segment_weights"],
        segment_advantages=merged_batch["segment_advantages"],
        **loss_options,
    )
    assert torch.allclose(loss_per_step, loss_merged, atol=1e-5)

    # 梯度：对同一损失反传，可训练参数梯度应一致（用独立拷贝）
    model_a = GPT2LMHeadModel(GPT2Config(
        vocab_size=48, n_positions=64, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        eos_token_id=0, bos_token_id=0))
    model_b = GPT2LMHeadModel(GPT2Config(
        vocab_size=48, n_positions=64, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        eos_token_id=0, bos_token_id=0))
    model_b.load_state_dict(model_a.state_dict())

    def grads(model, batch):
        model.zero_grad(set_to_none=True)
        log, mask = action_log_probabilities(model, batch)
        loss, _ = grpo_loss(
            log, log.detach(), mask, batch["advantages"],
            row_weights=batch["row_weights"],
            segment_ids=batch["segment_ids"],
            segment_weights=batch["segment_weights"],
            segment_advantages=batch["segment_advantages"],
            length_normalization="per_sequence",
        )
        loss.backward()
        return {name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters() if parameter.grad is not None}

    grads_a = grads(model_a, per_step_batch)
    grads_b = grads(model_b, merged_batch)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        assert torch.allclose(grads_a[name], grads_b[name], atol=1e-5), name


def test_non_append_only_falls_back_to_per_step_units():
    trajectory = Trajectory("g", 0, 1.0, steps=[
        make_step([1, 2], [3], 1.0),
        make_step([9, 9, 3], [4], -1.0),
    ])
    units = prepare_update_units([trajectory], merge_append_only=True)
    assert len(units) == 2
