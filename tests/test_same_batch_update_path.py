"""同批更新路径回归：merge / 段预算切分不应改变累积梯度（float32）。"""
import copy
import dataclasses

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from grpx.config import TrainingConfig
from grpx.loop import accumulate_gradients, update_policy
from grpx.types import Step, Trajectory


def make_append_only_trajectory(group_id, step_count, advantage, start=1):
    steps = []
    prompt = [start, start + 1]
    token = start + 2
    for _ in range(step_count):
        action = [token, token + 1]
        steps.append(Step(list(prompt), action, advantage=advantage))
        prompt = prompt + action
        token += 2
    return Trajectory(group_id, 0, 1.0, steps)


def tiny_gpt2(seed=0):
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_positions=128, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        eos_token_id=0, bos_token_id=0,
    ))
    model.to(dtype=torch.float32)
    return model


def collect_update_gradients(model, kept, config, update_steps):
    """跑 update_policy，但在每次 optimizer.step 前记下可训练参数梯度。"""
    padding_token_id = 0
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    # 复用 update_policy 的切批与旧对数概率，但自己循环以便抓梯度
    from grpx.loop import measure_backend_log_probability_gap, split_into_row_chunks
    from grpx.trainer import (action_log_probabilities, build_batch, merged_trajectory_fraction,
                              prepare_update_units)

    units = prepare_update_units(kept, merge_append_only=config.merge_append_only_turns)
    chunks = split_into_row_chunks(
        kept, config.max_rows_per_minibatch, config.max_tokens_per_minibatch,
        merge_append_only=config.merge_append_only_turns)
    batches = [build_batch(chunk, padding_token_id, model.device, config.row_weighting)
               for chunk in chunks]
    model.eval()
    with torch.no_grad():
        old_log_probabilities = [action_log_probabilities(model, batch)[0] for batch in batches]
    measure_backend_log_probability_gap(batches, old_log_probabilities)

    grad_steps = []
    param_snapshot = None
    for step_index in range(update_steps):
        model.train()
        optimizer.zero_grad()
        averaged = accumulate_gradients(model, batches, old_log_probabilities, config)
        if (config.max_approximate_kl is not None
                and averaged["approximate_kl"] > config.max_approximate_kl):
            break
        grad_steps.append([parameter.grad.detach().clone() for parameter in trainable])
        torch.nn.utils.clip_grad_norm_(trainable, config.max_gradient_norm)
        optimizer.step()
        if step_index == update_steps - 1:
            param_snapshot = [parameter.detach().clone() for parameter in trainable]
    return grad_steps, param_snapshot, merged_trajectory_fraction(kept, units)


def assert_grad_steps_close(left, right, abs_tol=1e-5):
    assert len(left) == len(right)
    for step_left, step_right in zip(left, right):
        assert len(step_left) == len(step_right)
        for gradient_left, gradient_right in zip(step_left, step_right):
            assert torch.allclose(gradient_left, gradient_right, atol=abs_tol, rtol=0), (
                f"max abs diff {(gradient_left - gradient_right).abs().max().item()}"
            )


@pytest.mark.parametrize("max_rows", [4, 8, 32])
def test_merge_on_off_same_accumulated_gradients(max_rows):
    kept = [
        make_append_only_trajectory("a", 4, 1.0, start=1),
        make_append_only_trajectory("a", 4, -1.0, start=3),
        make_append_only_trajectory("b", 3, 0.5, start=5),
        make_append_only_trajectory("b", 3, -0.5, start=7),
    ]
    base = TrainingConfig(
        learning_rate=1e-3, clip_range=0.2, length_normalization="per_sequence",
        max_rows_per_minibatch=max_rows, max_tokens_per_minibatch=10**6,
        updates_per_iteration=2, merge_append_only_turns=True,
    )
    model_on = tiny_gpt2(0)
    model_off = tiny_gpt2(0)
    grads_on, params_on, merge_fraction = collect_update_gradients(
        model_on, copy.deepcopy(kept), base, update_steps=2)
    assert merge_fraction > 0
    grads_off, params_off, _ = collect_update_gradients(
        model_off, copy.deepcopy(kept),
        dataclasses.replace(base, merge_append_only_turns=False), update_steps=2)
    assert_grad_steps_close(grads_on, grads_off)
    for left, right in zip(params_on, params_off):
        assert torch.allclose(left, right, atol=1e-5, rtol=0)


@pytest.mark.parametrize("merge", [True, False])
def test_segment_budget_4_8_32_same_gradients(merge):
    kept = [
        make_append_only_trajectory("g0", 6, 1.2, start=1),
        make_append_only_trajectory("g0", 6, -0.8, start=2),
        make_append_only_trajectory("g1", 5, 0.4, start=4),
        make_append_only_trajectory("g1", 5, -0.3, start=6),
    ]
    reference = None
    for max_rows in (4, 8, 32):
        config = TrainingConfig(
            learning_rate=1e-3, clip_range=0.2, length_normalization="per_sequence",
            max_rows_per_minibatch=max_rows, max_tokens_per_minibatch=10**6,
            updates_per_iteration=2, merge_append_only_turns=merge,
        )
        model = tiny_gpt2(1)
        grads, params, _ = collect_update_gradients(
            model, copy.deepcopy(kept), config, update_steps=2)
        if reference is None:
            reference = (grads, params)
        else:
            assert_grad_steps_close(reference[0], grads)
            for left, right in zip(reference[1], params):
                assert torch.allclose(left, right, atol=1e-5, rtol=0)


def test_update_policy_two_steps_matches_manual_path():
    """sanity：上面的手工路径与 update_policy 最终参数一致。"""
    kept = [
        make_append_only_trajectory("x", 3, 1.0, start=1),
        make_append_only_trajectory("x", 3, -1.0, start=2),
    ]
    config = TrainingConfig(
        learning_rate=1e-3, clip_range=0.2, length_normalization="per_sequence",
        max_rows_per_minibatch=8, max_tokens_per_minibatch=10**6,
        updates_per_iteration=2, merge_append_only_turns=True,
    )
    model_manual = tiny_gpt2(2)
    _, params_manual, _ = collect_update_gradients(
        model_manual, copy.deepcopy(kept), config, update_steps=2)

    model_api = tiny_gpt2(2)
    trainable = [parameter for parameter in model_api.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    update_policy(model_api, optimizer, copy.deepcopy(kept), 0, config)
    for left, right in zip(params_manual, [parameter.detach() for parameter in trainable]):
        assert torch.allclose(left, right, atol=1e-5, rtol=0)
