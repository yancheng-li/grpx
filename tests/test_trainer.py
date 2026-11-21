import math
from types import SimpleNamespace

import pytest
import torch

from grpx.trainer import action_log_probabilities, grpo_loss, train_on_batch, build_batch
from grpx.types import Step, Trajectory

class TinyLanguageModel(torch.nn.Module):
    """每个位置的输出只依赖当前 token，天然满足因果性"""
    def __init__(self, vocabulary_size=16, hidden_size=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.output_layer(self.embedding(input_ids)))


class TinyBackbone(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class SplitTinyLanguageModel(TinyLanguageModel):
    """和 transformers 模型一样提供骨干网络与输出层接口，走只算回答位置的路径"""
    def __init__(self, config=None, **keyword_arguments):
        super().__init__(**keyword_arguments)
        self.backbone = TinyBackbone(self.embedding)
        self.config = config
        self.full_forward_calls = 0

    @property
    def base_model(self):
        return self.backbone

    def get_output_embeddings(self):
        return self.output_layer

    def forward(self, input_ids, attention_mask=None):
        self.full_forward_calls += 1
        return super().forward(input_ids, attention_mask)


class PeftLikeWrapper(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def get_base_model(self):
        return self.inner

    def forward(self, input_ids, attention_mask=None):
        return self.inner(input_ids, attention_mask)


def make_trajectory(prompt_token_ids, action_token_ids, advantage, sampling_temperature=1.0):
    step = Step(prompt_token_ids=prompt_token_ids, action_token_ids=action_token_ids,
                advantage=advantage)
    return Trajectory("group", 0, sampling_temperature=sampling_temperature, steps=[step])


def make_batch(*trajectories):
    return build_batch(list(trajectories), padding_token_id=0, device="cpu")


def masked_log_probabilities(model, batch):
    with torch.no_grad():
        log_probabilities, mask = action_log_probabilities(model, batch)
    return [row[row_mask].tolist() for row, row_mask in zip(log_probabilities, mask)]


# ---------- build_batch ----------

def test_action_mask_selects_exactly_the_action_tokens():
    batch = make_batch(make_trajectory([1, 2, 3], [4, 5], 1.0))
    targets = batch["input_token_ids"][:, 1:]
    shifted_mask = batch["action_mask"][:, 1:]
    assert targets[shifted_mask].tolist() == [4, 5]


def test_shorter_sequence_is_padded_on_the_right():
    batch = make_batch(make_trajectory([1, 2, 3], [4, 5], 1.0), make_trajectory([6], [7, 8], -1.0))
    assert batch["input_token_ids"][1].tolist() == [6, 7, 8, 0, 0]
    assert batch["attention_mask"].sum(1).tolist() == [5, 3]
    assert batch["action_mask"].sum(1).tolist() == [2, 2]
    assert batch["advantages"].tolist() == [1.0, -1.0]


def test_build_batch_sampling_log_probabilities_mask_skips_none_and_missing():
    with_server = Trajectory("g", 0, 1.0, [Step(
        [1, 2], [3, 4, 5], advantage=1.0,
        backend_log_probabilities=[-0.1, -0.2, None],  # 补上的结束符无服务端概率
    )])
    local = Trajectory("g", 0, 1.0, [Step(
        [6], [7, 8], advantage=-1.0, backend_log_probabilities=None,
    )])
    batch = build_batch([with_server, local], padding_token_id=0, device="cpu")
    assert batch["sampling_log_probabilities"].shape == batch["action_mask"].shape
    assert batch["sampling_log_probability_mask"].shape == batch["action_mask"].shape
    # 回答落在 prompt 之后：位置 2、3 有值，位置 4（结束符）与本地行整行无值
    assert batch["sampling_log_probability_mask"][0].tolist() == [False, False, True, True, False]
    assert batch["sampling_log_probabilities"][0, 2:4].tolist() == pytest.approx([-0.1, -0.2])
    assert not batch["sampling_log_probability_mask"][1].any()


def test_gap_from_old_log_probabilities_ignores_unmasked_positions():
    from grpx.loop import measure_backend_log_probability_gap

    batch = {
        "sampling_log_probabilities": torch.tensor([[0.0, 0.0, -1.0, -2.0, 0.0]]),
        "sampling_log_probability_mask": torch.tensor([[False, False, True, True, False]]),
    }
    # old 已是错开一位后的形状，与 action_mask[:, 1:] 一致
    old = torch.tensor([[0.0, -1.1, -2.5, 9.0]])
    metrics = measure_backend_log_probability_gap([batch], [old])
    # 只比较 mask 为 True 的错开位置：|-1.1-(-1.0)|=0.1 与 |-2.5-(-2.0)|=0.5
    expected = torch.tensor([0.1, 0.5])
    assert metrics["backend_log_probability_gap_median"] == pytest.approx(expected.median().item())
    assert metrics["backend_log_probability_gap_p95"] == pytest.approx(
        torch.quantile(expected, 0.95).item())
    assert metrics["backend_log_probability_gap_max"] == pytest.approx(0.5)
    # |Δ|>0.5 不含恰好等于 0.5
    assert metrics["backend_log_probability_gap_above_half_fraction"] == pytest.approx(0.0)


def test_action_log_probabilities_builds_graph_unless_no_grad():
    """除更新外所有前向应在 no_grad 下；无包装时会建图，显存是无梯度时的数倍。"""
    torch.manual_seed(0)
    model = TinyLanguageModel()
    batch = make_batch(make_trajectory([1, 2], [3, 4], 1.0))
    with torch.enable_grad():
        log_probabilities, _ = action_log_probabilities(model, batch)
    assert log_probabilities.grad_fn is not None
    with torch.no_grad():
        detached, _ = action_log_probabilities(model, batch)
    assert detached.grad_fn is None


# ---------- action_log_probabilities ----------

def test_padding_does_not_change_log_probabilities():
    torch.manual_seed(0)
    model = TinyLanguageModel()
    short = make_trajectory([6], [7, 8], -1.0)
    alone = masked_log_probabilities(model, make_batch(short))[0]
    together = masked_log_probabilities(
        model, make_batch(make_trajectory([1, 2, 3, 4], [5, 9, 10], 1.0), short))[1]
    assert together == pytest.approx(alone)


def test_temperature_divides_logits():
    torch.manual_seed(0)
    model = TinyLanguageModel()
    batch = make_batch(make_trajectory([1, 2], [3], 1.0, sampling_temperature=2.0))
    with torch.no_grad():
        logits = model(batch["input_token_ids"]).logits
    # 回答 token 3 位于位置 2，由位置 1 的 logits 预测
    expected = torch.log_softmax(logits[0, 1] / 2.0, dim=-1)[3].item()
    assert masked_log_probabilities(model, batch)[0] == pytest.approx([expected])


def test_temperature_one_matches_undivided_logits():
    torch.manual_seed(0)
    model = TinyLanguageModel()
    batch = make_batch(make_trajectory([1, 2], [3, 4], 1.0, sampling_temperature=1.0))
    with torch.no_grad():
        logits = model(batch["input_token_ids"]).logits.float()
    expected = [
        torch.log_softmax(logits[0, 1], dim=-1)[3].item(),
        torch.log_softmax(logits[0, 2], dim=-1)[4].item(),
    ]
    assert masked_log_probabilities(model, batch)[0] == pytest.approx(expected)


def full_logits_reference(model, batch):
    """旧实现：对全部位置算 logits，再取回答位置"""
    with torch.no_grad():
        logits = model(input_ids=batch["input_token_ids"]).logits[:, :-1].float()
        logits = logits / batch["temperatures"][:, None, None]
        log_probabilities = torch.log_softmax(logits, dim=-1).gather(
            -1, batch["input_token_ids"][:, 1:, None]).squeeze(-1)
    mask = batch["action_mask"][:, 1:]
    return log_probabilities * mask


def make_mixed_batch():
    return make_batch(make_trajectory([1, 2, 3], [4, 5], 1.0, sampling_temperature=0.7),
                      make_trajectory([6], [7, 8, 9], -1.0, sampling_temperature=1.3))


@pytest.mark.parametrize("wrap", [False, True])
def test_action_only_logits_match_full_computation(wrap):
    torch.manual_seed(0)
    inner = SplitTinyLanguageModel()
    model = PeftLikeWrapper(inner) if wrap else inner
    batch = make_mixed_batch()
    with torch.no_grad():
        log_probabilities, mask = action_log_probabilities(model, batch)
    assert inner.full_forward_calls == 0
    assert torch.equal(mask, batch["action_mask"][:, 1:])
    assert log_probabilities[~mask].abs().max().item() == 0
    assert log_probabilities.flatten().tolist() == pytest.approx(full_logits_reference(inner, batch).flatten().tolist(), abs=1e-6)


def test_chunked_output_log_probs_match_unchunked():
    torch.manual_seed(0)
    model = SplitTinyLanguageModel()
    batch = make_mixed_batch()
    with torch.no_grad():
        full, mask = action_log_probabilities(model, batch, output_logit_chunk_size=None)
        chunked, mask2 = action_log_probabilities(model, batch, output_logit_chunk_size=1)
    assert torch.equal(mask, mask2)
    assert chunked.flatten().tolist() == pytest.approx(full.flatten().tolist(), abs=1e-6)


def test_chunked_output_backward_matches_unchunked():
    batch = make_mixed_batch()

    def loss_and_grads(chunk_size):
        torch.manual_seed(0)
        model = SplitTinyLanguageModel()
        log_probabilities, mask = action_log_probabilities(
            model, batch, output_logit_chunk_size=chunk_size)
        loss = -(log_probabilities[mask]).sum()
        loss.backward()
        return {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    full = loss_and_grads(None)
    chunked = loss_and_grads(1)
    assert full.keys() == chunked.keys()
    for name in full:
        assert torch.allclose(full[name], chunked[name], atol=1e-5, rtol=1e-4), name


def test_fallback_path_matches_full_computation():
    torch.manual_seed(0)
    model = TinyLanguageModel()
    batch = make_mixed_batch()
    with torch.no_grad():
        log_probabilities, _ = action_log_probabilities(model, batch)
    assert log_probabilities.flatten().tolist() == pytest.approx(full_logits_reference(model, batch).flatten().tolist(), abs=1e-6)


def test_models_with_logit_postprocessing_use_full_forward():
    torch.manual_seed(0)
    model = SplitTinyLanguageModel(config=SimpleNamespace(final_logit_softcapping=30.0))
    with torch.no_grad():
        action_log_probabilities(model, make_mixed_batch())
    assert model.full_forward_calls == 1


def test_action_only_gradients_match_full_computation():
    torch.manual_seed(0)
    split_model = SplitTinyLanguageModel()
    plain_model = TinyLanguageModel()
    plain_model.load_state_dict(split_model.state_dict(), strict=False)
    batch = make_mixed_batch()
    gradients = []
    for model in (split_model, plain_model):
        log_probabilities, mask = action_log_probabilities(model, batch)
        (log_probabilities * mask).sum().backward()
        gradients.append([parameter.grad.flatten().tolist() for parameter in
                          (model.embedding.weight, model.output_layer.weight, model.output_layer.bias)])
    for split_gradient, plain_gradient in zip(*gradients):
        assert split_gradient == pytest.approx(plain_gradient, abs=1e-6)


# ---------- grpo_loss ----------

def test_identical_policies_give_zero_divergence():
    log_probabilities = torch.randn(2, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    advantages = torch.tensor([2.0, -0.5])
    loss, metrics = grpo_loss(log_probabilities, log_probabilities.clone(), mask, advantages)
    assert loss.item() == pytest.approx(-advantages.mean().item())
    assert metrics["clip_fraction"] == 0
    assert metrics["approximate_kl"] == pytest.approx(0)
    assert metrics["max_ratio_deviation"] == pytest.approx(0)


@pytest.mark.parametrize("ratio, advantage, expect_gradient", [
    (1.5, 1.0, False),    # 已经涨过头的好动作：裁剪，不再推
    (1.5, -1.0, True),    # 概率变高的坏动作：必须能压回去
    (0.5, 1.0, True),     # 概率变低的好动作：必须能拉回来
    (0.5, -1.0, False),   # 已经压过头的坏动作：裁剪，不再压
])
def test_clipping_is_asymmetric(ratio, advantage, expect_gradient):
    new_log_probabilities = torch.tensor([[math.log(ratio)]], requires_grad=True)
    loss, _ = grpo_loss(new_log_probabilities, torch.zeros(1, 1),
                        torch.ones(1, 1, dtype=torch.bool), torch.tensor([advantage]))
    loss.backward()
    assert (new_log_probabilities.grad.abs().item() > 0) == expect_gradient


def test_length_normalization_modes():
    log_probabilities = torch.zeros(2, 4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    advantages = torch.tensor([1.0, -1.0])
    per_sequence_loss, _ = grpo_loss(log_probabilities, log_probabilities, mask, advantages,
                                     length_normalization="per_sequence")
    constant_loss, _ = grpo_loss(log_probabilities, log_probabilities, mask, advantages,
                                 length_normalization="constant", constant_length=4)
    assert per_sequence_loss.item() == pytest.approx(0.0)       # -(4/4 + -1/1) / 2
    assert constant_loss.item() == pytest.approx(-0.375)        # -(4 - 1) / (2 * 4)


def test_masked_positions_are_ignored():
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    old_log_probabilities = torch.zeros(1, 4)
    clean = torch.tensor([[0.1, -0.1, 0.0, 0.0]])
    dirty = torch.tensor([[0.1, -0.1, 50.0, -50.0]])
    advantages = torch.tensor([1.0])
    clean_loss, clean_metrics = grpo_loss(clean, old_log_probabilities, mask, advantages)
    dirty_loss, dirty_metrics = grpo_loss(dirty, old_log_probabilities, mask, advantages)
    assert dirty_loss.item() == pytest.approx(clean_loss.item())
    assert dirty_metrics == pytest.approx(clean_metrics)


def test_unknown_length_normalization_raises():
    with pytest.raises(ValueError):
        grpo_loss(torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool),
                  torch.tensor([1.0]), length_normalization="unknown")


# ---------- train_on_batch ----------

def make_training_setup():
    torch.manual_seed(0)
    model = TinyLanguageModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = make_batch(make_trajectory([1, 2], [3], 1.0), make_trajectory([1, 2], [4], -1.0))
    return model, optimizer, batch


def test_first_update_has_ratio_exactly_one():
    model, optimizer, batch = make_training_setup()
    history = train_on_batch(model, optimizer, batch)
    assert history[0]["max_ratio_deviation"] == pytest.approx(0, abs=1e-6)


def test_good_action_becomes_more_likely_and_bad_action_less():
    model, optimizer, batch = make_training_setup()
    before = masked_log_probabilities(model, batch)
    for _ in range(5):
        train_on_batch(model, optimizer, batch)
    after = masked_log_probabilities(model, batch)
    assert after[0][0] > before[0][0]     # 优势为正的回答
    assert after[1][0] < before[1][0]     # 优势为负的回答


def test_repeated_updates_move_away_from_old_policy():
    model, optimizer, batch = make_training_setup()
    history = train_on_batch(model, optimizer, batch, number_of_updates=3)
    assert history[0]["approximate_kl"] == pytest.approx(0, abs=1e-6)
    assert history[-1]["approximate_kl"] > 0


def test_model_is_left_in_training_mode():
    model, optimizer, batch = make_training_setup()
    model.eval()
    train_on_batch(model, optimizer, batch)
    assert model.training


def test_stops_early_when_divergence_exceeds_limit():
    model, optimizer, batch = make_training_setup()
    history = train_on_batch(model, optimizer, batch, number_of_updates=5,
                             max_approximate_kl=1e-9)
    assert len(history) == 2                        # 第 0 次通过并更新，第 1 次超限
    assert history[-1]["approximate_kl"] > 1e-9

    reference_model, reference_optimizer, reference_batch = make_training_setup()
    train_on_batch(reference_model, reference_optimizer, reference_batch, number_of_updates=1)
    for parameter, reference_parameter in zip(model.parameters(),
                                              reference_model.parameters()):
        assert torch.equal(parameter, reference_parameter)   # 恰好只更新了一次
