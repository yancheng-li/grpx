import random
from types import SimpleNamespace

import pytest
import torch

from grpx.loop import TrainingConfig, accumulate_gradients, row_length, split_into_row_chunks
from grpx.trainer import action_log_probabilities, build_batch, grpo_loss
from grpx.types import Step, Trajectory


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocabulary_size=16, hidden_size=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.output_layer(self.embedding(input_ids)))


def make_random_trajectories(seed=0, count=6, max_steps=4):
    random_generator = random.Random(seed)
    trajectories = []
    for trajectory_index in range(count):
        advantage = random_generator.uniform(-2, 2)
        steps = [
            Step(prompt_token_ids=[random_generator.randrange(1, 16) for _ in range(random_generator.randint(1, 12))],
                 action_token_ids=[random_generator.randrange(1, 16) for _ in range(random_generator.randint(1, 4))],
                 advantage=advantage)
            for _ in range(random_generator.randint(1, max_steps))
        ]
        trajectories.append(Trajectory(f"t{trajectory_index}", 0, sampling_temperature=random_generator.choice([0.7, 1.0]),
                                       steps=steps))
    return trajectories


def step_identity(row):
    return id(row.steps[0])


@pytest.mark.parametrize("max_rows, max_tokens", [(1, 10**6), (3, 10**6), (64, 30), (4, 40), (64, 10**6)])
def test_every_row_appears_in_exactly_one_chunk(max_rows, max_tokens):
    trajectories = make_random_trajectories()
    chunks = split_into_row_chunks(trajectories, max_rows, max_tokens)
    chunked = sorted(step_identity(row) for chunk in chunks for row in chunk)
    # 超预算的行会被丢掉，不参与组批
    expected = sorted(
        id(step) for trajectory in trajectories for step in trajectory.steps
        if len(step.prompt_token_ids) + len(step.action_token_ids) <= max_tokens
    )
    assert chunked == expected


@pytest.mark.parametrize("max_rows, max_tokens", [(1, 10**6), (3, 10**6), (64, 30), (4, 40)])
def test_no_chunk_exceeds_limits(max_rows, max_tokens):
    from grpx.loop import segment_count
    for chunk in split_into_row_chunks(make_random_trajectories(), max_rows, max_tokens):
        assert sum(segment_count(row) for row in chunk) <= max_rows
        assert len(chunk) * max(row_length(row) for row in chunk) <= max_tokens


def test_merged_chunks_budget_by_segments_not_rows():
    from grpx.loop import segment_count
    from grpx.types import Step, Trajectory
    # 两条可合并的 6 步轨迹：按段预算 max_rows=8 只能装下一条（6 段），不能再塞第二条
    def make_append_only(group_id):
        steps = []
        prompt = [1, 2]
        for index in range(6):
            action = [10 + index]
            steps.append(Step(list(prompt), action, advantage=1.0))
            prompt = prompt + action
        return Trajectory(group_id, 0, 1.0, steps)

    chunks = split_into_row_chunks(
        [make_append_only("a"), make_append_only("b")],
        max_rows=8, max_tokens=10**6, merge_append_only=True,
    )
    assert [len(chunk) for chunk in chunks] == [1, 1]
    assert all(sum(segment_count(row) for row in chunk) <= 8 for chunk in chunks)


def compute_gradients(model, old_model, chunks, config):
    batches = [build_batch(chunk, padding_token_id=0, device="cpu", row_weighting=config.row_weighting)
               for chunk in chunks]
    with torch.no_grad():
        old_log_probabilities = [action_log_probabilities(old_model, batch)[0] for batch in batches]
    model.zero_grad()
    metrics = accumulate_gradients(model, batches, old_log_probabilities, config)
    return [parameter.grad.clone() for parameter in model.parameters()], metrics


@pytest.mark.parametrize("row_weighting", ["per_step", "per_trajectory"])
@pytest.mark.parametrize("length_normalization", ["per_sequence", "constant"])
@pytest.mark.parametrize("max_rows, max_tokens", [(1, 10**6), (3, 10**6), (64, 30)])
def test_split_accumulation_matches_single_batch_gradients(length_normalization, max_rows, max_tokens, row_weighting):
    torch.manual_seed(0)
    old_model = TinyLanguageModel()
    model = TinyLanguageModel()
    model.load_state_dict(old_model.state_dict())
    with torch.no_grad():    # 让新旧策略不同，ratio 偏离 1，裁剪分支也参与进来
        for parameter in model.parameters():
            parameter.add_(0.3 * torch.randn_like(parameter))
    config = TrainingConfig(length_normalization=length_normalization, constant_length=8,
                            row_weighting=row_weighting)

    trajectories = make_random_trajectories()
    all_rows = [row for chunk in split_into_row_chunks(trajectories, 10**6, 10**9) for row in chunk]
    whole_gradients, whole_metrics = compute_gradients(model, old_model, [all_rows], config)
    split_chunks = split_into_row_chunks(trajectories, max_rows, max_tokens)
    assert len(split_chunks) > 1
    split_gradients, split_metrics = compute_gradients(model, old_model, split_chunks, config)

    assert any(gradient.abs().sum() > 0 for gradient in whole_gradients)
    for whole_gradient, split_gradient in zip(whole_gradients, split_gradients):
        assert split_gradient.flatten().tolist() == pytest.approx(whole_gradient.flatten().tolist(), abs=1e-6)
    assert split_metrics["loss"] == pytest.approx(whole_metrics["loss"], abs=1e-6)


def test_max_ratio_deviation_is_max_across_batches_not_average():
    """max_* 指标取各批最大值：十个小批里一个偏差 0.005、其余为 0，平均会稀释到阈值以下"""
    torch.manual_seed(0)
    old_model = TinyLanguageModel()
    model = TinyLanguageModel()
    model.load_state_dict(old_model.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5 * torch.randn_like(parameter))
    config = TrainingConfig(length_normalization="per_sequence", max_rows_per_minibatch=1)
    trajectories = make_random_trajectories(count=8, max_steps=1)
    chunks = split_into_row_chunks(trajectories, max_rows=1, max_tokens=10**6)
    assert len(chunks) > 1
    batches = [build_batch(chunk, padding_token_id=0, device="cpu") for chunk in chunks]
    with torch.no_grad():
        old_log_probabilities = [action_log_probabilities(old_model, batch)[0] for batch in batches]
    model.zero_grad()
    metrics = accumulate_gradients(model, batches, old_log_probabilities, config)

    per_batch = []
    for batch, old_log_probability in zip(batches, old_log_probabilities):
        new_log_probability, mask = action_log_probabilities(model, batch)
        _, batch_metrics = grpo_loss(
            new_log_probability, old_log_probability, mask, batch["advantages"],
            row_weights=batch["row_weights"], length_normalization="per_sequence")
        per_batch.append(batch_metrics["max_ratio_deviation"])
    assert metrics["max_ratio_deviation"] == pytest.approx(max(per_batch))
    # 若误用加权平均，单批大偏差会被稀释；这里至少有一批偏差明显不同
    assert max(per_batch) - min(per_batch) > 1e-4
