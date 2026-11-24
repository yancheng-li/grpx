"""位移测量间隔：可配置、可跳过，且不得改变训练结果。"""
import json
from types import SimpleNamespace

import pytest
import torch

from grpx.config import TrainingConfig
from grpx.environment import SingleTurnEnvironment
from grpx.loop import measure_policy_shift, should_measure_shift, train
from grpx.types import Step, Trajectory


def test_should_measure_shift_matches_evaluation_style():
    assert should_measure_shift(0, 1) is True
    assert should_measure_shift(3, 1) is True
    assert should_measure_shift(0, 5) is True
    assert should_measure_shift(5, 5) is True
    assert should_measure_shift(1, 5) is False
    assert should_measure_shift(4, 5) is False
    assert should_measure_shift(0, 0) is False
    assert should_measure_shift(7, 0) is False


def test_shift_measurement_interval_allows_zero_rejects_negative():
    TrainingConfig(shift_measurement_interval=0)
    TrainingConfig(shift_measurement_interval=5)
    with pytest.raises(ValueError, match="shift_measurement_interval"):
        TrainingConfig(shift_measurement_interval=-1)
    with pytest.raises(ValueError, match="shift_measurement_interval"):
        TrainingConfig(shift_measurement_interval=True)


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocabulary_size=16, hidden_size=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, vocabulary_size)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.output_layer(self.embedding(input_ids)))

    def save_pretrained(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "weights.pt")


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return messages[0]["content"]

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=[ord(character) % 15 + 1 for character in text[:8]] or [1])

    def decode(self, token_ids, skip_special_tokens):
        return str(token_ids[0] % 3) if token_ids else "0"


class DeterministicBackend:
    """回答 token 由 torch 随机数决定，组内才能有回报差异。"""
    temperature = 1.0
    end_token_ids = {0}
    max_new_tokens = 2

    def synchronize_policy(self, model):
        pass

    def generate_batch(self, requests, greedy=False):
        return [[([int(torch.randint(1, 10, (1,)).item())], False)
                 for _ in range(request[1])]
                for request in requests]


def reward_from_action(prompt, completion, sample):
    return float(int(completion) % 3) / 2


SHIFT_FIELDS = (
    "shift_clip_fraction",
    "shift_median_absolute_log_ratio",
    "shift_percentile_95_absolute_log_ratio",
)


def run_training(tmp_path, name, iterations, interval, seed=0, measure_unreplayed_shift=False):
    torch.manual_seed(seed)
    model = TinyLanguageModel()
    config = TrainingConfig(
        iterations=iterations, groups_per_iteration=2, group_size=2,
        max_rows_per_minibatch=4, max_tokens_per_minibatch=10**6,
        warmup_iterations=1, checkpoint_interval=10,
        evaluation_interval=10, max_steps=1, seed=seed,
        shift_measurement_interval=interval,
        measure_unreplayed_shift=measure_unreplayed_shift,
        output_root=str(tmp_path), run_name=name,
        learning_rate=1e-3,
    )
    dataset = [{"prompt": f"q{index}", "answer": index} for index in range(20)]
    directory = train(
        model, FakeTokenizer(), DeterministicBackend(), dataset,
        lambda: SingleTurnEnvironment(reward_from_action), config)
    records = [json.loads(line)
               for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
               if "loss" in json.loads(line) or "iteration" in json.loads(line)]
    return model, records


def test_interval_five_only_writes_shift_on_iteration_zero(tmp_path):
    _model, records = run_training(tmp_path, "shift-every-5", iterations=5, interval=5)
    training = [record for record in records if "loss" in record]
    assert [record["iteration"] for record in training] == [0, 1, 2, 3, 4]
    for record in training:
        has_shift = any(name in record for name in SHIFT_FIELDS)
        if record["iteration"] == 0:
            assert has_shift
            assert record.get("shift_seconds", 0) >= 0
            assert all(name in record for name in SHIFT_FIELDS)
        else:
            assert not has_shift
            assert record.get("shift_seconds", 0) == 0


def test_shift_measurement_does_not_change_trainable_params_or_loss(tmp_path):
    """测不测位移，训练结果逐位相同。"""
    model_every, records_every = run_training(
        tmp_path, "shift-every", iterations=3, interval=1, seed=0)
    model_never, records_never = run_training(
        tmp_path, "shift-never", iterations=3, interval=0, seed=0)

    for left, right in zip(model_every.parameters(), model_never.parameters()):
        assert torch.equal(left, right)

    losses_every = [record["loss"] for record in records_every if "loss" in record]
    losses_never = [record["loss"] for record in records_never if "loss" in record]
    assert losses_every == losses_never
    assert all(any(name in record for name in SHIFT_FIELDS)
               for record in records_every if "loss" in record)
    assert all(not any(name in record for name in SHIFT_FIELDS)
               for record in records_never if "loss" in record)


def test_unreplayed_shift_is_logged_only_when_enabled(tmp_path):
    _model, records = run_training(
        tmp_path, "shift-both", iterations=2, interval=1, measure_unreplayed_shift=True)
    training = [record for record in records if "loss" in record]
    assert training
    for record in training:
        assert "shift_unreplayed_clip_fraction" in record
        assert "shift_unreplayed_percentile_95_absolute_log_ratio" in record
        assert record["shift_unreplayed_clip_fraction"] == record["shift_clip_fraction"]
    _model, plain = run_training(tmp_path, "shift-one", iterations=1, interval=1)
    logged = [record for record in plain if "loss" in record]
    assert logged and "shift_unreplayed_clip_fraction" not in logged[0]


def test_measure_unreplayed_shift_must_be_boolean():
    with pytest.raises(ValueError, match="measure_unreplayed_shift"):
        TrainingConfig(measure_unreplayed_shift="yes")


def test_measure_policy_shift_leaves_params_and_rng_unchanged():
    from grpx.trainer import action_log_probabilities, build_batch

    torch.manual_seed(0)
    model = TinyLanguageModel()
    trajectory = Trajectory("g", 0, 1.0, [Step([1, 2], [3, 4], advantage=1.0)])
    batch = build_batch([trajectory], padding_token_id=0, device="cpu")
    with torch.no_grad():
        old, _mask = action_log_probabilities(model, batch)
    config = TrainingConfig(clip_range=0.2)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    rng_before = torch.random.get_rng_state()
    measure_policy_shift(model, [batch], [old], config)
    rng_after = torch.random.get_rng_state()
    assert torch.equal(rng_before, rng_after)
    for left, right in zip(before, model.parameters()):
        assert torch.equal(left, right)
