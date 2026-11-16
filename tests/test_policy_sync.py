import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from grpx.config import TrainingConfig
from grpx.environment import SingleTurnEnvironment
from grpx.loop import train


class TrackingBackend:
    temperature = 1.0
    end_token_ids = {0}
    max_new_tokens = 4

    def __init__(self):
        self.sync_events = []
        self.adapter_directory = None

    def synchronize_policy(self, model):
        self.sync_events.append(("sync", getattr(model, "mark", None), str(self.adapter_directory)))

    def generate_batch(self, requests, greedy=False):
        return [[([1], False) for _ in range(request[1])]
                for request in requests]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.mark = "model"

    @property
    def device(self):
        return self.weight.device

    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8)
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, directory):
        directory.mkdir(parents=True, exist_ok=True)


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return messages[0]["content"]

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=[1, 2, 3])

    def decode(self, token_ids, skip_special_tokens):
        return "1"


def reward_from_action(prompt, completion, sample):
    return 1.0 if sample["answer"] % 2 == 0 else 0.0


def test_train_syncs_before_loop_and_after_updates(tmp_path, monkeypatch):
    from grpx import loop as loop_module
    from grpx.types import Step, Trajectory

    backend = TrackingBackend()
    model = TinyModel()
    config = TrainingConfig(
        iterations=2, groups_per_iteration=2, group_size=2,
        max_rows_per_minibatch=4, max_tokens_per_minibatch=10**6,
        warmup_iterations=1, checkpoint_interval=10,
        evaluation_interval=10, max_steps=1, seed=0,
        output_root=str(tmp_path), run_name="sync-run",
    )
    dataset = [{"prompt": f"q{index}", "answer": index} for index in range(20)]

    def fake_collect(backend, tokenizer, environment_factory, samples, config, iteration):
        return [
            Trajectory(f"iteration-{iteration}-group-0", iteration, 1.0,
                       [Step([1], [1], reward=1.0)]),
            Trajectory(f"iteration-{iteration}-group-0", iteration, 1.0,
                       [Step([1], [2], reward=0.0)]),
        ]

    monkeypatch.setattr(loop_module, "collect_trajectories", fake_collect)
    monkeypatch.setattr(loop_module, "update_policy", lambda *args, **kwargs: (
        [{"loss": 0.0, "max_ratio_deviation": 0.0, "approximate_kl": 0.0}],
        [], [], {}, 0.0, None, {}, None))
    monkeypatch.setattr(loop_module, "measure_policy_shift", lambda *args, **kwargs: {
        "shift_clip_fraction": 0.0,
        "shift_median_absolute_log_ratio": 0.0,
        "shift_percentile_95_absolute_log_ratio": 0.0,
    })

    train(model, FakeTokenizer(), backend, dataset,
          lambda: SingleTurnEnvironment(reward_from_action), config)

    assert backend.adapter_directory is not None
    assert "vllm_adapter" in str(backend.adapter_directory)
    # 循环前 1 次 + 每轮有更新再各 1 次
    assert len(backend.sync_events) == 1 + 2
    assert all(event[0] == "sync" for event in backend.sync_events)
    metrics = [
        json.loads(line)
        for line in (tmp_path / "sync-run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    timed = [row for row in metrics if "sampling_seconds" in row]
    assert len(timed) == 2
    phase_keys = ("evaluation_seconds", "sampling_seconds", "metrics_seconds",
                  "update_seconds", "synchronize_seconds")
    for row in timed:
        assert all(key in row for key in phase_keys)
        assert sum(row[key] for key in phase_keys) == pytest.approx(
            row["iteration_seconds"], abs=1e-2)


def test_resume_syncs_after_checkpoint_load(tmp_path, monkeypatch):
    from grpx import loop as loop_module
    from grpx.types import Step, Trajectory

    backend = TrackingBackend()
    model = TinyModel()
    run_directory = tmp_path / "resume-run"
    run_directory.mkdir()
    (run_directory / "config.json").write_text("{}", encoding="utf-8")
    checkpoint = run_directory / "checkpoint-0001"
    checkpoint.mkdir()
    (checkpoint / "training_state.pt").write_bytes(b"")

    def fake_collect(backend, tokenizer, environment_factory, samples, config, iteration):
        return [
            Trajectory(f"iteration-{iteration}-group-0", iteration, 1.0,
                       [Step([1], [1], reward=1.0)]),
            Trajectory(f"iteration-{iteration}-group-0", iteration, 1.0,
                       [Step([1], [2], reward=0.0)]),
        ]

    monkeypatch.setattr(loop_module, "find_latest_checkpoint", lambda directory: checkpoint)
    monkeypatch.setattr(loop_module, "load_checkpoint", lambda *args, **kwargs: 1)
    monkeypatch.setattr(loop_module, "truncate_metrics", lambda *args, **kwargs: 0)
    monkeypatch.setattr(loop_module, "collect_trajectories", fake_collect)
    monkeypatch.setattr(loop_module, "update_policy", lambda *args, **kwargs: (
        [{"loss": 0.0, "max_ratio_deviation": 0.0, "approximate_kl": 0.0}],
        [], [], {}, 0.0, None, {}, None))
    monkeypatch.setattr(loop_module, "measure_policy_shift", lambda *args, **kwargs: {
        "shift_clip_fraction": 0.0,
        "shift_median_absolute_log_ratio": 0.0,
        "shift_percentile_95_absolute_log_ratio": 0.0,
    })

    config = TrainingConfig(
        iterations=2, groups_per_iteration=2, group_size=2,
        max_rows_per_minibatch=4, max_tokens_per_minibatch=10**6,
        warmup_iterations=1, checkpoint_interval=10,
        evaluation_interval=10, max_steps=1, seed=0,
        output_root=str(tmp_path), run_name="ignored",
    )
    dataset = [{"prompt": f"q{index}", "answer": index} for index in range(20)]
    train(model, FakeTokenizer(), backend, dataset,
          lambda: SingleTurnEnvironment(reward_from_action), config,
          resume_from=run_directory)

    # 恢复后循环前同步 1 次 + 第 2 轮更新后再同步 1 次
    assert len(backend.sync_events) == 2
