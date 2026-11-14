import json
from types import SimpleNamespace

import pytest

from grpx.evaluation import evaluate
from grpx.loop import summarize_rollout
from grpx.trajectory_logging import (
    append_trajectory_samples,
    select_trajectory_samples,
    summarize_answer_lengths,
    truncate_trajectory_samples,
)
from grpx.types import Step, Trajectory


def make_trajectory(action_length, is_successful, reward=0.0, group_id="g",
                    termination_reason="done"):
    return Trajectory(
        group_id, 0, 1.0,
        steps=[Step([1, 2], list(range(10, 10 + action_length)), reward)],
        termination_reason=termination_reason,
        is_successful=is_successful,
    )


def test_summarize_answer_lengths_splits_correct_and_incorrect():
    trajectories = [
        make_trajectory(10, True, reward=1.0),
        make_trajectory(30, False, reward=0.0),
        make_trajectory(20, True, reward=1.0),
    ]
    metrics = summarize_answer_lengths(trajectories)
    assert metrics["mean_answer_tokens"] == pytest.approx(20.0)
    assert metrics["mean_answer_tokens_correct"] == pytest.approx(15.0)
    assert metrics["mean_answer_tokens_incorrect"] == pytest.approx(30.0)
    assert metrics["answer_tokens_correct_count"] == 2
    assert metrics["answer_tokens_incorrect_count"] == 1


def test_summarize_answer_lengths_omits_split_without_success_flags():
    trajectories = [make_trajectory(8, None), make_trajectory(12, None)]
    metrics = summarize_answer_lengths(trajectories)
    assert metrics["mean_answer_tokens"] == pytest.approx(10.0)
    assert "mean_answer_tokens_correct" not in metrics


def test_summarize_rollout_includes_answer_lengths():
    trajectories = [make_trajectory(4, True, 1.0), make_trajectory(12, False, 0.0)]
    summary = summarize_rollout(trajectories, kept=trajectories)
    assert summary["mean_answer_tokens_correct"] == pytest.approx(4.0)
    assert summary["mean_answer_tokens_incorrect"] == pytest.approx(12.0)


def test_evaluate_includes_prefixed_answer_lengths():
    class Backend:
        temperature = 1.0

        def generate_batch(self, requests, greedy=False):
            assert greedy
            # 成功题生成 2 token，失败题生成 6 token
            outputs = []
            for request in requests:
                prompt_token_ids, n = request[0], request[1]
                prompt = "".join(chr(t) for t in prompt_token_ids)
                length = 2 if prompt == "ok" else 6
                outputs.append([([3] * length, False)] * n)
            return outputs

    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return messages[0]["content"]

        def __call__(self, text, add_special_tokens):
            return SimpleNamespace(input_ids=[ord(c) for c in text])

        def decode(self, token_ids, skip_special_tokens):
            return "x" * len(token_ids)

    class Environment:
        def reset(self, sample, seed):
            self.sample = sample
            return [{"role": "user", "content": sample["prompt"]}]

        def step(self, action):
            return [], 1.0 if self.sample["ok"] else 0.0, True

        def is_successful(self):
            return self.sample["ok"]

        def state_key(self):
            return ""

    metrics = evaluate(
        Backend(), Tokenizer(), Environment,
        [{"prompt": "ok", "ok": True}, {"prompt": "bad", "ok": False}],
    )
    assert metrics["evaluation_mean_answer_tokens_correct"] == pytest.approx(2.0)
    assert metrics["evaluation_mean_answer_tokens_incorrect"] == pytest.approx(6.0)


def test_select_trajectory_samples_prefers_both_classes():
    class Tokenizer:
        def decode(self, token_ids, skip_special_tokens):
            return f"t{len(token_ids)}"

    trajectories = (
        [make_trajectory(5, True, 1.0, group_id=f"c{i}") for i in range(3)]
        + [make_trajectory(9, False, 0.0, group_id=f"w{i}") for i in range(3)]
    )
    samples = select_trajectory_samples(trajectories, Tokenizer(), per_class=1)
    assert len(samples) == 2
    assert {sample["is_successful"] for sample in samples} == {True, False}
    assert samples[0]["group_id"] == "c0" and samples[1]["group_id"] == "w0"
    assert all("completion" in sample and "answer_token_count" in sample for sample in samples)


def test_append_trajectory_samples_writes_jsonl(tmp_path):
    path = tmp_path / "trajectory_samples.jsonl"
    append_trajectory_samples(path, iteration=3, samples=[{"group_id": "g", "is_successful": True}])
    append_trajectory_samples(path, iteration=4, samples=[{"group_id": "h", "is_successful": False}])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["iteration"] for row in rows] == [3, 4]
    assert rows[0]["samples"][0]["group_id"] == "g"


def test_truncate_trajectory_samples_keeps_before_checkpoint(tmp_path):
    path = tmp_path / "trajectory_samples.jsonl"
    for iteration in (0, 1, 2, 3):
        append_trajectory_samples(path, iteration, [{"iteration_marker": iteration}])
    assert truncate_trajectory_samples(path, start_iteration=2) == 2
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["iteration"] for row in rows] == [0, 1]
