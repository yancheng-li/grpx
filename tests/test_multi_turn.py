import re
from types import SimpleNamespace

import pytest
import torch

from examples.guess_number import GuessNumberEnvironment
from grpx.advantage import episode_advantages
from grpx.loop import (TrainingConfig, collect_trajectories, split_into_row_chunks,
                       summarize_rollout, update_policy)
from grpx.rollout import rollout_group, rollout_groups
from grpx.trainer import build_batch
from grpx.types import Step, Trajectory

INITIAL_PROMPT = ("我们玩猜数字。我想了一个 1 到 50 之间的整数，你来猜。\n"
                  "规则：你每次只回复一个阿拉伯数字，不要回复任何其他文字。"
                  "我会告诉你大了还是小了。\n现在请给出第一个猜测。")

def encode(text):
    return [ord(character) for character in text]

def decode(token_ids):
    return "".join(chr(token_id) for token_id in token_ids)

class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "|".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=encode(text))

    def decode(self, token_ids, skip_special_tokens):
        return decode(token_ids)

class ScriptedBackend:
    """按请求顺序返回预先写好的回答；calls 记录每个请求的 (提示词文本, 样本数)，
    batch_sizes 记录每次 generate_batch 调用里有几个请求"""
    temperature = 1.0

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.batch_sizes = []

    def generate_batch(self, requests, greedy=False):
        self.batch_sizes.append(len(requests))
        results = []
        for request in requests:
            prompt_token_ids, number_of_samples = request[0], request[1]
            self.calls.append((decode(prompt_token_ids), number_of_samples))
            answers = self.script.pop(0)
            assert len(answers) == number_of_samples
            results.append([(encode(answer), False) for answer in answers])
        return results


class BinarySearchBackend:
    """按对话里的反馈二分猜数；回答只由提示词决定，用来对比不同的分组与合并方式"""
    temperature = 1.0

    def generate_batch(self, requests, greedy=False):
        results = []
        for request in requests:
            prompt_token_ids, number_of_samples = request[0], request[1]
            lowest, highest = 1, 50
            for guess, direction in re.findall(r"(\d+) (小了|大了)", decode(prompt_token_ids)):
                if direction == "小了":
                    lowest = int(guess) + 1
                else:
                    highest = int(guess) - 1
            results.append([(encode(str((lowest + highest) // 2)), False)] * number_of_samples)
        return results

def sample_for(target):
    return {"target": target, "lowest": 1, "highest": 50}

def test_groups_with_identical_first_prompt_merge_into_one_request():
    backend = ScriptedBackend([["25", "25", "40", "40"], ["30", "30"], ["10", "10"]])
    trajectories = rollout_groups(backend, FakeTokenizer(), GuessNumberEnvironment,
                                  [("a", sample_for(30), 0), ("b", sample_for(12), 1)],
                                  group_size=2, max_steps=2)
    assert backend.calls[0] == (INITIAL_PROMPT, 2 + 2)
    assert backend.batch_sizes == [1, 2]
    assert [trajectory.is_successful for trajectory in trajectories] == [True, True, False, False]
    assert [trajectory.group_id for trajectory in trajectories] == ["a", "a", "b", "b"]
    assert [trajectory.steps[0].state_key for trajectory in trajectories] == ["1-50"] * 4

def test_rollout_groups_matches_one_group_at_a_time():
    specifications = [(f"g{target}", sample_for(target), seed)
                      for seed, target in enumerate([25, 7, 30, 30, 44, 1])]
    together = rollout_groups(BinarySearchBackend(), FakeTokenizer(), GuessNumberEnvironment,
                              specifications, group_size=3, policy_version=2, max_steps=6)
    one_at_a_time = [
        trajectory
        for group_id, sample, seed in specifications
        for trajectory in rollout_group(BinarySearchBackend(), FakeTokenizer(), GuessNumberEnvironment,
                                        sample, group_id, group_size=3, seed=seed,
                                        policy_version=2, max_steps=6)
    ]
    assert together == one_at_a_time
    assert len({len(trajectory.steps) for trajectory in together}) > 1    # 各组确实走了不同的步数

def run(script, target=30, group_size=1, max_steps=6):
    backend = ScriptedBackend(script)
    sample = {"target": target, "lowest": 1, "highest": 50}
    trajectories = rollout_group(backend, FakeTokenizer(), GuessNumberEnvironment, sample, "g0",
                                 group_size=group_size, seed=0, max_steps=max_steps)
    return backend, trajectories

def test_second_prompt_contains_first_answer_and_feedback():
    backend, trajectories = run([["25"], ["30"]])
    assert [prompt for prompt, _ in backend.calls] == [
        INITIAL_PROMPT,
        f"{INITIAL_PROMPT}|25|25 小了，再猜。",
    ]
    trajectory = trajectories[0]
    assert decode(trajectory.steps[1].prompt_token_ids) == backend.calls[1][0]
    assert (trajectory.termination_reason, trajectory.total_reward, len(trajectory.steps)) == ("done", 1.0, 2)

def test_state_key_is_recorded_before_the_action():
    _, trajectories = run([["25"], ["40"], ["30"]])
    assert [step.state_key for step in trajectories[0].steps] == ["1-50", "26-50", "26-39"]

def test_unsolved_after_max_steps_is_step_limit():
    _, trajectories = run([["1"], ["2"], ["3"]], max_steps=3)
    trajectory = trajectories[0]
    assert (trajectory.termination_reason, len(trajectory.steps)) == ("step_limit", 3)
    assert trajectory.is_successful is False
    assert trajectory.total_reward == pytest.approx(0.5 * 3 / 50)    # 每猜一次把下界抬高 1

def test_different_first_guesses_split_into_separate_requests():
    backend, trajectories = run([["10", "40"], ["30"], ["30"]], group_size=2)
    assert backend.calls == [
        (INITIAL_PROMPT, 2),
        (f"{INITIAL_PROMPT}|10|10 小了，再猜。", 1),
        (f"{INITIAL_PROMPT}|40|40 大了，再猜。", 1),
    ]
    assert backend.batch_sizes == [1, 2]    # 每一步只调用一次后端
    assert [trajectory.steps[1].state_key for trajectory in trajectories] == ["11-50", "1-39"]
    assert trajectories[0].steps[1].prompt_token_ids != trajectories[1].steps[1].prompt_token_ids

def test_identical_first_guesses_share_one_generation():
    backend, trajectories = run([["25", "25"], ["30", "28"], ["30"]], group_size=2)
    assert backend.calls[:2] == [
        (INITIAL_PROMPT, 2),
        (f"{INITIAL_PROMPT}|25|25 小了，再猜。", 2),
    ]
    assert [trajectory.termination_reason for trajectory in trajectories] == ["done", "done"]
    assert [len(trajectory.steps) for trajectory in trajectories] == [2, 3]

def test_answer_without_number_continues_without_changing_interval():
    backend, trajectories = run([["不知道"], ["30"]])
    trajectory = trajectories[0]
    assert backend.calls[1][0] == f"{INITIAL_PROMPT}|不知道|请只回复一个阿拉伯数字。"
    assert [step.state_key for step in trajectory.steps] == ["1-50", "1-50"]
    assert [step.reward for step in trajectory.steps] == [0.0, 1.0]
    assert trajectory.termination_reason == "done"

def test_build_batch_makes_one_row_per_step_with_shared_advantage():
    _, trajectories = run([["25", "25"], ["40", "1"], ["30"], ["2"], ["3"]], group_size=2, max_steps=4)
    solved, unsolved = trajectories
    assert (len(solved.steps), solved.is_successful, unsolved.is_successful) == (3, True, False)
    assert solved.total_reward == pytest.approx(1.0) and unsolved.total_reward < 0.5
    episode_advantages(trajectories)

    from grpx.trainer import prepare_update_units
    units = prepare_update_units([solved], merge_append_only=False)
    batch = build_batch(units, padding_token_id=0, device="cpu")
    assert batch["input_token_ids"].shape[0] == 3
    advantages = batch["advantages"].tolist()
    assert advantages[0] > 0 and advantages == [advantages[0]] * 3
    targets, shifted_mask = batch["input_token_ids"][:, 1:], batch["action_mask"][:, 1:]
    for row_index, step in enumerate(solved.steps):
        assert targets[row_index][shifted_mask[row_index]].tolist() == step.action_token_ids

def make_step_trajectory(group_id, lengths):
    steps = [Step(prompt_token_ids=[1] * length, action_token_ids=[2], advantage=1.0) for length in lengths]
    return Trajectory(group_id, 0, sampling_temperature=0.7, steps=steps)

def test_row_chunks_split_steps_and_respect_budgets():
    trajectories = [make_step_trajectory("a", [9, 19, 29]), make_step_trajectory("b", [9, 39])]
    chunks = split_into_row_chunks(trajectories, max_rows=8, max_tokens=60)
    row_lengths = [[len(row.steps[0].prompt_token_ids) + 1 for row in chunk] for chunk in chunks]
    assert sorted(length for chunk in row_lengths for length in chunk) == [10, 10, 20, 30, 40]
    assert all(len(chunk) * max(chunk) <= 60 or len(chunk) == 1 for chunk in row_lengths)
    assert all(len(row.steps) == 1 and row.sampling_temperature == 0.7 for chunk in chunks for row in chunk)
    assert [len(chunk) for chunk in split_into_row_chunks(trajectories, max_rows=2, max_tokens=10**6)] == [2, 2, 1]

def test_oversized_row_is_dropped_with_one_summary_warning():
    # 1 超长 + 20 短 → 丢弃约 5% 以内，走普通警告（不升级）
    trajectories = (
        [make_step_trajectory("long", [99])]
        + [make_step_trajectory(f"s{i}", [4]) for i in range(20)]
    )
    with pytest.warns(UserWarning, match=r"本轮有 1 行超过 max_tokens_per_minibatch=50（最长 100"):
        chunks = split_into_row_chunks(trajectories, max_rows=8, max_tokens=50)
    assert sum(len(chunk) for chunk in chunks) == 20
    assert all(len(row.steps[0].prompt_token_ids) == 4 for chunk in chunks for row in chunk)


def test_oversized_row_escalates_when_drop_fraction_is_high():
    # 2/3 > 20% → 「严重」
    with pytest.warns(UserWarning, match=r"严重：本轮有 2 行"):
        chunks = split_into_row_chunks(
            [make_step_trajectory("a", [80]),
             make_step_trajectory("b", [90]),
             make_step_trajectory("c", [10])],
            max_rows=8, max_tokens=50)
    assert [len(chunk) for chunk in chunks] == [1]

    # 1/20 = 5% 边界不算升级；1/10 = 10% → 「注意」
    with pytest.warns(UserWarning, match=r"注意：本轮有 1 行"):
        split_into_row_chunks(
            [make_step_trajectory("a", [80])]
            + [make_step_trajectory(f"s{i}", [10]) for i in range(9)],
            max_rows=8, max_tokens=50)


def test_multiple_oversized_rows_warn_once_and_update_skips_cleanly():
    import warnings

    trajectories = [
        make_step_trajectory("a", [80]),
        make_step_trajectory("b", [90]),
        make_step_trajectory("c", [10]),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        chunks = split_into_row_chunks(trajectories, max_rows=8, max_tokens=50)
    oversized_warnings = [
        item for item in caught if "超过 max_tokens_per_minibatch" in str(item.message)
    ]
    assert len(oversized_warnings) == 1
    assert "本轮有 2 行" in str(oversized_warnings[0].message)
    assert "严重" in str(oversized_warnings[0].message)
    assert [len(chunk) for chunk in chunks] == [1]

    # 全部超长时 update_policy 不崩溃，返回空 history
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
            self.device = torch.device("cpu")

        def forward(self, input_ids, attention_mask=None):
            hidden = self.layer(torch.zeros(*input_ids.shape, 2))
            return SimpleNamespace(logits=hidden)

    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = TrainingConfig(
        iterations=1, groups_per_iteration=2, group_size=2,
        max_rows_per_minibatch=8, max_tokens_per_minibatch=50,
    )
    with pytest.warns(UserWarning, match="本轮有 2 行超过"):
        history, batches, *_rest = update_policy(
            model, optimizer,
            [make_step_trajectory("a", [80]), make_step_trajectory("b", [90])],
            0, config,
        )
    assert history == [] and batches == []

def test_collect_trajectories_passes_max_steps_and_summary_counts_steps():
    backend = ScriptedBackend([["1", "1"], ["2", "2"], ["3", "3"]])
    samples = [{"target": 30, "lowest": 1, "highest": 50}]
    config = TrainingConfig(group_size=2, max_steps=3)
    trajectories = collect_trajectories(backend, FakeTokenizer(), GuessNumberEnvironment, samples, config, iteration=0)
    assert all(len(trajectory.steps) == 3 for trajectory in trajectories)
    summary = summarize_rollout(trajectories, kept=[])
    assert (summary["mean_step_count"], summary["step_limit_fraction"]) == (3.0, 1.0)
