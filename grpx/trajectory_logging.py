"""回答长度汇总，以及训练时抽样轨迹落盘。"""
from __future__ import annotations

import json
import pathlib
import statistics


def answer_token_count(trajectory):
    return sum(len(step.action_token_ids) for step in trajectory.steps)


def summarize_answer_lengths(trajectories, prefix=""):
    lengths = [answer_token_count(trajectory) for trajectory in trajectories]
    metrics = {}
    if lengths:
        metrics[f"{prefix}mean_answer_tokens"] = statistics.fmean(lengths)
    if not trajectories or any(trajectory.is_successful is None for trajectory in trajectories):
        return metrics
    correct = [answer_token_count(trajectory) for trajectory in trajectories if trajectory.is_successful]
    incorrect = [answer_token_count(trajectory) for trajectory in trajectories if not trajectory.is_successful]
    metrics[f"{prefix}answer_tokens_correct_count"] = len(correct)
    metrics[f"{prefix}answer_tokens_incorrect_count"] = len(incorrect)
    if correct:
        metrics[f"{prefix}mean_answer_tokens_correct"] = statistics.fmean(correct)
    if incorrect:
        metrics[f"{prefix}mean_answer_tokens_incorrect"] = statistics.fmean(incorrect)
    return metrics


def _decode(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def trajectory_sample_record(trajectory, tokenizer):
    prompt_token_ids = trajectory.steps[0].prompt_token_ids if trajectory.steps else []
    completion_token_ids = [
        token_id for step in trajectory.steps for token_id in step.action_token_ids
    ]
    return {
        "group_id": trajectory.group_id,
        "is_successful": trajectory.is_successful,
        "total_reward": trajectory.total_reward,
        "termination_reason": trajectory.termination_reason,
        "answer_token_count": answer_token_count(trajectory),
        "prompt": _decode(tokenizer, prompt_token_ids),
        "completion": _decode(tokenizer, completion_token_ids),
    }


def select_trajectory_samples(trajectories, tokenizer, per_class=2):
    """每类取前 per_class 条；没有成功标志时取前 2 * per_class 条。

    不使用训练随机源，避免抽样落盘改变可复现的采样序列。
    """
    if not trajectories:
        return []
    if any(trajectory.is_successful is None for trajectory in trajectories):
        chosen = list(trajectories)[: 2 * per_class]
    else:
        correct = [trajectory for trajectory in trajectories if trajectory.is_successful]
        incorrect = [trajectory for trajectory in trajectories if not trajectory.is_successful]
        chosen = correct[:per_class] + incorrect[:per_class]
    return [trajectory_sample_record(trajectory, tokenizer) for trajectory in chosen]


def append_trajectory_samples(path, iteration, samples):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            {"iteration": iteration, "samples": samples}, ensure_ascii=False) + "\n")


def truncate_trajectory_samples(path, start_iteration):
    """丢掉检查点之后写入的抽样，与 metrics 截断对齐。"""
    path = pathlib.Path(path)
    if not path.exists():
        return 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = [line for line in lines if json.loads(line)["iteration"] < start_iteration]
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    return len(lines) - len(kept)
