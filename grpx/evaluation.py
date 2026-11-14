from grpx.rollout import rollout_groups
from grpx.trajectory_logging import summarize_answer_lengths
from grpx.types import mean_environment_metrics, success_fraction


def evaluate(backend, tokenizer, environment_factory, evaluation_dataset, max_steps=1,
             truncated_reward=0.0):
    trajectories = rollout_groups(
        backend, tokenizer, environment_factory,
        [(f"evaluation-{sample_index}", sample, sample_index)
         for sample_index, sample in enumerate(evaluation_dataset)],
        group_size=1, max_steps=max_steps, greedy=True,
        truncated_reward=truncated_reward)
    count = len(trajectories)
    successes = success_fraction(trajectories)
    return {
        "evaluation_mean_reward":
            sum(trajectory.total_reward for trajectory in trajectories) / count,
        **({"evaluation_success_fraction": successes} if successes is not None else {}),
        "evaluation_length_limit_fraction":
            sum(trajectory.termination_reason == "length_limit" for trajectory in trajectories) / count,
        "evaluation_mean_step_count":
            sum(len(trajectory.steps) for trajectory in trajectories) / count,
        **summarize_answer_lengths(trajectories, prefix="evaluation_"),
        **mean_environment_metrics(trajectories, prefix="evaluation_environment_"),
    }
