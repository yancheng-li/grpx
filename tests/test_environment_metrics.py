from types import SimpleNamespace

from grpx.evaluation import evaluate
from grpx.loop import summarize_rollout
from grpx.rollout import rollout_group
from grpx.types import Step, Trajectory, mean_environment_metrics


def test_mean_environment_metrics_averages_present_keys():
    trajectories = [
        Trajectory("a", 0, 1.0, environment_metrics={"x": 1.0, "y": 0.0}),
        Trajectory("a", 0, 1.0, environment_metrics={"x": 0.5}),
    ]
    assert mean_environment_metrics(trajectories) == {
        "environment_x": 0.75, "environment_y": 0.0}
    assert mean_environment_metrics(trajectories, prefix="evaluation_environment_") == {
        "evaluation_environment_x": 0.75, "evaluation_environment_y": 0.0}


def test_summarize_and_evaluate_include_environment_metrics():
    class CountingEnvironment:
        def reset(self, sample, seed):
            self.value = sample["value"]
            return [{"role": "user", "content": "go"}]

        def step(self, action):
            return [], self.value, True

        def state_key(self):
            return ""

        def metrics(self):
            return {"score": self.value}

    class Backend:
        temperature = 1.0

        def generate_batch(self, requests, greedy=False):
            return [[([ord("x")], False)] * request[1] for request in requests]

    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return "go"

        def __call__(self, text, add_special_tokens):
            return SimpleNamespace(input_ids=[1])

        def decode(self, token_ids, skip_special_tokens):
            return "x"

    trajectories = rollout_group(Backend(), Tokenizer(), CountingEnvironment,
                                 {"value": 0.5}, "g", group_size=2, seed=0)
    assert [trajectory.environment_metrics for trajectory in trajectories] == [{"score": 0.5}, {"score": 0.5}]
    assert summarize_rollout(trajectories, trajectories)["environment_score"] == 0.5

    metrics = evaluate(Backend(), Tokenizer(), CountingEnvironment,
                       [{"value": 1.0}, {"value": 0.0}], max_steps=1)
    assert metrics["evaluation_environment_score"] == 0.5
