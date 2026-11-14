from types import SimpleNamespace

from grpx.environment import SingleTurnEnvironment
from grpx.rollout import rollout_group

class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "|".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def decode(self, token_ids, skip_special_tokens):
        return "".join(chr(token_id) for token_id in token_ids)

class FakeBackend:
    def __init__(self, outputs, temperature=0.7):
        self.outputs, self.temperature = outputs, temperature
        self.calls = []

    def generate(self, prompt_token_ids, number_of_samples):
        self.calls.append((prompt_token_ids, number_of_samples))
        return self.outputs[:number_of_samples]

    def generate_batch(self, requests, greedy=False):
        return [self.generate(request[0], request[1]) for request in requests]

class NeverDoneEnv(SingleTurnEnvironment):
    def step(self, action):
        return [{"role": "user", "content": "继续"}], 0.0, False

def exact_match_reward(prompt, completion, sample):
    return 1.0 if completion == sample["answer"] else 0.0

def encode(text):
    return [ord(character) for character in text]

def run(outputs, group_size, environment_class=SingleTurnEnvironment, sample=None, **keyword_arguments):
    backend = FakeBackend(outputs)
    trajectories = rollout_group(
        backend, FakeTokenizer(), lambda: environment_class(exact_match_reward),
        sample or {"prompt": "1+1", "answer": "2"}, "g0", group_size=group_size, seed=0, **keyword_arguments,
    )
    return backend, trajectories

def test_single_turn_group():
    outputs = [(encode("2"), False), (encode("3"), False), (encode("2"), True)]
    backend, trajectories = run(outputs, group_size=3, policy_version=4)
    assert backend.calls == [(encode("1+1"), 3)]
    # 第三条被截断：文本虽匹配，框架覆盖回报为 0，且不算成功
    assert [trajectory.total_reward for trajectory in trajectories] == [1.0, 0.0, 0.0]
    assert [trajectory.termination_reason for trajectory in trajectories] == ["done", "done", "length_limit"]
    assert [trajectory.is_successful for trajectory in trajectories] == [None, None, False]
    for trajectory in trajectories:
        assert (trajectory.group_id, trajectory.policy_version, trajectory.sampling_temperature) == ("g0", 4, 0.7)
        assert len(trajectory.steps) == 1
        assert trajectory.steps[0].prompt_token_ids == encode("1+1")


def test_truncated_overrides_reward_and_success():
    def always_success(completion, sample):
        return True

    backend = FakeBackend([(encode("2"), True)])
    trajectories = rollout_group(
        backend, FakeTokenizer(),
        lambda: SingleTurnEnvironment(exact_match_reward, success_function=always_success),
        {"prompt": "1+1", "answer": "2"}, "g0", group_size=1, seed=0,
        truncated_reward=0.0,
    )
    assert trajectories[0].termination_reason == "length_limit"
    assert trajectories[0].total_reward == 0.0
    assert trajectories[0].is_successful is False


def test_truncated_reward_none_keeps_environment_outcome():
    def always_success(completion, sample):
        return True

    backend = FakeBackend([(encode("2"), True)])
    trajectories = rollout_group(
        backend, FakeTokenizer(),
        lambda: SingleTurnEnvironment(exact_match_reward, success_function=always_success),
        {"prompt": "1+1", "answer": "2"}, "g0", group_size=1, seed=0,
        truncated_reward=None,
    )
    assert trajectories[0].termination_reason == "length_limit"
    assert trajectories[0].total_reward == 1.0
    assert trajectories[0].is_successful is True


def test_custom_truncated_reward_value():
    backend = FakeBackend([(encode("2"), True)])
    trajectories = rollout_group(
        backend, FakeTokenizer(),
        lambda: SingleTurnEnvironment(exact_match_reward),
        {"prompt": "1+1", "answer": "2"}, "g0", group_size=1, seed=0,
        truncated_reward=-1.0,
    )
    assert trajectories[0].total_reward == -1.0
    assert trajectories[0].is_successful is False

def test_unfinished_after_max_steps_is_step_limit():
    _, trajectories = run([(encode("x"), False)], group_size=1, environment_class=NeverDoneEnv, max_steps=2)
    assert len(trajectories[0].steps) == 2
    assert trajectories[0].steps[1].prompt_token_ids == encode("1+1|x|继续")
    assert trajectories[0].termination_reason == "step_limit"

def test_length_limit_in_multi_turn_stops_early():
    _, trajectories = run([(encode("x"), True)], group_size=1, environment_class=NeverDoneEnv, max_steps=3)
    assert len(trajectories[0].steps) == 1
    assert trajectories[0].termination_reason == "length_limit"


def test_truncated_reward_replaces_only_truncated_step_not_episode():
    """截断覆盖的是被截断那一步的回报，不是整局回报；先前步保留原值。"""
    class StepRewardsEnv(NeverDoneEnv):
        def __init__(self, reward_function, success_function=None):
            super().__init__(reward_function, success_function)
            self._rewards = iter([0.3, -0.5])

        def step(self, action):
            reward = next(self._rewards)
            return [{"role": "user", "content": "继续"}], reward, False

    class SequentialBackend(FakeBackend):
        def generate(self, prompt_token_ids, number_of_samples):
            self.calls.append((prompt_token_ids, number_of_samples))
            batch = self.outputs[:number_of_samples]
            self.outputs = self.outputs[number_of_samples:]
            return batch

    backend = SequentialBackend([(encode("a"), False), (encode("b"), True)])
    trajectories = rollout_group(
        backend, FakeTokenizer(), lambda: StepRewardsEnv(exact_match_reward),
        {"prompt": "1+1", "answer": "2"}, "g0", group_size=1, seed=0,
        max_steps=3, truncated_reward=0.0,
    )
    trajectory = trajectories[0]
    assert trajectory.termination_reason == "length_limit"
    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].reward == 0.3
    assert trajectory.steps[0].environment_reward == 0.3
    assert trajectory.steps[1].environment_reward == -0.5
    assert trajectory.steps[1].reward == 0.0
    # 整局 = 先前步 + 被覆盖的截断步；不是整局被改成 truncated_reward
    assert trajectory.total_reward == 0.3


def test_truncated_raises_negative_environment_reward_to_zero():
    """默认 truncated_reward=0 时，截断会把负的环境步回报抬到 0，相对「正常落到负回报」占便宜。"""
    def negative_reward(prompt, completion, sample):
        return -0.5

    backend = FakeBackend([(encode("x"), True)])
    trajectories = rollout_group(
        backend, FakeTokenizer(),
        lambda: SingleTurnEnvironment(negative_reward),
        {"prompt": "1+1", "answer": "2"}, "g0", group_size=1, seed=0,
        truncated_reward=0.0,
    )
    step = trajectories[0].steps[0]
    assert step.environment_reward == -0.5
    assert step.reward == 0.0
    assert trajectories[0].total_reward == 0.0


def test_shared_reset_list_is_not_mutated():
    shared_messages = [{"role": "user", "content": "q"}]

    class SharedListEnv(NeverDoneEnv):
        def reset(self, sample, seed):
            self.sample = sample
            return sample["messages"]

    outputs = [(encode("a"), False), (encode("b"), False)]
    _, trajectories = run(
        outputs, group_size=2, environment_class=SharedListEnv,
        sample={"messages": shared_messages, "prompt": "q", "answer": "2"}, max_steps=2,
    )
    assert shared_messages == [{"role": "user", "content": "q"}]
    assert trajectories[0].steps[1].prompt_token_ids == encode("q|a|继续")
    assert trajectories[1].steps[1].prompt_token_ids == encode("q|b|继续")
