from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from grpx.backends.huggingface import HuggingFaceBackend
from grpx.environment import SingleTurnEnvironment
from grpx.evaluation import evaluate

END_TOKEN_ID = 2

class FakeTokenizer:
    pad_token_id, eos_token_id = None, END_TOKEN_ID

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "|".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def decode(self, token_ids, skip_special_tokens):
        return "".join(chr(token_id) for token_id in token_ids)

class FakeGreedyBackend:
    """回答固定为 "x"；提示词为 "long" 时报告长度截断"""
    temperature = 0.7

    def __init__(self):
        self.greedy_prompts = []
        self.batch_count = 0

    def generate_batch(self, requests, greedy=False):
        assert greedy, "评测不应走采样路径"
        self.batch_count += 1
        results = []
        for request in requests:
            prompt_token_ids, number_of_samples = request[0], request[1]
            self.greedy_prompts.append(prompt_token_ids)
            prompt = "".join(chr(token_id) for token_id in prompt_token_ids)
            results.append([([ord("x")], prompt == "long")] * number_of_samples)
        return results

def reward_from_sample(prompt, completion, sample):
    return sample["reward"]

def success_from_sample(completion, sample):
    return sample["reward"] == 1.0

EVALUATION_DATASET = [{"prompt": "a", "reward": 1.0}, {"prompt": "long", "reward": 0.4}, {"prompt": "c", "reward": 0.0}]

def test_evaluation_metrics():
    backend = FakeGreedyBackend()
    metrics = evaluate(backend, FakeTokenizer(),
                       lambda: SingleTurnEnvironment(reward_from_sample, success_from_sample), EVALUATION_DATASET)
    # "long" 被截断：回报覆盖为 0，不算成功（环境原回报是 0.4）
    assert metrics["evaluation_mean_reward"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)
    assert metrics["evaluation_success_fraction"] == pytest.approx(1 / 3)
    assert metrics["evaluation_length_limit_fraction"] == pytest.approx(1 / 3)
    assert metrics["evaluation_mean_step_count"] == pytest.approx(1.0)
    assert metrics["evaluation_mean_answer_tokens"] == pytest.approx(1.0)
    assert metrics["evaluation_mean_answer_tokens_correct"] == pytest.approx(1.0)
    assert metrics["evaluation_mean_answer_tokens_incorrect"] == pytest.approx(1.0)
    assert metrics["evaluation_answer_tokens_correct_count"] == 1
    assert metrics["evaluation_answer_tokens_incorrect_count"] == 2
    assert len(backend.greedy_prompts) == 3
    assert backend.batch_count == 1    # 全部评测题在同一次后端调用里完成

def test_success_fraction_omitted_when_environment_has_no_success_notion():
    metrics = evaluate(FakeGreedyBackend(), FakeTokenizer(),
                       lambda: SingleTurnEnvironment(reward_from_sample), EVALUATION_DATASET)
    assert "evaluation_success_fraction" not in metrics

def test_batched_greedy_matches_one_prompt_at_a_time():
    model = make_tiny_model()
    prompts = [[ord(character) for character in text] for text in ("1+1", "12+34", "7", "100+200")]
    batched = HuggingFaceBackend(model, FakeTokenizer(), max_new_tokens=6)
    one_at_a_time = HuggingFaceBackend(model, FakeTokenizer(), max_new_tokens=6, max_rows_per_batch=1)
    assert [outputs[0] for outputs in batched.generate_batch([(prompt, 1) for prompt in prompts], greedy=True)] \
        == [one_at_a_time.generate_greedy(prompt) for prompt in prompts]

def make_tiny_model():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=128, n_positions=64, n_embd=16, n_layer=1, n_head=2,
        eos_token_id=END_TOKEN_ID, bos_token_id=END_TOKEN_ID))

@pytest.mark.parametrize("initially_training", [True, False])
def test_evaluation_keeps_training_mode_and_creates_no_gradients(initially_training):
    model = make_tiny_model()
    model.train(initially_training)
    backend = HuggingFaceBackend(model, FakeTokenizer(), max_new_tokens=4)
    dataset = [{"prompt": "1+1", "reward": 1.0}, {"prompt": "2+2", "reward": 0.0}]
    evaluate(backend, FakeTokenizer(), lambda: SingleTurnEnvironment(reward_from_sample), dataset)
    assert model.training is initially_training
    assert all(parameter.grad is None for parameter in model.parameters())

def test_greedy_evaluation_is_deterministic():
    model = make_tiny_model()
    backend = HuggingFaceBackend(model, FakeTokenizer(), max_new_tokens=6)
    prompt_token_ids = [ord(character) for character in "1+1"]
    results = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        results.append(backend.generate_greedy(prompt_token_ids))
    assert results[0] == results[1]
