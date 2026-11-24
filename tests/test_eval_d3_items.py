"""500 题评测脚本：逐题对错和汇总一致。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from grpx.environment import SingleTurnEnvironment
from examples.arithmetic import is_correct


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_d3_500_items.py"
    spec = importlib.util.spec_from_file_location("eval_d3_500_items", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return messages[0]["content"]

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=[1])

    def decode(self, token_ids, skip_special_tokens):
        return "5"


class _Backend:
    temperature = 1.0

    def generate_batch(self, requests, greedy=False):
        return [[([5], False) for _ in range(request[1])] for request in requests]


def test_score_checkpoint_records_each_problem():
    module = _load_script()
    dataset = [
        {"prompt": "a", "answer": 5},
        {"prompt": "b", "answer": 7},
    ]
    environment_factory = lambda: SingleTurnEnvironment(
        lambda prompt, completion, sample: 0.0, success_function=is_correct)
    row = module.score_checkpoint(_Backend(), _Tokenizer(), environment_factory, dataset)
    assert row["n"] == 2
    assert row["correct_count"] == 1
    assert row["success_fraction"] == 0.5
    assert [item["correct"] for item in row["items"]] == [True, False]
    assert row["items"][1]["answer"] == 7
    assert row["items"][0]["completion"] == "5"
