"""停止串：三种结束原因、落在 token 中间、空配置行为不变。"""
from types import SimpleNamespace

import pytest
import torch

from grpx.backends.common import normalize_stop_strings, parse_generated_tokens
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.backends.vllm import VllmBackend
from grpx.rollout import rollout_groups
from tests.test_huggingface_backend import END_TOKEN_ID, PADDING_TOKEN_ID, FakeModel, make_tokenizer
from tests.test_vllm_backend import (
    FakeResponse, make_backend, make_choice, patch_request, default_sync_handler,
)


def test_normalize_stop_strings_empty():
    assert normalize_stop_strings(None) == ()
    assert normalize_stop_strings([]) == ()


def test_normalize_stop_strings_rejects_bare_string():
    with pytest.raises(ValueError, match="列表"):
        normalize_stop_strings("</calc>")


def test_parse_end_token_before_stop():
    pieces = {5: "hello", END_TOKEN_ID: "", 9: "</calc>"}

    def decode(ids, skip_special_tokens=False):
        return "".join(pieces[i] for i in ids)

    tokenizer = SimpleNamespace(decode=decode)
    ids, reason = parse_generated_tokens(
        [5, END_TOKEN_ID, 9], {END_TOKEN_ID}, stop_strings=["</calc>"], tokenizer=tokenizer)
    assert (ids, reason) == ([5, END_TOKEN_ID], "end_token")


def test_parse_stop_string_mid_token():
    """停止串落在一个 token 中间：保留使解码文本首次包含停止串的那个完整 token。"""
    # token 7 单独解码是 "c>\n"，拼上前面变成 "...</calc>\n"；结束 token 集不含这些 id
    pieces = {1: "<", 3: "/cal", 7: "c>\n", 8: "#### 1"}

    def decode(ids, skip_special_tokens=False):
        return "".join(pieces[i] for i in ids)

    tokenizer = SimpleNamespace(decode=decode)
    ids, reason = parse_generated_tokens(
        [1, 3, 7, 8], {END_TOKEN_ID}, stop_strings=["</calc>"], tokenizer=tokenizer)
    assert reason == "stop_string"
    assert ids == [1, 3, 7]
    assert "</calc>" in decode(ids)


def test_parse_length_limit():
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=False: "abc")
    ids, reason = parse_generated_tokens(
        [5, 6, 7], {END_TOKEN_ID}, stop_strings=["</calc>"],
        tokenizer=tokenizer, max_new_tokens=3)
    assert (ids, reason) == ([5, 6, 7], "length_limit")


def test_huggingface_empty_stop_unchanged():
    model = FakeModel([
        [5, 6, END_TOKEN_ID, PADDING_TOKEN_ID, PADDING_TOKEN_ID],
        [7, END_TOKEN_ID, PADDING_TOKEN_ID, PADDING_TOKEN_ID, PADDING_TOKEN_ID],
    ])
    results = HuggingFaceBackend(model, make_tokenizer(), stop_strings=[]).generate([10, 11], 2)
    assert results == [([5, 6, END_TOKEN_ID], False), ([7, END_TOKEN_ID], False)]


def test_huggingface_stop_string_finish_reason():
    pieces = {5: "hello ", 6: "<calc>", 7: "1+1", 8: "</calc>", 9: " more"}

    class DecodingTokenizer:
        pad_token_id = PADDING_TOKEN_ID
        eos_token_id = END_TOKEN_ID

        def decode(self, ids, skip_special_tokens=False):
            return "".join(pieces[i] for i in ids)

    model = FakeModel([[5, 6, 7, 8, 9, PADDING_TOKEN_ID]])
    backend = HuggingFaceBackend(
        model, DecodingTokenizer(), stop_strings=["</calc>"], max_new_tokens=8)
    results = backend.generate([10], 1)
    token_ids, truncated, _logprobs, finish_reason = results[0]
    assert truncated is False
    assert finish_reason == "stop_string"
    assert token_ids == [5, 6, 7, 8]
    assert "more" not in DecodingTokenizer().decode(token_ids)


def test_vllm_stop_string_does_not_append_end_token(tmp_path, monkeypatch):
    backend = make_backend(tmp_path, stop_strings=["</calc>"])
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse({"choices": [
        make_choice(0, [5, 6, 7], finish_reason="stop", stop_reason="</calc>")
    ]}))
    token_ids, truncated, logprobs, finish_reason = backend.generate_batch([([10], 1)])[0][0]
    assert truncated is False
    assert finish_reason == "stop_string"
    assert token_ids == [5, 6, 7]
    assert END_TOKEN_ID not in token_ids
    assert len(logprobs) == 3


def test_vllm_empty_stop_still_appends_end_token(tmp_path, monkeypatch):
    backend = make_backend(tmp_path, stop_strings=[])
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse({"choices": [
        make_choice(0, [5, 6], finish_reason="stop", stop_reason=END_TOKEN_ID)
    ]}))
    # generate() 仍只返回 (ids, truncated)，与旧行为一致
    assert backend.generate([10], 1) == [([5, 6, END_TOKEN_ID], False)]


def test_vllm_sends_stop_and_include_flag(tmp_path, monkeypatch):
    backend = make_backend(tmp_path, stop_strings=["</calc>"])
    captured = []

    def handler(method, url, payload, timeout):
        if url.endswith("/v1/completions"):
            captured.append(payload)
            return FakeResponse({"choices": [
                make_choice(0, [5], finish_reason="stop", stop_reason="</calc>")
            ]})
        return default_sync_handler(method, url, payload, timeout)

    patch_request(monkeypatch, handler)
    backend.generate([10], 1)
    assert captured[0]["stop"] == ["</calc>"]
    assert captured[0]["include_stop_str_in_output"] is True


def test_rollout_stop_string_not_truncated_reward():
    """停止串结束的一步不触发 truncated_reward，轨迹可继续。"""

    class Backend:
        temperature = 1.0
        end_token_ids = {END_TOKEN_ID}

        def generate_batch(self, requests, greedy=False):
            return [[([1, 2, 3], False, None, "stop_string")]]

        def synchronize_policy(self, model):
            return

    class Env:
        def __init__(self):
            self.turns = 0

        def reset(self, sample, seed):
            return [{"role": "user", "content": "q"}]

        def step(self, action_text):
            self.turns += 1
            if self.turns == 1:
                return [{"role": "user", "content": "Calculator: 1"}], 0.0, False
            return [], 1.0, True

        def state_key(self):
            return ""

        def is_successful(self):
            return True

    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            return "prompt"

        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[9, 9])

        def decode(self, ids, skip_special_tokens=True):
            return "<calc>1</calc>"

    call_count = {"n": 0}
    original = Backend.generate_batch

    def counting(self, requests, greedy=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [[([1, 2, 3], False, None, "stop_string")]]
        return [[([4, END_TOKEN_ID], False, None, "end_token")]]

    Backend.generate_batch = counting
    trajectories = rollout_groups(
        Backend(), Tokenizer(), lambda: Env(), [("g", {}, 0)],
        group_size=1, max_steps=4, truncated_reward=-1.0)
    assert len(trajectories[0].steps) == 2
    assert trajectories[0].steps[0].finish_reason == "stop_string"
    assert trajectories[0].steps[0].reward == 0.0
    assert trajectories[0].termination_reason == "done"
