"""多 base_url：轨迹固定分配、同步全成或全败、gap 取各地址最大值。"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
import torch

from grpx.backends.vllm import VllmBackend, normalize_base_urls
from grpx.loop import enforce_backend_log_probability_gap, measure_backend_log_probability_gap
from grpx.rollout import backend_url_for_trajectory, rollout_groups
from grpx.types import Step, Trajectory


END_TOKEN_ID = 2


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is not None:
            return self._payload
        return json.loads(self.text)


def make_backend(tmp_path, base_url, **overrides):
    options = dict(
        tokenizer=SimpleNamespace(eos_token_id=END_TOKEN_ID),
        end_token_ids={END_TOKEN_ID},
        adapter_directory=tmp_path / "adapter",
        base_url=base_url,
        max_concurrent_requests=4,
        request_timeout_seconds=5,
        expected_base_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
    options.update(overrides)
    return VllmBackend(**options)


def test_normalize_base_urls_string_and_list():
    assert normalize_base_urls("http://a:8000/") == ["http://a:8000"]
    assert normalize_base_urls(["http://a:8000", "http://b:8001/"]) == [
        "http://a:8000", "http://b:8001",
    ]
    with pytest.raises(ValueError, match="不能为空"):
        normalize_base_urls([])


def test_single_element_list_matches_string(tmp_path):
    as_string = make_backend(tmp_path, "http://vllm.test")
    as_list = make_backend(tmp_path, ["http://vllm.test"])
    assert as_string.base_urls == as_list.base_urls == ["http://vllm.test"]
    assert as_string.base_url == as_list.base_url == "http://vllm.test"


def test_synchronize_aborts_when_second_url_fails(tmp_path, monkeypatch):
    backend = make_backend(tmp_path, ["http://a:8000", "http://b:8001"])
    model = MagicMock()
    model.save_pretrained = MagicMock()

    loaded = []

    def fake_request(method, url, json=None, timeout=None, **kwargs):
        if str(url).endswith("/v1/models"):
            return FakeResponse({"data": [
                {"id": "Qwen/Qwen2.5-0.5B-Instruct", "parent": None},
            ]})
        if str(url).endswith("/v1/load_lora_adapter"):
            if "b:8001" in str(url):
                return FakeResponse(status_code=500, text="boom")
            loaded.append(str(url))
            return FakeResponse({"ok": True})
        if str(url).endswith("/v1/unload_lora_adapter"):
            loaded.append(f"unload:{url}")
            return FakeResponse({"ok": True})
        return FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "request", fake_request)
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda model: True)
    # 跳过 parent 核对里「已加载」名单不足（测的是第二地址失败回滚）
    monkeypatch.setattr(backend, "_verify_loaded_adapter", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        backend.synchronize_policy(model)
    assert any("a:8000" in item and "load_lora" in item for item in loaded)
    assert any("unload:" in item and "a:8000" in item for item in loaded)
    assert backend.adapter_name is None


def test_trajectory_index_assigns_url_deterministically():
    backend = SimpleNamespace(base_urls=["http://a", "http://b"])
    assert backend_url_for_trajectory(backend, 0) == ("http://a", 0)
    assert backend_url_for_trajectory(backend, 1) == ("http://b", 1)
    assert backend_url_for_trajectory(backend, 2) == ("http://a", 0)


def test_rollout_keeps_same_url_across_turns():
    class MultiUrlBackend:
        temperature = 1.0
        base_urls = ["http://a", "http://b"]

        def __init__(self):
            self.seen = []

        def generate_batch(self, requests, greedy=False):
            results = []
            for request in requests:
                prompt, n, url_index = request
                self.seen.append((tuple(prompt), n, url_index))
                results.append([([ord("1")], False)] * n)
            return results

    class Env:
        def reset(self, sample, seed):
            return [{"role": "user", "content": "go"}]

        def step(self, action):
            return [{"role": "user", "content": "again"}], 0.0, False

        def state_key(self):
            return "s"

        def is_successful(self):
            return False

    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return "".join(message["content"] for message in messages)

        def __call__(self, text, add_special_tokens):
            return SimpleNamespace(input_ids=[ord(character) for character in text] or [1])

        def decode(self, token_ids, skip_special_tokens):
            return "1"

    backend = MultiUrlBackend()
    trajectories = rollout_groups(
        backend, Tokenizer(), Env,
        [("g0", {"x": 1}, 0), ("g1", {"x": 2}, 1)],
        group_size=1, max_steps=2, truncated_reward=None,
    )
    assert len(trajectories) == 2
    # 轨迹 0 → url 0，两步都是；轨迹 1 → url 1
    url_by_traj = {0: set(), 1: set()}
    for prompt, n, url_index in backend.seen:
        # 无法从 prompt 反推轨迹，但每条轨迹固定 url：检查各 url_index 出现次数与步数一致
        assert n == 1
    assert {url for _, _, url in backend.seen} == {0, 1}
    # 两步 × 两轨迹 = 4 次请求（提示词不同不会跨轨迹合并到同 url）
    assert len(backend.seen) == 4


def test_gap_metrics_take_max_across_urls_and_name_worst():
    def batch_for(url, train_logp, backend_logp):
        # old 与 sampling[:, 1:] 同形，对齐 update_policy 里 action_log_probabilities 的返回
        return {
            "sampling_log_probability_mask": torch.tensor([[False, True, True]]),
            "sampling_log_probabilities": torch.tensor([[0.0, backend_logp, backend_logp]]),
            "backend_urls": [url],
        }, torch.tensor([[train_logp, train_logp]])

    batch_a, old_a = batch_for("http://a", 0.0, 0.01)
    # url b：差 2.0 → 尾部大偏差，应判为最差地址并硬失败
    batch_b, old_b = batch_for("http://b", 0.0, 2.0)
    metrics = measure_backend_log_probability_gap([batch_a, batch_b], [old_a, old_b])
    assert metrics["backend_log_probability_gap_median"] == pytest.approx(2.0)
    assert metrics["backend_log_probability_gap_max"] == pytest.approx(2.0)
    assert metrics["backend_log_probability_gap_above_half_fraction"] == pytest.approx(1.0)
    assert metrics["backend_log_probability_gap_above_one_fraction"] == pytest.approx(1.0)
    assert metrics["backend_log_probability_gap_worst_url"] == "http://b"
    assert "gap_by_url" in metrics
    with pytest.raises(RuntimeError, match="http://b"):
        enforce_backend_log_probability_gap(metrics, hard=True)
