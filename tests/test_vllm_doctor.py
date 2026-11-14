from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
import torch

from grpx.doctor import FAIL, PASS, check_vllm_reachable, check_vllm_service, check_vllm_sync_takes_effect


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or "{}"

    def json(self):
        return self._payload


def test_vllm_reachable_pass(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())
    result = check_vllm_reachable("http://localhost:8000", model_name="Qwen/Qwen2.5-0.5B")
    assert result.status == PASS


def test_vllm_reachable_fail_includes_start_command(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", boom)
    result = check_vllm_reachable("http://localhost:8000", model_name="Qwen/Qwen2.5-0.5B", lora_rank=16)
    assert result.status == FAIL
    assert "vllm serve" in result.suggestion
    assert "--enable-lora" in result.suggestion


def test_vllm_service_checks_base_model(monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda *args, **kwargs: FakeResponse(payload={
            "data": [{"id": "Qwen/Qwen2.5-1.5B-Instruct", "parent": None}],
        }),
    )
    results = check_vllm_service(
        "http://localhost:8000",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        expected_base_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
    assert results[0].status == PASS
    assert results[1].name == "基座模型" and results[1].status == FAIL

    results_ok = check_vllm_service(
        "http://localhost:8000",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        expected_base_model="Qwen/Qwen2.5-1.5B-Instruct",
    )
    assert results_ok[1].status == PASS


def test_sync_takes_effect_detects_unchanged_output(monkeypatch):
    backend = MagicMock()
    backend.generate_greedy.side_effect = [
        ([1, 2], False),
        ([1, 2], False),    # 扰动后仍相同 → 失败
        ([1, 2], False),
    ]
    backend.synchronize_policy = MagicMock()

    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    model = MagicMock()
    model.named_parameters.return_value = [("layer.lora_B.weight", parameter)]

    result = check_vllm_sync_takes_effect(backend, model, prompt_token_ids=[10, 11])
    assert result.status == FAIL
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" in result.suggestion


def test_sync_takes_effect_pass_when_output_changes(monkeypatch):
    backend = MagicMock()
    backend.generate_greedy.side_effect = [
        ([1, 2], False),
        ([9, 9], False),
        ([1, 2], False),
    ]
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    model = MagicMock()
    model.named_parameters.return_value = [("layer.lora_B.weight", parameter)]

    result = check_vllm_sync_takes_effect(backend, model, prompt_token_ids=[10, 11])
    assert result.status == PASS
