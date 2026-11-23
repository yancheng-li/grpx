"""训推基座一致性：启动比对、加载后核对 parent、第 0 轮 gap 硬中止。"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
import torch

from grpx.backends.vllm import (
    VllmBackend,
    base_model_ids_from_vllm_models,
    verify_vllm_base_model_match,
)
from grpx.doctor import FAIL, PASS, check_vllm_base_model
from grpx.loop import enforce_backend_log_probability_gap


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


def make_backend(tmp_path, **overrides):
    options = dict(
        tokenizer=SimpleNamespace(eos_token_id=END_TOKEN_ID),
        end_token_ids={END_TOKEN_ID},
        adapter_directory=tmp_path / "adapter",
        base_url="http://vllm.test",
        max_concurrent_requests=4,
        request_timeout_seconds=5,
        expected_base_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
    options.update(overrides)
    return VllmBackend(**options)


def test_base_model_ids_only_include_parent_none():
    payload = {"data": [
        {"id": "Qwen/Qwen2.5-0.5B-Instruct", "parent": None},
        {"id": "policy-run-1", "parent": "Qwen/Qwen2.5-0.5B-Instruct"},
        {"id": "also-base", "parent": None},
    ]}
    assert base_model_ids_from_vllm_models(payload) == [
        "Qwen/Qwen2.5-0.5B-Instruct", "also-base",
    ]


def test_verify_vllm_base_model_match_accepts_and_rejects():
    payload = {"data": [{"id": "Qwen/Qwen2.5-0.5B-Instruct", "parent": None}]}
    verify_vllm_base_model_match(payload, "Qwen/Qwen2.5-0.5B-Instruct")
    with pytest.raises(RuntimeError, match="不是同一个模型"):
        verify_vllm_base_model_match(payload, "Qwen/Qwen2.5-1.5B-Instruct")


def test_doctor_check_vllm_base_model_statuses():
    payload = {"data": [
        {"id": "Qwen/Qwen2.5-1.5B-Instruct", "parent": None},
        {"id": "policy-x-1", "parent": "Qwen/Qwen2.5-1.5B-Instruct"},
    ]}
    assert check_vllm_base_model(payload, "Qwen/Qwen2.5-1.5B-Instruct").status == PASS
    failed = check_vllm_base_model(payload, "Qwen/Qwen2.5-0.5B-Instruct")
    assert failed.status == FAIL
    assert "1.5B" in failed.detail


def test_load_adapter_rejects_when_trained_base_differs_from_server_parent(tmp_path, monkeypatch):
    """vLLM 对不匹配 LoRA 仍可能返回 200；必须按 adapter_config 与 parent 核对。"""
    backend = make_backend(tmp_path, expected_base_model="Qwen/Qwen2.5-0.5B-Instruct")
    directory = tmp_path / "adapter" / "policy-run-1"
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
    }), encoding="utf-8")

    def fake_request(method, url, json=None, timeout=None, **kwargs):
        if str(url).endswith("/v1/load_lora_adapter"):
            return FakeResponse({"ok": True})
        if str(url).endswith("/v1/models"):
            return FakeResponse({"data": [
                {"id": "Qwen/Qwen2.5-1.5B-Instruct", "parent": None},
                {"id": "policy-run-1", "parent": "Qwen/Qwen2.5-1.5B-Instruct",
                 "root": str(directory)},
            ]})
        return FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "request", fake_request)
    with pytest.raises(RuntimeError, match="服务端基座"):
        backend._load_adapter("policy-run-1", directory)


def test_load_adapter_accepts_matching_parent(tmp_path, monkeypatch):
    backend = make_backend(tmp_path, expected_base_model="Qwen/Qwen2.5-0.5B-Instruct")
    directory = tmp_path / "adapter" / "policy-run-1"
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
    }), encoding="utf-8")

    def fake_request(method, url, json=None, timeout=None, **kwargs):
        if str(url).endswith("/v1/load_lora_adapter"):
            return FakeResponse({"ok": True})
        if str(url).endswith("/v1/models"):
            return FakeResponse({"data": [
                {"id": "Qwen/Qwen2.5-0.5B-Instruct", "parent": None},
                {"id": "policy-run-1", "parent": "Qwen/Qwen2.5-0.5B-Instruct"},
            ]})
        return FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "request", fake_request)
    backend._load_adapter("policy-run-1", directory)


def test_load_adapter_accepts_local_path_when_server_model_name_set(tmp_path, monkeypatch):
    """adapter_config 记本地路径、服务端记模型名：用 server_model_name（expected）放行。"""
    local_path = "/data/models/Qwen2.5-0.5B-Instruct"
    server_name = "Qwen/Qwen2.5-0.5B-Instruct"
    backend = make_backend(
        tmp_path,
        expected_base_model=server_name,
        training_model_name=local_path,
    )
    directory = tmp_path / "adapter" / "policy-run-1"
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": local_path,
    }), encoding="utf-8")

    def fake_request(method, url, json=None, timeout=None, **kwargs):
        if str(url).endswith("/v1/load_lora_adapter"):
            return FakeResponse({"ok": True})
        if str(url).endswith("/v1/models"):
            return FakeResponse({"data": [
                {"id": server_name, "parent": None},
                {"id": "policy-run-1", "parent": server_name},
            ]})
        return FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "request", fake_request)
    backend._load_adapter("policy-run-1", directory)


def test_load_adapter_rejects_trained_base_mismatch_despite_server_alias(tmp_path, monkeypatch):
    """parent 已对齐 expected，但 adapter_config 既不是 parent 也不是训练端名字 → 仍拒绝。"""
    backend = make_backend(
        tmp_path,
        expected_base_model="Qwen/Qwen2.5-0.5B-Instruct",
        training_model_name="/data/models/Qwen2.5-0.5B-Instruct",
    )
    directory = tmp_path / "adapter" / "policy-run-1"
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    }), encoding="utf-8")

    def fake_request(method, url, json=None, timeout=None, **kwargs):
        if str(url).endswith("/v1/load_lora_adapter"):
            return FakeResponse({"ok": True})
        if str(url).endswith("/v1/models"):
            return FakeResponse({"data": [
                {"id": "Qwen/Qwen2.5-0.5B-Instruct", "parent": None},
                {"id": "policy-run-1", "parent": "Qwen/Qwen2.5-0.5B-Instruct"},
            ]})
        return FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "request", fake_request)
    with pytest.raises(RuntimeError, match="适配器基于"):
        backend._load_adapter("policy-run-1", directory)


def test_enforce_gap_hard_on_first_process_check_warns_later():
    # 硬失败看差超过 1 的比例、最大差和中位数
    metrics = {
        "backend_log_probability_gap_median": 1e-7,
        "backend_log_probability_gap_p95": 0.05,
        "backend_log_probability_gap_above_one_fraction": 0.03,
        "backend_log_probability_gap_max": 8.0,
    }
    with pytest.raises(RuntimeError, match="不是同一个模型"):
        enforce_backend_log_probability_gap(metrics, hard=True)
    # 恢复训练时第一轮可能是 iteration=20，仍应硬中止
    with pytest.raises(RuntimeError, match="不是同一个模型"):
        enforce_backend_log_probability_gap(metrics, hard=True)
    with pytest.warns(UserWarning, match="不是同一个模型"):
        enforce_backend_log_probability_gap(metrics, hard=False)


def test_enforce_gap_passes_when_small_or_missing():
    enforce_backend_log_probability_gap(
        {
            "backend_log_probability_gap_median": 0.003,
            "backend_log_probability_gap_p95": 0.16,
            "backend_log_probability_gap_above_one_fraction": 0.0,
            "backend_log_probability_gap_max": 1.64,
        },
        hard=True)
    # 差超过 1 的比例 0.011%，低于 0.3% 的警告线
    enforce_backend_log_probability_gap(
        {
            "backend_log_probability_gap_median": 1e-7,
            "backend_log_probability_gap_p95": 1.1e-5,
            "backend_log_probability_gap_above_one_fraction": 1.1e-4,
            "backend_log_probability_gap_max": 1.64,
        },
        hard=True)
    enforce_backend_log_probability_gap({}, hard=True)


def test_enforce_gap_above_one_warns_without_hard_fail():
    with pytest.warns(UserWarning, match="above_one_fraction"):
        enforce_backend_log_probability_gap(
            {
                "backend_log_probability_gap_median": 0.003,
                "backend_log_probability_gap_p95": 0.16,
                "backend_log_probability_gap_above_one_fraction": 0.006,
                "backend_log_probability_gap_max": 1.64,
            },
            hard=True,
        )


def test_enforce_gap_hard_fails_on_large_median():
    with pytest.raises(RuntimeError, match="不是同一个模型"):
        enforce_backend_log_probability_gap(
            {
                "backend_log_probability_gap_median": 0.46,
                "backend_log_probability_gap_p95": 0.5,
                "backend_log_probability_gap_above_one_fraction": 0.0,
                "backend_log_probability_gap_max": 0.5,
            },
            hard=True)
