import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from grpx.backends.vllm import VllmBackend

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


def make_tokenizer(eos_token_id=END_TOKEN_ID):
    return SimpleNamespace(eos_token_id=eos_token_id)


def make_choice(index, token_ids, finish_reason="stop", stop_reason=None):
    tokens = [f"token_id:{token_id}" for token_id in token_ids]
    choice = {
        "index": index,
        "finish_reason": finish_reason,
        "logprobs": {"tokens": tokens},
    }
    if stop_reason is not None:
        choice["stop_reason"] = stop_reason
    return choice


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_backend(tmp_path, **overrides):
    options = dict(
        tokenizer=make_tokenizer(),
        end_token_ids={END_TOKEN_ID},
        adapter_directory=tmp_path / "adapter",
        base_url="http://vllm.test",
        max_concurrent_requests=4,
        request_timeout_seconds=5,
        expected_base_model=DEFAULT_BASE_MODEL,
    )
    options.update(overrides)
    return VllmBackend(**options)


def empty_models_with_base(base=DEFAULT_BASE_MODEL, loaded_adapter=None):
    data = [{"id": base, "parent": None}]
    if loaded_adapter is not None:
        data.append({"id": loaded_adapter, "parent": base})
    return FakeResponse({"object": "list", "data": data})


def patch_request(monkeypatch, handler):
    """包装 handler：自动记录成功 load 的适配器名，未声明的 GET /v1/models 返回匹配基座。"""
    state = {"loaded": None}

    def fake_request(method, url, json=None, timeout=None, params=None, **kwargs):
        payload = json
        try:
            response = handler(method, url, payload, timeout)
        except AssertionError:
            if method.upper() == "GET" and str(url).rstrip("/").endswith("/v1/models"):
                return empty_models_with_base(loaded_adapter=state["loaded"])
            raise
        if (method.upper() == "POST" and str(url).endswith("/v1/load_lora_adapter")
                and response.status_code < 400 and isinstance(payload, dict)):
            state["loaded"] = payload.get("lora_name")
            # 供 _verify_loaded_adapter 读取；真实 PEFT 会写这个文件
            lora_path = payload.get("lora_path")
            if lora_path:
                from pathlib import Path
                directory = Path(lora_path)
                directory.mkdir(parents=True, exist_ok=True)
                config_path = directory / "adapter_config.json"
                if not config_path.exists():
                    config_path.write_text(
                        __import__("json").dumps(
                            {"base_model_name_or_path": DEFAULT_BASE_MODEL}),
                        encoding="utf-8",
                    )
        if (method.upper() == "GET" and str(url).rstrip("/").endswith("/v1/models")
                and response.status_code < 400):
            try:
                body = response.json()
            except Exception:  # noqa: BLE001
                return empty_models_with_base(loaded_adapter=state["loaded"])
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, list):
                return empty_models_with_base(loaded_adapter=state["loaded"])
            ids = {item.get("id") for item in data if isinstance(item, dict)}
            patched = [dict(item) for item in data if isinstance(item, dict)]
            if DEFAULT_BASE_MODEL not in ids:
                patched.insert(0, {"id": DEFAULT_BASE_MODEL, "parent": None})
            if state["loaded"] and state["loaded"] not in ids:
                patched.append({"id": state["loaded"], "parent": DEFAULT_BASE_MODEL})
            return FakeResponse({"object": "list", "data": patched})
        return response

    monkeypatch.setattr(requests, "request", fake_request)


def default_sync_handler(method, url, payload, timeout, *, completion=None):
    if url.endswith("/v1/load_lora_adapter") or url.endswith("/v1/unload_lora_adapter"):
        return FakeResponse({"ok": True})
    if completion is not None:
        return completion
    return FakeResponse({"choices": [make_choice(0, [END_TOKEN_ID])]})


def test_length_truncation_does_not_append_end_token(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse(
        {"choices": [make_choice(0, [5, 6, 7], finish_reason="length")]}))
    assert backend.generate([10], 1) == [([5, 6, 7], True)]


def test_appends_end_token_when_missing(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse({"choices": [
        make_choice(0, [5, 6], finish_reason="stop", stop_reason=END_TOKEN_ID)
    ]}))
    assert backend.generate([10], 1) == [([5, 6, END_TOKEN_ID], False)]


def test_does_not_duplicate_end_token_already_present(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse({"choices": [
        make_choice(0, [5, END_TOKEN_ID], finish_reason="stop")
    ]}))
    assert backend.generate([10], 1) == [([5, END_TOKEN_ID], False)]


def test_n_greater_than_one_sorted_by_index(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse({"choices": [
        make_choice(1, [9, END_TOKEN_ID]),
        make_choice(0, [8, END_TOKEN_ID]),
    ]}))
    assert backend.generate([10], 2) == [([8, END_TOKEN_ID], False), ([9, END_TOKEN_ID], False)]


def test_greedy_uses_temperature_zero(tmp_path, monkeypatch):
    backend = make_backend(tmp_path, temperature=0.8)
    captured = []

    def handler(method, url, payload, timeout):
        captured.append(payload)
        return FakeResponse({"choices": [make_choice(0, [END_TOKEN_ID])]})

    patch_request(monkeypatch, handler)
    backend.generate_greedy([10])
    assert captured[0]["temperature"] == 0.0


def test_greedy_batch_requests_n_one_and_replicates(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    captured = []

    def handler(method, url, payload, timeout):
        captured.append(payload)
        return FakeResponse({"choices": [make_choice(0, [5, END_TOKEN_ID])]})

    patch_request(monkeypatch, handler)
    samples = backend.generate_batch([([10], 3)], greedy=True)[0]
    assert [item[:2] for item in samples] == [([5, END_TOKEN_ID], False)] * 3
    assert captured[0]["n"] == 1
    assert captured[0]["temperature"] == 0.0


def test_synchronize_uses_versioned_adapter_names(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)
    model = MagicMock()
    loads, unloads = [], []

    def handler(method, url, payload, timeout):
        if url.endswith("/v1/load_lora_adapter"):
            loads.append(payload)
            return FakeResponse({"ok": True})
        if url.endswith("/v1/unload_lora_adapter"):
            unloads.append(payload)
            return FakeResponse({"ok": True})
        raise AssertionError(f"unexpected {url}")

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(model)
    assert loads[0]["lora_name"] == "policy-1"
    assert loads[0]["lora_path"].endswith("policy-1")
    assert "load_inplace" not in loads[0]
    assert backend.adapter_name == "policy-1"
    assert unloads == []
    assert (tmp_path / "adapter" / "policy-1").is_dir()

    backend.synchronize_policy(model)
    assert loads[1]["lora_name"] == "policy-2"
    assert unloads == [{"lora_name": "policy-1"}]
    assert backend.adapter_name == "policy-2"
    assert not (tmp_path / "adapter" / "policy-1").exists()
    assert (tmp_path / "adapter" / "policy-2").is_dir()


def test_synchronize_capacity_error_unloads_then_reloads(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)
    model = MagicMock()
    events = []

    def handler(method, url, payload, timeout):
        if url.endswith("/v1/load_lora_adapter"):
            events.append(("load", payload["lora_name"]))
            if payload["lora_name"] == "policy-2" and events.count(("load", "policy-2")) == 1:
                return FakeResponse(
                    status_code=400,
                    text='{"error":{"message":"Cannot add LoRA, max lora limit exceeded"}}',
                )
            return FakeResponse({"ok": True})
        if url.endswith("/v1/unload_lora_adapter"):
            events.append(("unload", payload["lora_name"]))
            return FakeResponse({"ok": True})
        raise AssertionError(url)

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(model)
    backend.synchronize_policy(model)
    assert events == [
        ("load", "policy-1"),
        ("load", "policy-2"),
        ("unload", "policy-1"),
        ("load", "policy-2"),
    ]
    assert backend.adapter_name == "policy-2"


def test_synchronize_unload_failure_is_ignored(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)

    def handler(method, url, payload, timeout):
        if url.endswith("/v1/load_lora_adapter"):
            return FakeResponse({"ok": True})
        if url.endswith("/v1/unload_lora_adapter"):
            return FakeResponse(status_code=500, text='{"error":{"message":"busy"}}')
        raise AssertionError(url)

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(MagicMock())
    backend.synchronize_policy(MagicMock())    # 卸载失败不应抛出
    assert backend.adapter_name == "policy-2"


def test_synchronize_already_loaded_unloads_and_retries(tmp_path, monkeypatch):
    backend = make_backend(tmp_path, adapter_name="policy-runA")
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)
    events = []

    loaded = []

    def handler(method, url, payload, timeout):
        if method.upper() == "GET" and url.endswith("/v1/models"):
            data = [{"id": DEFAULT_BASE_MODEL, "parent": None}]
            data.extend({"id": name, "parent": DEFAULT_BASE_MODEL} for name in loaded)
            return FakeResponse({"object": "list", "data": data})
        if url.endswith("/v1/load_lora_adapter"):
            events.append(("load", payload["lora_name"]))
            if events.count(("load", "policy-runA-1")) == 1:
                return FakeResponse(
                    status_code=400,
                    text='{"error":{"message":"The lora adapter \'policy-runA-1\' has already been loaded. '
                         'If you want to load the adapter in place, set \'load_inplace\' to True."}}',
                )
            loaded.append(payload["lora_name"])
            return FakeResponse({"ok": True})
        if url.endswith("/v1/unload_lora_adapter"):
            events.append(("unload", payload["lora_name"]))
            if payload["lora_name"] in loaded:
                loaded.remove(payload["lora_name"])
            return FakeResponse({"ok": True})
        raise AssertionError(url)

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(MagicMock())
    assert events == [
        ("load", "policy-runA-1"),
        ("unload", "policy-runA-1"),
        ("load", "policy-runA-1"),
    ]
    assert backend.adapter_name == "policy-runA-1"


def test_first_sync_clears_stale_prefix_adapters(tmp_path, monkeypatch):
    backend = make_backend(tmp_path, adapter_name="policy-runA")
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)
    events = []

    loaded = []

    def handler(method, url, payload, timeout):
        if method.upper() == "GET" and url.endswith("/v1/models"):
            events.append(("list", None))
            data = [
                {"id": DEFAULT_BASE_MODEL, "parent": None},
                {"id": "policy-runA-61", "parent": DEFAULT_BASE_MODEL},
                {"id": "policy-runA-extra-1", "parent": DEFAULT_BASE_MODEL},  # 不同前缀，不应卸
                {"id": "policy-runB-1", "parent": DEFAULT_BASE_MODEL},
            ]
            data.extend({"id": name, "parent": DEFAULT_BASE_MODEL} for name in loaded)
            return FakeResponse({"object": "list", "data": data})
        if url.endswith("/v1/unload_lora_adapter"):
            events.append(("unload", payload["lora_name"]))
            return FakeResponse({"ok": True})
        if url.endswith("/v1/load_lora_adapter"):
            events.append(("load", payload["lora_name"]))
            loaded.append(payload["lora_name"])
            return FakeResponse({"ok": True})
        raise AssertionError(url)

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(MagicMock())
    assert events.count(("unload", "policy-runA-61")) == 1
    assert ("unload", "policy-runA-extra-1") not in events
    assert ("unload", "policy-runB-1") not in events
    assert ("load", "policy-runA-1") in events
    # 第二次同步不再扫残留（加载核对仍会 GET /v1/models）
    backend.synchronize_policy(MagicMock())
    assert events.count(("unload", "policy-runA-61")) == 1
    assert ("load", "policy-runA-2") in events


def test_close_unloads_active_adapter(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)
    unloads = []

    loaded = []

    def handler(method, url, payload, timeout):
        if method.upper() == "GET" and url.endswith("/v1/models"):
            data = [{"id": DEFAULT_BASE_MODEL, "parent": None}]
            data.extend({"id": name, "parent": DEFAULT_BASE_MODEL} for name in loaded)
            return FakeResponse({"object": "list", "data": data})
        if url.endswith("/v1/load_lora_adapter"):
            loaded.append(payload["lora_name"])
            return FakeResponse({"ok": True})
        if url.endswith("/v1/unload_lora_adapter"):
            unloads.append(payload["lora_name"])
            return FakeResponse({"ok": True})
        raise AssertionError(url)

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(MagicMock())
    assert backend.adapter_name == "policy-1"
    backend.close()
    assert unloads == ["policy-1"]
    assert backend.adapter_name is None


def test_sanitize_adapter_name_component():
    from grpx.backends.vllm import sanitize_adapter_name_component
    assert sanitize_adapter_name_component("my run/名称") == "my-run"
    assert sanitize_adapter_name_component("versioned-adapters-5") == "versioned-adapters-5"


def test_is_versioned_adapter_name(tmp_path):
    backend = make_backend(tmp_path, adapter_name="policy-runA")
    assert backend._is_versioned_adapter_name("policy-runA-61")
    assert not backend._is_versioned_adapter_name("policy-runA")
    assert not backend._is_versioned_adapter_name("policy-runA-extra-1")
    assert not backend._is_versioned_adapter_name("policy-runB-1")


def test_http_error_includes_server_message(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse(
        status_code=400,
        text='{"error":{"message":"prompt too long for context"}}',
    ))
    with pytest.raises(RuntimeError, match="prompt too long for context"):
        backend.generate([10], 1)


def test_retries_timeout_then_succeeds(tmp_path, monkeypatch, capsys):
    backend = make_backend(tmp_path)
    calls = {"count": 0}

    def handler(method, url, payload, timeout):
        calls["count"] += 1
        if calls["count"] < 3:
            raise requests.Timeout("timed out")
        return FakeResponse({"choices": [make_choice(0, [END_TOKEN_ID])]})

    patch_request(monkeypatch, handler)
    monkeypatch.setattr("grpx.backends.vllm.time.sleep", lambda seconds: None)
    assert backend.generate([10], 1) == [([END_TOKEN_ID], False)]
    assert calls["count"] == 3
    logged = capsys.readouterr().out
    assert logged.count("请求重试：") == 2
    assert "超时" in logged


def test_model_missing_resyncs_once(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    model = MagicMock()
    model.save_pretrained = MagicMock()
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)

    posts = []

    def handler(method, url, payload, timeout):
        posts.append(url)
        if url.endswith("/v1/load_lora_adapter") or url.endswith("/v1/unload_lora_adapter"):
            return FakeResponse({"ok": True})
        if url.endswith("/v1/completions") and sum(1 for item in posts if item.endswith("/v1/completions")) == 1:
            return FakeResponse(
                status_code=404,
                text='{"error":{"message":"adapter policy-1 not found"}}',
            )
        return FakeResponse({"choices": [make_choice(0, [END_TOKEN_ID])]})

    patch_request(monkeypatch, handler)
    monkeypatch.setattr("grpx.backends.vllm.time.sleep", lambda seconds: None)
    backend.synchronize_policy(model)
    assert backend.generate([10], 1) == [([END_TOKEN_ID], False)]
    assert sum(1 for url in posts if url.endswith("/v1/load_lora_adapter")) >= 2


def test_synchronize_rejects_non_lora(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: False)
    with pytest.raises(ValueError, match="LoRA"):
        backend.synchronize_policy(SimpleNamespace())


def test_server_adapter_path_prefix_rewrite(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    adapter = local_root / "run" / "vllm_adapter"
    adapter.mkdir(parents=True)
    backend = make_backend(
        tmp_path,
        adapter_directory=adapter,
        adapter_path_local_prefix=str(local_root),
        adapter_path_server_prefix="/mnt/work",
    )
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)
    model = MagicMock()
    captured = []

    def handler(method, url, payload, timeout):
        if url.endswith("/v1/load_lora_adapter"):
            captured.append(payload)
        return FakeResponse({"ok": True})

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(model)
    assert captured[0]["lora_path"] == "/mnt/work/run/vllm_adapter/policy-1"
    model.save_pretrained.assert_called_once_with(adapter / "policy-1")


def test_relative_adapter_path_sent_as_absolute(tmp_path, monkeypatch):
    import os
    adapter = tmp_path / "outputs" / "run" / "vllm_adapter"
    adapter.mkdir(parents=True)
    relative = os.path.relpath(adapter, tmp_path)
    backend = make_backend(tmp_path, adapter_directory=relative)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)
    captured = []

    def handler(method, url, payload, timeout):
        if url.endswith("/v1/load_lora_adapter"):
            captured.append(payload)
        return FakeResponse({"ok": True})

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(MagicMock())
    assert os.path.isabs(captured[0]["lora_path"])
    assert os.path.normpath(captured[0]["lora_path"]) == os.path.normpath(
        str((adapter / "policy-1").resolve()))


def test_prefix_mismatch_raises(tmp_path):
    adapter = tmp_path / "elsewhere" / "vllm_adapter"
    adapter.mkdir(parents=True)
    backend = make_backend(
        tmp_path,
        adapter_directory=adapter,
        adapter_path_local_prefix=str(tmp_path / "local"),
        adapter_path_server_prefix="/mnt/work",
    )
    with pytest.raises(ValueError, match="adapter_path_local_prefix"):
        backend.server_adapter_path()


def test_prefix_does_not_match_projects2(tmp_path):
    projects = tmp_path / "projects"
    projects2 = tmp_path / "projects2" / "run" / "vllm_adapter"
    projects2.mkdir(parents=True)
    backend = make_backend(
        tmp_path,
        adapter_directory=projects2,
        adapter_path_local_prefix=str(projects),
        adapter_path_server_prefix="/mnt/projects",
    )
    with pytest.raises(ValueError, match="adapter_path_local_prefix"):
        backend.server_adapter_path()


def test_concurrent_model_missing_resyncs_only_once(tmp_path, monkeypatch):
    backend = make_backend(tmp_path, max_concurrent_requests=8)
    model = MagicMock()
    model.save_pretrained = MagicMock()
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)
    monkeypatch.setattr("grpx.backends.vllm.time.sleep", lambda seconds: None)

    import threading
    lock = threading.Lock()
    state = {"loads": 0, "baseline_loads": None}

    def handler(method, url, payload, timeout):
        with lock:
            if url.endswith("/v1/unload_lora_adapter"):
                return FakeResponse({"ok": True})
            if url.endswith("/v1/load_lora_adapter"):
                state["loads"] += 1
                return FakeResponse({"ok": True})
            # 模拟服务端：适配器被清掉后，直到再次 load 之前一律 404
            if state["loads"] == state["baseline_loads"]:
                return FakeResponse(
                    status_code=404,
                    text=f'{{"error":{{"message":"adapter {backend.adapter_name} not found"}}}}',
                )
            return FakeResponse({"choices": [make_choice(0, [END_TOKEN_ID])]})

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(model)
    state["baseline_loads"] = state["loads"]
    results = backend.generate_batch([([10], 1) for _ in range(8)])
    assert results == [[([END_TOKEN_ID], False, [None])]] * 8
    assert state["loads"] - state["baseline_loads"] == 1


def test_model_missing_on_last_attempt_still_retries(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    model = MagicMock()
    model.save_pretrained = MagicMock()
    monkeypatch.setattr("grpx.backends.vllm.is_lora_model", lambda candidate: True)
    monkeypatch.setattr("grpx.backends.vllm.time.sleep", lambda seconds: None)

    calls = []

    def handler(method, url, payload, timeout):
        calls.append(url)
        if url.endswith("/v1/load_lora_adapter") or url.endswith("/v1/unload_lora_adapter"):
            return FakeResponse({"ok": True})
        if not url.endswith("/v1/completions"):
            raise AssertionError(url)
        # 前两次超时，第三次模型丢失；同步后第四次成功
        completions = sum(1 for item in calls if item.endswith("/v1/completions"))
        if completions <= 2:
            raise requests.Timeout("timed out")
        if completions == 3:
            return FakeResponse(
                status_code=404,
                text=f'{{"error":{{"message":"adapter {backend.adapter_name} not found"}}}}',
            )
        return FakeResponse({"choices": [make_choice(0, [END_TOKEN_ID])]})

    patch_request(monkeypatch, handler)
    backend.synchronize_policy(model)
    assert backend.generate([10], 1) == [([END_TOKEN_ID], False)]
    assert sum(1 for url in calls if url.endswith("/v1/completions")) == 4


def test_null_logprobs_treated_as_empty_then_append_end(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)

    def handler(method, url, payload, timeout):
        return FakeResponse({"choices": [{
            "index": 0,
            "finish_reason": "stop",
            "logprobs": None,
            "stop_reason": END_TOKEN_ID,
        }]})

    patch_request(monkeypatch, handler)
    assert backend.generate([10], 1) == [([END_TOKEN_ID], False)]


def test_unexpected_token_format_raises_readable_error(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)

    def handler(method, url, payload, timeout):
        return FakeResponse({"choices": [{
            "index": 0,
            "finish_reason": "stop",
            "logprobs": {"tokens": ["▁hello", "token_id:2"]},
        }]})

    patch_request(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="token 格式不符合预期"):
        backend.generate([10], 1)


def _npy_b64(array):
    import base64
    import io
    import numpy as np
    buffer = io.BytesIO()
    np.save(buffer, array)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_record_routing_decodes_each_choice_and_pads_a_local_end_token(tmp_path, monkeypatch):
    import numpy as np
    backend = make_backend(tmp_path)
    backend.record_routing = True
    backend.routing_layer_count = 2
    backend.routing_experts_per_token = 2
    prompt = [10, 11]
    first = np.arange(3 * 2 * 2, dtype=np.uint8).reshape(3, 2, 2)
    second = np.full((3, 2, 2), 7, dtype=np.uint8)

    def handler(method, url, payload, timeout):
        assert payload["n"] == 2
        return FakeResponse({"choices": [
            {**make_choice(1, [9, 8], finish_reason="stop", stop_reason=END_TOKEN_ID),
             "routed_experts": _npy_b64(second)},
            {**make_choice(0, [5, 6], finish_reason="stop", stop_reason=END_TOKEN_ID),
             "routed_experts": _npy_b64(first)},
        ]})

    patch_request(monkeypatch, handler)
    samples = backend.generate_batch([(prompt, 2)])[0]
    assert [sample[0] for sample in samples] == [
        [5, 6, END_TOKEN_ID], [9, 8, END_TOKEN_ID]]
    # 提示词 2 + 服务端回答 2 = 4 行记录，本地补的结束符再占一行 -1
    assert samples[0][4].shape == (5, 2, 2)
    assert int(samples[0][4][0, 0, 0]) == 0
    assert int(samples[0][4][-2, 0, 0]) == -1
    assert int(samples[0][4][-1, 0, 0]) == -1
    assert int(samples[1][4][0, 0, 0]) == 7


def test_record_routing_rejects_a_response_without_the_field(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    backend.record_routing = True
    backend.routing_layer_count = 2
    backend.routing_experts_per_token = 2
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse(
        {"choices": [make_choice(0, [5, END_TOKEN_ID])]}))
    with pytest.raises(RuntimeError, match="routed_experts"):
        backend.generate_batch([([10], 1)])


def test_greedy_does_not_read_routing_even_when_recording_is_enabled(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    backend.record_routing = True
    backend.routing_layer_count = 2
    backend.routing_experts_per_token = 2
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse(
        {"choices": [make_choice(0, [END_TOKEN_ID])]}))
    sample = backend.generate_batch([([10], 1)], greedy=True)[0][0]
    assert len(sample) == 3


def test_probe_stops_when_the_server_omits_routing(tmp_path, monkeypatch):
    from torch import nn

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 2
            self.num_experts = 4
            self.norm_topk_prob = False

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Module()
            self.block.gate = Gate()

    backend = make_backend(tmp_path)
    patch_request(monkeypatch, lambda method, url, payload, timeout: FakeResponse(
        {"choices": [make_choice(0, [5])]}))
    with pytest.raises(RuntimeError, match="routed_experts"):
        backend.probe_routed_experts(Model())
    assert backend.routing_layer_count == 1
    assert backend.routing_experts_per_token == 2


def test_unknown_finish_reason_raises(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)

    def handler(method, url, payload, timeout):
        return FakeResponse({"choices": [{
            "index": 0,
            "finish_reason": "abort",
            "logprobs": {"tokens": ["token_id:5"]},
        }]})

    patch_request(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="finish_reason"):
        backend.generate([10], 1)
