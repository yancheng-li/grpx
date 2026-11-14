"""本地后端不应被 vLLM 依赖（如 requests）拖垮。"""
import sys

import pytest

from tests.test_huggingface_backend import END_TOKEN_ID, FakeModel, make_tokenizer


def _purge_backend_modules():
    for name in list(sys.modules):
        if name == "grpx.backends" or name.startswith("grpx.backends."):
            del sys.modules[name]


def test_huggingface_backend_imports_and_constructs_without_requests(monkeypatch):
    # 屏蔽 requests：已加载的也清掉，未加载的则阻止导入
    monkeypatch.setitem(sys.modules, "requests", None)
    _purge_backend_modules()

    from grpx.backends.huggingface import HuggingFaceBackend

    backend = HuggingFaceBackend(FakeModel([[END_TOKEN_ID]]), make_tokenizer())
    assert backend.generate([10], 1) == [([END_TOKEN_ID], False)]

    # 包入口仍可拿到本地后端；此时不应已加载 vllm 子模块
    import grpx.backends as backends

    assert backends.HuggingFaceBackend is HuggingFaceBackend
    assert "grpx.backends.vllm" not in sys.modules


def test_vllm_backend_still_requires_requests_when_accessed(monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    _purge_backend_modules()

    with pytest.raises(ModuleNotFoundError):
        from grpx.backends import VllmBackend  # noqa: F401  触发惰性导入
