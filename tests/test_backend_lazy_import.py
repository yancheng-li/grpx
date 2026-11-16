"""本地后端不应被 vLLM 依赖（如 requests）拖垮。

在子进程里测导入隔离，避免从 sys.modules 卸模块污染同进程的其他测试。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_huggingface_backend_imports_and_constructs_without_requests():
    script = r"""
import sys
sys.modules["requests"] = None
from grpx.backends.huggingface import HuggingFaceBackend
from tests.test_huggingface_backend import END_TOKEN_ID, FakeModel, make_tokenizer
backend = HuggingFaceBackend(FakeModel([[END_TOKEN_ID]]), make_tokenizer())
assert backend.generate([10], 1) == [([END_TOKEN_ID], False)]
import grpx.backends as backends
assert backends.HuggingFaceBackend is HuggingFaceBackend
assert "grpx.backends.vllm" not in sys.modules
print("ok")
"""
    result = _run_isolated(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_vllm_backend_still_requires_requests_when_accessed():
    script = r"""
import sys
sys.modules["requests"] = None
try:
    from grpx.backends import VllmBackend  # noqa: F401
except ModuleNotFoundError:
    print("ok")
else:
    raise SystemExit("expected ModuleNotFoundError")
"""
    result = _run_isolated(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout
