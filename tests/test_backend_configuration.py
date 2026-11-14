import pytest

from grpx.command_line import ConfigurationError, load_configuration


MINIMAL = """
[model]
name = "some-model"

[task]
dataset = "examples.guess_number:make_dataset"
environment = "examples.guess_number:GuessNumberEnvironment"
"""


def write_configuration(tmp_path, text=MINIMAL):
    path = tmp_path / "configuration.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_vllm_rejects_huggingface_batch_keys(tmp_path):
    with pytest.raises(ConfigurationError, match="max_rows_per_batch 只对本地后端有效"):
        load_configuration(write_configuration(tmp_path), [
            "backend.type=vllm",
            "backend.base_url=\"http://localhost:8000\"",
            "backend.max_rows_per_batch=8",
        ])


def test_huggingface_rejects_vllm_keys(tmp_path):
    with pytest.raises(ConfigurationError, match="max_concurrent_requests 只对 vLLM 后端有效"):
        load_configuration(write_configuration(tmp_path), [
            "backend.max_concurrent_requests=16",
        ])


def test_vllm_accepts_its_own_keys(tmp_path):
    configuration = load_configuration(write_configuration(tmp_path), [
        "backend.type=vllm",
        "backend.base_url=\"http://localhost:8000\"",
        "backend.max_concurrent_requests=16",
        "backend.adapter_path_local_prefix=\"D:/projects\"",
        "backend.adapter_path_server_prefix=\"/mnt/d/projects\"",
    ])
    assert configuration["backend"]["type"] == "vllm"
    assert configuration["backend"]["max_concurrent_requests"] == 16


def test_unknown_backend_type(tmp_path):
    with pytest.raises(ConfigurationError, match="未知的后端类型"):
        load_configuration(write_configuration(tmp_path), ["backend.type=llama_cpp"])
