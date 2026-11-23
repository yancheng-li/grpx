import pytest

from grpx.config import TrainingConfig
from grpx.doctor import shared_gpu_memory_hint
from grpx.command_line import ConfigurationError, load_configuration


def test_training_config_defaults_and_rejects_bad_memory_fraction():
    config = TrainingConfig()
    assert config.device == "cuda"
    assert config.devices == ("cuda",)
    assert config.memory_fraction == 0.6
    with pytest.raises(ValueError, match="memory_fraction"):
        TrainingConfig(memory_fraction=0)
    with pytest.raises(ValueError, match="memory_fraction"):
        TrainingConfig(memory_fraction=1.5)
    with pytest.raises(ValueError, match="device"):
        TrainingConfig(device="")


def test_recovery_flags_default_off_and_accumulation_needs_trajectories():
    config = TrainingConfig()
    assert config.gradient_checkpointing is False
    assert config.save_iteration_trajectories is False
    assert config.accumulation_checkpoint_interval == 0
    TrainingConfig(accumulation_checkpoint_interval=0, save_iteration_trajectories=False)
    TrainingConfig(accumulation_checkpoint_interval=2, save_iteration_trajectories=True)
    with pytest.raises(ValueError, match="save_iteration_trajectories"):
        TrainingConfig(accumulation_checkpoint_interval=2)
    with pytest.raises(ValueError, match="accumulation_checkpoint_interval"):
        TrainingConfig(accumulation_checkpoint_interval=-1)
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        TrainingConfig(gradient_checkpointing="yes")


def test_devices_list_keeps_device_as_shorthand_for_the_first():
    config = TrainingConfig(device="cpu")
    assert config.devices == ("cpu",)
    config = TrainingConfig(devices=["cuda:0", "cuda:1"])
    assert config.device == "cuda:0"
    assert config.devices == ("cuda:0", "cuda:1")
    TrainingConfig(device="cuda:0", devices=["cuda:0", "cuda:1"])
    with pytest.raises(ValueError, match="重复"):
        TrainingConfig(devices=["cuda:0", "cuda:0"])
    with pytest.raises(ValueError, match="简写"):
        TrainingConfig(device="cpu", devices=["cuda:0", "cuda:1"])
    with pytest.raises(ValueError, match="devices"):
        TrainingConfig(devices="cuda:0")


def test_same_device_switch_does_not_loosen_the_precheck(monkeypatch):
    from grpx.config import SAME_DEVICE_REPLICAS_ENV

    monkeypatch.setenv(SAME_DEVICE_REPLICAS_ENV, "1")
    config = TrainingConfig(devices=["cuda:0", "cuda:0"])
    assert config.devices == ("cuda:0", "cuda:0")
    from grpx.doctor import judge_training_devices

    rejected = judge_training_devices([
        {"name": "cuda:0", "status": "ok", "key": "cuda:0", "index": 0},
        {"name": "cuda:0", "status": "ok", "key": "cuda:0", "index": 0},
    ])
    assert any(item.status == "失败" and "同一张卡" in item.detail for item in rejected)
    monkeypatch.delenv(SAME_DEVICE_REPLICAS_ENV)
    with pytest.raises(ValueError, match="重复"):
        TrainingConfig(devices=["cuda:0", "cuda:0"])


def test_training_device_check_rejects_duplicates_and_inference_overlap():
    from grpx.doctor import judge_training_devices, parse_nvidia_smi_vllm_indices

    ok = judge_training_devices([
        {"name": "cuda:0", "status": "ok", "key": "cuda:0", "index": 0},
        {"name": "cuda:1", "status": "ok", "key": "cuda:1", "index": 1},
    ])
    assert ok[-1].status == "通过"
    duplicated = judge_training_devices([
        {"name": "cuda:0", "status": "ok", "key": "cuda:0", "index": 0},
        {"name": "cuda", "status": "ok", "key": "cuda:0", "index": 0},
    ])
    assert any(item.status == "失败" and "同一张卡" in item.detail for item in duplicated)
    missing = judge_training_devices([
        {"name": "cuda:3", "status": "missing", "index": 3, "count": 1},
    ])
    assert missing[0].status == "失败"
    overlap = judge_training_devices(
        [
            {"name": "cuda:0", "status": "ok", "key": "cuda:0", "index": 0},
            {"name": "cuda:1", "status": "ok", "key": "cuda:1", "index": 1},
        ],
        inference_indices={1},
    )
    assert any(item.status == "失败" and "推理服务" in item.detail for item in overlap)
    shared_one_card = judge_training_devices(
        [{"name": "cuda:0", "status": "ok", "key": "cuda:0", "index": 0}],
        inference_indices={0},
    )
    assert all(item.status == "通过" for item in shared_one_card)
    unknown = judge_training_devices(
        [
            {"name": "cuda:0", "status": "ok", "key": "cuda:0", "index": 0},
            {"name": "cuda:1", "status": "ok", "key": "cuda:1", "index": 1},
        ],
        inference_known=False,
    )
    assert any(item.status == "警告" for item in unknown)
    assert parse_nvidia_smi_vllm_indices(
        "0, GPU-aaa\n1, GPU-bbb\n",
        "GPU-bbb, VLLM::EngineCore\nGPU-aaa, python\n",
    ) == {1}


def test_shared_gpu_hint_only_when_fraction_below_threshold():
    assert shared_gpu_memory_hint(0.6) is not None
    assert shared_gpu_memory_hint(0.95) is None


def test_base_url_list_accepted_in_configuration(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        """
[model]
name = "m"

[task]
dataset = "examples.guess_number:make_dataset"
environment = "examples.guess_number:GuessNumberEnvironment"

[backend]
type = "vllm"
base_url = ["http://localhost:8000", "http://localhost:8001"]
""",
        encoding="utf-8",
    )
    configuration = load_configuration(path, [])
    assert configuration["backend"]["base_url"] == [
        "http://localhost:8000", "http://localhost:8001",
    ]


def test_base_url_empty_list_rejected(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        """
[model]
name = "m"

[task]
dataset = "examples.guess_number:make_dataset"
environment = "examples.guess_number:GuessNumberEnvironment"

[backend]
type = "vllm"
base_url = []
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="不能为空"):
        load_configuration(path, [])
