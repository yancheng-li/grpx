import pytest

from grpx.config import TrainingConfig
from grpx.doctor import shared_gpu_memory_hint
from grpx.command_line import ConfigurationError, load_configuration


def test_training_config_defaults_and_rejects_bad_memory_fraction():
    config = TrainingConfig()
    assert config.device == "cuda"
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
