"""多卡更新接到训练循环之后：两个 CPU 副本和单副本的结果应在浮点误差内一致。"""
import copy

import pytest
import torch

from grpx.gradient_checkpointing import gradient_checkpointing
from grpx.loop import update_policy
from grpx.multi_device_update import ReplicaSynchronizationError, trainable_parameter_fingerprint
from tests.test_same_batch_update_path import make_append_only_trajectory, tiny_gpt2

from grpx.config import TrainingConfig


def _kept():
    return [
        make_append_only_trajectory("a", 3, 1.0, start=1),
        make_append_only_trajectory("a", 3, -1.0, start=3),
        make_append_only_trajectory("b", 2, 0.4, start=5),
        make_append_only_trajectory("b", 2, -0.4, start=7),
    ]


def _config(**overrides):
    values = dict(
        learning_rate=1e-3,
        clip_range=0.2,
        length_normalization="per_sequence",
        max_rows_per_minibatch=1,
        max_tokens_per_minibatch=10**6,
        updates_per_iteration=1,
        merge_append_only_turns=True,
        device="cpu",
    )
    values.update(overrides)
    return TrainingConfig(**values)


def _replicas(count, seed=0):
    primary = tiny_gpt2(seed)
    return [primary] + [copy.deepcopy(primary) for _ in range(count - 1)]


def _run(model, kept, config, replicas=None, run_directory=None):
    targets = replicas if replicas is not None else [model]
    if config.gradient_checkpointing:
        for replica in targets:
            gradient_checkpointing(replica, True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    update_policy(
        model, optimizer, kept, 0, config,
        run_directory=run_directory,
        iteration=0,
        trajectory_hash_value="round",
        replicas=replicas,
    )
    return [parameter.detach().clone() for parameter in trainable]


def _assert_close(left, right):
    assert len(left) == len(right)
    for actual, expected in zip(left, right):
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5), (
            f"max abs diff {(actual - expected).abs().max().item()}"
        )


def test_two_replicas_match_one_replica():
    config = _config()
    single = _run(tiny_gpt2(0), copy.deepcopy(_kept()), config)
    replicas = _replicas(2)
    updated = _run(replicas[0], copy.deepcopy(_kept()), config, replicas=replicas)
    _assert_close(updated, single)
    assert trainable_parameter_fingerprint(replicas[0]) == trainable_parameter_fingerprint(replicas[1])


def test_replica_with_no_minibatch_still_matches():
    config = _config(max_rows_per_minibatch=32)
    kept = _kept()
    single = _run(tiny_gpt2(1), copy.deepcopy(kept), config)
    replicas = _replicas(3, seed=1)
    updated = _run(replicas[0], copy.deepcopy(kept), config, replicas=replicas)
    _assert_close(updated, single)
    assert len({trainable_parameter_fingerprint(replica) for replica in replicas}) == 1


def test_missed_broadcast_is_rejected_on_the_next_update():
    config = _config()
    replicas = _replicas(2)
    _run(replicas[0], copy.deepcopy(_kept()), config, replicas=replicas)
    with torch.no_grad():
        next(parameter for parameter in replicas[1].parameters() if parameter.requires_grad).add_(1e-3)
    trainable = [parameter for parameter in replicas[0].parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    with pytest.raises(ReplicaSynchronizationError):
        update_policy(
            replicas[0], optimizer, copy.deepcopy(_kept()), 0, config, replicas=replicas)


def test_checkpointing_and_accumulation_match_single_replica(tmp_path):
    config = _config(
        gradient_checkpointing=True,
        save_iteration_trajectories=True,
        accumulation_checkpoint_interval=1,
    )
    single = _run(
        tiny_gpt2(2), copy.deepcopy(_kept()), config, run_directory=tmp_path / "single")
    replicas = _replicas(2, seed=2)
    updated = _run(
        replicas[0], copy.deepcopy(_kept()), config, replicas=replicas,
        run_directory=tmp_path / "multi")
    _assert_close(updated, single)
    assert trainable_parameter_fingerprint(replicas[0]) == trainable_parameter_fingerprint(replicas[1])
    assert not (tmp_path / "multi" / "accumulation.pt").exists()
