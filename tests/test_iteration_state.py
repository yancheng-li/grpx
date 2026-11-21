"""轨迹落盘与累积中途梯度：中断再恢复的结果相同，指纹不对或写到一半则不能接着用。"""
import random

import pytest
import torch

from grpx.config import TrainingConfig
from grpx.iteration_state import (
    accumulation_checkpoint_path,
    atomic_torch_save,
    load_accumulation_checkpoint,
    load_iteration_trajectories,
    make_fingerprint,
    save_iteration_trajectories,
)
from grpx.loop import accumulate_gradients
from grpx.trainer import action_log_probabilities, build_batch
from grpx.types import Step, Trajectory


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocabulary_size=16, hidden_size=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, vocabulary_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None):
        return type("Out", (), {"logits": self.output_layer(self.embedding(input_ids))})()


def fresh_model():
    torch.manual_seed(0)
    return TinyLanguageModel()


def make_rows():
    specs = [([1, 2], [3], 1.0), ([1, 2], [4], -1.0), ([2, 3], [5], 0.5), ([3, 4], [6], -0.5)]
    return [Trajectory("g", 0, 1.0, steps=[Step(prompt, action, advantage=advantage)])
            for prompt, action, advantage in specs]


def make_config():
    return TrainingConfig(
        learning_rate=1e-3, clip_range=0.2, max_rows_per_minibatch=1,
        max_tokens_per_minibatch=10**6, updates_per_iteration=1, warmup_iterations=1,
    )


def batches_and_old(model, rows):
    batches = [build_batch([row], padding_token_id=0, device="cpu") for row in rows]
    model.eval()
    with torch.no_grad():
        old = [action_log_probabilities(model, batch)[0] for batch in batches]
    return batches, old


def test_interrupted_accumulation_matches_uninterrupted_update(tmp_path, monkeypatch):
    rows = make_rows()
    config = make_config()
    model_full = fresh_model()
    batches, old = batches_and_old(model_full, rows)
    full_metrics = accumulate_gradients(model_full, batches, old, config)
    full_grads = [parameter.grad.detach().clone()
                  for parameter in model_full.parameters() if parameter.requires_grad]

    model_interrupt = fresh_model()
    content_hash = "same-round"
    fingerprint = make_fingerprint(config, 3, content_hash, model_interrupt)
    path = accumulation_checkpoint_path(tmp_path)
    accumulation = {"path": path, "interval": 2, "fingerprint": fingerprint}
    import grpx.loop as loop_module
    real_save = loop_module.save_accumulation_checkpoint

    def save_then_stop(path, fingerprint, next_index, *args, **kwargs):
        real_save(path, fingerprint, next_index, *args, **kwargs)
        if next_index == 2:
            raise RuntimeError("累积到第 2 个小批时中断")

    monkeypatch.setattr(loop_module, "save_accumulation_checkpoint", save_then_stop)
    with pytest.raises(RuntimeError, match="中断"):
        accumulate_gradients(model_interrupt, batches, old, config, accumulation=accumulation)
    assert path.is_file()
    assert not path.with_name(path.name + ".partial").exists()

    model_resumed = fresh_model()
    resumed_fingerprint = make_fingerprint(config, 3, content_hash, model_resumed)
    assert resumed_fingerprint == fingerprint
    monkeypatch.setattr(loop_module, "save_accumulation_checkpoint", real_save)
    resumed_metrics = accumulate_gradients(
        model_resumed, batches, old, config,
        accumulation={"path": path, "interval": 2, "fingerprint": resumed_fingerprint})
    assert resumed_metrics == full_metrics
    resumed_grads = [parameter.grad.detach().clone()
                     for parameter in model_resumed.parameters() if parameter.requires_grad]
    for left, right in zip(full_grads, resumed_grads):
        assert torch.equal(left, right)

    def stepped(model):
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
        torch.nn.utils.clip_grad_norm_(trainable, config.max_gradient_norm)
        optimizer.step()
        return [parameter.detach().clone() for parameter in trainable]

    for left, right in zip(stepped(model_full), stepped(model_resumed)):
        assert torch.equal(left, right)
    assert not path.exists()


def test_accumulation_fingerprint_mismatch_refuses_resume(tmp_path):
    model = fresh_model()
    config = make_config()
    rows = make_rows()
    batches, old = batches_and_old(model, rows)
    fingerprint = make_fingerprint(config, 1, "round", model)
    path = tmp_path / "accumulation.pt"
    accumulate_gradients(
        model, batches, old, config,
        accumulation={"path": path, "interval": 2, "fingerprint": fingerprint})
    # 正常结束会删掉中途文件。这里直接写一份，再拿错的指纹去读。
    from grpx.iteration_state import capture_gradients, save_accumulation_checkpoint
    save_accumulation_checkpoint(
        path, fingerprint, next_index=2, total_weight=1.0, batch_count=4,
        weighted_metrics=[(0.5, {"loss": 0.0})], gradients=capture_gradients(model))
    wrong = make_fingerprint(config, 2, "round", model)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    with pytest.warns(UserWarning, match="轮次不一致"):
        restored = load_accumulation_checkpoint(
            path, wrong, total_weight=1.0, batch_count=4, trainable_names=trainable_names)
    assert restored is None
    assert not path.exists()


def test_interrupted_write_leaves_no_usable_accumulation_state(tmp_path, monkeypatch):
    path = tmp_path / "accumulation.pt"

    def fail_replace(*args, **kwargs):
        raise RuntimeError("写到一半被中断")

    monkeypatch.setattr("grpx.iteration_state.os.replace", fail_replace)
    with pytest.raises(RuntimeError, match="写到一半"):
        atomic_torch_save(path, {"ok": True})
    assert not path.exists()
    partial = path.with_name(path.name + ".partial")
    assert partial.exists()
    assert load_accumulation_checkpoint(
        path, {"config_hash": "x", "iteration": 0, "trajectory_hash": "y",
               "adapter_weight_hash": "z"},
        total_weight=1.0, batch_count=1, trainable_names=[]) is None


def test_saved_trajectories_round_trip_and_reject_weight_mismatch(tmp_path):
    model = fresh_model()
    config = make_config()
    rows = make_rows()
    generator = random.Random(0)
    torch.manual_seed(1)
    generator.random()
    torch.rand(1)
    python_state = generator.getstate()
    torch_state = torch.get_rng_state().clone()
    content_hash = save_iteration_trajectories(
        tmp_path, 4, model, config, rows, rows, {"step_signal_coverage": 0.0},
        [[0], [1], [2], [3]], generator, device="cpu")
    generator.random()
    torch.rand(1)
    loaded = load_iteration_trajectories(
        tmp_path, 4, model, config, generator, device="cpu")
    assert loaded.content_hash == content_hash
    assert loaded.minibatch_plan == [[0], [1], [2], [3]]
    assert [step.advantage for step in loaded.kept[0].steps] == [1.0]
    assert generator.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)

    with torch.no_grad():
        next(model.parameters()).add_(1)
    generator2 = random.Random(5)
    before = generator2.getstate()
    with pytest.warns(UserWarning, match="适配器权重哈希"):
        assert load_iteration_trajectories(
            tmp_path, 4, model, config, generator2, device="cpu") is None
    assert generator2.getstate() == before
