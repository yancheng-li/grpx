"""梯度检查点：训练模式才生效，冻结基座上的 LoRA 仍有梯度，无梯度前向不启用。"""
import torch
import pytest
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from grpx.config import TrainingConfig
from grpx.gradient_checkpointing import gradient_bearing_forward, gradient_checkpointing
from grpx.loop import update_policy
from grpx.trainer import action_log_probabilities, build_batch
from grpx.types import Step, Trajectory


def tiny_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        eos_token_id=0, bos_token_id=0,
    ))


def one_batch():
    rows = [
        Trajectory("g", 0, 1.0, steps=[Step([1, 2, 3], [4, 5], advantage=1.0)]),
        Trajectory("g", 0, 1.0, steps=[Step([1, 2, 3], [6, 7], advantage=-1.0)]),
    ]
    return build_batch(rows, padding_token_id=0, device="cpu")


class RecordingModel:
    """只提供新名字。查询旧属性应直接失败，避免 transformers 6 的弃用警告。"""

    def __init__(self):
        self.enabled = False
        self.kwargs = None

    def activation_checkpointing_enable(self, activation_checkpointing_kwargs=None):
        self.enabled = True
        self.kwargs = activation_checkpointing_kwargs

    def activation_checkpointing_disable(self):
        self.enabled = False

    @property
    def is_activation_checkpointing(self):
        return self.enabled

    @property
    def is_gradient_checkpointing(self):
        raise AssertionError("不应再查询 is_gradient_checkpointing")

    def gradient_checkpointing_enable(self, **kwargs):
        raise AssertionError("有新接口时不应再调用 gradient_checkpointing_enable")


def test_enable_and_query_use_the_current_names_and_non_reentrant():
    model = RecordingModel()
    assert gradient_checkpointing(model, True) is True
    assert model.kwargs == {"use_reentrant": False}
    assert gradient_checkpointing(model) is True
    assert gradient_checkpointing(model, False) is False
    assert gradient_checkpointing(model) is False


def test_enable_rejects_models_without_the_api():
    with pytest.raises(ValueError, match="梯度检查点"):
        gradient_checkpointing(torch.nn.Linear(2, 2), True)


def test_gradient_bearing_forward_restores_eval_mode():
    model = torch.nn.Linear(2, 2)
    model.eval()
    with gradient_bearing_forward(model):
        assert model.training
    assert not model.training


def test_checkpointing_runs_only_during_the_gradient_forward():
    model = tiny_gpt2()
    batch = one_batch()
    gradient_checkpointing(model, True)
    calls = {"n": 0}
    for block in model.transformer.h:
        assert block._gradient_checkpointing_func.keywords["use_reentrant"] is False
        real = block._gradient_checkpointing_func

        def wrapped(*args, _real=real, **kwargs):
            calls["n"] += 1
            return _real(*args, **kwargs)

        block._gradient_checkpointing_func = wrapped

    model.eval()
    with torch.no_grad():
        action_log_probabilities(model, batch)
    assert calls["n"] == 0

    with gradient_bearing_forward(model):
        log_probabilities, _mask = action_log_probabilities(model, batch)
        log_probabilities.sum().backward()
    assert calls["n"] > 0
    assert not model.training


def test_frozen_base_lora_gradients_are_nonzero_without_input_grad_hook():
    torch.manual_seed(0)
    base = tiny_gpt2()
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and "lora_B" in name:
                parameter.add_(0.02)
    gradient_checkpointing(model, True)
    inner = model.get_base_model()
    if getattr(inner, "_require_grads_hook", None) is not None:
        inner.disable_input_require_grads()
    batch = one_batch()
    model.eval()
    with torch.no_grad():
        old = action_log_probabilities(model, batch)[0]
    from grpx.loop import accumulate_gradients
    accumulate_gradients(
        model, [batch], [old],
        TrainingConfig(learning_rate=1e-3, max_rows_per_minibatch=4,
                       max_tokens_per_minibatch=10**6))
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    assert trainable
    for name, parameter in trainable:
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum().item() > 0, name


def test_first_step_ratio_stays_one_with_checkpointing():
    model = tiny_gpt2()
    gradient_checkpointing(model, True)
    model.eval()
    kept = [
        Trajectory("g", 0, 1.0, steps=[Step([1, 2], [3, 4], advantage=1.0)]),
        Trajectory("g", 0, 1.0, steps=[Step([1, 2], [5, 6], advantage=-1.0)]),
    ]
    config = TrainingConfig(
        learning_rate=1e-3, clip_range=0.2, max_rows_per_minibatch=8,
        max_tokens_per_minibatch=10**6, updates_per_iteration=1,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate)
    history, *_rest = update_policy(model, optimizer, kept, 0, config)
    assert history[0]["max_ratio_deviation"] == pytest.approx(0, abs=1e-5)


def test_first_step_ratio_sees_train_mode_dropout():
    """预检的首步 ratio 来自 update_policy：旧概率在评测模式，新概率在带梯度的训练模式。"""
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0.5, embd_pdrop=0.5, attn_pdrop=0.5,
        eos_token_id=0, bos_token_id=0,
    ))
    gradient_checkpointing(model, True)
    model.eval()
    kept = [
        Trajectory("g", 0, 1.0, steps=[Step([1, 2, 3], [4, 5, 6], advantage=1.0)]),
        Trajectory("g", 0, 1.0, steps=[Step([1, 2, 3], [7, 8, 9], advantage=-1.0)]),
    ]
    config = TrainingConfig(
        learning_rate=1e-3, clip_range=0.2, max_rows_per_minibatch=8,
        max_tokens_per_minibatch=10**6, updates_per_iteration=1,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate)
    history, *_rest = update_policy(model, optimizer, kept, 0, config)
    assert history[0]["max_ratio_deviation"] > 1e-3
