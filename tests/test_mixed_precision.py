"""混合精度：可训练参数 float32，logits float32。"""
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from grpx.trainer import action_log_probabilities, build_batch
from grpx.types import Step, Trajectory


def make_tiny_gpt2(dtype=torch.float32):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_positions=32, n_embd=32, n_layer=1, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        eos_token_id=0, bos_token_id=0))
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    return get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]))


def test_trainable_parameters_are_float32_when_base_is_bfloat16():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA 测 bfloat16")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("当前 GPU 不支持 bfloat16")
    model = make_tiny_gpt2(dtype=torch.bfloat16).cuda()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.dtype == torch.float32 for parameter in trainable)


def test_action_logits_are_float32_under_mixed_precision():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("当前 GPU 不支持 bfloat16")
    model = make_tiny_gpt2(dtype=torch.bfloat16).cuda()
    model.grpx_mixed_precision = True
    batch = build_batch(
        [Trajectory("g", 0, 1.0, [Step([1, 2, 3], [4, 5], advantage=1.0)])],
        padding_token_id=0, device="cuda",
    )
    log_probs, mask = action_log_probabilities(model, batch)
    assert log_probs.dtype == torch.float32
    assert mask.any()
