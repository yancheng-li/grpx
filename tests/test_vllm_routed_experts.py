"""假的 vLLM 响应对齐到训练用的 int8 记录。不需要显卡。"""
import base64
import io

import numpy as np
import pytest
import torch

from grpx.routing_replay import decode_vllm_routed_experts


def encode(array):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_decoded_rows_match_tokens_fed_to_the_model_and_the_last_is_missing():
    # 提示词 4、回答 3：服务端 6 行，训练记录 7 行，最后一行是 -1
    source = (np.arange(6 * 16 * 8) % 64).astype(np.uint8).reshape(6, 16, 8)
    routing = decode_vllm_routed_experts(encode(source), 4, 3, 16, 8)
    assert routing.dtype == torch.int8
    assert routing.shape == (7, 16, 8)
    assert torch.equal(routing[:6].long(), torch.as_tensor(source).long())
    assert torch.all(routing[6] == -1)


def test_locally_appended_end_token_is_also_unrecorded():
    source = np.zeros((2, 2, 2), dtype=np.uint8)
    source[0] = 1
    source[1] = 4
    routing = decode_vllm_routed_experts(
        encode(source), prompt_length=1, action_length=2, layer_count=2,
        experts_per_token=2, extra_unrecorded=1)
    assert routing.shape == (4, 2, 2)
    assert torch.all(routing[2] == -1)
    assert torch.all(routing[3] == -1)


def test_expert_ids_at_most_127_stay_non_negative():
    source = np.array([[[63, 0]]], dtype=np.uint8)
    routing = decode_vllm_routed_experts(encode(source), 1, 1, 1, 2)
    assert int(routing[0, 0, 0]) == 63
    assert torch.all(routing[1] == -1)


def test_expert_id_128_is_rejected_before_it_becomes_negative():
    source = np.array([[[128]]], dtype=np.uint16)
    with pytest.raises(ValueError, match="127"):
        decode_vllm_routed_experts(encode(source), 1, 1, 1, 1)


def test_shape_must_match_layers_experts_and_fed_tokens():
    source = np.zeros((5, 16, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="16"):
        decode_vllm_routed_experts(encode(source), 4, 2, 15, 8)
    with pytest.raises(ValueError, match="8"):
        decode_vllm_routed_experts(encode(source), 4, 2, 16, 4)
    with pytest.raises(ValueError, match="5 行"):
        decode_vllm_routed_experts(encode(source), 4, 3, 16, 8)


def test_missing_field_is_an_error():
    with pytest.raises(RuntimeError, match="routed_experts"):
        decode_vllm_routed_experts(None, 4, 2, 16, 8)


def test_pickled_payload_is_not_executed():
    buffer = io.BytesIO()
    np.save(buffer, np.array([{"not": "a routing record"}], dtype=object))
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    with pytest.raises(RuntimeError, match="pickle"):
        decode_vllm_routed_experts(encoded, 1, 1, 1, 1)
