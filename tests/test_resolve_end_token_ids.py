from types import SimpleNamespace

import pytest

from grpx.backends.common import resolve_end_token_ids


def test_prefers_model_generation_config():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 9]))
    tokenizer = SimpleNamespace(eos_token_id=3)
    assert resolve_end_token_ids(model, tokenizer) == {2, 9}


def test_falls_back_to_tokenizer():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None))
    tokenizer = SimpleNamespace(eos_token_id=3)
    assert resolve_end_token_ids(model, tokenizer) == {3}


def test_single_id_becomes_set():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
    tokenizer = SimpleNamespace(eos_token_id=None)
    assert resolve_end_token_ids(model, tokenizer) == {2}


def test_missing_raises():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None))
    tokenizer = SimpleNamespace(eos_token_id=None)
    with pytest.raises(ValueError, match="eos_token_id"):
        resolve_end_token_ids(model, tokenizer)
