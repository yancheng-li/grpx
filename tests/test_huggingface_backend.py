from types import SimpleNamespace

import pytest
import torch

from grpx.backends.huggingface import HuggingFaceBackend

END_TOKEN_ID, PADDING_TOKEN_ID = 2, 0

class FakeModel:
    def __init__(self, completions, eos_token_id=END_TOKEN_ID, training=False):
        self.completions = completions
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self.device = torch.device("cpu")
        self.training = training
        self.calls = []
        self.inputs = []
        self.training_during_generate = []

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def generate(self, input_ids, **keyword_arguments):
        """输入几行就返回几行：第 i 行接上 completions[i]"""
        self.calls.append(keyword_arguments)
        self.inputs.append(input_ids.tolist())
        self.training_during_generate.append(self.training)
        rows = [row + self.completions[row_index] for row_index, row in enumerate(input_ids.tolist())]
        return torch.tensor(rows)

class EchoModel(FakeModel):
    """每行回显提示词的最后一个 token 再接结束符，用来核对结果是否回到了自己的请求"""
    def __init__(self, **keyword_arguments):
        super().__init__([], **keyword_arguments)

    def generate(self, input_ids, **keyword_arguments):
        self.calls.append(keyword_arguments)
        self.inputs.append(input_ids.tolist())
        return torch.tensor([row + [row[-1], END_TOKEN_ID] for row in input_ids.tolist()])

class FailingModel(FakeModel):
    def generate(self, input_ids, **keyword_arguments):
        raise RuntimeError("generate failed")

def make_tokenizer(pad_token_id=PADDING_TOKEN_ID, eos_token_id=None):
    return SimpleNamespace(pad_token_id=pad_token_id, eos_token_id=eos_token_id)

def test_splits_on_end_token_and_strips_padding():
    model = FakeModel([
        [5, 6, END_TOKEN_ID, PADDING_TOKEN_ID, PADDING_TOKEN_ID],
        [7, END_TOKEN_ID, PADDING_TOKEN_ID, PADDING_TOKEN_ID, PADDING_TOKEN_ID],
    ])
    results = HuggingFaceBackend(model, make_tokenizer()).generate([10, 11], 2)
    assert results == [([5, 6, END_TOKEN_ID], False), ([7, END_TOKEN_ID], False)]

def test_first_token_is_end_token():
    model = FakeModel([[END_TOKEN_ID, PADDING_TOKEN_ID, PADDING_TOKEN_ID]])
    assert HuggingFaceBackend(model, make_tokenizer()).generate([10], 1) == [([END_TOKEN_ID], False)]

def test_no_end_token_is_truncated():
    model = FakeModel([[5, 6, 7], [5, 6, END_TOKEN_ID]])
    results = HuggingFaceBackend(model, make_tokenizer()).generate([10], 2)
    assert results == [([5, 6, 7], True), ([5, 6, END_TOKEN_ID], False)]

def test_multiple_end_token_ids():
    model = FakeModel([[5, 9, PADDING_TOKEN_ID], [5, END_TOKEN_ID, PADDING_TOKEN_ID]], eos_token_id=[END_TOKEN_ID, 9])
    results = HuggingFaceBackend(model, make_tokenizer()).generate([10], 2)
    assert results == [([5, 9], False), ([5, END_TOKEN_ID], False)]

def test_end_token_falls_back_to_tokenizer():
    model = FakeModel([[5, 3, PADDING_TOKEN_ID]], eos_token_id=None)
    backend = HuggingFaceBackend(model, make_tokenizer(eos_token_id=3))
    assert backend.end_token_ids == {3}
    assert backend.generate([10], 1) == [([5, 3], False)]

def test_missing_end_token_raises():
    with pytest.raises(ValueError):
        HuggingFaceBackend(FakeModel([], eos_token_id=None), make_tokenizer())

@pytest.mark.parametrize("temperature", [0, -1.0, float("nan")])
def test_non_positive_temperature_raises(temperature):
    with pytest.raises(ValueError, match="temperature"):
        HuggingFaceBackend(FakeModel([]), make_tokenizer(), temperature=temperature)

def test_padding_falls_back_to_end_token():
    model = FakeModel([[5, END_TOKEN_ID, END_TOKEN_ID]])
    HuggingFaceBackend(model, make_tokenizer(pad_token_id=None)).generate([10], 1)
    assert model.calls[0]["generation_config"].pad_token_id == END_TOKEN_ID

def test_generation_config_is_pure_sampling():
    model = FakeModel([[5, END_TOKEN_ID]])
    HuggingFaceBackend(model, make_tokenizer(), max_new_tokens=7, temperature=0.8).generate([10, 11], 1)
    keyword_arguments = model.calls[0]
    generation_config = keyword_arguments["generation_config"]
    assert keyword_arguments["attention_mask"].tolist() == [[1, 1]]
    assert (generation_config.do_sample, generation_config.temperature) == (True, 0.8)
    assert (generation_config.top_p, generation_config.top_k, generation_config.repetition_penalty) == (1.0, 0, 1.0)
    assert (generation_config.max_new_tokens, generation_config.num_return_sequences) == (7, 1)

def test_generate_greedy_returns_single_pair():
    model = FakeModel([[5, END_TOKEN_ID, PADDING_TOKEN_ID]])
    result = HuggingFaceBackend(model, make_tokenizer(), max_new_tokens=7).generate_greedy([10, 11])
    assert result == ([5, END_TOKEN_ID], False)
    keyword_arguments = model.calls[0]
    generation_config = keyword_arguments["generation_config"]
    assert generation_config.do_sample is False
    assert (generation_config.num_return_sequences, generation_config.max_new_tokens) == (1, 7)
    assert generation_config.repetition_penalty == 1.0
    assert model.training_during_generate == [False]

@pytest.mark.parametrize("initially_training", [True, False])
def test_generates_in_eval_mode_and_restores(initially_training):
    model = FakeModel([[5, END_TOKEN_ID]], training=initially_training)
    HuggingFaceBackend(model, make_tokenizer()).generate([10], 1)
    assert model.training_during_generate == [False]
    assert model.training is initially_training

def test_left_padding_layout():
    model = EchoModel()
    HuggingFaceBackend(model, make_tokenizer()).generate_batch([([10, 11, 12], 1), ([20], 1)])
    keyword_arguments = model.calls[0]
    # 按提示词长度升序排列，所以短的那条在第一行；它靠右对齐，左边是填充
    assert model.inputs[0] == [[PADDING_TOKEN_ID, PADDING_TOKEN_ID, 20], [10, 11, 12]]
    assert keyword_arguments["attention_mask"].tolist() == [[0, 0, 1], [1, 1, 1]]

def test_results_return_to_their_requests():
    requests = [([1, 2, 3, 4, 5, 6, 7], 2), ([8], 3), ([9, 10, 11, 12], 1), ([13, 14], 4)]
    backend = HuggingFaceBackend(EchoModel(), make_tokenizer(), max_new_tokens=2,
                                 max_rows_per_batch=3)
    results = backend.generate_batch(requests)
    assert [len(outputs) for outputs in results] == [2, 3, 1, 4]
    for (prompt_token_ids, _), outputs in zip(requests, results):
        assert outputs == [([prompt_token_ids[-1], END_TOKEN_ID], False)] * len(outputs)

def test_generate_and_generate_greedy_go_through_batch():
    model = EchoModel()
    backend = HuggingFaceBackend(model, make_tokenizer())
    assert backend.generate([4, 5], 3) == [([5, END_TOKEN_ID], False)] * 3
    assert backend.generate_greedy([6]) == ([6, END_TOKEN_ID], False)
    assert len(model.inputs[0]) == 3 and model.calls[0]["generation_config"].do_sample is True
    assert model.calls[1]["generation_config"].do_sample is False

def test_empty_request_list():
    model = EchoModel()
    assert HuggingFaceBackend(model, make_tokenizer()).generate_batch([]) == []
    assert model.calls == []

@pytest.mark.parametrize("max_rows, max_tokens", [(1, 10**6), (3, 10**6), (64, 40), (4, 30)])
def test_generation_chunks_respect_limits(max_rows, max_tokens):
    max_new_tokens = 5
    requests = [([1] * length, count) for length, count in [(3, 2), (12, 1), (7, 3), (1, 4), (20, 1)]]
    model = EchoModel()
    backend = HuggingFaceBackend(model, make_tokenizer(), max_new_tokens=max_new_tokens,
                                 max_rows_per_batch=max_rows, max_tokens_per_batch=max_tokens)
    backend.generate_batch(requests)
    assert sum(len(chunk) for chunk in model.inputs) == sum(count for _, count in requests)
    for chunk in model.inputs:
        padded_tokens = len(chunk) * (len(chunk[0]) + max_new_tokens)
        assert len(chunk) <= max_rows
        assert padded_tokens <= max_tokens or len(chunk) == 1

def test_oversized_prompt_gets_its_own_chunk():
    model = EchoModel()
    backend = HuggingFaceBackend(model, make_tokenizer(), max_new_tokens=4, max_tokens_per_batch=20)
    backend.generate_batch([([1] * 3, 2), ([1] * 30, 1)])
    assert [len(chunk) for chunk in model.inputs] == [2, 1]

def test_restores_training_mode_when_generate_fails():
    model = FailingModel([], training=True)
    with pytest.raises(RuntimeError):
        HuggingFaceBackend(model, make_tokenizer()).generate([10], 1)
    assert model.training is True
