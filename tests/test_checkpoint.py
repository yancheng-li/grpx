import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch

from grpx.checkpoint import (find_latest_checkpoint, load_checkpoint, save_checkpoint,
                              truncate_metrics)
from grpx.command_line import ConfigurationError, load_resume_configuration
from grpx.config import TrainingConfig
from grpx.environment import SingleTurnEnvironment
from grpx.loop import train


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
        self.saved_directories = []

    def save_pretrained(self, directory):
        self.saved_directories.append(directory)


def make_optimizer_and_scheduler(model, learning_rate=1e-3, warmup=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup))
    return optimizer, scheduler


def test_load_checkpoint_accepts_legacy_cuda_rng_list(tmp_path):
    """旧检查点 cuda_random_states 是 get_rng_state_all() 的列表，恢复时按设备序号取。"""
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = RecordingModel().to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    random_generator = random.Random(0)
    loss = model.layer(torch.randn(4, 2, device=device)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint_directory = save_checkpoint(
        tmp_path, 3, model, optimizer, scheduler, random_generator, device=device)
    state_path = checkpoint_directory / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    # 写成旧格式：整卡列表
    legacy = list(torch.cuda.get_rng_state_all())
    state["cuda_random_states"] = legacy
    torch.save(state, state_path)

    other = RecordingModel().to(device)
    other_optimizer, other_scheduler = make_optimizer_and_scheduler(other)
    other_random = random.Random(99)
    torch.cuda.manual_seed_all(99)
    load_checkpoint(
        checkpoint_directory, other, other_optimizer, other_scheduler, other_random,
        device=device)
    restored = torch.cuda.get_rng_state(device)
    expected = legacy[torch.cuda.current_device()]
    assert torch.equal(restored.cpu(), expected.cpu())


def test_save_and_load_restores_everything(tmp_path):
    torch.manual_seed(0)
    model = RecordingModel()
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    random_generator = random.Random(0)
    # 让优化器和调度器、随机状态都离开初值
    loss = model.layer(torch.randn(4, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    random_generator.random()
    torch.rand(3)

    before_parameters = {name: parameter.detach().clone()
                         for name, parameter in model.named_parameters()}
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_learning_rate = scheduler.get_last_lr()[0]
    before_python = random_generator.getstate()
    before_torch = torch.get_rng_state()

    checkpoint_directory = save_checkpoint(tmp_path, 10, model, optimizer, scheduler, random_generator)
    assert checkpoint_directory == tmp_path / "checkpoint-0010"
    # save_pretrained 先写到 .partial，改名后正式目录在，.partial 不再存在
    assert model.saved_directories == [tmp_path / "checkpoint-0010.partial"]
    assert checkpoint_directory.is_dir() and not (tmp_path / "checkpoint-0010.partial").exists()

    # 弄乱所有状态后再加载
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    optimizer.zero_grad()
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    random_generator.seed(99)
    torch.manual_seed(99)

    other = RecordingModel()
    other.load_state_dict(model.state_dict())
    other_optimizer, other_scheduler = make_optimizer_and_scheduler(other)
    other_random = random.Random(99)
    completed, _horizon = load_checkpoint(
        checkpoint_directory, other, other_optimizer, other_scheduler, other_random)

    assert completed == 10
    for name, parameter in other.named_parameters():
        assert torch.equal(parameter, before_parameters[name])
    assert other_optimizer.state_dict()["state"].keys() == before_optimizer["state"].keys()
    for index in before_optimizer["state"]:
        for key, value in before_optimizer["state"][index].items():
            loaded = other_optimizer.state_dict()["state"][index][key]
            assert torch.equal(loaded, value) if torch.is_tensor(value) else loaded == value
    assert other_scheduler.get_last_lr()[0] == before_learning_rate
    assert other_random.getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_save_overwrites_same_iteration_atomically(tmp_path):
    model = RecordingModel()
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    random_generator = random.Random(0)
    first = save_checkpoint(tmp_path, 10, model, optimizer, scheduler, random_generator)
    second = save_checkpoint(tmp_path, 10, model, optimizer, scheduler, random_generator)
    assert first == second == tmp_path / "checkpoint-0010"
    assert (first / "training_state.pt").exists()
    assert list(tmp_path.glob("checkpoint-*.partial")) == []


def test_find_latest_ignores_partial_and_picks_highest(tmp_path):
    model = RecordingModel()
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    random_generator = random.Random(0)
    save_checkpoint(tmp_path, 5, model, optimizer, scheduler, random_generator)
    save_checkpoint(tmp_path, 20, model, optimizer, scheduler, random_generator)
    partial = tmp_path / "checkpoint-0030.partial"
    partial.mkdir()
    (partial / "training_state.pt").write_bytes(b"not a real checkpoint")
    empty = tmp_path / "checkpoint-0040"
    empty.mkdir()    # 没有 training_state.pt
    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-0020"
    assert find_latest_checkpoint(tmp_path / "empty") is None


def test_truncate_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps({"iteration": index, "loss": index}) + "\n"
                             for index in (0, 1, 2, 3, 4, 5)), encoding="utf-8")
    assert truncate_metrics(path, 3) == 3
    kept = [json.loads(line)["iteration"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert kept == [0, 1, 2]


def test_truncate_metrics_missing_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    assert truncate_metrics(path, 1) == 0
    assert path.read_text(encoding="utf-8") == ""


def test_load_rejects_mismatched_trainable_parameters(tmp_path):
    model = RecordingModel()
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    random_generator = random.Random(0)
    checkpoint_directory = save_checkpoint(tmp_path, 1, model, optimizer, scheduler, random_generator)

    class BiggerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
            self.extra = torch.nn.Linear(2, 2)

        def save_pretrained(self, directory):
            pass

    bigger = BiggerModel()
    bigger_optimizer, bigger_scheduler = make_optimizer_and_scheduler(bigger)
    with pytest.raises(ValueError, match="可训练参数与当前模型不一致"):
        load_checkpoint(checkpoint_directory, bigger, bigger_optimizer, bigger_scheduler, random.Random(0))


class TinyLanguageModel(torch.nn.Module):
    def __init__(self, vocabulary_size=16, hidden_size=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocabulary_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, vocabulary_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.output_layer(self.embedding(input_ids)))

    def save_pretrained(self, directory):
        torch.save(self.state_dict(), directory / "weights.pt")


class DeterministicBackend:
    """回答 token 由 torch 随机数决定，用来检查恢复后采样序列是否一致"""
    temperature = 1.0
    end_token_ids = {0}

    def synchronize_policy(self, model):
        return

    def generate_batch(self, requests, greedy=False):
        return [[([int(torch.randint(1, 10, (1,)).item())], False) for _ in range(request[1])]
                for request in requests]


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return messages[0]["content"]

    def __call__(self, text, add_special_tokens):
        return SimpleNamespace(input_ids=[ord(character) % 15 + 1 for character in text] or [1])

    def decode(self, token_ids, skip_special_tokens):
        return str(token_ids[0])


def reward_from_action(prompt, completion, sample):
    return float(int(completion) % 3) / 2


def run_tiny_training(tmp_path, name, iterations, resume_from=None, checkpoint_interval=2):
    torch.manual_seed(0)
    model = TinyLanguageModel()
    if resume_from is not None:
        # 恢复路径会覆盖这些参数；这里故意弄乱，确认检查点能拉回来
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(0.0).add_(1.0)
    config = TrainingConfig(
        iterations=iterations, groups_per_iteration=2, group_size=2,
        max_rows_per_minibatch=4, max_tokens_per_minibatch=10**6,
        warmup_iterations=2, checkpoint_interval=checkpoint_interval,
        evaluation_interval=10, max_steps=1, seed=0,
        output_root=str(tmp_path), run_name=name,
    )
    dataset = [{"prompt": f"q{index}", "answer": index} for index in range(20)]
    return train(model, FakeTokenizer(), DeterministicBackend(), dataset,
                 lambda: SingleTurnEnvironment(reward_from_action), config,
                 resume_from=resume_from), model


def test_resume_matches_uninterrupted_run(tmp_path):
    unbroken_directory, unbroken_model = run_tiny_training(tmp_path, "unbroken", iterations=4)
    first_directory, _ = run_tiny_training(tmp_path, "resumed", iterations=2)
    resumed_directory, resumed_model = run_tiny_training(
        tmp_path, "resumed", iterations=4, resume_from=first_directory)

    assert resumed_directory == first_directory
    for left, right in zip(unbroken_model.parameters(), resumed_model.parameters()):
        assert torch.equal(left, right)

    def records(path):
        return [json.loads(line) for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]

    unbroken, resumed = records(unbroken_directory), records(resumed_directory)
    assert [record["iteration"] for record in unbroken] == [record["iteration"] for record in resumed]
    timing_keys = {
        "iteration_seconds", "sampling_seconds", "metrics_seconds", "update_seconds",
        "shift_seconds", "synchronize_seconds", "evaluation_seconds", "peak_memory_gigabytes",
        "peak_reserved_memory_gigabytes",
    }
    for left, right in zip(unbroken, resumed):
        comparable = {name: left[name] for name in left
                      if name in right and name not in timing_keys}
        for name, value in comparable.items():
            assert right[name] == value


def test_resume_of_completed_run_returns_immediately(tmp_path):
    directory, finished_model = run_tiny_training(tmp_path, "done", iterations=2)
    finished_parameters = [parameter.detach().clone() for parameter in finished_model.parameters()]
    lines_before = (directory / "metrics.jsonl").read_text(encoding="utf-8")

    directory_again, resumed_model = run_tiny_training(
        tmp_path, "done", iterations=2, resume_from=directory)
    assert directory_again == directory
    assert (directory / "metrics.jsonl").read_text(encoding="utf-8") == lines_before
    for left, right in zip(finished_parameters, resumed_model.parameters()):
        assert torch.equal(left, right)


def test_crash_after_log_before_checkpoint_leaves_no_gap(tmp_path, monkeypatch):
    """先写日志再保存：保存时被打断，恢复后截掉检查点之后的日志并重跑，轮次连续、不缺轮"""
    import grpx.loop as loop_module

    real_save = loop_module.save_checkpoint
    first, _ = run_tiny_training(tmp_path, "gap-lock", iterations=2, checkpoint_interval=2)
    assert find_latest_checkpoint(first) == first / "checkpoint-0002"

    def crash_on_four(run_directory, completed_iterations, *arguments, **keyword_arguments):
        if completed_iterations == 4:
            raise RuntimeError("模拟保存 checkpoint-0004 时被杀")
        return real_save(run_directory, completed_iterations, *arguments, **keyword_arguments)

    monkeypatch.setattr(loop_module, "save_checkpoint", crash_on_four)
    with pytest.raises(RuntimeError, match="checkpoint-0004"):
        run_tiny_training(tmp_path, "gap-lock", iterations=4, checkpoint_interval=2, resume_from=first)

    # 第 2、3 轮日志已写入，但 checkpoint-0004 没有；最新检查点仍是 0002
    orphaned = [json.loads(line)["iteration"]
                for line in (first / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert orphaned == [0, 1, 2, 3]
    assert find_latest_checkpoint(first) == first / "checkpoint-0002"

    monkeypatch.setattr(loop_module, "save_checkpoint", real_save)
    resumed, _ = run_tiny_training(tmp_path, "gap-lock", iterations=4, checkpoint_interval=2, resume_from=first)
    iterations = [json.loads(line)["iteration"]
                  for line in (resumed / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert iterations == [0, 1, 2, 3, 4]


def test_resume_rejects_disallowed_overrides(tmp_path):
    configuration = {
        "model": {"name": "m"},
        "backend": {"type": "huggingface"},
        "task": {"dataset": "examples.arithmetic:make_dataset",
                 "reward_function": "examples.arithmetic:reward_function"},
        "training": {"algorithm": "grpo", "iterations": 10,
                     "max_tokens_per_minibatch": 4096, "max_rows_per_minibatch": 8},
    }
    (tmp_path / "resolved_configuration.json").write_text(
        json.dumps(configuration), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="只能修改"):
        load_resume_configuration(tmp_path, ["training.learning_rate=1e-4"])
    updated = load_resume_configuration(tmp_path, [
        "training.iterations=40",
        "training.max_tokens_per_minibatch=1024",
        "training.max_rows_per_minibatch=4",
        "training.shift_measurement_interval=5",
        "backend.max_rows_per_batch=8",
    ])
    assert updated["training"]["iterations"] == 40
    assert updated["training"]["max_tokens_per_minibatch"] == 1024
    assert updated["training"]["max_rows_per_minibatch"] == 4
    assert updated["training"]["shift_measurement_interval"] == 5
    assert updated["backend"]["max_rows_per_batch"] == 8
