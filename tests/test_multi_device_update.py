import copy

import pytest
import torch

from grpx.multi_device_update import (
    ReplicaGroup,
    ReplicaSynchronizationError,
    ReplicaWorkerError,
    assign_minibatches,
    trainable_parameter_fingerprint,
)

FEATURE_COUNT = 6


class ToyPolicy(torch.nn.Module):
    """冻结的基座加一个可训练的低秩增量，和 LoRA 的结构相同。"""

    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(FEATURE_COUNT, FEATURE_COUNT)
        self.base.requires_grad_(False)
        self.adapter_down = torch.nn.Parameter(torch.randn(2, FEATURE_COUNT) * 0.3)
        self.adapter_up = torch.nn.Parameter(torch.randn(FEATURE_COUNT, 2) * 0.3)
        self.unused_adapter = torch.nn.Parameter(torch.zeros(3))  # 始终拿不到梯度

    def forward(self, inputs):
        return self.base(inputs) + inputs @ self.adapter_down.T @ self.adapter_up.T


def make_minibatches(count=7, seed=0):
    generator = torch.Generator().manual_seed(seed)
    minibatches = []
    for index in range(count):
        row_count = 1 + (index * 3) % 5
        minibatches.append(
            (torch.randn(row_count, FEATURE_COUNT, generator=generator),
             torch.randn(row_count, FEATURE_COUNT, generator=generator))
        )
    return minibatches


def make_replicas(count, seed=0):
    torch.manual_seed(seed)
    primary = ToyPolicy()
    return [primary] + [copy.deepcopy(primary) for _ in range(count - 1)]


def make_loss_function(minibatches):
    denominator = float(sum(inputs.shape[0] for inputs, _ in minibatches))  # 全局分母，事先定好

    def compute_minibatch_loss(model, minibatch_index, device):
        inputs, targets = minibatches[minibatch_index]
        inputs, targets = inputs.to(device), targets.to(device)
        loss = ((model(inputs) - targets) ** 2).sum() / denominator
        return loss, {"rows": inputs.shape[0], "loss": float(loss.detach())}

    return compute_minibatch_loss


def single_device_reference(minibatches, seed=0):
    model = make_replicas(1, seed)[0]
    compute = make_loss_function(minibatches)
    for index in range(len(minibatches)):
        loss, _ = compute(model, index, torch.device("cpu"))
        loss.backward()
    return model


def gradients(model):
    return {name: parameter.grad for name, parameter in model.named_parameters() if parameter.requires_grad}


# ---- 小批分配 -----------------------------------------------------------------


def test_assignment_is_balanced_deterministic_and_complete():
    token_counts = [100, 900, 300, 300, 50, 700, 250]
    first = assign_minibatches(token_counts, 2)
    assert first == assign_minibatches(token_counts, 2)
    assert sorted(index for share in first for index in share) == list(range(len(token_counts)))
    loads = [sum(token_counts[index] for index in share) for share in first]
    assert abs(loads[0] - loads[1]) <= max(token_counts)
    assert all(share == sorted(share) for share in first)


def test_assignment_with_fewer_minibatches_than_replicas():
    assert assign_minibatches([10], 3) == [[0], [], []]
    assert assign_minibatches([], 2) == [[], []]
    with pytest.raises(ValueError):
        assign_minibatches([1, 2], 0)


# ---- 和单卡等价 ---------------------------------------------------------------


@pytest.mark.parametrize("replica_count", [1, 2, 3])
def test_gradients_and_updated_parameters_match_single_device(replica_count):
    minibatches = make_minibatches()
    reference = single_device_reference(minibatches)

    replicas = make_replicas(replica_count)
    group = ReplicaGroup(replicas, ["cpu"] * replica_count)
    group.assert_synchronized()
    assignment = assign_minibatches([inputs.numel() for inputs, _ in minibatches], replica_count)
    metrics = group.accumulate_gradients(assignment, make_loss_function(minibatches))
    group.reduce_gradients()

    assert sorted(metrics) == list(range(len(minibatches)))
    reference_gradients = gradients(reference)
    for name, gradient in gradients(group.primary).items():
        if reference_gradients[name] is None:
            assert gradient is None
        else:
            torch.testing.assert_close(gradient, reference_gradients[name], rtol=1e-5, atol=1e-7)
    for model in replicas[1:]:
        assert all(gradient is None for gradient in gradients(model).values())

    # 用同一个优化器各更新一步，参数也应相同
    for model in (reference, group.primary):
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad and p.grad is not None], lr=1e-2)
        optimizer.step()
    for (name, updated), (_, expected) in zip(
        group.primary.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(updated, expected, rtol=1e-5, atol=1e-7)

    group.broadcast_parameters()
    group.assert_synchronized()
    group.close()


def test_replica_without_minibatches_contributes_nothing():
    minibatches = make_minibatches(count=1)
    reference = single_device_reference(minibatches)
    replicas = make_replicas(3)
    group = ReplicaGroup(replicas, ["cpu"] * 3)
    group.accumulate_gradients([[0], [], []], make_loss_function(minibatches))
    group.reduce_gradients()
    torch.testing.assert_close(group.primary.adapter_up.grad, reference.adapter_up.grad)
    group.close()


def test_primary_without_minibatches_receives_gradients_from_others():
    minibatches = make_minibatches(count=2)
    reference = single_device_reference(minibatches)
    group = ReplicaGroup(make_replicas(2), ["cpu", "cpu"])
    group.accumulate_gradients([[], [0, 1]], make_loss_function(minibatches))
    assert group.primary.adapter_up.grad is None
    group.reduce_gradients()
    torch.testing.assert_close(group.primary.adapter_up.grad, reference.adapter_up.grad, rtol=1e-5, atol=1e-7)
    group.close()


def test_stale_gradients_are_cleared_before_accumulation():
    minibatches = make_minibatches(count=3)
    reference = single_device_reference(minibatches)
    replicas = make_replicas(2)
    for model in replicas:
        model.adapter_up.grad = torch.full_like(model.adapter_up, 123.0)
    group = ReplicaGroup(replicas, ["cpu", "cpu"])
    group.accumulate_gradients([[0, 2], [1]], make_loss_function(minibatches))
    group.reduce_gradients()
    torch.testing.assert_close(group.primary.adapter_up.grad, reference.adapter_up.grad, rtol=1e-5, atol=1e-7)
    group.close()


# ---- 同步 ---------------------------------------------------------------------


def test_missed_broadcast_is_detected():
    group = ReplicaGroup(make_replicas(2), ["cpu", "cpu"])
    group.assert_synchronized()
    with torch.no_grad():
        group.primary.adapter_up.add_(1e-6)  # 主副本更新了，但没有同步
    with pytest.raises(ReplicaSynchronizationError):
        group.assert_synchronized()
    group.broadcast_parameters()
    group.assert_synchronized()
    group.close()


def test_broadcast_leaves_frozen_parameters_untouched():
    replicas = make_replicas(2)
    with torch.no_grad():
        replicas[1].base.weight.add_(1.0)  # 基座故意不同：同步只管可训练参数
    before = replicas[1].base.weight.clone()
    group = ReplicaGroup(replicas, ["cpu", "cpu"])
    group.broadcast_parameters()
    torch.testing.assert_close(replicas[1].base.weight, before)
    group.close()


def test_fingerprint_depends_on_values_and_names():
    first, second = make_replicas(2)
    assert trainable_parameter_fingerprint(first) == trainable_parameter_fingerprint(second)
    with torch.no_grad():
        second.adapter_down.mul_(1.0 + 1e-6)
    assert trainable_parameter_fingerprint(first) != trainable_parameter_fingerprint(second)


def test_mismatched_trainable_names_rejected():
    first, second = make_replicas(2)
    second.unused_adapter.requires_grad_(False)
    with pytest.raises(ValueError):
        ReplicaGroup([first, second], ["cpu", "cpu"])


# ---- 线程里的梯度开关 ---------------------------------------------------------


def test_map_without_gradient_disables_gradient_inside_worker_threads():
    minibatches = make_minibatches(count=4)
    group = ReplicaGroup(make_replicas(2), ["cpu", "cpu"])

    def forward(model, minibatch_index, device):
        return model(minibatches[minibatch_index][0].to(device))

    assert torch.is_grad_enabled()  # 主线程没有关梯度
    outputs = group.map_without_gradient([[0, 1], [2, 3]], forward)
    assert sorted(outputs) == [0, 1, 2, 3]
    assert all(not output.requires_grad for output in outputs.values())
    group.close()


def test_accumulation_works_even_if_caller_is_inside_no_grad():
    minibatches = make_minibatches(count=3)
    reference = single_device_reference(minibatches)
    group = ReplicaGroup(make_replicas(2), ["cpu", "cpu"])
    with torch.no_grad():
        group.accumulate_gradients([[0, 1], [2]], make_loss_function(minibatches))
    group.reduce_gradients()
    torch.testing.assert_close(group.primary.adapter_up.grad, reference.adapter_up.grad, rtol=1e-5, atol=1e-7)
    group.close()


# ---- 出错 ---------------------------------------------------------------------


def test_worker_error_names_the_device_and_does_not_hang():
    group = ReplicaGroup(make_replicas(2), ["cpu", "cpu"])

    def failing(model, minibatch_index, device):
        if minibatch_index == 1:
            raise RuntimeError("out of memory")
        return torch.zeros((), requires_grad=True) * model.adapter_up.sum(), {}

    with pytest.raises(ReplicaWorkerError, match="out of memory"):
        group.accumulate_gradients([[0], [1]], failing)
    group.close()


def test_assignment_size_must_match_replica_count():
    group = ReplicaGroup(make_replicas(2), ["cpu", "cpu"])
    with pytest.raises(ValueError):
        group.map_without_gradient([[0]], lambda model, index, device: None)
    group.close()
