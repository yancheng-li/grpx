import torch
import torch.nn.functional as functional

from grpx.gradient_checkpointing import gradient_bearing_forward
from grpx.types import Trajectory


def row_weight(trajectory, row_weighting):
    if row_weighting == "per_step":
        return 1.0
    if row_weighting == "per_trajectory":
        return 1.0 / (trajectory.source_step_count or len(trajectory.steps))
    raise ValueError(f"未知的 row_weighting: {row_weighting}")


def is_append_only(trajectory):
    """第 t+1 步提示词是否恰好是「第 t 步提示词 + 第 t 步回答」。"""
    for current, following in zip(trajectory.steps, trajectory.steps[1:]):
        expected_prefix = current.prompt_token_ids + current.action_token_ids
        actual = following.prompt_token_ids
        if actual[:len(expected_prefix)] != expected_prefix:
            return False
    return True


def prepare_update_units(trajectories, merge_append_only=False):
    """把轨迹变成组批单元：默认可合并时整条轨迹一行，否则每步一行。"""
    units = []
    for trajectory in trajectories:
        if (merge_append_only and len(trajectory.steps) > 1
                and is_append_only(trajectory)):
            units.append(trajectory)
            continue
        for step in trajectory.steps:
            units.append(Trajectory(
                trajectory.group_id, trajectory.policy_version,
                sampling_temperature=trajectory.sampling_temperature,
                steps=[step],
                source_step_count=trajectory.source_step_count or len(trajectory.steps),
            ))
    return units


def merged_trajectory_fraction(trajectories, units):
    if not trajectories:
        return 0.0
    mergeable = sum(
        1 for trajectory in trajectories
        if len(trajectory.steps) > 1 and is_append_only(trajectory)
    )
    if mergeable == 0:
        return 0.0
    merged = sum(1 for unit in units if len(unit.steps) > 1)
    return merged / mergeable


def build_batch(trajectories, padding_token_id, device, row_weighting="per_step"):
    """每条轨迹（或单步单元）是一行；多步且追加式时一行内多段回答。"""
    rows = []
    for trajectory in trajectories:
        weight = row_weight(trajectory, row_weighting)
        if len(trajectory.steps) == 1:
            step = trajectory.steps[0]
            tokens = step.prompt_token_ids + step.action_token_ids
            segments = [(
                len(step.prompt_token_ids),
                len(step.action_token_ids),
                step.advantage,
                weight,
                step.backend_log_probabilities,
            )]
        else:
            last = trajectory.steps[-1]
            tokens = last.prompt_token_ids + last.action_token_ids
            segments = [
                (
                    len(step.prompt_token_ids),
                    len(step.action_token_ids),
                    step.advantage,
                    weight,
                    step.backend_log_probabilities,
                )
                for step in trajectory.steps
            ]
        backend_url = next(
            (step.backend_base_url for step in trajectory.steps if step.backend_base_url),
            None,
        )
        rows.append((tokens, segments, trajectory.sampling_temperature, backend_url))

    batch_size = len(rows)
    max_length = max(len(tokens) for tokens, *_ in rows)

    input_token_ids = torch.full((batch_size, max_length), padding_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    action_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    segment_ids = torch.full((batch_size, max_length), -1, dtype=torch.long)
    sampling_log_probabilities = torch.zeros((batch_size, max_length), dtype=torch.float32)
    sampling_log_probability_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    segment_advantages = []
    segment_weights = []
    row_advantages = []
    row_weights = []
    backend_urls = []
    segment_id = 0
    for row_index, (tokens, segments, _, backend_url) in enumerate(rows):
        length = len(tokens)
        input_token_ids[row_index, :length] = torch.tensor(tokens)
        attention_mask[row_index, :length] = 1
        backend_urls.append(backend_url)
        # 行级优势/权重：取第一段，供旧调用方兼容；真正损失用 segment_* 
        row_advantages.append(segments[0][2])
        row_weights.append(segments[0][3])
        for action_start, action_length, advantage, weight, backend_log_probabilities in segments:
            action_end = action_start + action_length
            action_mask[row_index, action_start:action_end] = True
            segment_ids[row_index, action_start:action_end] = segment_id
            segment_advantages.append(advantage)
            segment_weights.append(weight)
            if backend_log_probabilities is not None:
                for offset, value in enumerate(backend_log_probabilities):
                    if value is None:
                        continue
                    position = action_start + offset
                    sampling_log_probabilities[row_index, position] = value
                    sampling_log_probability_mask[row_index, position] = True
            segment_id += 1

    batch = {
        "input_token_ids": input_token_ids,
        "attention_mask": attention_mask,
        "action_mask": action_mask,
        "segment_ids": segment_ids,
        "advantages": torch.tensor(row_advantages, dtype=torch.float32),
        "temperatures": torch.tensor([row[2] for row in rows], dtype=torch.float32),
        "row_weights": torch.tensor(row_weights, dtype=torch.float32),
        "segment_advantages": torch.tensor(segment_advantages, dtype=torch.float32),
        "segment_weights": torch.tensor(segment_weights, dtype=torch.float32),
        "sampling_log_probabilities": sampling_log_probabilities,
        "sampling_log_probability_mask": sampling_log_probability_mask,
    }
    moved = {name: tensor.to(device) for name, tensor in batch.items()}
    moved["backend_urls"] = backend_urls
    return moved


# 这些配置项表示 CausalLM.forward 会在输出层之后再处理 logits，绕过 forward 会算错
LOGIT_POSTPROCESSING_CONFIG_NAMES = ("final_logit_softcapping", "logit_scale")


def _supports_action_only_logits(language_model):
    if not (hasattr(language_model, "get_output_embeddings") and hasattr(language_model, "base_model")):
        return False
    config = getattr(language_model, "config", None)
    return not any(getattr(config, name, None) not in (None, 1.0) for name in LOGIT_POSTPROCESSING_CONFIG_NAMES)


def _mixed_precision_enabled(model):
    return bool(getattr(model, "grpx_mixed_precision", False))


def _cached_float32_parameter(module, attribute_name, parameter):
    """半精度参数缓存一份 float32，避免每次 .float()；已是 float32 则直接用。"""
    if parameter is None:
        return None
    if parameter.dtype == torch.float32:
        return parameter
    cached = getattr(module, attribute_name, None)
    if cached is None or cached.shape != parameter.shape or cached.device != parameter.device:
        cached = parameter.detach().float().clone()
        setattr(module, attribute_name, cached)
    return cached


def _resolve_output_logit_chunk_size(model, chunk_size):
    if chunk_size is not None:
        return chunk_size
    return getattr(model, "grpx_output_logit_chunk_size", None)


def _cross_entropy_log_probs(logits, targets, temperatures):
    if logits.dtype != torch.float32:
        logits = logits.float()
    if temperatures is not None and not torch.all(temperatures == 1):
        logits = logits / temperatures[:, None]
    return -functional.cross_entropy(logits, targets, reduction="none")


def _output_chunk_log_probs(selected_chunk, weight, bias, targets_chunk, temperatures_chunk):
    """单块：线性层 + CE。供 checkpoint 包住，反向时逐块重算，峰值只留一块 logits。"""
    logits = functional.linear(selected_chunk, weight, bias)
    return _cross_entropy_log_probs(logits, targets_chunk, temperatures_chunk)


def _chunked_output_log_probs(selected, weight, bias, targets, temperatures, chunk_size):
    """对回答位置分块做输出层 + CE。

    用 checkpoint 包每一块：前向不保留块内 logits，反向再逐块重算，
    否则「循环算完再拼接」会把所有块的 logits 同时留在计算图里，峰值和不分块一样。
    """
    token_count = selected.shape[0]
    if token_count == 0:
        return selected.new_zeros((0,))
    if chunk_size is None or chunk_size <= 0 or token_count <= chunk_size:
        return _output_chunk_log_probs(selected, weight, bias, targets, temperatures)
    pieces = []
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        temp = None if temperatures is None else temperatures[start:end]
        # use_reentrant=False：与自动混合精度、多次 checkpoint 更合拍
        pieces.append(torch.utils.checkpoint.checkpoint(
            _output_chunk_log_probs,
            selected[start:end], weight, bias, targets[start:end], temp,
            use_reentrant=False,
        ))
    return torch.cat(pieces, dim=0)


def action_log_probabilities(model, batch, output_logit_chunk_size=None):
    # 位置 i 的输出预测第 i+1 个 token，所以掩码和目标都错开一位
    mask = batch["action_mask"][:, 1:]
    targets = batch["input_token_ids"][:, 1:][mask]
    temperatures = batch["temperatures"][:, None].expand_as(mask)[mask]
    chunk_size = _resolve_output_logit_chunk_size(model, output_logit_chunk_size)

    language_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    use_mixed = _mixed_precision_enabled(model)
    device_type = batch["input_token_ids"].device.type
    if _supports_action_only_logits(language_model):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_mixed):
            hidden_states = language_model.base_model(
                input_ids=batch["input_token_ids"],
                attention_mask=batch["attention_mask"]).last_hidden_state
        selected = hidden_states[:, :-1][mask].float()
        output_layer = language_model.get_output_embeddings()
        weight = _cached_float32_parameter(output_layer, "_grpx_float32_weight", output_layer.weight)
        bias = _cached_float32_parameter(output_layer, "_grpx_float32_bias", output_layer.bias)
        with torch.autocast(device_type=device_type, enabled=False):
            selected_log_probabilities = _chunked_output_log_probs(
                selected, weight, bias, targets, temperatures, chunk_size)
    else:
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_mixed):
            logits = model(input_ids=batch["input_token_ids"],
                           attention_mask=batch["attention_mask"]).logits
        selected = logits[:, :-1][mask]
        if selected.dtype != torch.float32:
            selected = selected.float()
        # 全量 forward 已物化整段 logits；这里仍可按行分块做 CE，峰值略降
        if chunk_size is None or chunk_size <= 0 or selected.shape[0] <= chunk_size:
            selected_log_probabilities = _cross_entropy_log_probs(selected, targets, temperatures)
        else:
            pieces = []
            for start in range(0, selected.shape[0], chunk_size):
                end = min(start + chunk_size, selected.shape[0])
                pieces.append(_cross_entropy_log_probs(
                    selected[start:end], targets[start:end], temperatures[start:end]))
            selected_log_probabilities = torch.cat(pieces, dim=0)

    log_probabilities = torch.zeros(
        mask.shape, dtype=selected_log_probabilities.dtype, device=mask.device
    ).masked_scatter(mask, selected_log_probabilities)
    return log_probabilities, mask


def grpo_loss(new_log_probs, old_log_probs, mask, advantages, clip_range=0.2,
              length_normalization="per_sequence", constant_length=None, row_weights=None,
              segment_ids=None, segment_advantages=None, segment_weights=None):
    mask_float = mask.float()
    log_ratio = (new_log_probs - old_log_probs) * mask_float
    ratio = torch.exp(log_ratio)

    if segment_ids is None:
        advantages = advantages[:, None]
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages) * mask_float
        if length_normalization == "per_sequence":
            per_segment = surrogate.sum(1) / mask_float.sum(1).clamp(min=1)
        elif length_normalization == "constant":
            if constant_length is None:
                raise ValueError("length_normalization='constant' 需要提供 constant_length")
            per_segment = surrogate.sum(1) / constant_length
        else:
            raise ValueError(f"未知的 length_normalization: {length_normalization}")
        weights = row_weights if row_weights is not None else torch.ones_like(per_segment)
        loss = -(per_segment * weights).sum() / weights.sum()
    else:
        token_mask = mask.bool()
        segment_index = segment_ids[:, 1:][token_mask]
        if (segment_index < 0).any():
            raise ValueError("回答位置的 segment_ids 不应为 -1")
        ratio_tokens = ratio[token_mask]
        advantage_tokens = segment_advantages[segment_index]
        clipped_ratio = torch.clamp(ratio_tokens, 1 - clip_range, 1 + clip_range)
        surrogate_tokens = torch.min(ratio_tokens * advantage_tokens, clipped_ratio * advantage_tokens)
        segment_count = segment_advantages.shape[0]
        segment_sums = torch.zeros(segment_count, device=surrogate_tokens.device,
                                   dtype=surrogate_tokens.dtype)
        segment_sums.index_add_(0, segment_index, surrogate_tokens)
        if length_normalization == "per_sequence":
            token_counts = torch.zeros_like(segment_sums)
            token_counts.index_add_(0, segment_index, torch.ones_like(surrogate_tokens))
            per_segment = segment_sums / token_counts.clamp(min=1)
        elif length_normalization == "constant":
            if constant_length is None:
                raise ValueError("length_normalization='constant' 需要提供 constant_length")
            per_segment = segment_sums / constant_length
        else:
            raise ValueError(f"未知的 length_normalization: {length_normalization}")
        weights = segment_weights if segment_weights is not None else torch.ones_like(per_segment)
        loss = -(per_segment * weights).sum() / weights.sum()

    token_count = mask_float.sum().clamp(min=1)
    absolute_log_ratio = log_ratio.detach().abs()[mask.bool()]
    metrics = {
        "loss": loss.item(),
        "clip_fraction": (((ratio - 1).abs() > clip_range).float() * mask_float).sum().item() / token_count.item(),
        "approximate_kl": (((ratio - 1) - log_ratio) * mask_float).sum().item() / token_count.item(),
        "max_ratio_deviation": ((ratio - 1).abs() * mask_float).max().item(),
        "median_absolute_log_ratio": absolute_log_ratio.median().item(),
        "percentile_95_absolute_log_ratio": torch.quantile(absolute_log_ratio, 0.95).item(),
    }
    return loss, metrics


def train_on_batch(model, optimizer, batch, number_of_updates=1, max_gradient_norm=1.0,
                   max_approximate_kl=None, **loss_options):
    model.eval()
    with torch.no_grad():
        old_log_probs, mask = action_log_probabilities(model, batch)

    model.train()
    history = []
    for _ in range(number_of_updates):
        # 反向会重算前向，训练模式要保持到 backward 结束
        with gradient_bearing_forward(model):
            new_log_probs, _ = action_log_probabilities(model, batch)
            loss, metrics = grpo_loss(
                new_log_probs, old_log_probs, mask, batch["advantages"],
                row_weights=batch.get("row_weights"),
                segment_ids=batch.get("segment_ids"),
                segment_advantages=batch.get("segment_advantages"),
                segment_weights=batch.get("segment_weights"),
                **loss_options
            )
            should_step = not (max_approximate_kl is not None
                               and metrics["approximate_kl"] > max_approximate_kl)
            if should_step:
                optimizer.zero_grad()
                loss.backward()
        history.append(metrics)
        if not should_step:
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
    return history
