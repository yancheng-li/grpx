from collections import defaultdict
from grpx.backends.base import GenerationBackend
from grpx.types import Trajectory, Step

def encode_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def backend_url_count(backend):
    urls = getattr(backend, "base_urls", None)
    if urls:
        return len(urls)
    return 1


def backend_url_for_trajectory(backend, trajectory_index):
    urls = getattr(backend, "base_urls", None)
    if not urls:
        return None, 0
    url_index = trajectory_index % len(urls)
    return urls[url_index], url_index


def rollout_groups(backend: GenerationBackend, tokenizer, environment_factory, group_specifications,
                   group_size, policy_version=0, max_steps=1, greedy=False, truncated_reward=0.0):
    """group_specifications: [(group_id, sample, seed), ...]；所有组同时推进，每一步只调用一次后端。

    truncated_reward：生成被长度截断时覆盖**该步**回报（不是整局），并把轨迹标为不成功。
    多轮时先前步的回报保留；整局回报是各步之和。传 None 则保留环境回报与成功判定（旧行为）。

    多 base_url 时按轨迹序号对地址数取余固定分配，便于前缀缓存，并保证每个地址都有样本。
    """
    environments, conversations, trajectories = [], [], []
    for group_id, sample, seed in group_specifications:
        for _ in range(group_size):
            environment = environment_factory()
            environments.append(environment)
            # 下面会原地追加消息；环境可能返回共享的列表（如 sample["messages"]），必须各自复制
            conversations.append(list(environment.reset(sample, seed)))
            trajectories.append(Trajectory(
                group_id, policy_version, sampling_temperature=backend.temperature))
    active_indices = set(range(len(trajectories)))
    url_count = backend_url_count(backend)

    for _ in range(max_steps):
        # 相同提示词合并成一个请求；多地址时只在同一地址内合并，以免打散前缀缓存
        indices_by_key = defaultdict(list)
        for index in sorted(active_indices):
            prompt_token_ids = tuple(encode_messages(tokenizer, conversations[index]))
            url_index = index % url_count
            indices_by_key[(url_index, prompt_token_ids)].append(index)

        request_keys = list(indices_by_key.items())
        requests = [
            (list(prompt_token_ids), len(indices), url_index)
            for (url_index, prompt_token_ids), indices in request_keys
        ]
        responses = backend.generate_batch(requests, greedy=greedy)

        for ((url_index, prompt_token_ids), indices), outputs in zip(
                request_keys, responses, strict=True):
            base_url = (
                backend.base_urls[url_index]
                if getattr(backend, "base_urls", None) else None
            )
            for index, output in zip(indices, outputs, strict=True):
                action_token_ids, was_truncated = output[0], output[1]
                # 兼容旧假后端：(ids, truncated) / (ids, truncated, logprobs)
                # 新后端：(ids, truncated, logprobs|None, finish_reason)
                backend_log_probabilities = None
                finish_reason = None
                if len(output) > 3:
                    backend_log_probabilities = output[2]
                    finish_reason = output[3]
                elif len(output) > 2:
                    if isinstance(output[2], str):
                        finish_reason = output[2]
                    else:
                        backend_log_probabilities = output[2]
                if finish_reason is None:
                    finish_reason = "length_limit" if was_truncated else "end_token"
                # 只有长度截断算 truncated；停止串是正常结束
                was_truncated = finish_reason == "length_limit"
                # 动作文本必须由动作 token 解码，保证停止串落在 token 中间时两者一致
                action_text = tokenizer.decode(action_token_ids, skip_special_tokens=True)
                state_key = environments[index].state_key()
                new_messages, reward, is_done = environments[index].step(action_text)
                environment_reward = reward
                if was_truncated and truncated_reward is not None:
                    reward = truncated_reward
                trajectories[index].steps.append(
                    Step(list(prompt_token_ids), action_token_ids, reward, state_key,
                         backend_log_probabilities=backend_log_probabilities,
                         backend_base_url=base_url,
                         environment_reward=environment_reward,
                         finish_reason=finish_reason)
                )
                conversations[index] += (
                    [{"role": "assistant", "content": action_text}] + new_messages
                )
                # 回答不完整优先于环境结束：训练端要据此过滤或惩罚
                if was_truncated:
                    trajectories[index].termination_reason = "length_limit"
                elif is_done:
                    trajectories[index].termination_reason = "done"
                if was_truncated or is_done:
                    active_indices.discard(index)
        if not active_indices:
            break

    for index in active_indices:
        trajectories[index].termination_reason = "step_limit"
    for index, environment in enumerate(environments):
        if hasattr(environment, "is_successful"):
            trajectories[index].is_successful = environment.is_successful()
        if (truncated_reward is not None
                and trajectories[index].termination_reason == "length_limit"):
            trajectories[index].is_successful = False
        if hasattr(environment, "metrics"):
            trajectories[index].environment_metrics = dict(environment.metrics())
    return trajectories

def rollout_group(backend: GenerationBackend, tokenizer, environment_factory, sample, group_id,
                  group_size, seed, policy_version=0, max_steps=1, truncated_reward=0.0):
    return rollout_groups(backend, tokenizer, environment_factory,
                          [(group_id, sample, seed)], group_size, policy_version, max_steps,
                          truncated_reward=truncated_reward)
