"""批量生成的真实模型验收：贪心等价、训练/推理一致性、速度。

用法：python -m examples.check_batched_generation outputs/<运行名> [检查点迭代号，默认取最后一个]
"""
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.guess_number import GuessNumberEnvironment, make_dataset
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.loop import split_into_row_chunks
from grpx.rollout import rollout_groups
from grpx.trainer import action_log_probabilities, build_batch
from grpx.types import success_fraction

MAX_STEPS = 6
GROUPS, GROUP_SIZE = 16, 8


class OnePromptAtATime:
    """改动前的行为：每个提示词单独一次生成，同一提示词的样本在一起，没有填充"""
    def __init__(self, backend):
        self.backend, self.temperature = backend, backend.temperature

    def generate_batch(self, requests, greedy=False):
        return [self.backend.generate_batch([request], greedy=greedy)[0] for request in requests]


def environment_factory():
    return GuessNumberEnvironment(reveal_interval=True)


def timed_rollout(backend, specifications, group_size, greedy):
    torch.cuda.synchronize()
    start = time.perf_counter()
    trajectories = rollout_groups(backend, tokenizer, environment_factory, specifications,
                                  group_size, max_steps=MAX_STEPS, greedy=greedy)
    torch.cuda.synchronize()
    return trajectories, time.perf_counter() - start


def describe(trajectories):
    count = len(trajectories)
    return (f"成功率={success_fraction(trajectories):.3f}  "
            f"平均奖励={sum(trajectory.total_reward for trajectory in trajectories) / count:.4f}  "
            f"截断比例={sum(trajectory.termination_reason == 'length_limit' for trajectory in trajectories) / count:.3f}  "
            f"平均步数={sum(len(trajectory.steps) for trajectory in trajectories) / count:.2f}")


@torch.no_grad()
def mean_action_log_probability(trajectories):
    total, token_count = 0.0, 0
    for chunk in split_into_row_chunks(trajectories, 8, 4096):
        log_probabilities, mask = action_log_probabilities(
            model, build_batch(chunk, tokenizer.pad_token_id, "cuda"))
        total += log_probabilities[mask].sum().item()
        token_count += mask.sum().item()
    return total / token_count


run_directory = Path(sys.argv[1])
if len(sys.argv) > 2:
    checkpoint_directory = run_directory / f"checkpoint-{int(sys.argv[2]):04d}"
else:
    checkpoint_directory = sorted(run_directory.glob("checkpoint-*"))[-1]

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, checkpoint_directory).cuda().eval()
batched = HuggingFaceBackend(model, tokenizer)
one_prompt_at_a_time = OnePromptAtATime(HuggingFaceBackend(model, tokenizer))
print(f"检查点：{checkpoint_directory}")

print("\n== 1. 贪心等价（50 道评测题）==")
evaluation_specifications = [(f"evaluation-{index}", sample, index)
                             for index, sample in enumerate(make_dataset(50, seed=1))]
old, old_seconds = timed_rollout(one_prompt_at_a_time, evaluation_specifications, 1, greedy=True)
new, new_seconds = timed_rollout(batched, evaluation_specifications, 1, greedy=True)
print(f"  逐提示词：{describe(old)}  耗时={old_seconds:.1f}s")
print(f"  批量：    {describe(new)}  耗时={new_seconds:.1f}s")
differing = [index for index, (old_trajectory, new_trajectory) in enumerate(zip(old, new))
             if [step.action_token_ids for step in old_trajectory.steps]
             != [step.action_token_ids for step in new_trajectory.steps]]
print(f"  回答序列不同的题：{len(differing)} 道 {differing}")
print(f"  成功与否不同的题：{sum(a.is_successful != b.is_successful for a, b in zip(old, new))} 道")

print(f"\n== 2. 训练/推理一致性与速度（{GROUPS} 组 × {GROUP_SIZE}，采样）==")
training_specifications = [(f"group-{index}", sample, index)
                           for index, sample in enumerate(make_dataset(GROUPS, seed=0))]
for name, backend in [("逐提示词", one_prompt_at_a_time), ("批量", batched)]:
    torch.manual_seed(0)
    trajectories, seconds = timed_rollout(backend, training_specifications, GROUP_SIZE, greedy=False)
    row_count = sum(len(trajectory.steps) for trajectory in trajectories)
    print(f"  {name}：{describe(trajectories)}")
    print(f"    采样耗时={seconds:.1f}s  训练行数={row_count}  "
          f"回答 token 平均对数概率={mean_action_log_probability(trajectories):.4f}")
