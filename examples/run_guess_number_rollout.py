import collections
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.rollout import rollout_group
from grpx.advantage import episode_advantages
from grpx.types import success_fraction
from examples.guess_number import GuessNumberEnvironment, make_dataset

if not torch.cuda.is_available():
    raise SystemExit(f"当前 Python（torch {torch.__version__}）看不到 CUDA，请换到装了 GPU 版 torch 的环境运行")

torch.manual_seed(0)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda()
backend = HuggingFaceBackend(model, tokenizer)

group_size, group_count, max_steps, trajectories = 8, 32, 6, []
for group_index, sample in enumerate(make_dataset(group_count)):
    trajectories += rollout_group(
        backend, tokenizer, lambda: GuessNumberEnvironment(reveal_interval=True), sample, f"g{group_index}",
        group_size=group_size, seed=group_index, max_steps=max_steps,
    )

kept = episode_advantages(trajectories)
print(f"成功率={success_fraction(trajectories):.2f}  "
      f"平均回报={sum(trajectory.total_reward for trajectory in trajectories) / len(trajectories):.2f}  "
      f"有信号的组={len(kept) // group_size}/{group_count}  "
      f"平均步数={sum(len(trajectory.steps) for trajectory in trajectories) / len(trajectories):.2f}")
reason_counts = collections.Counter(trajectory.termination_reason for trajectory in trajectories)
for reason, count in sorted(reason_counts.items()):
    print(f"  {reason}: {count} 条")

# 最后一步的提示词里已经包含了此前所有回答和环境反馈，再拼上最后一步的回答就是整段对话
for trajectory in random.Random(0).sample(trajectories, 2):
    last_step = trajectory.steps[-1]
    print(f"\n===== {trajectory.group_id}  步数={len(trajectory.steps)}  "
          f"结束原因={trajectory.termination_reason}  成功={trajectory.is_successful}  "
          f"回报={trajectory.total_reward:.3f} =====")
    print("各步状态：", [step.state_key for step in trajectory.steps])
    print("各步回报：", [round(step.reward, 3) for step in trajectory.steps])
    print(tokenizer.decode(last_step.prompt_token_ids + last_step.action_token_ids))
