import collections

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from grpx.environment import SingleTurnEnvironment
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.rollout import rollout_group
from grpx.advantage import episode_advantages
from examples.arithmetic import make_dataset, reward_function

if not torch.cuda.is_available():
    raise SystemExit(f"当前 Python（torch {torch.__version__}）看不到 CUDA，请换到装了 GPU 版 torch 的环境运行")

torch.manual_seed(0)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).cuda()
backend = HuggingFaceBackend(model, tokenizer)

group_size, group_count, trajectories = 8, 32, []
for group_index, sample in enumerate(make_dataset(group_count)):
    trajectories += rollout_group(
        backend, tokenizer, lambda: SingleTurnEnvironment(reward_function), sample, f"g{group_index}",
        group_size=group_size, seed=group_index,
    )

kept = episode_advantages(trajectories)
print(f"正确率={sum(trajectory.total_reward for trajectory in trajectories) / len(trajectories):.2f}  "
      f"有信号的组={len(kept) // group_size}/{group_count}")

rewards_by_reason = collections.defaultdict(list)
for trajectory in trajectories:
    rewards_by_reason[trajectory.termination_reason].append(trajectory.total_reward)
for reason, rewards in sorted(rewards_by_reason.items()):
    print(f"  {reason}: {len(rewards)} 条，平均回报={sum(rewards) / len(rewards):.2f}")
