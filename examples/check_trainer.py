import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.arithmetic import make_dataset, reward_function
from grpx.advantage import episode_advantages
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.environment import SingleTurnEnvironment
from grpx.rollout import rollout_group
from grpx.trainer import action_log_probabilities, build_batch, train_on_batch

torch.manual_seed(0)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 用 float32：bfloat16 下 1e-5 量级的参数更新会被舍入掉
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
backend = HuggingFaceBackend(model, tokenizer)

group_size, trajectories = 8, []
for group_index, sample in enumerate(make_dataset(16)):
    trajectories += rollout_group(
        backend, tokenizer, lambda: SingleTurnEnvironment(reward_function), sample,
        f"group-{group_index}", group_size=group_size, seed=group_index)
kept = episode_advantages(trajectories)
print(f"参与训练的轨迹：{len(kept)} 条")

padding_token_id = tokenizer.pad_token_id
if padding_token_id is None:
    padding_token_id = min(backend.end_token_ids)
batch = build_batch(kept, padding_token_id, model.device)

model.eval()
with torch.no_grad():
    log_probabilities, mask = action_log_probabilities(model, batch)
    # 对照：对全部位置算 logits，再取回答位置
    full_logits = model(input_ids=batch["input_token_ids"], attention_mask=batch["attention_mask"]).logits
    full_logits = full_logits[:, :-1].float() / batch["temperatures"][:, None, None]
    reference = torch.log_softmax(full_logits, dim=-1).gather(-1, batch["input_token_ids"][:, 1:, None]).squeeze(-1)
mean_log_probability = ((log_probabilities * mask).sum() / mask.sum()).item()
print(f"回答 token 的平均对数概率：{mean_log_probability:.3f}  "
      f"（全量 logits 对照：{((reference * mask).sum() / mask.sum()).item():.3f}，"
      f"逐 token 最大差 {(log_probabilities - reference)[mask].abs().max().item():.2e}）")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
history = train_on_batch(model, optimizer, batch, number_of_updates=10)
print(f"首步 ratio 最大偏差：{history[0]['max_ratio_deviation']:.2e}")
for update_index, metrics in enumerate(history):
    print(f"第 {update_index} 次更新  loss={metrics['loss']:+.4f}  "
          f"approximate_kl={metrics['approximate_kl']:.5f}  "
          f"clip_fraction={metrics['clip_fraction']:.3f}  "
          f"median_absolute_log_ratio={metrics['median_absolute_log_ratio']:.5f}  "
          f"percentile_95_absolute_log_ratio={metrics['percentile_95_absolute_log_ratio']:.5f}")