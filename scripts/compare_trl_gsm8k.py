"""用 TRL 的 GRPOTrainer 在 GSM8K 上训练一个和 grpx 设置对齐的模型，作为对照。

提示词和回报函数直接复用 examples/gsm8k.py，保证两边完全相同。
训练结束后只保存 LoRA 适配器；评测请用 scripts/eval_trl_gsm8k_adapter.py
对这个适配器跑完整测试集，这样两边的评测口径也完全相同。

依赖（不要装进 grpx 本身）：请另建环境安装 TRL 1.18，以及与之匹配的 vLLM
（对照实验用过 vLLM 0.33；版本需满足 TRL 文档对 colocate 模式的要求）。
GRPOConfig 的参数名随 TRL 版本变化，请以你安装的版本的文档为准。

在项目根目录下运行：
    python scripts/compare_trl_gsm8k.py --seed 0 --output outputs/trl-gsm8k-s0 --use-vllm
"""

import argparse

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from examples import gsm8k

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GROUPS_PER_UPDATE = 32          # grpx 的 groups_per_iteration
GROUP_SIZE = 8                  # grpx 的 group_size
UPDATE_COUNT = 60               # grpx 的 iterations（每轮只更新一次）
COMPLETIONS_PER_FORWARD = 8     # 只影响显存，放不下就调小，并相应调大累积步数


def build_dataset():
    samples = gsm8k.make_dataset(split="train")
    return Dataset.from_list([
        {"prompt": [{"role": "user", "content": sample["prompt"]}], "answer": sample["answer"]}
        for sample in samples
    ])


def reward(prompts, completions, answer, **kwargs):
    """和 grpx 用同一个回报函数：答对 1 分，格式对但答错 0.1 分，其余 0 分。"""
    return [
        gsm8k.reward_function(prompt[-1]["content"], completion[0]["content"], {"answer": gold})
        for prompt, completion, gold in zip(prompts, completions, answer)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=UPDATE_COUNT,
                        help="更新次数；冒烟可先设 2～3")
    parser.add_argument("--groups-per-update", type=int, default=GROUPS_PER_UPDATE,
                        help="每次更新的题数；正式对照用 32，冒烟可先设 2")
    parser.add_argument("--completions-per-forward", type=int, default=COMPLETIONS_PER_FORWARD,
                        help="每次前向的完成条数，须能被 num_generations 整除；显存不够就调小")
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--use-vllm", action="store_true",
                        help="用 vLLM 采样（同卡时只能用 colocate 模式，且需要 TRL 支持的 vLLM 版本）")
    arguments = parser.parse_args()

    groups_per_update = arguments.groups_per_update
    completions_per_forward = arguments.completions_per_forward
    total_completions = groups_per_update * GROUP_SIZE
    if completions_per_forward % GROUP_SIZE != 0:
        raise SystemExit(
            f"completions_per_forward={completions_per_forward} 必须能被 "
            f"num_generations={GROUP_SIZE} 整除（TRL 要求 generation batch 含完整题组）"
        )
    if total_completions % completions_per_forward != 0:
        raise SystemExit(
            f"completions_per_forward={completions_per_forward} 必须整除 "
            f"groups_per_update*group_size={total_completions}"
        )
    accumulation_steps = total_completions // completions_per_forward
    configuration = GRPOConfig(
        output_dir=arguments.output,
        seed=arguments.seed,
        # —— 和 grpx 对齐的部分 ——
        num_generations=GROUP_SIZE,
        per_device_train_batch_size=completions_per_forward,
        gradient_accumulation_steps=accumulation_steps,   # 正式：32 题 × 8 条；冒烟可缩小
        max_steps=arguments.max_steps,
        learning_rate=5e-5,
        warmup_steps=min(5, arguments.max_steps),
        lr_scheduler_type="constant_with_warmup",
        max_completion_length=arguments.max_completion_length,
        temperature=1.0,
        loss_type="grpo",                # 每条回答内取平均，对应 grpx 的 grpo 预设
        beta=0.0,                        # 不加 KL 项
        epsilon=0.2,
        mask_truncated_completions=True, # 见说明：grpx 是把被截断的回答记 0 分，这里是把它们排除在损失之外
        # —— 运行相关 ——
        bf16=True,
        use_vllm=arguments.use_vllm,
        vllm_mode="colocate",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=configuration,
        reward_funcs=reward,
        train_dataset=build_dataset(),
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.model.save_pretrained(arguments.output)


if __name__ == "__main__":
    main()
