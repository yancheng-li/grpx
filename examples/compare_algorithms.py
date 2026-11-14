"""猜数字上的算法对比：一次运行一个配置、一个种子。

用法：python -m examples.compare_algorithms --algorithm gigpo --seed 0 [--row-weighting per_trajectory]
      [--learning-rate 1e-5] [--iterations 30]
结果写到 outputs/comparison/<配置名>-seed<种子>/，用 examples.summarize_comparison 汇总。
"""
import argparse

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.guess_number import GuessNumberEnvironment, make_dataset
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.loop import ALGORITHM_PRESETS, TrainingConfig, train

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", choices=sorted(ALGORITHM_PRESETS), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--row-weighting", choices=["per_step", "per_trajectory"], default="per_step")
parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
parser.add_argument("--iterations", type=int, default=30)
parser.add_argument("--output-root", default="outputs/comparison")
arguments = parser.parse_args()

configuration_name = arguments.algorithm
if arguments.row_weighting != "per_step":
    configuration_name += f"-{arguments.row_weighting}"
if arguments.learning_rate != TrainingConfig.learning_rate:
    configuration_name += f"-lr{arguments.learning_rate:g}"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
torch.manual_seed(arguments.seed)    # LoRA 的初始化也随种子变
model = get_peft_model(
    model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    )
)

config = TrainingConfig.from_algorithm(
    arguments.algorithm,
    iterations=arguments.iterations,
    groups_per_iteration=16,
    learning_rate=arguments.learning_rate,
    row_weighting=arguments.row_weighting,
    seed=arguments.seed,
    max_steps=6,
    evaluation_interval=5,
    output_root=arguments.output_root,
    run_name=f"{configuration_name}-seed{arguments.seed}",
)
train(model, tokenizer, HuggingFaceBackend(model, tokenizer), make_dataset(2000, seed=0),
      lambda: GuessNumberEnvironment(reveal_interval=True), config,
      evaluation_dataset=make_dataset(200, seed=1))
