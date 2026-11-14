"""对一组已完成运行的最终检查点重新贪心评测（含新环境指标）。

用法：python -m examples.reevaluate_checkpoints outputs/comparison-hidden
"""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.guess_number import GuessNumberEnvironment, make_dataset
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.evaluation import evaluate

root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/comparison-hidden")
configuration = json.loads((root / "grpo-seed0" / "resolved_configuration.json").read_text(encoding="utf-8"))
model_name = configuration["model"]["name"]
max_steps = configuration["training"].get("max_steps", 6)
evaluation_options = configuration["task"].get("evaluation_dataset_options", {"size": 200, "seed": 1})
environment_options = configuration["task"].get("environment_options", {})
evaluation_dataset = make_dataset(**evaluation_options)

from peft import LoraConfig, get_peft_model

print(f"加载基座 {model_name}……")
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
lora = configuration["model"]
torch.manual_seed(0)
model = get_peft_model(base_model, LoraConfig(
    r=lora.get("lora_rank", 16),
    lora_alpha=lora.get("lora_alpha", 32),
    lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=lora.get("lora_target_modules", "all-linear")))
backend = HuggingFaceBackend(model, tokenizer)

def run_eval(label):
    metrics = evaluate(backend, tokenizer, lambda: GuessNumberEnvironment(**environment_options),
                       evaluation_dataset, max_steps)
    print(f"{label}:")
    for name in ("evaluation_success_fraction", "evaluation_mean_reward",
                 "evaluation_environment_informative_step_fraction",
                 "evaluation_environment_split_quality"):
        print(f"  {name}: {metrics.get(name)}")
    return metrics

results = [("baseline", run_eval("基线（随机初始化 LoRA）"))]
for run_directory in sorted(path for path in root.iterdir() if path.is_dir()):
    checkpoint = run_directory / "checkpoint-0080"
    if not (checkpoint / "training_state.pt").exists():
        print(f"跳过 {run_directory.name}：没有 checkpoint-0080")
        continue
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["trainable_parameters"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    results.append((run_directory.name, run_eval(run_directory.name)))

print("\n汇总:")
for name, metrics in results:
    print(f"  {name:14s}  success={metrics.get('evaluation_success_fraction'):.3f}"
          f"  informative={metrics.get('evaluation_environment_informative_step_fraction'):.3f}"
          f"  split_quality={metrics.get('evaluation_environment_split_quality'):.3f}"
          f"  reward={metrics.get('evaluation_mean_reward'):.3f}")
