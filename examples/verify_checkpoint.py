"""在新进程里加载基座模型和最后一个检查点重新评测，结果应与训练结束时的最终评测逐位一致。

用法：python -m examples.verify_checkpoint outputs/<运行名>
"""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.arithmetic import is_correct, make_dataset, reward_function
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.environment import SingleTurnEnvironment
from grpx.evaluation import evaluate

run_directory = Path(sys.argv[1])
config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
records = [json.loads(line) for line in (run_directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
final_record = next(record for record in reversed(records) if record["iteration"] == config["iterations"])
checkpoint_directory = run_directory / f"checkpoint-{config['iterations']:04d}"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)    # 须与训练时的精度一致
model = PeftModel.from_pretrained(base_model, checkpoint_directory).cuda()
backend = HuggingFaceBackend(model, tokenizer)

reloaded = evaluate(backend, tokenizer, lambda: SingleTurnEnvironment(reward_function, is_correct),
                    make_dataset(200, seed=1), config["max_steps"])
# 旧版本训练记录里可能没有新加的指标，只比较两边都有的
compared_names = [name for name in reloaded if name in final_record]
mismatches = {name: (final_record[name], reloaded[name]) for name in compared_names if final_record[name] != reloaded[name]}
print(f"检查点：{checkpoint_directory}")
for name in compared_names:
    print(f"  {name}: 训练结束时={final_record[name]}  重新加载后={reloaded[name]}")
if mismatches:
    raise SystemExit(f"不一致：{mismatches}")
print("逐位一致")
