"""固定 500 道三位数加法，贪心评测检查点，并记下每题对错。

评测集和 scripts/eval_d3_500.py 相同：
  make_dataset(size=500, digits=3, seed=1, prompt_variant="english_strict")

用法：

    python scripts/eval_d3_500_items.py --checkpoints outputs/<run>/checkpoint-0030 outputs/<run>/checkpoint-0060 --output outputs/<run>-d3-500.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.arithmetic import is_correct, make_dataset
from grpx.backends.huggingface import HuggingFaceBackend
from grpx.environment import SingleTurnEnvironment
from grpx.rollout import rollout_groups

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
EVAL_SIZE = 500
EVAL_DIGITS = 3
EVAL_SEED = 1


def score_checkpoint(backend, tokenizer, environment_factory, dataset):
    trajectories = rollout_groups(
        backend, tokenizer, environment_factory,
        [(f"evaluation-{index}", sample, index) for index, sample in enumerate(dataset)],
        group_size=1, max_steps=1, greedy=True, truncated_reward=0.0)
    items = []
    for index, (sample, trajectory) in enumerate(zip(dataset, trajectories, strict=True)):
        completion = tokenizer.decode(
            trajectory.steps[-1].action_token_ids, skip_special_tokens=True) if trajectory.steps else ""
        correct = is_correct(completion, sample)
        if trajectory.is_successful is not None and bool(trajectory.is_successful) != correct:
            raise RuntimeError(
                f"第 {index} 题的成功标志和环境判定不一致：{trajectory.is_successful} / {correct}")
        items.append({
            "index": index,
            "prompt": sample["prompt"],
            "answer": sample["answer"],
            "completion": completion,
            "correct": correct,
        })
    correct_count = sum(item["correct"] for item in items)
    return {
        "n": len(items),
        "correct_count": correct_count,
        "success_fraction": correct_count / len(items) if items else 0.0,
        "items": items,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-rows-per-batch", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    dataset = make_dataset(
        size=EVAL_SIZE, digits=EVAL_DIGITS, seed=EVAL_SEED, prompt_variant="english_strict")
    environment_factory = lambda: SingleTurnEnvironment(lambda prompt, completion, sample: 0.0, success_function=is_correct)

    print(f"loading base {MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    results = {
        "dataset": {
            "size": EVAL_SIZE,
            "digits": EVAL_DIGITS,
            "seed": EVAL_SEED,
            "prompt_variant": "english_strict",
            "builder": "examples.arithmetic:make_dataset",
        },
        "checkpoints": [],
    }
    for checkpoint in arguments.checkpoints:
        checkpoint = checkpoint.resolve()
        print(f"=== {checkpoint} ===", flush=True)
        model = PeftModel.from_pretrained(base, str(checkpoint))
        model.eval()
        backend = HuggingFaceBackend(
            model, tokenizer,
            max_new_tokens=arguments.max_new_tokens,
            max_rows_per_batch=arguments.max_rows_per_batch,
        )
        row = score_checkpoint(backend, tokenizer, environment_factory, dataset)
        row["checkpoint"] = str(checkpoint)
        results["checkpoints"].append({key: value for key, value in row.items() if key != "items"})
        results["checkpoints"][-1]["items"] = row["items"]
        print(f"success={row['success_fraction']:.4f}  correct={row['correct_count']}/{row['n']}", flush=True)
        base = model.unload()
        del model
        torch.cuda.empty_cache()
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", arguments.output, flush=True)


if __name__ == "__main__":
    main()
