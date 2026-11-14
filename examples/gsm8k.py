import random
import re

from datasets import load_dataset

INSTRUCTION = ("Solve the problem step by step. "
               "End your response with a line of the form '#### <number>'.")
FORMAT_REWARD = 0.1


# 千分位：1,234 或 12,345,678；不接受 1,43（旧实现会剥掉逗号读成 143，判分偏松）
_THOUSAND_SEP_NUMBER = re.compile(r"-?\d{1,3}(,\d{3})+(\.\d+)?")
_PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(text):
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
    if match is None:
        return None
    raw = match.group()
    if "," in raw:
        if _THOUSAND_SEP_NUMBER.fullmatch(raw) is None:
            return None
        raw = raw.replace(",", "")
    elif _PLAIN_NUMBER.fullmatch(raw) is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def extract_answer(completion):
    """只认 '####' 之后的数字；没有按格式作答则返回 None"""
    if "####" not in completion:
        return None
    return parse_number(completion.split("####")[-1])


def make_dataset(split="train", size=None, seed=0):
    rows = load_dataset("openai/gsm8k", "main", split=split)
    samples = [{"prompt": f"{row['question']}\n\n{INSTRUCTION}",
                "question": row["question"],
                "answer": parse_number(row["answer"].split("####")[-1])}
               for row in rows]
    if size is not None:
        random.Random(seed).shuffle(samples)
        samples = samples[:size]
    return samples


def success_function(completion, sample):
    predicted = extract_answer(completion)
    return predicted is not None and abs(predicted - sample["answer"]) < 1e-6


def reward_function(prompt, completion, sample):
    predicted = extract_answer(completion)
    if predicted is None:
        return 0.0                    # 没按格式作答，或被截断
    if abs(predicted - sample["answer"]) < 1e-6:
        return 1.0
    return FORMAT_REWARD              # 格式对、答案错
