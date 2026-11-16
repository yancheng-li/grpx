import random
import re

from grpx.environment import SingleTurnEnvironment

ALLOWED_OPERATIONS = ("addition", "multiplication")


def make_dataset(size, digits=4, seed=0, digit_count=None, operation="addition"):
    """生成算术题。digits / digit_count 为每个操作数的位数，默认 4。

    operation 取 "addition" 或 "multiplication"（默认加法，与历史评测集兼容）。
    """
    if digit_count is not None:
        digits = digit_count
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(
            f"operation 只能是 {ALLOWED_OPERATIONS!r}，收到 {operation!r}")
    if not isinstance(digits, int) or isinstance(digits, bool) or digits < 1:
        raise ValueError(f"digits 必须是正整数，收到 {digits!r}")
    random_generator = random.Random(seed)
    lowest, highest = 10 ** (digits - 1), 10 ** digits - 1
    operator_symbol = "+" if operation == "addition" else "*"
    dataset = []
    for _ in range(size):
        left = random_generator.randint(lowest, highest)
        right = random_generator.randint(lowest, highest)
        answer = left + right if operation == "addition" else left * right
        dataset.append(
            {
                "prompt": (
                    f"Compute {left} {operator_symbol} {right}. "
                    "Output only the final integer answer with no equation, "
                    "explanation, or other text."
                ),
                "answer": answer,
                "digits": digits,
                "operation": operation,
            }
        )
    return dataset


def make_mixed_digit_dataset(size, seed=0, digit_counts=(2, 3), operation="addition"):
    """按位数均分的混合算术集；每条样本带 digits / operation，便于评测时分开报成功率。"""
    digit_counts = tuple(digit_counts)
    if not digit_counts:
        raise ValueError("digit_counts 不能为空")
    if size % len(digit_counts) != 0:
        raise ValueError(
            f"size={size} 不能被位数种类数 {len(digit_counts)} 整除，无法均分")
    per_bucket = size // len(digit_counts)
    dataset = []
    for offset, digits in enumerate(digit_counts):
        dataset.extend(make_dataset(
            size=per_bucket, digits=digits, seed=seed + 10007 * (offset + 1),
            operation=operation))
    random.Random(seed).shuffle(dataset)
    return dataset


# 规范的千分位数字：首段 1 到 3 位，之后每段恰好三位，前后不与其他数字或分隔符相连
THOUSANDS_NUMBER = re.compile(
    r"(?<!\d)(?<!\d[,，])\d{1,3}(?:[,，]\d{3})+(?!\d)(?![,，]\d)")

PARTIAL_CREDIT_CEILING = 0.5


def remove_thousands_separators(text):
    return THOUSANDS_NUMBER.sub(
        lambda match: re.sub(r"[,，]", "", match.group()), text)


def extract_final_number(completion):
    numbers = re.findall(r"-?\d+", remove_thousands_separators(completion))
    return numbers[-1] if numbers else None


def is_correct(completion, sample):
    final_number = extract_final_number(completion)
    return final_number is not None and int(final_number) == sample["answer"]


def reward_function(prompt, completion, sample):
    final_number = extract_final_number(completion)
    if final_number is None:
        return 0.0
    if int(final_number) == sample["answer"]:
        return 1.0
    # 没答对：从个位起逐位对齐，按正确位数给部分分，上限为 PARTIAL_CREDIT_CEILING
    predicted_digits = final_number.lstrip("-")
    expected_digits = str(sample["answer"])
    matching_digit_count = sum(
        predicted_digit == expected_digit
        for predicted_digit, expected_digit
        in zip(reversed(predicted_digits), reversed(expected_digits)))
    longer_length = max(len(predicted_digits), len(expected_digits))
    return PARTIAL_CREDIT_CEILING * matching_digit_count / longer_length


class ArithmeticEnvironment(SingleTurnEnvironment):
    """单轮算术；metrics 按样本 digits / operation 分开记成功率，供混合集评测。"""

    def __init__(self):
        super().__init__(reward_function, success_function=is_correct)

    def metrics(self):
        if not hasattr(self, "sample") or self.successful is None:
            return {}
        digits = self.sample.get("digits")
        operation = self.sample.get("operation")
        if digits is None:
            return {}
        # success_digits_* 历史上专指加法；乘法只用 success_multiplication_digits_*，避免读日志时混淆
        metrics = {}
        if operation is None or operation == "addition":
            metrics[f"success_digits_{digits}"] = float(self.successful)
        if operation is not None:
            metrics[f"success_{operation}_digits_{digits}"] = float(self.successful)
        return metrics