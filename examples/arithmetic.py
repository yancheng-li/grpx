import random
import re


def make_dataset(size, digits=4, seed=0):
    random_generator = random.Random(seed)
    lowest, highest = 10 ** (digits - 1), 10 ** digits - 1
    dataset = []
    for _ in range(size):
        left = random_generator.randint(lowest, highest)
        right = random_generator.randint(lowest, highest)
        dataset.append(
            {"prompt": f"计算 {left} + {right}，只输出数字。", "answer": left + right}
        )
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