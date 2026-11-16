import pytest

from examples.arithmetic import (
    PARTIAL_CREDIT_CEILING, extract_final_number, is_correct, make_dataset,
    remove_thousands_separators, reward_function,
)

SAMPLE = {"answer": 11802}


@pytest.mark.parametrize("completion", [
    "11802", "11,802", "11，802", "答案是 11,802。", "1234 + 10568 = 11802", "答案是11,802，对吗",
])
def test_correct_answers(completion):
    assert reward_function("", completion, SAMPLE) == 1.0
    assert is_correct(completion, SAMPLE)


@pytest.mark.parametrize("completion, expected_reward", [
    ("1,1802", 0.4),              # 取到 1802，五位里对了四位
    ("1,11,802", 0.3),            # 取到 802，五位里对了三位
    ("11802 还是 11803", 0.4),     # 以最后一个数为准
    ("-11802", 0.5),              # 数字全对但符号错，拿到部分分上限
    ("11,8020", 0.0),             # 取到 8020，逐位都对不上
    ("", 0.0),
    ("没有数字", 0.0),
])
def test_wrong_or_malformed_answers(completion, expected_reward):
    assert reward_function("", completion, SAMPLE) == pytest.approx(expected_reward)
    assert not is_correct(completion, SAMPLE)


def test_partial_credit_aligns_from_units_digit():
    # 15201 对 8221：个位和百位相同，分母取较长的五位
    assert reward_function("", "8221", {"answer": 15201}) == pytest.approx(0.2)


def test_partial_credit_uses_longer_length_as_denominator():
    # 预测比答案长：111802 对 11802，后五位全对，分母是六
    assert reward_function("", "111802", SAMPLE) == pytest.approx(PARTIAL_CREDIT_CEILING * 5 / 6)


@pytest.mark.parametrize("completion", ["1,1802", "11803", "-11802", "111802", "8221", "99999"])
def test_wrong_answer_never_reaches_full_reward(completion):
    assert reward_function("", completion, SAMPLE) <= PARTIAL_CREDIT_CEILING


@pytest.mark.parametrize("text", ["5678,912", "1,11,802", "1,1802", "11,8020", "12,34"])
def test_malformed_numbers_are_not_merged(text):
    assert remove_thousands_separators(text) == text


def test_multiple_thousands_separators():
    assert reward_function("", "1,234,567", {"answer": 1234567}) == 1.0


@pytest.mark.parametrize("completion, expected_number", [
    ("11,802", "11802"), ("3 加 4 等于 7", "7"), ("-5", "-5"), ("", None),
])
def test_extract_final_number(completion, expected_number):
    assert extract_final_number(completion) == expected_number


def test_dataset_size_and_answers():
    dataset = make_dataset(32, digits=4)
    assert len(dataset) == 32
    assert all(2000 <= sample["answer"] <= 19998 for sample in dataset)
    assert all(sample["operation"] == "addition" for sample in dataset)


def test_dataset_prompt_is_english_and_digit_count_alias():
    sample = make_dataset(1, digits=2, seed=0)[0]
    assert sample["prompt"].startswith("Compute ")
    assert " + " in sample["prompt"]
    assert "Output only the final integer" in sample["prompt"]
    assert make_dataset(4, digit_count=3, seed=1) == make_dataset(4, digits=3, seed=1)


def test_dataset_is_reproducible():
    assert make_dataset(8, seed=3) == make_dataset(8, seed=3)
    assert make_dataset(8, seed=3) != make_dataset(8, seed=4)


def test_default_addition_matches_pre_operation_extension():
    # 改动前 make_dataset(size=500, seed=1, digits=3) 的前几道题；RNG 与题面必须保持不变
    expected = [
        ("Compute 237 + 682. ", 919),
        ("Compute 967 + 921. ", 1888),
        ("Compute 882 + 164. ", 1046),
        ("Compute 361 + 220. ", 581),
        ("Compute 607 + 879. ", 1486),
    ]
    dataset = make_dataset(500, seed=1, digits=3)
    for index, (prefix, answer) in enumerate(expected):
        assert dataset[index]["prompt"].startswith(prefix)
        assert dataset[index]["answer"] == answer
        assert dataset[index]["digits"] == 3
        assert dataset[index]["operation"] == "addition"
    assert make_dataset(8, seed=3) == make_dataset(8, seed=3, operation="addition")


def test_multiplication_dataset():
    sample = make_dataset(1, digits=2, seed=0, operation="multiplication")[0]
    assert " * " in sample["prompt"]
    assert " + " not in sample["prompt"]
    assert sample["operation"] == "multiplication"
    assert sample["digits"] == 2
    # 两位×两位：答案 2–4 位
    assert 10 <= sample["answer"] <= 99 * 99
    # 与加法同 seed 时操作数相同、答案不同
    addition = make_dataset(1, digits=2, seed=0, operation="addition")[0]
    assert sample["prompt"].split(" * ")[0] == addition["prompt"].split(" + ")[0]
    assert sample["answer"] != addition["answer"]


def test_operation_must_be_known():
    with pytest.raises(ValueError, match="operation"):
        make_dataset(1, operation="mul")


def test_digits_can_exceed_ten():
    sample = make_dataset(1, digits=12, seed=0)[0]
    assert sample["digits"] == 12
    assert len(str(sample["answer"])) >= 12


def test_environment_metrics_split_by_operation_and_digits():
    from examples.arithmetic import ArithmeticEnvironment

    environment = ArithmeticEnvironment()
    multiplication = make_dataset(1, digits=3, seed=0, operation="multiplication")[0]
    messages = environment.reset(multiplication, seed=0)
    environment.step("0")
    metrics = environment.metrics()
    # 乘法不再报 success_digits_*（该名历史上专指加法）
    assert "success_digits_3" not in metrics
    assert metrics["success_multiplication_digits_3"] == 0.0
    assert "success_addition_digits_3" not in metrics

    environment.reset(multiplication, seed=0)
    environment.step(str(multiplication["answer"]))
    metrics = environment.metrics()
    assert "success_digits_3" not in metrics
    assert metrics["success_multiplication_digits_3"] == 1.0

    addition = make_dataset(1, digits=2, seed=0, operation="addition")[0]
    environment.reset(addition, seed=0)
    environment.step(str(addition["answer"]))
    metrics = environment.metrics()
    assert metrics["success_digits_2"] == 1.0
    assert metrics["success_addition_digits_2"] == 1.0
    assert messages  # reset 返回了对话