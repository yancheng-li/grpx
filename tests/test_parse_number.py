from examples.gsm8k import extract_answer, parse_number, reward_function, success_function


def test_parse_number_accepts_thousand_separators():
    assert parse_number("#### 1,234") == 1234.0
    assert parse_number("12,345,678") == 12345678.0
    assert parse_number("-1,234.5") == -1234.5


def test_parse_number_rejects_non_thousand_commas():
    # 旧实现会把 1,43 读成 143
    assert parse_number("#### 1,43") is None
    assert parse_number("1,2,3") is None
    assert parse_number("12,34") is None


def test_parse_number_plain_and_decimals():
    assert parse_number("42") == 42.0
    assert parse_number("-3.14") == -3.14
    assert parse_number("no number") is None


def test_extract_and_scoring_use_strict_commas():
    sample = {"answer": 143.0}
    assert extract_answer("reasoning\n#### 1,43") is None
    assert success_function("#### 1,43", sample) is False
    assert reward_function("", "#### 1,43", sample) == 0.0
    assert success_function("#### 143", sample) is True
    assert success_function("#### 1,234", {"answer": 1234.0}) is True
