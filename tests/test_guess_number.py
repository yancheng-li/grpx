import pytest

from examples.guess_number import GuessNumberEnvironment

def play(guesses, target=30, reveal_interval=False):
    environment = GuessNumberEnvironment(reveal_interval=reveal_interval)
    environment.reset({"target": target, "lowest": 1, "highest": 50}, seed=0)
    results = [environment.step(guess) for guess in guesses]
    return environment, results

@pytest.mark.parametrize("guesses", [["30"], ["25", "30"], ["10", "40", "20", "35", "30"], ["1", "2", "3", "30"]])
def test_success_totals_exactly_one_regardless_of_step_count(guesses):
    environment, results = play(guesses)
    assert sum(reward for _, reward, _ in results) == pytest.approx(1.0)
    assert results[-1][2] is True
    assert environment.is_successful() is True

def test_narrowing_reward_is_proportional_to_interval_shrink():
    _, results = play(["25", "40"])
    assert [reward for _, reward, _ in results] == pytest.approx([0.5 * 25 / 50, 0.5 * 11 / 50])

@pytest.mark.parametrize("guess", ["80", "0", "我觉得是二十"])
def test_useless_guesses_earn_nothing(guess):
    environment, results = play([guess])
    assert results[0][1] == 0.0
    assert environment.state_key() == "1-50"
    assert environment.is_successful() is False

def test_already_excluded_guess_earns_nothing():
    environment, results = play(["12", "40", "10"], target=30)
    assert results[2][1] == 0.0
    assert environment.state_key() == "13-39"

def test_reward_ordering_verbose_lt_random_lt_narrowing_lt_success():
    random_total = sum(reward for _, reward, _ in play(["80", "90"])[1])
    narrowing_total = sum(reward for _, reward, _ in play(["10", "40", "20"])[1])
    success_total = sum(reward for _, reward, _ in play(["10", "30"])[1])
    assert 0.0 <= random_total < narrowing_total < 0.5 < success_total == pytest.approx(1.0)

def test_reveal_interval_appends_known_range():
    _, hidden = play(["25"])
    _, revealed = play(["25"], reveal_interval=True)
    assert hidden[0][0][0]["content"] == "25 小了，再猜。"
    assert revealed[0][0][0]["content"] == "25 小了，再猜。现在答案在 26 到 50 之间。"

def test_informative_step_fraction_counts_non_numeric_in_denominator():
    # 废话、区间内、区间外 → 1/3（旧定义是 1/2）
    environment, _ = play(["请给提示", "25", "80"])
    assert environment.metrics()["informative_step_fraction"] == pytest.approx(1 / 3)

def test_only_non_numeric_step_is_zero_not_one():
    environment, _ = play(["请给提示"])
    assert environment.metrics()["informative_step_fraction"] == 0.0

def test_first_step_hit_is_one():
    environment, _ = play(["30"])
    assert environment.metrics()["informative_step_fraction"] == 1.0

@pytest.mark.parametrize("guess, expected", [("25", 1.0), ("26", 1.0), ("1", 0.0), ("13", 0.5)])
def test_split_quality_on_full_interval(guess, expected):
    environment, _ = play([guess])
    assert environment.metrics()["split_quality"] == pytest.approx(expected)

def test_split_quality_two_left_is_always_one():
    environment = GuessNumberEnvironment()
    environment.reset({"target": 30, "lowest": 1, "highest": 50}, seed=0)
    environment.low, environment.high = 30, 31
    environment.step_count = 0
    environment.informative_guess_count = 0
    environment.split_quality_total = 0.0
    environment.step("30")
    assert environment.metrics()["split_quality"] == pytest.approx(1.0)

def test_split_quality_zero_for_non_numeric_or_out_of_range():
    environment, _ = play(["废话"])
    assert environment.metrics()["split_quality"] == 0.0
    environment, _ = play(["80"])
    assert environment.metrics()["split_quality"] == 0.0
