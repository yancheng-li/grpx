from examples.gsm8k_calculator import INVALID_TURN_FEEDBACK, CalculatorEnvironment


def test_invalid_turn_cost_deducts_on_success_only():
    env = CalculatorEnvironment(
        success_function="examples.gsm8k:success_function",
        reward_function="examples.gsm8k:reward_function",
        question_field="question",
        invalid_turn_cost=0.02,
    )
    env.reset({"question": "What is 2+2?", "answer": 4.0}, seed=0)
    messages, reward, done = env.step("I need to think.")
    assert done is False
    assert reward == 0.0
    assert env.invalid_turn_count == 1
    assert messages == [{"role": "user", "content": INVALID_TURN_FEEDBACK}]

    _messages, reward, done = env.step("#### 4")
    assert done is True
    assert reward == 0.98


def test_invalid_turn_cost_does_not_penalize_wrong_answer():
    from examples.gsm8k import FORMAT_REWARD

    env = CalculatorEnvironment(
        success_function="examples.gsm8k:success_function",
        reward_function="examples.gsm8k:reward_function",
        question_field="question",
        invalid_turn_cost=0.02,
    )
    env.reset({"question": "What is 2+2?", "answer": 4.0}, seed=0)
    env.step("placeholder")
    _messages, reward, done = env.step("#### 7")
    assert done is True
    # 答错只拿格式分，不扣无效轮代价（与 call_cost 同口径）
    assert reward == FORMAT_REWARD


def test_invalid_turn_default_still_zero():
    env = CalculatorEnvironment(
        success_function="examples.gsm8k:success_function",
        reward_function="examples.gsm8k:reward_function",
        question_field="question",
    )
    env.reset({"question": "What is 2+2?", "answer": 4.0}, seed=0)
    _messages, reward, done = env.step("no tools here")
    assert done is False and reward == 0.0
