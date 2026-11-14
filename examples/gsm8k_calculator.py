"""带计算器的 GSM8K：多轮工具调用任务。

模型每一轮要么调用计算器，要么给出最终答案：
- 调用计算器：在回答里写 <calc>表达式</calc>，环境在下一条消息里返回结果。一轮里可以写多个。
- 最终答案：回答里出现 "####"，并且它后面没有计算器调用，这一局结束，用和单轮版相同的函数判定对错。
  如果 "####" 后面还跟着一次调用（比如模型想在答案那一行里用计算器），按调用处理，
  这一局继续，模型拿到结果后再给出答案。
- 两者都没有：算无效回答，环境给一句提示，占用一步。

回报只看最终答案，调用计算器的次数不加分也不扣分。

答案的判定和打分都不在这个文件里实现，而是通过 success_function、reward_function 两个选项
引用单轮版的函数，保证两个版本的准确率和回报可以直接比较。没有给出最终答案的一局得 0 分。
"""

import ast
import importlib
import re
import warnings
from fractions import Fraction

MAXIMUM_EXPRESSION_LENGTH = 100
MAXIMUM_CALLS_PER_TURN = 5
MAXIMUM_ABSOLUTE_VALUE = Fraction(10) ** 15

CALCULATOR_CALL = re.compile(r"<calc>(.*?)</calc>", re.DOTALL)
FINAL_ANSWER_MARKER = "####"
THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
PERCENTAGE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# 两种提示。"optional_tool" 以单轮版的提示为主体，只在后面补一段说明计算器可用，
# 目的是让不用计算器的模型表现和单轮版一样；"worked_example" 带一段完整的示范对话。
SINGLE_TURN_INSTRUCTION = (
    "Solve the problem step by step. "
    "End your response with a line of the form '#### <number>'."
)

PROMPT_TEMPLATES = {
    "optional_tool": (
        "{question}\n\n"
        + SINGLE_TURN_INSTRUCTION
        + "\n\nYou also have a calculator. To use it, write <calc>expression</calc> "
        "(numbers, + - * / and parentheses) and end your message right there, before giving the final answer. "
        "The result will be sent to you, and then you continue solving. "
        "You can use it several times, or not at all."
    ),
    "worked_example": (
        SINGLE_TURN_INSTRUCTION
        + "\n\nYou also have a calculator. To use it, write <calc>expression</calc> "
        "(numbers, + - * / and parentheses) and end your message right there. "
        "The result will be sent to you, and then you continue solving.\n\n"
        "Example\n\n"
        "Problem: A crate holds 12 rows of 15 apples. 37 of the apples are rotten. "
        "How many good apples are there?\n"
        "You: The crate holds 12 rows with 15 apples in each row, so first I need the total. <calc>12*15</calc>\n"
        "Calculator: 12*15 = 180\n"
        "You: There are 180 apples in total. 37 of them are rotten, so I subtract them. <calc>180-37</calc>\n"
        "Calculator: 180-37 = 143\n"
        "You: So 143 apples are good.\n#### 143\n\n"
        "Now solve this problem.\n\nProblem: {question}"
    ),
}

INVALID_TURN_FEEDBACK = (
    "Your message had no calculator call and no final answer. "
    "Either write <calc>expression</calc>, or finish with a line '#### <number>'."
)


class CalculatorError(Exception):
    pass


def _evaluate_node(node):
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        # 用源文本里的写法构造分数，避免 0.1 + 0.2 这类浮点误差。
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        else:
            if right == 0:
                raise CalculatorError("division by zero")
            result = left / right
        if abs(result) > MAXIMUM_ABSOLUTE_VALUE:
            raise CalculatorError("result is too large")
        return result
    raise CalculatorError("only numbers, + - * / and parentheses are allowed")


def format_number(value):
    """整数原样输出；非整数最多保留 6 位小数，去掉末尾的 0。"""
    if value.denominator == 1:
        return str(value.numerator)
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


def calculate(expression):
    """安全地计算一个算术表达式，返回结果的字符串。不使用 eval。出错时抛出 CalculatorError。"""
    cleaned = expression.strip()
    if not cleaned:
        raise CalculatorError("the expression is empty")
    if len(cleaned) > MAXIMUM_EXPRESSION_LENGTH:
        raise CalculatorError("the expression is too long")
    cleaned = cleaned.replace("×", "*").replace("÷", "/").replace("$", "")
    cleaned = THOUSANDS_SEPARATOR.sub("", cleaned)
    cleaned = PERCENTAGE.sub(r"(\1/100)", cleaned)
    try:
        with warnings.catch_warnings():
            # 形如 "3kg" 的输入会让解析器发出 SyntaxWarning，这里不需要它。
            warnings.simplefilter("ignore")
            tree = ast.parse(cleaned, mode="eval")
    except (SyntaxError, ValueError):
        raise CalculatorError("the expression could not be parsed") from None
    try:
        return format_number(_evaluate_node(tree))
    except (RecursionError, OverflowError, ValueError):
        raise CalculatorError("the expression could not be evaluated") from None


def import_object(path):
    module_path, _, object_name = path.partition(":")
    if not module_path or not object_name:
        raise ValueError(f"应写成 '模块路径:对象名'，实际是 {path!r}")
    return getattr(importlib.import_module(module_path), object_name)


class CalculatorEnvironment:
    """success_function：单轮版的成功判定，写成 '模块路径:函数名'，签名是 (completion, sample)。
    reward_function：单轮版的回报函数，签名是 (prompt, completion, sample)，用来给最终答案打分。
        不填时答对 1 分、其余 0 分。
    question_field：样本里题目文本所在的字段名。
    prompt_variant："optional_tool"（默认）或 "worked_example"。
    maximum_response_characters：一局里模型写出的总字符数上限。某一轮结束后累计超过它、
        而这一轮又没有给出最终答案，这一局就以失败结束（0 分）。它限制对话的总长度，
        避免个别特别长的对话在训练时占用过多显存。设为 None 则不限制。
    call_cost：每次调用计算器的代价。只在答对时扣：答对的回报是"原回报 - call_cost × 调用次数"，
        不低于 0；答错和没作答的回报不变。这样在不需要工具的题上，少调用的正确回答得分更高；
        而答错的回答不会因为尝试过调用而被多扣分，不妨碍模型在需要工具的题上摸索。默认 0，即不扣。
    invalid_turn_cost：无效轮代价。和 call_cost 一样只在答对时从最终回报里扣
        （每次无效轮扣这么多，不低于 0）；无效轮本身仍得 0，回报始终非负，
        不会让 truncated_reward=0 的截断相对「负分失败」占便宜。默认 0。
    source_field：样本里标明题目来源的字段名。样本带这个字段时，额外按来源分别统计
        成功率和计算器使用率（指标名里带上来源），用于混合数据集。没有这个字段时不受影响。"""

    def __init__(self, success_function, reward_function=None, question_field="question",
                 prompt_variant="optional_tool", maximum_response_characters=3500,
                 source_field="source", call_cost=0.0, invalid_turn_cost=0.0):
        if call_cost < 0:
            raise ValueError("call_cost 不能为负数")
        if invalid_turn_cost < 0:
            raise ValueError("invalid_turn_cost 不能为负数")
        self.call_cost = call_cost
        self.invalid_turn_cost = float(invalid_turn_cost)
        self.source_field = source_field
        if maximum_response_characters is not None and maximum_response_characters <= 0:
            raise ValueError("maximum_response_characters 必须是正数或 None")
        self.maximum_response_characters = maximum_response_characters
        if prompt_variant not in PROMPT_TEMPLATES:
            raise ValueError(f"未知的 prompt_variant：{prompt_variant!r}，可选 {sorted(PROMPT_TEMPLATES)}")
        self.prompt_template = PROMPT_TEMPLATES[prompt_variant]
        self.success_function = (
            import_object(success_function) if isinstance(success_function, str) else success_function
        )
        self.reward_function = (
            import_object(reward_function) if isinstance(reward_function, str) else reward_function
        )
        self.question_field = question_field

    def reset(self, sample, seed):
        if self.question_field not in sample:
            raise KeyError(
                f"样本里没有 {self.question_field!r} 字段；"
                "用 environment_options 的 question_field 指定题目文本所在的字段"
            )
        self.sample = sample
        self.successful = False
        self.answered = False
        self.turn_count = 0
        self.invalid_turn_count = 0
        self.calculator_call_count = 0
        self.calculator_error_count = 0
        self.answered_without_waiting = False
        self.response_character_count = 0
        self.exceeded_response_budget = False
        self.call_after_answer_marker = False
        self.prompt = self.prompt_template.format(question=sample[self.question_field])
        return [{"role": "user", "content": self.prompt}]

    def step(self, action_text):
        self.turn_count += 1
        self.response_character_count += len(action_text)

        calls = list(CALCULATOR_CALL.finditer(action_text))
        marker_position = action_text.rfind(FINAL_ANSWER_MARKER)
        call_follows_marker = bool(calls) and marker_position != -1 and calls[-1].end() > marker_position
        if call_follows_marker:
            self.call_after_answer_marker = True

        if marker_position != -1 and not call_follows_marker:
            self.answered = True
            # 同一条消息里既调用了计算器又给了最终答案：模型没有停下来等结果，
            # 调用后面的内容（包括它自己编的计算结果）都是在没有真实结果的情况下写的。
            self.answered_without_waiting = bool(calls)
            self.successful = bool(self.success_function(action_text, self.sample))
            if self.reward_function is None:
                reward = 1.0 if self.successful else 0.0
            else:
                reward = float(self.reward_function(self.prompt, action_text, self.sample))
            if self.successful:
                penalty = (self.call_cost * self.calculator_call_count
                           + self.invalid_turn_cost * self.invalid_turn_count)
                if penalty > 0:
                    reward = max(0.0, reward - penalty)
            return [], reward, True

        if (self.maximum_response_characters is not None
                and self.response_character_count > self.maximum_response_characters):
            self.exceeded_response_budget = True
            return [], 0.0, True

        expressions = [call.group(1) for call in calls]
        if not expressions:
            self.invalid_turn_count += 1
            return [{"role": "user", "content": INVALID_TURN_FEEDBACK}], 0.0, False

        lines = []
        for expression in expressions[:MAXIMUM_CALLS_PER_TURN]:
            self.calculator_call_count += 1
            shown = " ".join(expression.split())[:MAXIMUM_EXPRESSION_LENGTH]
            try:
                lines.append(f"Calculator: {shown} = {calculate(expression)}")
            except CalculatorError as error:
                self.calculator_error_count += 1
                lines.append(f"Calculator: {shown} -> error: {error}")
        if len(expressions) > MAXIMUM_CALLS_PER_TURN:
            lines.append(f"(Only the first {MAXIMUM_CALLS_PER_TURN} calls in a message are evaluated.)")
        return [{"role": "user", "content": "\n".join(lines)}], 0.0, False

    def state_key(self):
        return ""

    def is_successful(self):
        return self.successful

    def metrics(self):
        result = self._overall_metrics()
        source = self.sample.get(self.source_field)
        if source is not None:
            # 这两项只出现在对应来源的轨迹里，取平均时就是这一类题上的比例。
            result[f"{source}_success_fraction"] = 1.0 if self.successful else 0.0
            result[f"{source}_used_calculator_fraction"] = 1.0 if self.calculator_call_count > 0 else 0.0
            result[f"{source}_calculator_call_count"] = float(self.calculator_call_count)
            result[f"{source}_answered_fraction"] = 1.0 if self.answered else 0.0
            result[f"{source}_step_count"] = float(self.turn_count)
        return result

    def _overall_metrics(self):
        turn_count = max(self.turn_count, 1)
        return {
            "calculator_call_count": float(self.calculator_call_count),
            "used_calculator_fraction": 1.0 if self.calculator_call_count > 0 else 0.0,
            "calculator_error_fraction": (
                self.calculator_error_count / self.calculator_call_count
                if self.calculator_call_count > 0 else 0.0
            ),
            "invalid_turn_fraction": self.invalid_turn_count / turn_count,
            "answered_fraction": 1.0 if self.answered else 0.0,
            "answered_without_waiting_fraction": 1.0 if self.answered_without_waiting else 0.0,
            "exceeded_response_budget_fraction": 1.0 if self.exceeded_response_budget else 0.0,
            "call_after_answer_marker_fraction": 1.0 if self.call_after_answer_marker else 0.0,
        }
