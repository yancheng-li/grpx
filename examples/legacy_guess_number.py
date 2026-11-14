"""最初版本的猜数字：旧提示词、不提示区间、稀疏回报。

用来对照预检：成功率接近 0、截断比例约一半、有信号的组偏少。
正式训练请用 examples.guess_number。
"""
import random
import re

from examples.guess_number import make_dataset  # noqa: F401  配置里复用同一份数据集函数


class LegacyGuessNumberEnvironment:
    def reset(self, sample, seed):
        self.target = sample["target"]
        self.low, self.high = sample["lowest"], sample["highest"]
        self.successful = False
        return [{"role": "user", "content":
                 f"我想了一个 {self.low} 到 {self.high} 之间的整数。"
                 "你每次猜一个数，我会告诉你大了还是小了。每次只输出一个整数。"}]

    def is_successful(self):
        return self.successful

    def state_key(self):
        return f"{self.low}-{self.high}"

    def step(self, action_text):
        numbers = re.findall(r"\d+", action_text)
        if not numbers:
            return [{"role": "user", "content": "请输出一个整数。"}], 0.0, False
        guess = int(numbers[-1])
        if guess == self.target:
            self.successful = True
            return [], 1.0, True
        if guess < self.target:
            self.low = max(self.low, guess + 1)
            feedback = "小了"
        else:
            self.high = min(self.high, guess - 1)
            feedback = "大了"
        return [{"role": "user", "content": f"{guess} {feedback}。"}], 0.0, False
