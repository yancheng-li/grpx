import random
import re


def make_dataset(size, lowest=1, highest=50, seed=0):
    random_generator = random.Random(seed)
    return [{"target": random_generator.randint(lowest, highest),
             "lowest": lowest, "highest": highest} for _ in range(size)]


class GuessNumberEnvironment:
    """分级回报：每缩小一次区间按缩小比例给分（合计最多 0.5），猜中再补齐到恰好 1.0，与用了几步无关"""

    def __init__(self, reveal_interval=False):
        self.reveal_interval = reveal_interval

    def reset(self, sample, seed):
        self.target = sample["target"]
        self.low, self.high = sample["lowest"], sample["highest"]
        self.initial_size = self.high - self.low + 1
        self.successful = False
        self.step_count = 0
        self.informative_guess_count = 0
        self.split_quality_total = 0.0
        return [{"role": "user", "content":
                 f"我们玩猜数字。我想了一个 {self.low} 到 {self.high} 之间的整数，你来猜。\n"
                 "规则：你每次只回复一个阿拉伯数字，不要回复任何其他文字。"
                 "我会告诉你大了还是小了。\n现在请给出第一个猜测。"}]

    def is_successful(self):
        return self.successful

    def state_key(self):
        # 目前已知答案所在的区间：不同轨迹走到同一区间，就是同一个状态
        return f"{self.low}-{self.high}"

    def metrics(self):
        if self.step_count == 0:
            return {"informative_step_fraction": 0.0, "split_quality": 0.0}
        return {
            "informative_step_fraction": self.informative_guess_count / self.step_count,
            "split_quality": self.split_quality_total / self.step_count,
        }

    def step(self, action_text):
        self.step_count += 1                      # 每一步都计入，包括没有输出数字的
        numbers = re.findall(r"\d+", action_text)
        if not numbers:
            return [{"role": "user", "content": "请只回复一个阿拉伯数字。"}], 0.0, False
        guess = int(numbers[-1])
        if self.low <= guess <= self.high:        # 在更新区间之前判断
            self.informative_guess_count += 1
            size = self.high - self.low + 1
            best_smaller_part = (size - 1) // 2      # 对半切时较小一段的长度
            if best_smaller_part == 0:               # 只剩一两个数，怎么猜都是最优
                self.split_quality_total += 1.0
            else:
                smaller_part = min(guess - self.low, self.high - guess)
                self.split_quality_total += smaller_part / best_smaller_part
        size_before = self.high - self.low + 1
        if guess == self.target:
            self.successful = True
            return [], 0.5 + 0.5 * size_before / self.initial_size, True
        if guess < self.target:
            self.low = max(self.low, guess + 1)
            feedback = "小了"
        else:
            self.high = min(self.high, guess - 1)
            feedback = "大了"
        size_after = self.high - self.low + 1
        reward = 0.5 * (size_before - size_after) / self.initial_size
        message = f"{guess} {feedback}，再猜。"
        if self.reveal_interval:
            message += f"现在答案在 {self.low} 到 {self.high} 之间。"
        return [{"role": "user", "content": message}], reward, False
