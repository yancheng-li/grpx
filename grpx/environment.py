class SingleTurnEnvironment:
    # 有状态：每条轨迹需要独立的实例，同组 G 条轨迹不能共用一个
    # success_function(completion, sample) 可选；不传时 is_successful 返回 None
    def __init__(self, reward_function, success_function=None):
        self.reward_function = reward_function
        self.success_function = success_function
        self.successful = None    # 子类重写 reset 时可能不调用父类，这里先给初值

    def reset(self, sample, seed):
        self.sample = sample
        self.successful = None
        return [{"role": "user", "content": sample["prompt"]}]

    def step(self, action: str):
        reward = self.reward_function(self.sample["prompt"], action, self.sample)
        if self.success_function is not None:
            self.successful = bool(self.success_function(action, self.sample))
        return [], reward, True         # 新增的 messages， reward，是否结束

    def is_successful(self):
        return self.successful

    def state_key(self):
        return ""
