"""GSM8K 和计算密集应用题的混合数据集。

用来检验模型能不能学会"按需使用计算器"：GSM8K 的数字小，直接算就行；
计算题的数字很大，不用计算器基本做不对。两类题混在一起训练，
每道题带一个 source 字段（"gsm8k" 或 "calculation"），环境据此分别统计两类题上的成功率和使用率。
"""

import random


def make_dataset(split="train", size=4000, seed=0, calculation_fraction=0.5):
    if not 0.0 <= calculation_fraction <= 1.0:
        raise ValueError("calculation_fraction 必须在 0 和 1 之间")
    if split not in ("train", "test"):
        raise ValueError(f"未知的 split：{split!r}，可选 'train' 或 'test'")
    # 放在函数里导入：只用其中一个数据集时不必安装另一个的依赖。
    from examples import calculation_problems, gsm8k

    calculation_count = round(size * calculation_fraction)
    gsm8k_count = size - calculation_count
    samples = []
    if gsm8k_count > 0:
        for sample in gsm8k.make_dataset(split=split, size=gsm8k_count, seed=seed):
            samples.append({**sample, "source": "gsm8k"})
    if calculation_count > 0:
        for sample in calculation_problems.make_dataset(split=split, size=calculation_count, seed=seed):
            samples.append({**sample, "source": "calculation"})
    random.Random(seed).shuffle(samples)
    return samples
