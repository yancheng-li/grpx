"""grpx doctor：训练前的预检。每项给出 通过 / 警告 / 失败 和一句建议。

判定都在 judge_* 纯函数里，阈值集中在下面，不需要模型就能测试；
run_doctor 负责加载模型、采样一轮、试更新一步，再把测到的数交给它们。
"""
import itertools
import math
import random
import re
import sys
import time
from dataclasses import dataclass

from grpx.command_line import (ConfigurationError, apply_lora, build_task,
                               build_training_config, load_base_model_and_tokenizer)

PASS, WARNING, FAIL = "通过", "警告", "失败"

MINIMUM_VERSIONS = {"transformers": "4.56", "peft": None}
PROBE_ACTIONS = ["42", "", "我不知道"]          # 环境要能接住任意文本，模型的输出没有保证
LOW_SUCCESS, HIGH_SUCCESS = 0.0, 0.9
LOW_SIGNAL = 0.25
HIGH_LENGTH_LIMIT = 0.2
NEAR_LENGTH_LIMIT = 0.9                            # 回答长度 95 分位占生成上限的比例
# 截断回答仍拿到接近满分的环境回报：说明回报函数看不见截断，依赖框架覆盖
HIGH_TRUNCATED_ENVIRONMENT_REWARD = 0.5
IMPOSSIBLE_LOG_PROBABILITY = -15.0                 # 按采样分布几乎不可能被采到的 token
HIGH_IMPOSSIBLE_FRACTION = 0.005
LOW_MEAN_LOG_PROBABILITY = -3.0
HIGH_FIRST_STEP_RATIO_DEVIATION = 1e-3
HIGH_SHIFT_CLIP_FRACTION = 0.2
HIGH_SHIFT_PERCENTILE_95 = 1.0                     # 5% 的 token 概率变化超过 e 倍
LOW_SHIFT_PERCENTILE_95 = 0.01                     # 过小看 95 分位，中位数反映的是策略确定性
# 与训练 set_per_process_memory_fraction 对齐；超过上限 FAIL，超过上限的 HEADROOM 比例 WARN
MEMORY_HEADROOM_FRACTION = 0.8
LOW_STEP_SIGNAL_COVERAGE = 0.05                    # 开了 GiGPO 但几乎没有步级信号
# 超长行丢弃比例：预算设太小时会静默丢掉大量数据，训练看似在跑却几乎没学
OVERSIZED_ROW_WARN_FRACTION = 0.05
OVERSIZED_ROW_FAIL_FRACTION = 0.20


@dataclass
class CheckResult:
    category: str
    name: str
    status: str
    detail: str
    suggestion: str = ""

    def format(self):
        line = f"[{self.status}] {self.category} · {self.name}：{self.detail}"
        if self.suggestion and self.status != PASS:
            line += f"\n       建议：{self.suggestion}"
        return line


class Report:
    def __init__(self, output=print):
        self.results, self.output = [], output

    def add(self, *results):
        for result in results:
            self.results.append(result)
            self.output(result.format())

    @property
    def failed(self):
        return any(result.status == FAIL for result in self.results)

    def summary(self):
        counts = {status: sum(result.status == status for result in self.results)
                  for status in (PASS, WARNING, FAIL)}
        return "汇总：" + "，".join(f"{status} {count}" for status, count in counts.items())


def _one_line(error):
    return f"{type(error).__name__}: {error}".splitlines()[0]


def is_out_of_memory(error):
    """本项目真实遇到过的是 RuntimeError: CUBLAS_STATUS_ALLOC_FAILED，不是 OutOfMemoryError"""
    try:
        import torch
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return "out of memory" in message or "cublas_status_alloc_failed" in message


def _version_tuple(version):
    """'2.5.1+cu121' -> (2, 5, 1)"""
    return tuple(int(part) for part in re.findall(r"\d+", version.split("+")[0])[:3])


def judge_package_version(package, installed, minimum):
    if installed is None:
        return CheckResult("环境", package, FAIL, "没有安装", f"pip install {package}")
    if minimum and _version_tuple(installed) < _version_tuple(minimum):
        return CheckResult("环境", package, FAIL, f"版本 {installed}，需要 ≥ {minimum}",
                           f"pip install -U '{package}>={minimum}'")
    return CheckResult("环境", package, PASS, f"版本 {installed}")


def check_environment():
    # Python 版本已在 command_line.main 入口检查；这里只报告当前版本，不再重复判定
    from importlib.metadata import PackageNotFoundError, version

    results = [CheckResult("环境", "Python", PASS, f"版本 {'.'.join(map(str, sys.version_info[:3]))}")]
    try:
        import torch
    except ImportError as error:
        results.append(CheckResult("环境", "torch", FAIL, _one_line(error), "安装 GPU 版 torch"))
        return results
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        results.append(CheckResult("环境", "CUDA", PASS, f"torch {torch.__version__}，{properties.name}，"
                                   f"显存 {properties.total_memory / 2 ** 30:.1f} GB"))
    else:
        results.append(CheckResult("环境", "CUDA", FAIL, f"torch {torch.__version__} 看不到 CUDA",
                                   "安装 GPU 版 torch，并确认驱动正常（nvidia-smi）"))
    for package, minimum in MINIMUM_VERSIONS.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        results.append(judge_package_version(package, installed, minimum))
    return results


def judge_dataset(train_dataset, evaluation_dataset, groups_per_iteration):
    results = []
    try:
        size = len(train_dataset)
        train_dataset[0]
    except Exception as error:    # noqa: BLE001  用户代码返回什么都有可能
        return [CheckResult("任务", "训练集", FAIL, f"不是可按下标取样的列表：{_one_line(error)}",
                            "数据集函数应返回 list，每个元素是一道题的样本")]
    if size < groups_per_iteration:
        results.append(CheckResult("任务", "训练集", FAIL, f"只有 {size} 道题，每轮要抽 {groups_per_iteration} 组",
                                   "增大 train_dataset_options 里的数据量，或减小 training.groups_per_iteration"))
    else:
        results.append(CheckResult("任务", "训练集", PASS, f"{size} 道题"))
    if evaluation_dataset is None:
        results.append(CheckResult("任务", "评测集", WARNING, "没有评测集，训练中看不到贪心评测",
                                   "在 [task] 里加 evaluation_dataset_options"))
    elif len(evaluation_dataset) == 0:
        results.append(CheckResult("任务", "评测集", FAIL, "评测集为空", "检查 evaluation_dataset_options"))
    else:
        results.append(CheckResult("任务", "评测集", PASS, f"{len(evaluation_dataset)} 道题"))
    return results


def _is_message_list(messages):
    return isinstance(messages, list) and all(
        isinstance(message, dict) and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str) for message in messages)


def walk_through_environment(environment_factory, sample, max_steps):
    """用几句固定的探测回答把环境走一遍，检查接口的形状"""
    name = "环境走一遍"
    stage = "environment_factory()"
    try:
        environment = environment_factory()
        stage = "environment_factory() 第二次调用"
        if environment is environment_factory():
            return CheckResult("任务", name, FAIL, "环境工厂两次返回了同一个对象",
                               "环境有状态，每条轨迹需要独立实例：工厂里每次都要新建")
        stage = "reset(sample, seed=0)"
        messages = environment.reset(sample, 0)
        if not _is_message_list(messages) or not messages:
            return CheckResult("任务", name, FAIL, f"reset 返回 {messages!r:.80}",
                               "reset 应返回非空的消息列表，如 [{'role': 'user', 'content': '...'}]")
        rewards, done = [], False
        for action in itertools.islice(itertools.cycle(PROBE_ACTIONS), max_steps):
            stage = f"step({action!r})"
            result = environment.step(action)
            if not (isinstance(result, tuple) and len(result) == 3):
                return CheckResult("任务", name, FAIL, f"step 返回 {result!r:.80}",
                                   "step 应返回 (新消息列表, 奖励, 是否结束)")
            new_messages, reward, done = result
            if not isinstance(new_messages, list) or (new_messages and not _is_message_list(new_messages)):
                return CheckResult("任务", name, FAIL, f"step 返回的新消息是 {new_messages!r:.80}",
                                   "新消息应是消息列表，没有就返回 []")
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                return CheckResult("任务", name, FAIL, f"奖励是 {reward!r}",
                                   "奖励应是有限的 Python int 或 float（numpy 数值请用 float(...) 转一下），"
                                   "否则写不进 JSON 日志")
            rewards.append(reward)
            if done:
                break
        if hasattr(environment, "state_key"):
            stage = "state_key()"
            if not isinstance(environment.state_key(), str):
                return CheckResult("任务", name, FAIL, "state_key() 返回的不是字符串",
                                   "state_key 应返回字符串，环境没有状态概念时返回 \"\"")
        if hasattr(environment, "is_successful"):
            stage = "is_successful()"
            if environment.is_successful() not in (None, True, False):
                return CheckResult("任务", name, FAIL, f"is_successful() 返回 {environment.is_successful()!r}",
                                   "is_successful 应返回 True、False 或 None")
        if hasattr(environment, "metrics"):
            stage = "metrics()"
            metrics = environment.metrics()
            if not isinstance(metrics, dict):
                return CheckResult("任务", name, FAIL, f"metrics() 返回 {metrics!r:.80}",
                                   "metrics 应返回字典，键是字符串，值是有限的 int 或 float")
            for key, value in metrics.items():
                if not isinstance(key, str):
                    return CheckResult("任务", name, FAIL, f"metrics() 的键 {key!r} 不是字符串",
                                       "键名应是字符串，会写进日志")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    return CheckResult("任务", name, FAIL, f"metrics()['{key}'] = {value!r}",
                                       "值应是有限的 Python int 或 float（numpy 数值请用 float(...) 转一下）")
    except Exception as error:    # noqa: BLE001
        return CheckResult("任务", name, FAIL, f"{stage} 引发 {_one_line(error)}",
                           "环境要能处理任意文本（包括空串）；在 Python 里直接调用对应方法复现，看完整堆栈")
    ending = "已结束" if done else f"{max_steps} 步内未结束"
    metrics_text = f"，环境指标 {environment.metrics()}" if hasattr(environment, "metrics") else ""
    return CheckResult("任务", name, PASS, f"走了 {len(rewards)} 步，奖励 {rewards}，{ending}{metrics_text}")


def _percentile(sorted_values, fraction):
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]


def judge_sampling(trajectories, kept, max_new_tokens, truncated_reward=0.0, stop_strings=()):
    from grpx.types import success_fraction

    count = len(trajectories)
    results = []
    successes = success_fraction(trajectories)
    if successes is None:
        results.append(CheckResult("采样", "成功率", PASS, "环境没有成功判定，只看回报"))
    elif successes <= LOW_SUCCESS:
        results.append(CheckResult("采样", "成功率", WARNING, f"{successes:.3f}，一次都没有成功",
                                   "学习信号只能来自部分得分；考虑降低任务难度或给中间奖励"))
    elif successes > HIGH_SUCCESS:
        results.append(CheckResult("采样", "成功率", WARNING, f"{successes:.3f}，任务对当前模型太容易",
                                   "大部分组内没有差异，可以提高难度"))
    else:
        results.append(CheckResult("采样", "成功率", PASS, f"{successes:.3f}"))

    signal = len(kept) / count
    if not kept:
        results.append(CheckResult("采样", "有信号的比例", FAIL, "0，所有组内回报都没有差异，这一轮不会产生梯度",
                                   "检查奖励函数是否总返回同一个值；或增大 group_size、调整难度"))
    elif signal < LOW_SIGNAL:
        results.append(CheckResult("采样", "有信号的比例", WARNING, f"{signal:.3f}，大部分采样被浪费",
                                   "增大 group_size，或调整难度让组内结果有好有坏"))
    else:
        results.append(CheckResult("采样", "有信号的比例", PASS, f"{signal:.3f}"))

    fractions = {reason: sum(trajectory.termination_reason == reason for trajectory in trajectories) / count
                 for reason in ("done", "length_limit", "step_limit")}
    detail = "  ".join(f"{reason} {fraction:.3f}" for reason, fraction in fractions.items())
    if fractions["length_limit"] > HIGH_LENGTH_LIMIT:
        results.append(CheckResult("采样", "结束原因", WARNING, f"{detail}；截断比例偏高",
                                   "增大 backend.max_new_tokens，或在提示词里要求回答简短"))
    else:
        results.append(CheckResult("采样", "结束原因", PASS, detail))

    step_finish = {"end_token": 0, "stop_string": 0, "length_limit": 0, "unknown": 0}
    step_total = 0
    for trajectory in trajectories:
        for step in trajectory.steps:
            step_total += 1
            reason = getattr(step, "finish_reason", None)
            if reason in step_finish:
                step_finish[reason] += 1
            else:
                step_finish["unknown"] += 1
    if step_total:
        step_detail = "  ".join(
            f"{name} {step_finish[name] / step_total:.3f}"
            for name in ("end_token", "stop_string", "length_limit")
        )
        if stop_strings and step_finish["stop_string"] == 0:
            results.append(CheckResult(
                "采样", "生成结束", WARNING,
                f"{step_detail}；已配置 stop_strings={list(stop_strings)!r} 但一次都没触发",
                "检查停止串是否写错，或提示是否从未让模型写出该串"))
        else:
            results.append(CheckResult("采样", "生成结束", PASS, step_detail))

    lengths = sorted(len(step.action_token_ids) for trajectory in trajectories for step in trajectory.steps)
    mean_length = sum(lengths) / len(lengths)
    percentile_95 = _percentile(lengths, 0.95)
    detail = f"平均 {mean_length:.1f} 个 token，95 分位 {percentile_95}，上限 {max_new_tokens}"
    if percentile_95 >= NEAR_LENGTH_LIMIT * max_new_tokens:
        results.append(CheckResult("采样", "回答长度", WARNING, f"{detail}；接近生成上限",
                                   "增大 backend.max_new_tokens，否则长回答会被截断"))
    else:
        results.append(CheckResult("采样", "回答长度", PASS, detail))

    results.append(judge_truncated_beats_normal(trajectories, truncated_reward))
    results.append(judge_truncated_high_reward(trajectories))
    return results


def judge_oversized_row_fraction(trajectories, max_tokens, merge_append_only=False):
    """组批前会丢掉长度 > max_tokens 的行；比例过高说明预算设小了，训练几乎没在学。"""
    from grpx.loop import row_length
    from grpx.trainer import prepare_update_units

    rows = prepare_update_units(trajectories, merge_append_only=merge_append_only)
    if not rows:
        return CheckResult("采样", "超长行丢弃比例", PASS, "没有更新单元")
    oversized = sum(1 for row in rows if row_length(row) > max_tokens)
    total = len(rows)
    fraction = oversized / total
    percent = 100.0 * fraction
    longest = max((row_length(row) for row in rows if row_length(row) > max_tokens), default=0)
    detail = (
        f"{percent:.0f}%（{oversized}/{total}）超过 max_tokens_per_minibatch={max_tokens}"
        + (f"，最长 {longest}" if oversized else "")
    )
    suggestion = (
        "增大 training.max_tokens_per_minibatch（或缩短对话 / 降低 max_new_tokens）；"
        "否则超限行全部不参与更新，训练看似在跑却几乎没学"
    )
    if fraction > OVERSIZED_ROW_FAIL_FRACTION:
        return CheckResult("采样", "超长行丢弃比例", FAIL, detail, suggestion)
    if fraction > OVERSIZED_ROW_WARN_FRACTION:
        return CheckResult("采样", "超长行丢弃比例", WARNING, detail, suggestion)
    return CheckResult("采样", "超长行丢弃比例", PASS, detail)


def _environment_reward(trajectory):
    if not trajectory.steps:
        return 0.0
    step = trajectory.steps[-1]
    if step.environment_reward is not None:
        return step.environment_reward
    return step.reward


def judge_truncated_beats_normal(trajectories, truncated_reward=0.0):
    """正常结局（非 length_limit）的整局回报低于 truncated_reward：截断会得分更高。

    不依赖本轮是否出现截断轨迹；比例 > 0 即警告。
    """
    if truncated_reward is None:
        return CheckResult("采样", "截断优于正常结局", PASS, "未启用 truncated_reward 覆盖")
    normal = [trajectory for trajectory in trajectories
              if trajectory.termination_reason != "length_limit"]
    if not normal:
        return CheckResult("采样", "截断优于正常结局", PASS, "没有正常结局可对照")
    beaten = sum(1 for trajectory in normal if trajectory.total_reward < truncated_reward)
    fraction = beaten / len(normal)
    percent = 100.0 * fraction
    detail = (f"截断比 {percent:.0f}% 的正常结局得分更高"
              f"（{beaten}/{len(normal)} 条整局回报 < truncated_reward={truncated_reward:g}）")
    if beaten > 0:
        return CheckResult(
            "采样", "截断优于正常结局", WARNING, detail,
            "调低 truncated_reward，或让正常失败结局的回报不低于截断覆盖值；"
            "否则长度截断会相对负回报结局占便宜")
    return CheckResult("采样", "截断优于正常结局", PASS, detail)


def judge_truncated_high_reward(trajectories):
    """截断但环境仍给高回报：回报函数看不见截断，必须靠 truncated_reward 覆盖。"""
    count = len(trajectories)
    if count == 0:
        return CheckResult("采样", "截断却高回报", PASS, "没有轨迹")
    bad = sum(
        1 for trajectory in trajectories
        if trajectory.termination_reason == "length_limit"
        and _environment_reward(trajectory) >= HIGH_TRUNCATED_ENVIRONMENT_REWARD
    )
    fraction = bad / count
    detail = f"{fraction:.3f}（{bad}/{count}），环境回报 ≥ {HIGH_TRUNCATED_ENVIRONMENT_REWARD:g}"
    if fraction > 0:
        return CheckResult(
            "采样", "截断却高回报", WARNING, detail,
            "回报函数看不到截断，会把不完整回答当高分；确认 training.truncated_reward 已覆盖"
            "（默认 0），不要用 null 关掉")
    return CheckResult("采样", "截断却高回报", PASS, detail)


def judge_step_signal_coverage(coverage, step_advantage_weight):
    """只在开了步级优势时调用；覆盖率接近 0 说明当前配置等同于 GRPO"""
    detail = f"覆盖率 {coverage:.3f}（ω = {step_advantage_weight:g}）"
    if coverage < LOW_STEP_SIGNAL_COVERAGE:
        return CheckResult("采样", "步级信号覆盖率", WARNING, f"{detail}；步级分组没有生效，当前配置等同于 GRPO",
                           "单轮环境或 state_key 恒为空时换回 algorithm=grpo；多轮任务检查 state_key 是否返回有区分度的状态")
    return CheckResult("采样", "步级信号覆盖率", PASS, detail)


def judge_log_probabilities(action_log_probabilities):
    values = sorted(action_log_probabilities)
    mean = sum(values) / len(values)
    impossible = sum(value < IMPOSSIBLE_LOG_PROBABILITY for value in values) / len(values)
    detail = f"平均 {mean:.3f}，低于 {IMPOSSIBLE_LOG_PROBABILITY:g} 的 token 占 {impossible:.4f}"
    if impossible > HIGH_IMPOSSIBLE_FRACTION:
        return CheckResult("训推一致", "回答 token 的对数概率", FAIL, f"{detail}；训练端认为这些采到的 token 几乎不可能",
                           "采样和训练看到的分布不一致：检查对话模板、温度、填充方向")
    if mean < LOW_MEAN_LOG_PROBABILITY:
        return CheckResult("训推一致", "回答 token 的对数概率", WARNING, f"{detail}；平均值偏低",
                           "温度过高时也会这样；若温度正常，检查采样与训练的输入是否一致")
    return CheckResult("训推一致", "回答 token 的对数概率", PASS, detail)


def judge_first_step_ratio(max_ratio_deviation):
    detail = f"首步 |ratio - 1| 最大 {max_ratio_deviation:.2e}"
    if max_ratio_deviation > HIGH_FIRST_STEP_RATIO_DEVIATION:
        return CheckResult("训推一致", "首步 ratio", WARNING, f"{detail}；参数还没更新，ratio 应恰好为 1",
                           "训练模式和评估模式的前向不同：检查 lora_dropout、模型的 attention_dropout")
    return CheckResult("训推一致", "首步 ratio", PASS, detail)


def judge_shift(shift, learning_rate, mixed_precision=False):
    median = shift["shift_median_absolute_log_ratio"]
    percentile_95 = shift["shift_percentile_95_absolute_log_ratio"]
    clip_fraction = shift["shift_clip_fraction"]
    detail = (f"学习率 {learning_rate:g} 下一步更新：位移中位数 {median:.2e}，"
              f"95 分位 {percentile_95:.3f}，超出裁剪范围 {clip_fraction:.3f}")
    if median == 0 and percentile_95 == 0:
        return CheckResult("试更新", "位移", FAIL, f"{detail}；更新没有改变模型", "检查是否有可训练参数、梯度是否为零")
    # bf16 下 1e-8 量级的 LoRA 扰动就会把位移中位数抬到 ~5e-3、95 分位 ~0.1（与训推 gap 同量级），
    # 不能据此判断学习率过大/过小；只保留「完全没动」和「裁剪比例爆了」两类硬信号。
    if mixed_precision:
        if clip_fraction > HIGH_SHIFT_CLIP_FRACTION:
            return CheckResult("试更新", "位移", WARNING, f"{detail}；一步超出裁剪过多",
                               "学习率偏大，建议减小 training.learning_rate")
        return CheckResult(
            "试更新", "位移", PASS,
            f"{detail}（bf16 位移有底噪，不作学习率过大/过小判断）")
    if clip_fraction > HIGH_SHIFT_CLIP_FRACTION or percentile_95 > HIGH_SHIFT_PERCENTILE_95:
        return CheckResult("试更新", "位移", WARNING, f"{detail}；一步的位移过大",
                           "学习率偏大，建议减小 training.learning_rate")
    if percentile_95 < LOW_SHIFT_PERCENTILE_95:
        return CheckResult("试更新", "位移", WARNING, f"{detail}；95 分位位移很小",
                           "学习率可能偏小；中位数小不一定有问题（策略很确定时中位数本来就接近 0）")
    return CheckResult("试更新", "位移", PASS, detail)


def judge_memory(peak_by_phase, total, memory_fraction=None):
    """按训练同款显存上限判定：超上限 FAIL，超上限的 MEMORY_HEADROOM_FRACTION 则 WARN。"""
    from grpx.loop import TRAINING_MEMORY_FRACTION

    if memory_fraction is None:
        memory_fraction = TRAINING_MEMORY_FRACTION
    phase, peak = max(peak_by_phase.items(), key=lambda item: item[1])
    limit = total * memory_fraction
    detail = (f"峰值 {peak / 2 ** 30:.1f} GB（{phase}），占显存 {peak / total:.0%}，"
              f"训练上限 {limit / 2 ** 30:.1f} GB（{memory_fraction:.0%}）；"
              + "，".join(f"{name} {value / 2 ** 30:.1f} GB" for name, value in peak_by_phase.items()))
    if phase == "采样":
        suggestion = ("减小 backend.max_tokens_per_batch 或 max_concurrent_requests；"
                      "或设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    else:
        suggestion = ("减小 training.max_tokens_per_minibatch 或 max_rows_per_minibatch；"
                      "或设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    if peak > limit:
        return CheckResult(
            "试更新", "显存", FAIL,
            f"{detail}；已超过训练显存上限，按当前小批预算训练会撞上限", suggestion)
    if peak > MEMORY_HEADROOM_FRACTION * limit:
        return CheckResult(
            "试更新", "显存", WARNING,
            f"{detail}；余量偏紧（已超上限的 {MEMORY_HEADROOM_FRACTION:.0%}），长一些的轮次可能显存不足",
            suggestion)
    return CheckResult("试更新", "显存", PASS, detail)


def summarize_lora_targets(model, lora_target_modules):
    """在套 PEFT 之前统计 LoRA 目标会命中哪些 Linear（含专家层 / 路由）。"""
    import torch

    linear_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if lora_target_modules == "all-linear":
        matched_module_names = linear_names
    else:
        targets = list(lora_target_modules)
        matched_module_names = [
            name for name in linear_names
            if any(name == target or name.endswith("." + target) for target in targets)
        ]
    expert_module_names = [name for name in matched_module_names if ".experts." in name]
    router_module_names = [name for name in matched_module_names if name.endswith(".gate")]
    modules_by_name = dict(model.named_modules())
    expert_parameter_count = sum(
        parameter.numel()
        for name in expert_module_names
        for parameter in modules_by_name[name].parameters()
    )
    return {
        "matched_module_count": len(matched_module_names),
        "expert_module_count": len(expert_module_names),
        "router_module_count": len(router_module_names),
        "expert_parameter_count": expert_parameter_count,
    }


def judge_lora_targets(summary):
    """matched=0 失败；命中路由 / 专家层则警告。"""
    results = []
    matched = summary["matched_module_count"]
    if matched == 0:
        results.append(CheckResult(
            "模型", "LoRA 目标模块", FAIL,
            "没有任何 Linear 模块匹配 lora_target_modules",
            "检查 [model].lora_target_modules 是否写对了模块名后缀"
            "（例如 q_proj）；dense 模型可省略该项，默认 all-linear"))
        return results
    results.append(CheckResult(
        "模型", "LoRA 目标模块", PASS,
        f"匹配到 {matched} 个 Linear 模块"))
    if summary["router_module_count"] > 0:
        results.append(CheckResult(
            "模型", "LoRA 与路由", WARNING,
            f"有 {summary['router_module_count']} 个匹配模块名以 .gate 结尾，"
            "路由门控可能被挂上 LoRA",
            "MoE 上请把 lora_target_modules 显式写成注意力投影"
            "（q_proj/k_proj/v_proj/o_proj），不要包含 gate"))
    if summary["expert_module_count"] > 0:
        parameter_count = summary.get("expert_parameter_count", 0)
        results.append(CheckResult(
            "模型", "LoRA 与专家层", WARNING,
            f"有 {summary['expert_module_count']} 个匹配模块落在 .experts. 下"
            f"（约 {parameter_count:,} 个参数）",
            "专家层挂 LoRA 会急剧增大可训练参数与显存；"
            "MoE 上请显式指定注意力投影作为 lora_target_modules"))
    return results


def judge_chat_template(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    if template is None or not str(template).strip():
        return CheckResult(
            "模型", "对话模板", FAIL,
            "tokenizer.chat_template 为空",
            "换用带 chat_template 的 Instruct 模型，或自行设置 tokenizer.chat_template")
    return CheckResult("模型", "对话模板", PASS, "tokenizer.chat_template 已设置")


def shared_gpu_memory_hint(memory_fraction, threshold=None):
    """memory_fraction 低于阈值时提示与推理同卡要留余量；不加独立配置项。"""
    from grpx.loop import SHARED_GPU_MEMORY_FRACTION_HINT

    if threshold is None:
        threshold = SHARED_GPU_MEMORY_FRACTION_HINT
    if memory_fraction >= threshold:
        return None
    return CheckResult(
        "配置", "同卡显存", PASS,
        f"training.memory_fraction={memory_fraction:.0%}（<{threshold:.0%}），"
        "若推理服务与训练在同一张卡上，请确认服务 --gpu-memory-utilization 已留出余量；"
        "独占一张卡时可把 memory_fraction 调到 0.95",
    )


def estimate_duration(config, sampling_seconds, update_seconds, evaluation_size):
    per_iteration = sampling_seconds + update_seconds
    trajectory_count = config.groups_per_iteration * config.group_size
    # 评测按题数折算采样时间，只是粗略估计
    evaluation_seconds = sampling_seconds * evaluation_size / trajectory_count if evaluation_size else 0.0
    evaluation_count = (len(range(0, config.iterations, config.evaluation_interval)) + 1) if evaluation_size else 0
    total_minutes = (per_iteration * config.iterations + evaluation_seconds * evaluation_count) / 60
    return CheckResult("试更新", "预计耗时", PASS,
                       f"每轮约 {per_iteration:.0f} 秒（采样 {sampling_seconds:.0f}，更新 {update_seconds:.0f}），"
                       f"{config.iterations} 轮加 {evaluation_count} 次评测约 {total_minutes:.0f} 分钟")


def longest_first_prompt_token_ids(tokenizer, environment_factory, dataset):
    """训练集里首轮提示词最长的那条（按 tokenize 后长度）。"""
    from grpx.rollout import encode_messages

    longest = []
    for sample in dataset:
        messages = environment_factory().reset(sample, seed=0)
        token_ids = encode_messages(tokenizer, messages)
        if len(token_ids) > len(longest):
            longest = token_ids
    return longest


def max_prompt_growth_tokens(trajectories, minimum=0):
    """采样里相邻两步提示长度之差的最大值（含回答 + 环境反馈 + 模板）。"""
    maximum = minimum
    for trajectory in trajectories:
        for current, following in zip(trajectory.steps, trajectory.steps[1:]):
            growth = len(following.prompt_token_ids) - len(current.prompt_token_ids)
            if growth > maximum:
                maximum = growth
    return maximum


@dataclass(frozen=True)
class WorstCaseSpec:
    """最坏小批候选：均匀行（每行 steps_per_unit 步）。"""
    steps_per_unit: int
    row_count: int
    row_length: int
    segments: int
    area: int
    answer_positions: int

    @property
    def shape(self):
        return "long" if self.steps_per_unit > 1 else "short"


def _worst_case_step_growth(max_new_tokens, tokens_per_step):
    growth = max_new_tokens if tokens_per_step is None else tokens_per_step
    return max(growth, max_new_tokens)


def worst_case_row_length(prompt_length, max_new_tokens, steps_per_unit, tokens_per_step=None):
    growth = _worst_case_step_growth(max_new_tokens, tokens_per_step)
    if steps_per_unit <= 1:
        return prompt_length + max_new_tokens
    return prompt_length + (steps_per_unit - 1) * growth + max_new_tokens


def enumerate_worst_case_specs(prompt_length, max_new_tokens, max_rows, max_tokens,
                               max_steps=1, merge_append_only=False, tokens_per_step=None):
    """按行数枚举可行形状。

    对每个行数，行长取 min(完整最坏长度, 预算 // 行数)，这样会摸到「用满 token 预算」的批；
    完整最坏长度超过预算时，另加一条单行超长形状（训练里也会单独成批/被筛掉前的峰值）。
    """
    step_options = range(1, max_steps + 1) if (merge_append_only and max_steps > 1) else (1,)
    specs = []
    seen = set()

    def add_spec(steps_per_unit, row_count, row_length):
        if row_count < 1 or row_length <= 0:
            return
        segments = row_count * steps_per_unit
        key = (steps_per_unit, row_count, row_length)
        if key in seen:
            return
        seen.add(key)
        action_len = min(max_new_tokens, max(1, row_length - 1))
        specs.append(WorstCaseSpec(
            steps_per_unit=steps_per_unit,
            row_count=row_count,
            row_length=row_length,
            segments=segments,
            area=row_count * row_length,
            answer_positions=segments * action_len,
        ))

    for steps_per_unit in step_options:
        full_length = worst_case_row_length(
            prompt_length, max_new_tokens, steps_per_unit, tokens_per_step)
        if full_length <= 0:
            continue
        max_rows_for_steps = max_rows // steps_per_unit
        for row_count in range(1, max_rows_for_steps + 1):
            # 用满预算：多行时行长可以短于完整最坏长度
            capped = min(full_length, max_tokens // row_count)
            add_spec(steps_per_unit, row_count, capped)
        if full_length > max_tokens:
            # 超预算单独成批的长对话：即使用 min 截断后仍要摸这条
            add_spec(steps_per_unit, 1, full_length)
    if not specs:
        row_length = worst_case_row_length(prompt_length, max_new_tokens, 1, tokens_per_step)
        add_spec(1, 1, row_length)
    return specs


def select_worst_case_probe_targets(specs):
    """摸底目标：总面积最大，以及回答位置最多；相同时只保留一个。

    面积并列时优先更长的行（注意力按长度平方涨，比「同样面积、更多短行」更吃显存）。
    """
    if not specs:
        raise ValueError("没有可摸底的最坏小批形状")
    by_area = max(specs, key=lambda spec: (spec.area, spec.row_length, spec.row_count))
    by_answers = max(specs, key=lambda spec: (spec.answer_positions, spec.area, spec.row_length))
    targets = [by_area]
    if by_answers != by_area:
        targets.append(by_answers)
    return targets


def _append_only_worst_case_trajectory(name, temperature, prompt_token_ids, action_token_ids,
                                       step_count, tokens_per_step):
    """追加式多段：每步回答固定长度，步间提示增长用 tokens_per_step（含反馈）。"""
    from grpx.types import Step, Trajectory
    if tokens_per_step < len(action_token_ids):
        raise ValueError("tokens_per_step 不能短于回答长度")
    steps = []
    prefix = list(prompt_token_ids)
    growth = [action_token_ids[0]] * tokens_per_step
    for step_index in range(step_count):
        steps.append(Step(list(prefix), list(action_token_ids), advantage=1.0))
        if step_index + 1 < step_count:
            prefix = prefix + growth
    return Trajectory(name, 0, temperature, steps)


def _fit_tokens_to_row_length(prompt_token_ids, max_new_tokens, target_length, filler):
    """构造 prompt/action，使 len(prompt)+len(action) == target_length。"""
    if target_length < 2:
        raise ValueError(f"target_length 至少为 2，收到 {target_length}")
    action_length = min(max_new_tokens, target_length - 1)
    prompt_length = target_length - action_length
    if prompt_length <= len(prompt_token_ids):
        prompt = list(prompt_token_ids[:prompt_length])
    else:
        prompt = list(prompt_token_ids) + [filler] * (prompt_length - len(prompt_token_ids))
    if not prompt:
        prompt = [filler]
        action_length = target_length - 1
    action = [filler] * action_length
    return prompt, action


def _fit_merged_tokens_to_row_length(prompt_token_ids, max_new_tokens, target_length,
                                    steps_per_unit, step_growth, filler):
    """追加式多段：调整首提示/步增，使最后一步总长为 target_length。

    约束：tokens_per_step >= action_length（追加式轨迹的增长至少含本步回答）。
    """
    action_length = min(max_new_tokens, max(1, target_length - 1))
    gaps = max(steps_per_unit - 1, 0)
    # final = prompt0 + gaps * growth + action
    growth = max(step_growth, action_length)
    prompt0_length = target_length - gaps * growth - action_length
    if prompt0_length < 1 and gaps > 0:
        # 压缩步增：均分「总长 - 首提示最少 1 - 回答」
        remaining = target_length - 1 - action_length
        growth = max(action_length, remaining // gaps if gaps else action_length)
        prompt0_length = target_length - gaps * growth - action_length
    prompt0_length = max(1, prompt0_length)
    action_length = target_length - prompt0_length - gaps * growth
    if action_length < 1:
        action_length = 1
        prompt0_length = max(1, target_length - gaps * growth - action_length)
    growth = max(growth, action_length)
    # 再按最终 growth 校正 prompt，保证总长精确
    prompt0_length = target_length - gaps * growth - action_length
    if prompt0_length < 1:
        prompt0_length = 1
        action_length = max(1, target_length - gaps * growth - prompt0_length)
        growth = max(growth, action_length)
        prompt0_length = max(1, target_length - gaps * growth - action_length)
    if prompt0_length <= len(prompt_token_ids):
        prompt0 = list(prompt_token_ids[:prompt0_length])
    else:
        prompt0 = list(prompt_token_ids) + [filler] * (prompt0_length - len(prompt_token_ids))
    action = [filler] * action_length
    return prompt0, action, growth


def build_worst_case_batch(prompt_token_ids, max_new_tokens, max_rows, max_tokens,
                           padding_token_id, device, temperature=1.0, row_weighting="per_step",
                           max_steps=1, merge_append_only=False, shape="long",
                           tokens_per_step=None, row_count=None, steps_per_unit=None,
                           target_row_length=None):
    """按 WorstCaseSpec 构造摸底小批；未给 row_count 时按 shape 填满预算（兼容旧测试）。"""
    from grpx.trainer import build_batch
    from grpx.types import Step, Trajectory

    if shape not in ("long", "short"):
        raise ValueError(f"未知的最坏小批 shape: {shape!r}")
    filler = padding_token_id if padding_token_id is not None else 0
    step_growth = _worst_case_step_growth(max_new_tokens, tokens_per_step)

    if row_count is None or steps_per_unit is None:
        want_long = shape == "long" and merge_append_only and max_steps > 1
        specs = enumerate_worst_case_specs(
            len(prompt_token_ids), max_new_tokens, max_rows, max_tokens,
            max_steps=max_steps, merge_append_only=merge_append_only,
            tokens_per_step=tokens_per_step,
        )
        if want_long:
            pool = [spec for spec in specs if spec.steps_per_unit > 1] or specs
        else:
            pool = [spec for spec in specs if spec.steps_per_unit == 1] or specs
        chosen = max(pool, key=lambda spec: (spec.area, spec.row_length, spec.row_count))
        row_count = chosen.row_count
        steps_per_unit = chosen.steps_per_unit
        if target_row_length is None:
            target_row_length = chosen.row_length

    if target_row_length is None:
        target_row_length = worst_case_row_length(
            len(prompt_token_ids), max_new_tokens, steps_per_unit, tokens_per_step)

    use_merge = steps_per_unit > 1
    trajectories = []
    for index in range(row_count):
        if use_merge:
            prompt0, action_token_ids, growth = _fit_merged_tokens_to_row_length(
                prompt_token_ids, max_new_tokens, target_row_length,
                steps_per_unit, step_growth, filler)
            trajectories.append(_append_only_worst_case_trajectory(
                f"worst-case-{index}", temperature, prompt0, action_token_ids,
                steps_per_unit, growth))
        else:
            prompt, action_token_ids = _fit_tokens_to_row_length(
                prompt_token_ids, max_new_tokens, target_row_length, filler)
            trajectories.append(Trajectory(
                f"worst-case-{index}", 0, temperature,
                [Step(prompt, action_token_ids, advantage=1.0)],
            ))
    if not trajectories:
        raise ValueError("最坏小批未能构造任何轨迹")
    batch = build_batch(trajectories, filler, device, row_weighting)
    length = max(len(trajectory.steps[-1].prompt_token_ids) + len(trajectory.steps[-1].action_token_ids)
                 for trajectory in trajectories)
    return batch, len(trajectories), length, steps_per_unit


def probe_worst_case_shapes(model, prompt_token_ids, max_new_tokens, config, padding_token_id,
                            device, temperature, restore_trainable, tokens_per_step=None):
    """枚举行数，摸「总面积最大」与「回答位最多」两个形状（可重合），取峰值更大者。"""
    import torch

    specs = enumerate_worst_case_specs(
        len(prompt_token_ids), max_new_tokens,
        config.max_rows_per_minibatch, config.max_tokens_per_minibatch,
        max_steps=config.max_steps,
        merge_append_only=config.merge_append_only_turns,
        tokens_per_step=tokens_per_step,
    )
    targets = select_worst_case_probe_targets(specs)
    best = None
    details = []
    for spec in targets:
        restore_trainable()
        batch, unit_count, row_length, steps_per_unit = build_worst_case_batch(
            prompt_token_ids, max_new_tokens,
            config.max_rows_per_minibatch, config.max_tokens_per_minibatch,
            padding_token_id, device, temperature=temperature,
            row_weighting=config.row_weighting,
            max_steps=config.max_steps,
            merge_append_only=config.merge_append_only_turns,
            shape=spec.shape,
            tokens_per_step=tokens_per_step,
            row_count=spec.row_count,
            steps_per_unit=spec.steps_per_unit,
            target_row_length=spec.row_length,
        )
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        run_worst_case_update_probe(model, batch, config)
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
        else:
            peak = 0
        segment_total = int(batch["segment_advantages"].shape[0])
        detail = {
            "shape": spec.shape,
            "unit_count": unit_count,
            "segments": segment_total,
            "steps_per_unit": steps_per_unit,
            "row_length": row_length,
            "area": unit_count * row_length,
            "answer_positions": int(batch["action_mask"].sum().item()),
            "tokens_per_step": (_worst_case_step_growth(max_new_tokens, tokens_per_step)
                                if steps_per_unit > 1 else max_new_tokens),
            "peak_bytes": peak,
        }
        details.append(detail)
        if best is None or peak > best["peak_bytes"]:
            best = detail
    restore_trainable()
    return best, details


def run_worst_case_update_probe(model, batch, config):
    """对最坏小批做一次带梯度的前向+反向，用于显存摸底。"""
    from grpx.loop import loss_options
    from grpx.trainer import action_log_probabilities, grpo_loss

    model.train()
    model.zero_grad(set_to_none=True)
    log_probabilities, mask = action_log_probabilities(model, batch)
    loss, metrics = grpo_loss(
        log_probabilities, log_probabilities.detach(), mask, batch["advantages"],
        row_weights=batch.get("row_weights"),
        segment_ids=batch.get("segment_ids"),
        segment_advantages=batch.get("segment_advantages"),
        segment_weights=batch.get("segment_weights"),
        **loss_options(config),
    )
    loss.backward()
    return metrics


def vllm_start_command(model_name, lora_rank=16, port=8000):
    return (f"vllm serve {model_name} --enable-lora --max-lora-rank {lora_rank} "
            f"--port {port}")


def check_vllm_base_model(payload, expected_name):
    """纯函数：/v1/models 载荷里是否包含期望的基座。"""
    from grpx.backends.vllm import base_model_ids_from_vllm_models

    bases = base_model_ids_from_vllm_models(payload)
    detail = f"期望 {expected_name}；服务端基座 {bases or '（无）'}"
    if expected_name in bases:
        return CheckResult("后端", "基座模型", PASS, detail)
    return CheckResult(
        "后端", "基座模型", FAIL, detail,
        "用同一模型启动 vLLM，或在 [backend] 设置 server_model_name 声明服务端用的名字")


def check_vllm_reachable(base_url, model_name, lora_rank=16, timeout_seconds=5):
    import requests

    url = base_url.rstrip("/") + "/v1/models"
    try:
        response = requests.get(url, timeout=timeout_seconds)
    except requests.RequestException as error:
        return CheckResult(
            "后端", "vLLM 服务", FAIL, f"无法连接 {url}：{error}",
            f"先启动服务，例如：{vllm_start_command(model_name, lora_rank)}")
    if response.status_code >= 400:
        return CheckResult(
            "后端", "vLLM 服务", FAIL, f"{url} 返回 HTTP {response.status_code}",
            f"检查服务是否正常；启动示例：{vllm_start_command(model_name, lora_rank)}")
    return CheckResult("后端", "vLLM 服务", PASS, f"已响应 {url}")


def check_vllm_service(base_url, model_name, lora_rank=16, timeout_seconds=5,
                       expected_base_model=None):
    """连通性 + 基座名；返回若干 CheckResult。"""
    import requests

    expected = expected_base_model or model_name
    url = base_url.rstrip("/") + "/v1/models"
    try:
        response = requests.get(url, timeout=timeout_seconds)
    except requests.RequestException as error:
        return [CheckResult(
            "后端", "vLLM 服务", FAIL, f"无法连接 {url}：{error}",
            f"先启动服务，例如：{vllm_start_command(model_name, lora_rank)}")]
    if response.status_code >= 400:
        return [CheckResult(
            "后端", "vLLM 服务", FAIL, f"{url} 返回 HTTP {response.status_code}",
            f"检查服务是否正常；启动示例：{vllm_start_command(model_name, lora_rank)}")]
    results = [CheckResult("后端", "vLLM 服务", PASS, f"已响应 {url}")]
    try:
        payload = response.json()
    except ValueError:
        results.append(CheckResult(
            "后端", "基座模型", FAIL, f"{url} 返回的不是 JSON", "检查服务是否为 vLLM"))
        return results
    results.append(check_vllm_base_model(payload, expected))
    return results


def _lora_b_parameters(model):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_B" in name]


def check_vllm_sync_takes_effect(backend, model, prompt_token_ids):
    """同步 → 贪心 → 扰动 LoRA B → 再同步再贪心；两次输出必须不同。"""
    import torch

    parameters = _lora_b_parameters(model)
    if not parameters:
        return CheckResult("后端", "同步是否生效", FAIL, "模型里没有可训练的 lora_B 参数",
                           "确认 [model] 已启用 LoRA")
    backend.synchronize_policy(model)
    before = backend.generate_greedy(prompt_token_ids)
    snapshot = {name: parameter.detach().clone() for name, parameter in parameters}
    with torch.no_grad():
        for _, parameter in parameters:
            parameter.copy_(torch.randn_like(parameter) * 10)
    try:
        backend.synchronize_policy(model)
        after = backend.generate_greedy(prompt_token_ids)
    finally:
        with torch.no_grad():
            for name, parameter in parameters:
                parameter.copy_(snapshot[name])
        backend.synchronize_policy(model)
    if before == after:
        return CheckResult(
            "后端", "同步是否生效", FAIL,
            "扰动 LoRA 后贪心输出不变，服务端可能仍在用旧权重",
            "检查：1) VLLM_ALLOW_RUNTIME_LORA_UPDATING=1；"
            "2) 版本号适配器（policy-N）是否成功加载；"
            "3) 服务端是否仍挂着旧名字或旧权重")
    return CheckResult("后端", "同步是否生效", PASS, "扰动 LoRA 后贪心输出已变化")


def run_doctor(configuration, output=print):
    """返回退出码：有失败项时为 1"""
    report = Report(output)
    report.add(*check_environment())
    if report.failed:
        output(report.summary())
        return 1

    try:
        config = build_training_config(configuration)
        train_dataset, evaluation_dataset, environment_factory = build_task(configuration["task"])
    except ConfigurationError as error:
        report.add(CheckResult("配置", "构建", FAIL, str(error), "按提示修改配置文件"))
        output(report.summary())
        return 1
    report.add(*judge_dataset(train_dataset, evaluation_dataset, config.groups_per_iteration))
    if report.failed:
        output(report.summary())
        return 1
    report.add(walk_through_environment(environment_factory, train_dataset[0], config.max_steps))
    if report.failed:
        output(report.summary())
        return 1

    import torch
    from pathlib import Path

    backend_section = configuration.get("backend", {})
    backend_type = backend_section.get("type", "huggingface")
    adapter_directory = None
    if backend_type == "vllm":
        # 每次预检用唯一运行目录名编进适配器前缀，避免连跑 doctor 撞名
        from datetime import datetime
        from grpx.backends.vllm import normalize_base_urls
        doctor_run = Path.cwd() / "outputs" / f"doctor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        adapter_directory = doctor_run / "vllm_adapter"
        adapter_directory.mkdir(parents=True, exist_ok=True)
        expected_base = backend_section.get("server_model_name") or configuration["model"]["name"]
        for url in normalize_base_urls(
                backend_section.get("base_url", "http://localhost:8000")):
            report.add(*check_vllm_service(
                url,
                model_name=configuration["model"]["name"],
                lora_rank=configuration["model"].get("lora_rank", 16),
                expected_base_model=expected_base,
            ))
            if report.failed:
                output(report.summary())
                return 1
        hint = shared_gpu_memory_hint(config.memory_fraction)
        if hint is not None:
            report.add(hint)

    output("加载模型……")
    backend = None
    try:
        training_device = configuration.get("training", {}).get("device", "cuda")
        try:
            model, tokenizer, dtype_name = load_base_model_and_tokenizer(
                configuration["model"], device=training_device)
        except ConfigurationError as error:
            report.add(CheckResult("模型", "加载", FAIL, str(error), "按提示修改 [model] 或 [backend]"))
            output(report.summary())
            return 1

        report.add(judge_chat_template(tokenizer))
        lora_targets = configuration["model"].get("lora_target_modules", "all-linear")
        lora_summary = summarize_lora_targets(model, lora_targets)
        report.add(*judge_lora_targets(lora_summary))
        if report.failed:
            output(report.summary())
            return 1

        try:
            model = apply_lora(
                model, configuration["model"], config.seed, dtype_name=dtype_name)
            # 复用 build_model_and_backend 的后端装配：临时塞回已加载的模型
            model, tokenizer, backend = _attach_backend(
                model, tokenizer, configuration, adapter_directory=adapter_directory)
        except ConfigurationError as error:
            report.add(CheckResult("模型", "加载", FAIL, str(error), "按提示修改 [model] 或 [backend]"))
            output(report.summary())
            return 1
        model.grpx_output_logit_chunk_size = config.output_logit_chunk_size

        if backend_type == "vllm":
            probe_prompt = [token_id for token_id in range(1, 9)]
            try:
                report.add(check_vllm_sync_takes_effect(backend, model, probe_prompt))
            except Exception as error:    # noqa: BLE001
                report.add(CheckResult(
                    "后端", "同步是否生效", FAIL, _one_line(error),
                    f"确认适配器目录对服务端可读；启动示例："
                    f"{vllm_start_command(configuration['model']['name'], configuration['model'].get('lora_rank', 16))}"))
            if report.failed:
                output(report.summary())
                return 1
        else:
            backend.synchronize_policy(model)

        _check_with_model(report, model, tokenizer, backend, train_dataset, evaluation_dataset,
                          environment_factory, config, torch)
        output(report.summary())
        return 1 if report.failed else 0
    finally:
        # 预检不是训练：结束后卸掉最后一个适配器，避免服务端堆积
        if backend is not None and hasattr(backend, "close"):
            try:
                backend.close()
            except Exception:    # noqa: BLE001
                pass


def _attach_backend(model, tokenizer, configuration, adapter_directory=None):
    """在已加载的（含 LoRA）模型上装配后端；逻辑与 build_model_and_backend 对齐。"""
    from pathlib import Path

    from grpx.backends.common import resolve_end_token_ids
    from grpx.backends.huggingface import HuggingFaceBackend
    from grpx.command_line import validate_backend_section

    backend_section = dict(configuration.get("backend", {}))
    validate_backend_section(backend_section)
    backend_type = backend_section.pop("type", "huggingface")
    try:
        if backend_type == "vllm":
            from grpx.backends.vllm import VllmBackend, sanitize_adapter_name_component
            if adapter_directory is None:
                raise ConfigurationError("构建 vLLM 后端时需要 adapter_directory")
            end_token_ids = resolve_end_token_ids(model, tokenizer)
            base_name = sanitize_adapter_name_component(backend_section.pop("adapter_name", "policy"))
            run_tag = sanitize_adapter_name_component(Path(adapter_directory).resolve().parent.name)
            backend_section["adapter_name"] = f"{base_name}-{run_tag}"
            server_model_name = backend_section.pop("server_model_name", None)
            model_name = configuration["model"]["name"]
            backend_section["expected_base_model"] = server_model_name or model_name
            backend_section["training_model_name"] = model_name
            backend = VllmBackend(
                tokenizer, end_token_ids, adapter_directory=adapter_directory, **backend_section)
            try:
                backend.verify_server_base_model()
            except RuntimeError as error:
                raise ConfigurationError(str(error)) from error
        else:
            backend = HuggingFaceBackend(model, tokenizer, **backend_section)
    except ValueError as error:
        raise ConfigurationError(f"[backend] {error}") from error
    return model, tokenizer, backend


def _check_with_model(report, model, tokenizer, backend, train_dataset, evaluation_dataset,
                      environment_factory, config, torch):
    from grpx.advantage import compute_advantages
    from grpx.device import indexed_cuda_device
    from grpx.loop import collect_trajectories, measure_policy_shift, split_into_row_chunks, update_policy
    from grpx.trainer import action_log_probabilities, build_batch

    device = config.device
    cuda_device = indexed_cuda_device(device)
    use_cuda = cuda_device is not None and torch.cuda.is_available()
    # 与训练同款显存上限：超额时先清缓存再明确报错，预检据此把撞上限转成失败项
    if use_cuda:
        torch.cuda.set_per_process_memory_fraction(
            config.memory_fraction, device=cuda_device)
    # 与训练第 0 轮相同的种子和题目，看到的就是训练开头会遇到的情况
    torch.manual_seed(config.seed)
    samples = random.Random(config.seed).sample(train_dataset, config.groups_per_iteration)
    peak_by_phase = {}
    report.output(f"采样一轮（{config.groups_per_iteration} 组 × {config.group_size}）……")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    start = time.perf_counter()
    try:
        trajectories = collect_trajectories(backend, tokenizer, environment_factory, samples, config, iteration=0)
    except Exception as error:    # noqa: BLE001
        if is_out_of_memory(error):
            report.add(CheckResult(
                "采样", "显存", FAIL, f"采样时显存不足（{_one_line(error)}）",
                "减小 backend.max_tokens_per_batch 或 max_concurrent_requests；"
                "或设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"))
            return
        raise
    sampling_seconds = time.perf_counter() - start
    if use_cuda:
        peak_by_phase["采样"] = torch.cuda.max_memory_allocated(cuda_device)
    diagnostics = {}
    kept = compute_advantages(trajectories, normalize_std=config.normalize_std,
                              step_advantage_weight=config.step_advantage_weight,
                              discount_factor=config.discount, diagnostics=diagnostics)
    report.add(*judge_sampling(
        trajectories, kept, backend.max_new_tokens, truncated_reward=config.truncated_reward,
        stop_strings=getattr(backend, "stop_strings", ())))
    report.add(judge_oversized_row_fraction(
        trajectories, config.max_tokens_per_minibatch,
        merge_append_only=config.merge_append_only_turns))
    from grpx.types import mean_environment_metrics
    for name, value in mean_environment_metrics(trajectories, prefix="").items():
        report.add(CheckResult("采样", name, PASS, f"{value:.3f}"))
    if config.step_advantage_weight > 0:
        report.add(judge_step_signal_coverage(diagnostics["step_signal_coverage"], config.step_advantage_weight))

    padding_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(backend.end_token_ids)
    model.eval()
    values = []
    with torch.no_grad():
        for chunk in split_into_row_chunks(
                trajectories, config.max_rows_per_minibatch, config.max_tokens_per_minibatch,
                merge_append_only=config.merge_append_only_turns):
            log_probabilities, mask = action_log_probabilities(
                model, build_batch(chunk, padding_token_id, model.device, config.row_weighting))
            values += log_probabilities[mask].tolist()
    report.add(judge_log_probabilities(values))

    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    snapshot = {name: parameter.detach().clone() for name, parameter in trainable.items()}

    def restore_trainable():
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(snapshot[name])

    # 最坏小批：按行数枚举，摸总面积最大与回答位最多两个形状，取峰值更大者；
    # 步间增量用采样里相邻提示长度差的最大值（含环境反馈）
    report.output("最坏小批显存摸底（按行数枚举；面积最大 / 回答位最多）……")
    try:
        prompt_token_ids = longest_first_prompt_token_ids(
            tokenizer, environment_factory, train_dataset)
        tokens_per_step = max_prompt_growth_tokens(trajectories, minimum=backend.max_new_tokens)
        best, details = probe_worst_case_shapes(
            model, prompt_token_ids, backend.max_new_tokens, config, padding_token_id,
            model.device, backend.temperature, restore_trainable,
            tokens_per_step=tokens_per_step)
        peak_by_phase["最坏小批"] = best["peak_bytes"]
        shape_summary = "；".join(
            f"{item['shape']} {item['unit_count']}行×{item['segments']}段×长{item['row_length']}"
            f"（步增{item['tokens_per_step']}）→{item['peak_bytes'] / 2 ** 30:.1f}GB"
            for item in details
        )
        report.add(CheckResult(
            "试更新", "最坏小批", PASS,
            f"取 {best['shape']}：{best['unit_count']} 行共 {best['segments']} 段 × 最长 {best['row_length']}"
            f"（首提示 {len(prompt_token_ids)} + 步增 {tokens_per_step}，回答上限 "
            f"{backend.max_new_tokens}），峰值 "
            f"{best['peak_bytes'] / 2 ** 30:.1f} GB｜{shape_summary}"))
    except Exception as error:    # noqa: BLE001
        if is_out_of_memory(error):
            report.add(CheckResult(
                "试更新", "最坏小批", FAIL,
                f"最坏小批显存不足（{_one_line(error)}）",
                "减小 training.max_tokens_per_minibatch、max_rows_per_minibatch 或 backend.max_new_tokens；"
                "或设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"))
            restore_trainable()
            return
        raise
    finally:
        restore_trainable()

    update_seconds = 0.0
    if not kept:
        report.add(CheckResult("试更新", "跳过", WARNING, "这一轮没有有信号的组，无法试更新",
                               "先解决上面的采样问题"))
    else:
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=config.learning_rate)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        start = time.perf_counter()
        try:
            (history, batches, old_log_probabilities, _gap_metrics, _merge_fraction,
             peak_batch, _oversized) = update_policy(
                model, optimizer, kept, padding_token_id, config)
            if not history:
                report.add(CheckResult(
                    "试更新", "跳过", WARNING,
                    "有信号的组全部因超过 max_tokens_per_minibatch 被跳过，无法试更新",
                    "增大 training.max_tokens_per_minibatch，或缩短对话"))
                return
            shift = measure_policy_shift(model, batches, old_log_probabilities, config)
            if use_cuda:
                torch.cuda.synchronize(cuda_device)
        except Exception as error:    # noqa: BLE001
            if is_out_of_memory(error):
                report.add(CheckResult(
                    "试更新", "显存", FAIL, f"更新时显存不足（{_one_line(error)}）",
                    "减小 training.max_tokens_per_minibatch 或 max_rows_per_minibatch；"
                    "或设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"))
                return
            raise
        finally:
            restore_trainable()
            del optimizer
        update_seconds = time.perf_counter() - start
        if use_cuda:
            peak_by_phase["更新"] = torch.cuda.max_memory_allocated(cuda_device)
        report.add(judge_first_step_ratio(history[0]["max_ratio_deviation"]))
        report.add(judge_shift(
            shift, config.learning_rate,
            mixed_precision=bool(getattr(model, "grpx_mixed_precision", False))))
        if peak_batch is not None:
            report.add(CheckResult(
                "试更新", "峰值小批", PASS,
                f"{peak_batch['rows']} 行 × 最长 {peak_batch['length']} × {peak_batch['segments']} 段，"
                f"该批峰值 {peak_batch['peak_gigabytes']:.1f} GB（{peak_batch['phase']}）"))

    if use_cuda and peak_by_phase:
        total_memory = torch.cuda.get_device_properties(cuda_device).total_memory
        report.add(judge_memory(
            peak_by_phase, total_memory, memory_fraction=config.memory_fraction))
    restored = all(torch.equal(parameter, snapshot[name]) for name, parameter in trainable.items())
    report.add(CheckResult("试更新", "还原权重", PASS if restored else FAIL,
                           "可训练参数已逐位还原" if restored else "还原后的参数与更新前不一致",
                           "这是 grpx 的问题，请报告"))
    report.add(estimate_duration(config, sampling_seconds, update_seconds,
                                 len(evaluation_dataset) if evaluation_dataset else 0))
