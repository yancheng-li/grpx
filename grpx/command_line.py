import argparse
import copy
import difflib
import importlib
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

from grpx.config import ALGORITHM_PRESETS, TrainingConfig
from grpx.environment import SingleTurnEnvironment

BACKEND_HUGGINGFACE_KEYS = {"max_rows_per_batch", "max_tokens_per_batch"}
BACKEND_VLLM_KEYS = {
    "base_url", "max_concurrent_requests", "request_timeout_seconds", "adapter_name",
    "adapter_path_local_prefix", "adapter_path_server_prefix",
    "server_model_name",  # 服务端基座别名 / 本地路径；默认与 [model].name 相同
}
BACKEND_COMMON_KEYS = {"type", "max_new_tokens", "temperature", "stop_strings"}

ALLOWED_KEYS = {
    "model": {"name", "dtype", "lora_rank", "lora_alpha", "lora_target_modules"},
    "backend": BACKEND_COMMON_KEYS | BACKEND_HUGGINGFACE_KEYS | BACKEND_VLLM_KEYS,
    "task": {"environment", "environment_options", "reward_function", "success_function",
             "dataset", "train_dataset_options", "evaluation_dataset_options"},
    "training": {"algorithm"} | {field.name for field in fields(TrainingConfig)},
}

MODEL_DTYPES = {
    "float32": "float32",
    "bfloat16": "bfloat16",
}

# 恢复时改这些以外的项会让一次运行前后不一致。
# 显存相关项只影响批大小，不改变已采样轨迹上的学习结果，允许中途调小。
RESUME_ALLOWED_OVERRIDES = {
    ("training", "iterations"),
    ("training", "evaluation_interval"),
    ("training", "checkpoint_interval"),
    ("training", "shift_measurement_interval"),
    ("training", "max_tokens_per_minibatch"),
    ("training", "max_rows_per_minibatch"),
    ("training", "truncated_reward"),
    ("training", "memory_fraction"),
    ("backend", "max_tokens_per_batch"),
    ("backend", "max_rows_per_batch"),
    ("backend", "max_concurrent_requests"),
}

MINIMUM_PYTHON = (3, 11)    # 配置文件用标准库 tomllib

class ConfigurationError(Exception):
    pass


def import_object(path):
    module_name, _, object_name = path.partition(":")
    if not module_name or not object_name:
        raise ConfigurationError(f"'{path}' 应写成 '模块路径:对象名'")
    try:
        return getattr(importlib.import_module(module_name), object_name)
    except ModuleNotFoundError as error:
        raise ConfigurationError(
            f"无法导入 '{path}'：{error}。"
            "请在项目根目录下运行，或把任务代码所在的目录加入 PYTHONPATH"
        ) from error
    except (ImportError, AttributeError) as error:
        raise ConfigurationError(f"无法导入 '{path}'：{error}") from error


def parse_value(raw_value):
    import tomllib
    # TOML 没有 null；命令行里用 null/none 表示 Python 的 None（如 truncated_reward）
    if raw_value.strip().lower() in ("null", "none"):
        return None
    try:
        return tomllib.loads(f"value = {raw_value}")["value"]
    except tomllib.TOMLDecodeError:
        return raw_value                      # 解析不了就当作字符串


def load_configuration(path, overrides):
    import tomllib    # 3.11 才有；放在函数里，旧版本 Python 才能走到 main 开头的版本检查
    try:
        with open(path, "rb") as configuration_file:
            configuration = tomllib.load(configuration_file)
    except FileNotFoundError:
        raise ConfigurationError(f"找不到配置文件 '{path}'") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"'{path}' 不是合法的 TOML：{error}") from None
    for override in overrides:                # 形如 training.learning_rate=2e-5
        key, equals_sign, raw_value = override.partition("=")
        section, dot, name = key.strip().partition(".")
        if not equals_sign or not dot or not section or not name:
            raise ConfigurationError(f"--set '{override}' 应写成 '段.字段=值'，例如 training.seed=1")
        configuration.setdefault(section, {})[name] = parse_value(raw_value.strip())

    for section, values in configuration.items():
        if section not in ALLOWED_KEYS:
            raise ConfigurationError(f"未知的配置段 [{section}]，可用的有：{sorted(ALLOWED_KEYS)}")
        if not isinstance(values, dict):
            raise ConfigurationError(f"'{section}' 应该是一个配置段 [{section}]，而不是单个值")
        for name in values:
            if name not in ALLOWED_KEYS[section]:
                suggestions = difflib.get_close_matches(name, ALLOWED_KEYS[section], n=1)
                hint = f"，是不是想写 '{suggestions[0]}'？" if suggestions else ""
                raise ConfigurationError(f"[{section}] 里没有 '{name}' 这个字段{hint}")
    if "task" not in configuration or "dataset" not in configuration["task"]:
        raise ConfigurationError("[task] 里必须有 dataset")
    if "name" not in configuration.get("model", {}):
        raise ConfigurationError("[model] 里必须有 name")
    algorithm = configuration.get("training", {}).get("algorithm", "grpo")
    if algorithm not in ALGORITHM_PRESETS:
        raise ConfigurationError(f"未知的算法 '{algorithm}'，可用的有：{sorted(ALGORITHM_PRESETS)}")
    validate_backend_section(configuration.get("backend", {}))
    build_training_config(configuration)    # 取值范围的问题也在这里一并报出
    return configuration


def validate_backend_section(backend):
    backend_type = backend.get("type", "huggingface")
    if backend_type not in ("huggingface", "vllm"):
        raise ConfigurationError(f"未知的后端类型 '{backend_type}'，可用的有：['huggingface', 'vllm']")
    if backend_type == "vllm":
        for name in BACKEND_HUGGINGFACE_KEYS:
            if name in backend:
                raise ConfigurationError(
                    f"{name} 只对本地后端有效；vLLM 后端请用 max_concurrent_requests")
        if "base_url" not in backend:
            raise ConfigurationError("[backend] type = \"vllm\" 时需要 base_url")
        base_url = backend["base_url"]
        if isinstance(base_url, str):
            pass
        elif isinstance(base_url, list):
            if not base_url:
                raise ConfigurationError("[backend] base_url 列表不能为空")
            if not all(isinstance(item, str) and item.strip() for item in base_url):
                raise ConfigurationError("[backend] base_url 列表里每一项都必须是非空字符串")
        else:
            raise ConfigurationError(
                "[backend] base_url 必须是字符串或字符串列表，"
                f"收到 {type(base_url).__name__}")
    else:
        for name in BACKEND_VLLM_KEYS:
            if name in backend:
                raise ConfigurationError(f"{name} 只对 vLLM 后端有效")


def call_with_options(function, options, description):
    """用户代码的参数写错时给一行报错；用户代码内部的其他异常照常抛出，堆栈对排查有用"""
    try:
        return function(**options)
    except TypeError as error:
        raise ConfigurationError(f"{description} 的参数有误：{error}") from error


def build_environment_factory(task):
    if "environment" in task and "reward_function" in task:
        raise ConfigurationError("[task] 里 environment（多轮）和 reward_function（单轮）只能写一个")
    if "environment" in task:
        environment_class = import_object(task["environment"])
        options = task.get("environment_options", {})
        call_with_options(environment_class, options, f"环境 {task['environment']}")    # 先试建一个，参数错误立刻暴露
        return lambda: environment_class(**options)
    if "reward_function" not in task:
        raise ConfigurationError("[task] 里需要 environment（多轮）或 reward_function（单轮）之一")
    reward_function = import_object(task["reward_function"])
    success_function = (import_object(task["success_function"])
                        if "success_function" in task else None)
    return lambda: SingleTurnEnvironment(reward_function, success_function=success_function)


def load_model(model_section, seed, device="cuda"):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from grpx.device import resolve_device

    try:
        torch_device = resolve_device(device)
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    name = model_section["name"]
    dtype_name = model_section.get("dtype", "float32")
    if dtype_name not in MODEL_DTYPES:
        raise ConfigurationError(
            f"[model] dtype 只能是 {sorted(MODEL_DTYPES)}，收到 {dtype_name!r}")
    torch_dtype = getattr(torch, MODEL_DTYPES[dtype_name])
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype).to(torch_device)
    torch.manual_seed(seed)    # LoRA 的初始化也由种子决定
    model = get_peft_model(model, LoraConfig(
        r=model_section.get("lora_rank", 16),
        lora_alpha=model_section.get("lora_alpha", 32),
        lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=model_section.get("lora_target_modules", "all-linear")))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not all(parameter.dtype == torch.float32 for parameter in trainable):
        raise ConfigurationError("可训练参数必须是 float32（PEFT 在半精度基座上应保持适配器为 float32）")
    model.grpx_mixed_precision = dtype_name == "bfloat16"
    model.print_trainable_parameters()
    return model, tokenizer


def build_training_config(configuration):
    training = dict(configuration.get("training", {}))
    try:
        return TrainingConfig.from_algorithm(training.pop("algorithm", "grpo"), **training)
    except ValueError as error:
        raise ConfigurationError(f"[training] {error}") from None


def build_task(task):
    """返回 (训练集, 评测集或 None, 环境工厂)；不加载模型"""
    dataset_builder = import_object(task["dataset"])
    train_dataset = call_with_options(dataset_builder, task.get("train_dataset_options", {}),
                                      f"数据集 {task['dataset']}")
    evaluation_dataset = (call_with_options(dataset_builder, task["evaluation_dataset_options"],
                                            f"数据集 {task['dataset']}")
                          if "evaluation_dataset_options" in task else None)
    return train_dataset, evaluation_dataset, build_environment_factory(task)


def build_model_and_backend(configuration, seed, adapter_directory=None):
    from grpx.backends.common import resolve_end_token_ids
    from grpx.backends.huggingface import HuggingFaceBackend

    training_device = configuration.get("training", {}).get("device", "cuda")
    model, tokenizer = load_model(configuration["model"], seed, device=training_device)
    backend_section = dict(configuration.get("backend", {}))
    validate_backend_section(backend_section)
    backend_type = backend_section.pop("type", "huggingface")
    try:
        if backend_type == "vllm":
            # 延迟导入：本地后端不该被 vLLM 依赖（如 requests）拖垮
            from grpx.backends.vllm import VllmBackend, sanitize_adapter_name_component
            if adapter_directory is None:
                raise ConfigurationError("构建 vLLM 后端时需要 adapter_directory")
            end_token_ids = resolve_end_token_ids(model, tokenizer)
            # 把运行目录名编进适配器前缀，避免跨运行 policy-N 撞名命中旧前缀缓存
            base_name = sanitize_adapter_name_component(backend_section.pop("adapter_name", "policy"))
            run_tag = sanitize_adapter_name_component(Path(adapter_directory).resolve().parent.name)
            backend_section["adapter_name"] = f"{base_name}-{run_tag}"
            server_model_name = backend_section.pop("server_model_name", None)
            model_name = configuration["model"]["name"]
            backend_section["expected_base_model"] = server_model_name or model_name
            backend_section["training_model_name"] = model_name
            backend = VllmBackend(
                tokenizer, end_token_ids, adapter_directory=adapter_directory, **backend_section)
            # 训练也必须核对基座，不能只依赖 doctor
            try:
                backend.verify_server_base_model()
            except RuntimeError as error:
                raise ConfigurationError(str(error)) from error
        else:
            backend = HuggingFaceBackend(model, tokenizer, **backend_section)
    except ValueError as error:
        raise ConfigurationError(f"[backend] {error}") from error
    return model, tokenizer, backend


def prepare_vllm_run_directory(configuration, training_config, resume_from=None):
    """vLLM 同步适配器需要服务端可读的目录；新运行提前建好运行目录，恢复则用原目录。"""
    from grpx.logging import create_run_directory

    if resume_from is not None:
        run_directory = Path(resume_from)
    else:
        run_directory = create_run_directory(training_config)
    adapter_directory = run_directory / "vllm_adapter"
    adapter_directory.mkdir(parents=True, exist_ok=True)
    return run_directory, adapter_directory


def build_components(configuration, resume_from=None):
    training_config = build_training_config(configuration)
    train_dataset, evaluation_dataset, environment_factory = build_task(configuration["task"])
    backend_type = configuration.get("backend", {}).get("type", "huggingface")
    run_directory = None
    adapter_directory = None
    if backend_type == "vllm":
        run_directory, adapter_directory = prepare_vllm_run_directory(
            configuration, training_config, resume_from=resume_from)
    model, tokenizer, backend = build_model_and_backend(
        configuration, training_config.seed, adapter_directory=adapter_directory)
    return (model, tokenizer, backend, train_dataset, environment_factory,
            training_config, evaluation_dataset, run_directory)


def apply_overrides(configuration, overrides, *, allowed=None):
    """把 --set 项合并进配置；allowed 不为 None 时，只允许改列出的 (段, 字段)"""
    configuration = copy.deepcopy(configuration)
    for override in overrides:
        key, equals_sign, raw_value = override.partition("=")
        section, dot, name = key.strip().partition(".")
        if not equals_sign or not dot or not section or not name:
            raise ConfigurationError(f"--set '{override}' 应写成 '段.字段=值'，例如 training.seed=1")
        if allowed is not None and (section, name) not in allowed:
            allowed_text = "、".join(f"{item[0]}.{item[1]}" for item in sorted(allowed))
            raise ConfigurationError(
                f"恢复训练时只能修改 {allowed_text}；要换其他配置请开始一次新的运行")
        if section not in ALLOWED_KEYS or name not in ALLOWED_KEYS[section]:
            suggestions = difflib.get_close_matches(name, ALLOWED_KEYS.get(section, []), n=1)
            hint = f"，是不是想写 '{suggestions[0]}'？" if suggestions else ""
            raise ConfigurationError(f"[{section}] 里没有 '{name}' 这个字段{hint}"
                                     if section in ALLOWED_KEYS
                                     else f"未知的配置段 [{section}]，可用的有：{sorted(ALLOWED_KEYS)}")
        configuration.setdefault(section, {})[name] = parse_value(raw_value.strip())
    return configuration


def load_resume_configuration(run_directory, overrides):
    path = Path(run_directory) / "resolved_configuration.json"
    if not path.exists():
        raise ConfigurationError(f"{run_directory} 里没有 resolved_configuration.json，"
                                 "无法恢复（旧运行或非命令行启动的运行）")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    configuration = apply_overrides(configuration, overrides, allowed=RESUME_ALLOWED_OVERRIDES)
    algorithm = configuration.get("training", {}).get("algorithm", "grpo")
    if algorithm not in ALGORITHM_PRESETS:
        raise ConfigurationError(f"未知的算法 '{algorithm}'，可用的有：{sorted(ALGORITHM_PRESETS)}")
    build_training_config(configuration)
    return configuration


def main(arguments=None):
    if sys.version_info < MINIMUM_PYTHON:
        raise SystemExit(f"grpx 需要 Python {'.'.join(map(str, MINIMUM_PYTHON))} 及以上，"
                         f"当前是 {sys.version.split()[0]}")
    parser = argparse.ArgumentParser(prog="grpx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_source = train_parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument("--config")
    train_source.add_argument("--resume", metavar="运行目录")
    train_parser.add_argument("--set", action="append", default=[], metavar="段.字段=值")

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--config", required=True)
    doctor_parser.add_argument("--set", action="append", default=[], metavar="段.字段=值")

    parsed = parser.parse_args(arguments)
    # 安装后的 grpx 命令不会把当前目录放进 sys.path，配置里引用的用户模块（如 examples.guess_number）就找不到
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    backend = None
    try:
        if parsed.command == "doctor":
            configuration = load_configuration(parsed.config, parsed.set)
            from grpx.doctor import run_doctor
            raise SystemExit(run_doctor(configuration))
        if parsed.resume:
            configuration = load_resume_configuration(parsed.resume, parsed.set)
            resume_from = parsed.resume
        else:
            configuration = load_configuration(parsed.config, parsed.set)
            resume_from = None
        components = build_components(configuration, resume_from=resume_from)
        from grpx.loop import train
        (model, tokenizer, backend, train_dataset, environment_factory,
         training_config, evaluation_dataset, run_directory) = components
        train(model, tokenizer, backend, train_dataset, environment_factory, training_config,
              evaluation_dataset=evaluation_dataset, resolved_configuration=configuration,
              resume_from=resume_from, run_directory=run_directory)
    except ConfigurationError as error:
        raise SystemExit(f"配置错误：{error}") from None
    finally:
        if backend is not None and hasattr(backend, "close"):
            backend.close()
