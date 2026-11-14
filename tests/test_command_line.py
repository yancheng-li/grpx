import pytest

from examples.guess_number import GuessNumberEnvironment
from grpx.command_line import (ConfigurationError, build_environment_factory, import_object,
                               load_configuration, main, parse_value)
from grpx.environment import SingleTurnEnvironment

MINIMAL = """
[model]
name = "some-model"

[task]
dataset = "examples.guess_number:make_dataset"
environment = "examples.guess_number:GuessNumberEnvironment"
"""


def write_configuration(tmp_path, text=MINIMAL):
    path = tmp_path / "configuration.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_override_values_are_parsed(tmp_path):
    configuration = load_configuration(write_configuration(tmp_path), [
        "training.learning_rate=2e-5", "training.seed=1", "training.algorithm=dr_grpo",
        "training.normalize_std=false", "task.environment_options={ reveal_interval = true }",
        "training.run_name = my-run",
    ])
    training = configuration["training"]
    assert training["learning_rate"] == 2e-5 and isinstance(training["learning_rate"], float)
    assert training["seed"] == 1 and isinstance(training["seed"], int)
    assert training["algorithm"] == "dr_grpo"
    assert training["normalize_std"] is False
    assert training["run_name"] == "my-run"
    assert configuration["task"]["environment_options"] == {"reveal_interval": True}

def test_override_replaces_file_value(tmp_path):
    path = write_configuration(tmp_path, MINIMAL + '\n[training]\nalgorithm = "gigpo"\nseed = 0\n')
    assert load_configuration(path, ["training.seed=3"])["training"] == {"algorithm": "gigpo", "seed": 3}

def test_parse_value():
    assert parse_value("1e-5") == 1e-5
    assert parse_value('"quoted"') == "quoted"
    assert parse_value("not toml at all") == "not toml at all"
    assert parse_value("null") is None
    assert parse_value("None") is None
    assert parse_value("none") is None

def test_unknown_section(tmp_path):
    with pytest.raises(ConfigurationError, match=r"未知的配置段 \[trainer\]"):
        load_configuration(write_configuration(tmp_path, MINIMAL + "\n[trainer]\nseed = 1\n"), [])

def test_unknown_field_gets_suggestion(tmp_path):
    with pytest.raises(ConfigurationError, match="是不是想写 'learning_rate'"):
        load_configuration(write_configuration(tmp_path), ["training.learnng_rate=1e-5"])

def test_unknown_field_without_close_match(tmp_path):
    with pytest.raises(ConfigurationError) as error:
        load_configuration(write_configuration(tmp_path), ["backend.zzzz=1"])
    assert "没有 'zzzz'" in str(error.value) and "是不是" not in str(error.value)

def test_missing_dataset(tmp_path):
    with pytest.raises(ConfigurationError, match="必须有 dataset"):
        load_configuration(write_configuration(tmp_path, '[model]\nname = "m"\n[task]\nenvironment = "a:b"\n'), [])

def test_unknown_algorithm(tmp_path):
    with pytest.raises(ConfigurationError, match="未知的算法 'gipgo'"):
        load_configuration(write_configuration(tmp_path), ["training.algorithm=gipgo"])

@pytest.mark.parametrize("override", ["training", "seed=1", "training.=1", ".seed=1"])
def test_malformed_override(tmp_path, override):
    with pytest.raises(ConfigurationError, match="段.字段=值"):
        load_configuration(write_configuration(tmp_path), [override])

def test_top_level_value_outside_section(tmp_path):
    text = 'model = "m"\n[task]\ndataset = "a:b"\nenvironment = "c:d"\n'
    with pytest.raises(ConfigurationError, match=r"应该是一个配置段 \[model\]"):
        load_configuration(write_configuration(tmp_path, text), [])


def test_import_object_without_colon():
    with pytest.raises(ConfigurationError, match="模块路径:对象名"):
        import_object("examples.guess_number.make_dataset")

def test_import_object_missing_module():
    with pytest.raises(
            ConfigurationError,
            match=r"无法导入 'examples.no_such_module:thing'.*项目根目录.*PYTHONPATH"):
        import_object("examples.no_such_module:thing")

def test_import_object_missing_attribute():
    with pytest.raises(ConfigurationError, match="无法导入 'examples.guess_number:NoSuchThing'"):
        import_object("examples.guess_number:NoSuchThing")
    # 属性缺失不是路径问题，不应提示 PYTHONPATH
    with pytest.raises(ConfigurationError) as caught:
        import_object("examples.guess_number:NoSuchThing")
    assert "PYTHONPATH" not in str(caught.value)

def test_import_object_success():
    assert import_object("examples.guess_number:GuessNumberEnvironment") is GuessNumberEnvironment


def test_multi_turn_environment_factory():
    factory = build_environment_factory({
        "environment": "examples.guess_number:GuessNumberEnvironment",
        "environment_options": {"reveal_interval": True},
    })
    first, second = factory(), factory()
    assert isinstance(first, GuessNumberEnvironment) and first is not second    # 每条轨迹一个独立实例
    assert first.reveal_interval is True

def test_single_turn_environment_factory():
    factory = build_environment_factory({
        "reward_function": "examples.arithmetic:reward_function",
        "success_function": "examples.arithmetic:is_correct",
    })
    environment = factory()
    assert isinstance(environment, SingleTurnEnvironment)
    environment.reset({"prompt": "1+1=?", "answer": 2}, seed=0)
    assert environment.success_function is not None

def test_environment_factory_needs_one_of_the_two():
    with pytest.raises(ConfigurationError, match="environment（多轮）或 reward_function（单轮）"):
        build_environment_factory({"dataset": "examples.guess_number:make_dataset"})

def test_environment_factory_rejects_both():
    with pytest.raises(ConfigurationError, match="只能写一个"):
        build_environment_factory({"environment": "examples.guess_number:GuessNumberEnvironment",
                                   "reward_function": "examples.arithmetic:reward_function"})

def test_wrong_environment_option_is_one_line_error():
    with pytest.raises(ConfigurationError, match="参数有误"):
        build_environment_factory({"environment": "examples.guess_number:GuessNumberEnvironment",
                                   "environment_options": {"reveal_intervals": True}})


def test_main_reports_configuration_errors_as_one_line(tmp_path):
    with pytest.raises(SystemExit) as exit_information:
        main(["train", "--config", str(write_configuration(tmp_path)), "--set", "training.learnng_rate=1e-5"])
    message = str(exit_information.value.code)
    assert message.startswith("配置错误：") and "\n" not in message
