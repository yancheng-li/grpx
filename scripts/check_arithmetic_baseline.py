"""快速开始第 0 轮贪心评测。需要显卡和模型，不进单元测试。

改动示例任务之后手动跑一次：

    python scripts/check_arithmetic_baseline.py

用 configs/arithmetic.toml 的默认提示词（chinese_short）训一轮，
只看训练开始前的 evaluation_success_fraction，应落在 0.60 到 0.70。
"""
import json
import os
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from grpx.command_line import main as grpx_main

    output_root = Path(tempfile.mkdtemp(prefix="grpx-arithmetic-baseline-", dir=root / "tmp"))
    try:
        grpx_main([
            "train",
            "--config", "configs/arithmetic.toml",
            "--set", "training.iterations=1",
            "--set", "training.run_name=baseline",
            "--set", f"training.output_root={output_root}",
        ])
    except SystemExit as exit_status:
        code = 0 if exit_status.code is None else exit_status.code
        if code != 0:
            raise SystemExit(code)
    metrics_path = output_root / "baseline" / "metrics.jsonl"
    row = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    value = row["evaluation_success_fraction"]
    print(f"evaluation_success_fraction={value}")
    if not 0.60 <= value <= 0.70:
        raise SystemExit(f"第 0 轮贪心评测 {value} 不在 0.60 到 0.70 之间")


if __name__ == "__main__":
    main()
