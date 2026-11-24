import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

SCIENTIFIC_NOTATION_FIELDS = {
    "learning_rate", "shift_median_absolute_log_ratio",
    "shift_unreplayed_median_absolute_log_ratio", "approximate_kl"}

def format_record(record):
    parts = []
    for name, value in record.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            parts.append(f"{name}={value}")
        elif isinstance(value, int):
            parts.append(f"{name}={value}")
        elif name in SCIENTIFIC_NOTATION_FIELDS:
            parts.append(f"{name}={value:.1e}")
        else:
            parts.append(f"{name}={value:.3f}")
    # 用竖线分隔：部分终端捕获会把连续空格吃掉，导致字段粘连
    return " | ".join(parts)

def create_run_directory(config):
    run_name = config.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_directory = Path(config.output_root) / run_name
    run_directory.mkdir(parents=True, exist_ok=False)    # 已存在就报错，避免覆盖旧实验
    (run_directory / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")
    return run_directory

