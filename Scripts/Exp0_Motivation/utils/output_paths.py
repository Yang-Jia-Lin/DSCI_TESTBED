"""
Scripts/Exp0_Motivation/utils/output_paths.py
时间戳输出目录与 latest.txt 读写。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

EXP0_ROOT = Path("Results") / "Exp0_Motivation"
LATEST_FILE = EXP0_ROOT / "latest.txt"


def create_run_output_dirs(project_root: Path | None = None) -> tuple[str, Path]:
    """
    创建带时间戳的结果目录，并写入 latest.txt。

    Returns:
        (timestamp, output_dir)
    """
    root = project_root or Path.cwd()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = root / EXP0_ROOT / timestamp
    for sub in ("logs", "data", "figures"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    latest = root / LATEST_FILE
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(timestamp, encoding="utf-8")
    return timestamp, output_dir


def resolve_output_dir(
    project_root: Path | None = None, timestamp: str | None = None
) -> Path:
    """从参数或 latest.txt 解析输出目录。"""
    root = project_root or Path.cwd()
    if timestamp is None:
        latest = root / LATEST_FILE
        if not latest.exists():
            raise FileNotFoundError(
                f"未找到 {latest}，请先运行 run_exp*.py 或通过 --timestamp 指定"
            )
        timestamp = latest.read_text(encoding="utf-8").strip()
    return root / EXP0_ROOT / timestamp
