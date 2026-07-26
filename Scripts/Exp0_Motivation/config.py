"""Shared configuration and output helpers for Exp0 motivation studies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULT_DATA_DIR = EXPERIMENT_ROOT / "result_data"
RESULT_FIGURE_DIR = EXPERIMENT_ROOT / "result_figure"


@dataclass(frozen=True)
class MotivationConfig:
    bundle_id: str = "resnet50-cifar10-ee-v1"
    dataset_split: str = "val"
    accuracy_drop_tolerance_pp: float = 0.5
    curve_batch_size: int = 64
    device_profile_id: str = "device-nx1-pytorch-resnet50-cifar10"
    edge_profile_id: str = "edge-jialindesktop-pytorch-resnet50-cifar10"
    cloud_profile_id: str = "cloud-v100-pytorch-resnet50-cifar10"
    bandwidth_d2e_mbps: tuple[float, ...] = (
        60.0,
        70.0,
        80.0,
        84.0,
        86.0,
        88.0,
        89.0,
        90.0,
        91.0,
        92.0,
        94.0,
        96.0,
        98.0,
        100.0,
        102.0,
        104.0,
        106.0,
        108.0,
        110.0,
        112.0,
        114.0,
        116.0,
        118.0,
        120.0,
        122.0,
        125.0,
        130.0,
        140.0,
        150.0,
    )
    bandwidth_e2c_mbps: float = 50.0
    tensor_transport_dtype: str = "float32"
    protocol_overhead_s: float = 0.0


DEFAULT_CONFIG = MotivationConfig()


def prepare_result_dirs(*, create: bool = True) -> Path:
    if create:
        RESULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return EXPERIMENT_ROOT


def data_dir(run_dir: Path) -> Path:
    del run_dir
    return RESULT_DATA_DIR


def figure_dir(run_dir: Path) -> Path:
    del run_dir
    return RESULT_FIGURE_DIR


def canonical_curve_path(run_dir: Path) -> Path:
    return data_dir(run_dir) / "canonical_exit_curves.csv"


def config_path(run_dir: Path) -> Path:
    return data_dir(run_dir) / "config.json"


def paper_numbers_path(run_dir: Path) -> Path:
    return data_dir(run_dir) / "paper_numbers.json"


def save_config(run_dir: Path, cfg: MotivationConfig = DEFAULT_CONFIG) -> Path:
    payload = asdict(cfg)
    path = config_path(run_dir)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def update_paper_numbers(run_dir: Path, section: str, values: dict[str, Any]) -> Path:
    path = paper_numbers_path(run_dir)
    payload = load_json(path, {})
    payload[section] = values
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def required_curve_columns(cfg: MotivationConfig = DEFAULT_CONFIG) -> set[str]:
    exit_ids = ("after_layer2", "after_layer3")
    columns = {"threshold", "final_accuracy", "final_rate", "overall_accuracy"}
    for exit_id in exit_ids:
        columns.update(
            {
                f"{exit_id}_rate",
                f"{exit_id}_accuracy",
                f"{exit_id}_sequential_rate",
                f"{exit_id}_sequential_accuracy",
            }
        )
    return columns
