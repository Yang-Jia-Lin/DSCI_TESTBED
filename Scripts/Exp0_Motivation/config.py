"""Shared configuration and output helpers for Exp0 motivation studies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "Scripts" / "Results" / "Exp0_Motivation"
LATEST_RUN_FILE = RESULTS_ROOT / "latest_run.txt"


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
    exp3_users: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    exp3_state_vector_dim: int = 20
    exp3_config_bytes: int = 16
    exp3_control_bandwidth_mbps: float = 4.0
    exp3_rtt_ms: float = 20.0
    exp3_decision_latency_ms: float = 2.0
    exp3_slow_optimization_latency_ms: float = 500.0
    exp3_slow_fastpath_latency_ms: float = 2.0
    exp3_schedule_period_s: float = 30.0
    exp3_inference_latency_ms: float = 50.0


DEFAULT_CONFIG = MotivationConfig()


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def subdirs(run_dir: Path) -> dict[str, Path]:
    return {
        "data": run_dir / "Data",
        "figures": run_dir / "Figures",
        "logs": run_dir / "Logs",
    }


def prepare_run_dir(
    run_id: str | None = None,
    *,
    prefer_latest: bool = False,
    create: bool = True,
) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    selected = run_id
    if selected is None and prefer_latest and LATEST_RUN_FILE.is_file():
        selected = LATEST_RUN_FILE.read_text(encoding="utf-8").strip() or None
    if selected is None:
        selected = new_run_id()
    run_dir = RESULTS_ROOT / selected
    if create:
        for path in subdirs(run_dir).values():
            path.mkdir(parents=True, exist_ok=True)
        LATEST_RUN_FILE.write_text(selected + "\n", encoding="utf-8")
    return run_dir


def data_dir(run_dir: Path) -> Path:
    return subdirs(run_dir)["data"]


def figure_dir(run_dir: Path) -> Path:
    return subdirs(run_dir)["figures"]


def log_dir(run_dir: Path) -> Path:
    return subdirs(run_dir)["logs"]


def canonical_curve_path(run_dir: Path) -> Path:
    return data_dir(run_dir) / "canonical_exit_curves.csv"


def config_path(run_dir: Path) -> Path:
    return run_dir / "config.json"


def paper_numbers_path(run_dir: Path) -> Path:
    return run_dir / "paper_numbers.json"


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
