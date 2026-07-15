"""Evaluation constants used by Scripts/Exp1-Exp3."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable
from itertools import product

from Src.Shared.Config.paths import RESULT_DIR

EXP1_RESULT_DIR = RESULT_DIR / "Exp1_SEAM"
EXP2_RESULT_DIR = RESULT_DIR / "Exp2_Ablation"
EXP3_RESULT_DIR = RESULT_DIR / "Exp3_Convergency_and_Overhead"

DEFAULT_SOLUTION_NPZ = Path("Data/Runtime/SolutionCache/latest_solution.npz")
DEFAULT_SOLUTION_META = Path("Data/Runtime/SolutionCache/latest_solution_meta.json")
DEFAULT_TRAINING_EVENTS = Path("Data/Runtime/SolutionCache/training_events.jsonl")

PRIMARY_BUNDLES = (
    "resnet50-cifar10",
    "resnet50-neucls64",
    "resnet50-imagenet100",
)

PENDING_BUNDLES = (
    "vit-base-cifar10",
    "vit-base-neucls64",
    "vit-base-imagenet100",
)

EXPECTED_BUNDLES = PRIMARY_BUNDLES + PENDING_BUNDLES

BW_D2E_MBPS = (1.0, 5.0, 10.0, 20.0, 50.0)
BW_E2C_MBPS = (10.0, 50.0, 100.0)

SOLUTION_INDEX_COLUMNS = (
    "solution_npz",
    "solution_meta",
    "bundle_id",
    "model",
    "dataset",
    "bw_d2e_mbps",
    "bw_e2c_mbps",
)

KNOWN_DATASETS = ("cifar10", "neucls64", "imagenet100")


def split_bundle_id(bundle_id: str) -> tuple[str, str]:
    base = str(bundle_id).removesuffix("-ee-v1")
    for dataset in KNOWN_DATASETS:
        suffix = f"-{dataset}"
        if base.endswith(suffix):
            return base[: -len(suffix)], dataset
    return base, ""


def iter_bandwidth_cases(bundle_ids: Iterable[str] = EXPECTED_BUNDLES) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle_id, bw_d2e, bw_e2c in product(bundle_ids, BW_D2E_MBPS, BW_E2C_MBPS):
        model, dataset = split_bundle_id(bundle_id)
        rows.append(
            {
                "solution_npz": "",
                "solution_meta": "",
                "bundle_id": bundle_id,
                "model": model,
                "dataset": dataset,
                "bw_d2e_mbps": bw_d2e,
                "bw_e2c_mbps": bw_e2c,
            }
        )
    return rows
