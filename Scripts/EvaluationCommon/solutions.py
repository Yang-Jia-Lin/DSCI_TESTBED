"""Load cached DSCI solutions and evaluate strategy matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Scripts.EvaluationCommon.config import (
    DEFAULT_SOLUTION_META,
    DEFAULT_SOLUTION_NPZ,
    SOLUTION_INDEX_COLUMNS,
)
from Src.Phase2_Scheduler.Objective.compute_accuracy import compute_expected_accuracy
from Src.Phase2_Scheduler.Objective.compute_latency import compute_total_latency
from Src.Phase2_Scheduler.Objective.compute_P import compute_layer_exit_probs
from Src.Phase2_Scheduler.Objective.objective import objective
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Partitioning.split_actions import (
    decode_split_row,
    deployment_pair_kind,
    encode_split_row,
)
from Src.Shared.Profiles.segment_profile import load_segment_profile


@dataclass(frozen=True)
class SolutionBundle:
    solution_path: Path
    meta_path: Path
    meta: dict[str, Any]
    paras: Paras
    X: np.ndarray
    Y: np.ndarray
    F_e: np.ndarray
    F_c: np.ndarray


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _profile_owner(profile_id: str, bundle_id: str, manifest_id: str, model_hash: str) -> dict[str, Any]:
    profile = load_segment_profile(profile_id)
    metadata = profile.metadata
    return {
        "bundle_id": bundle_id,
        "manifest_id": manifest_id,
        "model_hash": model_hash,
        "backend": str(metadata["backend"]),
        "execution_profile_id": profile_id,
        "worker_count": int(metadata["worker_count"]),
    }


def state_from_solution_meta(meta: dict[str, Any]) -> dict[str, Any]:
    signature = meta.get("state_signature") or {}
    model = signature.get("model") or {}
    bundle_id = str(model.get("bundle_id") or "")
    if not bundle_id:
        raise ValueError("Solution metadata does not contain state_signature.model.bundle_id")
    manifest_id = str(signature["manifest_id"])
    model_hash = str(signature["model_hash"])
    users = []
    for index, user in enumerate(signature.get("users") or []):
        profile_id = str(user.get("execution_profile_id") or "")
        if not profile_id:
            raise ValueError(f"Solution metadata user {index} lacks execution_profile_id")
        users.append(
            {
                **_profile_owner(profile_id, bundle_id, manifest_id, model_hash),
                "user_id": int(user.get("user_id", index)),
                "BW_d2e": float(user.get("BW_d2e", 0.0)),
            }
        )

    edge_sig = signature.get("edge") or {}
    cloud_sig = signature.get("cloud") or {}
    edge_profile_id = str(edge_sig.get("execution_profile_id") or "")
    cloud_profile_id = str(cloud_sig.get("execution_profile_id") or "")
    if not edge_profile_id or not cloud_profile_id:
        raise ValueError("Solution metadata lacks edge/cloud execution profiles")

    return {
        "bundle_id": bundle_id,
        "resource_mode": str(signature.get("resource_mode", "fixed_worker_pool")),
        "users": users,
        "edge": _profile_owner(edge_profile_id, bundle_id, manifest_id, model_hash),
        "cloud": {
            **_profile_owner(cloud_profile_id, bundle_id, manifest_id, model_hash),
            "BW_e2c": float(cloud_sig.get("BW_e2c", 0.0)),
        },
    }


def load_solution_bundle(
    solution_path: str | Path = DEFAULT_SOLUTION_NPZ,
    meta_path: str | Path = DEFAULT_SOLUTION_META,
) -> SolutionBundle:
    solution_path = Path(solution_path)
    meta_path = Path(meta_path)
    meta = _load_json(meta_path)
    data = np.load(solution_path, allow_pickle=True)
    required = {"X", "Y", "F_e", "F_c"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{solution_path} missing arrays: {missing}")
    paras = Paras.from_state(state_from_solution_meta(meta))
    return SolutionBundle(
        solution_path=solution_path,
        meta_path=meta_path,
        meta=meta,
        paras=paras,
        X=np.asarray(data["X"], dtype=np.float32),
        Y=np.asarray(data["Y"], dtype=np.float32),
        F_e=np.asarray(data["F_e"], dtype=np.float32),
        F_c=np.asarray(data["F_c"], dtype=np.float32),
    )


def read_solution_index(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    frame = pd.read_csv(path)
    required = {"solution_npz", "solution_meta"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Solution index {path} missing columns: {missing}")
    rows = []
    for row in frame.to_dict("records"):
        item = dict(row)
        for column in ("solution_npz", "solution_meta"):
            value = Path(str(item[column]))
            if not value.is_absolute():
                value = path.parent / value
            item[column] = value
        rows.append(item)
    return rows


def evaluate_solution_row(row: dict[str, object], *, name: str = "Ours") -> dict[str, object]:
    bundle = load_solution_bundle(row["solution_npz"], row["solution_meta"])
    result = evaluate_matrices(
        name=name,
        X=bundle.X,
        Y=bundle.Y,
        F_e=bundle.F_e,
        F_c=bundle.F_c,
        paras=bundle.paras,
        group="ours",
    )
    for column in SOLUTION_INDEX_COLUMNS:
        if column in row and column not in {"solution_npz", "solution_meta"}:
            result[column] = row[column]
    result["solution_npz"] = str(row["solution_npz"])
    result["solution_meta"] = str(row["solution_meta"])
    return result


def no_early_exit_matrix(paras: Paras, n: int) -> np.ndarray:
    return np.ones((n, paras.m), dtype=np.float32)


def split_matrix(paras: Paras, first: int, second: int, n: int) -> np.ndarray:
    row = encode_split_row(int(first), int(second), int(paras.m), dtype=np.float32)
    return np.tile(row, (int(n), 1))


def split_summary(X: np.ndarray, paras: Paras) -> str:
    pairs = [decode_split_row(row) for row in np.asarray(X)]
    unique = []
    for pair in pairs:
        if pair not in unique:
            unique.append(pair)
    final = int(paras.partition_manifest.final_boundary_id)
    return "; ".join(
        f"b1={first},b2={second},{deployment_pair_kind(first, second, final)}"
        for first, second in unique
    )


def threshold_summary(Y: np.ndarray, paras: Paras) -> str:
    if not paras.E:
        return ""
    pieces = []
    row = np.asarray(Y)[0]
    for exit_id, boundary in zip(paras.exit_ids, paras.E):
        pieces.append(f"{exit_id}={float(row[int(boundary)]):.4f}")
    return "; ".join(pieces)


def evaluate_matrices(
    *,
    name: str,
    X: np.ndarray,
    Y: np.ndarray,
    F_e: np.ndarray,
    F_c: np.ndarray,
    paras: Paras,
    group: str = "",
) -> dict[str, object]:
    P = compute_layer_exit_probs(Y, paras)
    latency_vec = compute_total_latency(X, P, F_e, F_c, paras)
    acc_vec = compute_expected_accuracy(Y, P, paras)
    utility = float(objective(X, Y, F_e, F_c, paras))
    return {
        "group": group,
        "name": name,
        "bundle_id": paras.bundle_id,
        "num_users": int(paras.n),
        "accuracy_mean": float(np.mean(acc_vec)),
        "accuracy_sum": float(np.sum(acc_vec)),
        "latency_mean_s": float(np.mean(latency_vec)),
        "latency_sum_s": float(np.sum(latency_vec)),
        "latency_mean_ms": float(np.mean(latency_vec) * 1000.0),
        "latency_sum_ms": float(np.sum(latency_vec) * 1000.0),
        "utility": utility,
        "split": split_summary(X, paras),
        "thresholds": threshold_summary(Y, paras),
    }


def write_rows(rows: list[dict[str, object]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path
