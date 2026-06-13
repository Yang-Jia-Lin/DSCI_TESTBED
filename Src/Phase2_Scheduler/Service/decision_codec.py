"""Encode algorithm matrices (X, Y, F_e, F_c) into deploy-facing JSON."""

import numpy as np

from Src.Phase2_Scheduler.Utils.parsing_data import split_points_matrix
from Src.Phase2_Scheduler.paras import Paras

_ALLOC_TOL = 1e-6


class DecisionCodecError(ValueError):
    """Raised when decision matrices fail deploy contract validation."""


def _as_2d_resource(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    if out.size != n:
        raise DecisionCodecError(f"Resource vector length {out.size} != num_users {n}")
    return out.reshape(n, 1)


def validate_decision(
    X: np.ndarray,
    Y: np.ndarray,
    F_e: np.ndarray,
    F_c: np.ndarray,
    paras: Paras,
) -> None:
    """Validate (X, Y, F_e, F_c) before encoding for deploy."""
    n, m = int(paras.n), int(paras.m)
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape != (n, m):
        raise DecisionCodecError(f"X shape {X.shape} != ({n}, {m})")
    if Y.shape != (n, m):
        raise DecisionCodecError(f"Y shape {Y.shape} != ({n}, {m})")

    F_e_v = _as_2d_resource(F_e, n)
    F_c_v = _as_2d_resource(F_c, n)

    split_pts = split_points_matrix(X)
    for i in range(n):
        ones = np.flatnonzero(X[i] > 0.5)
        if len(ones) != 2:
            raise DecisionCodecError(
                f"User {i}: X_row must have exactly two 1s, got {len(ones)}"
            )
        s1, s2 = int(split_pts[i, 0]), int(split_pts[i, 1])
        if paras.resource_mode == "fixed_worker_pool":
            paras.partition_manifest.validate_boundary_pair(s1, s2)
        if not (0 <= s1 < s2 < m):
            raise DecisionCodecError(
                f"User {i}: require 0 <= partition_s1 < partition_s2 < {m}, "
                f"got ({s1}, {s2})"
            )
        for layer in paras.E:
            if not (0 <= layer < m):
                raise DecisionCodecError(f"Invalid early-exit layer {layer} for m={m}")
            thr = float(Y[i, layer])
            if not (0.0 <= thr <= 1.0):
                raise DecisionCodecError(
                    f"User {i}, layer {layer}: threshold {thr} not in [0, 1]"
                )

    if paras.resource_mode == "fixed_worker_pool":
        return

    edge_limit = float(paras.f_e_max)
    cloud_limit = float(paras.f_c_max)
    edge_tol = max(_ALLOC_TOL, abs(edge_limit) * _ALLOC_TOL)
    cloud_tol = max(_ALLOC_TOL, abs(cloud_limit) * _ALLOC_TOL)
    if float(F_e_v.sum()) > edge_limit + edge_tol:
        raise DecisionCodecError(
            f"sum(F_e)={float(F_e_v.sum()):.6f} exceeds f_e_max={paras.f_e_max}"
        )
    if float(F_c_v.sum()) > cloud_limit + cloud_tol:
        raise DecisionCodecError(
            f"sum(F_c)={float(F_c_v.sum()):.6f} exceeds f_c_max={paras.f_c_max}"
        )


def encode(
    X: np.ndarray,
    Y: np.ndarray,
    F_e: np.ndarray,
    F_c: np.ndarray,
    paras: Paras,
    *,
    decision_id: str | None = None,
    bundle_id: str | None = None,
    user_ids: list[int] | None = None,
    include_debug_rows: bool = True,
) -> dict:
    """
    Encode one batch decision for the deploy module (section 4.2 contract).

    Args:
        X, Y, F_e, F_c: Algorithm outputs for ``paras.n`` users.
        paras: Runtime parameters used for this round.
        decision_id: Copied from request ``round_id`` when provided.
        bundle_id: Defaults to ``paras.bundle_id``.
        user_ids: Per-user ids; defaults to ``0 .. n-1``.
        include_debug_rows: Include ``X_row`` / ``Y_row`` for debugging.
    """
    validate_decision(X, Y, F_e, F_c, paras)

    n, m = int(paras.n), int(paras.m)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    F_e_v = _as_2d_resource(F_e, n)
    F_c_v = _as_2d_resource(F_c, n)
    split_pts = split_points_matrix(X)

    bundle_id = bundle_id or paras.bundle_id
    user_ids = user_ids if user_ids is not None else list(range(n))
    if len(user_ids) != n:
        raise DecisionCodecError(f"user_ids length {len(user_ids)} != num_users {n}")

    f_e_max = float(paras.f_e_max)
    f_c_max = float(paras.f_c_max)

    users_out = []
    for i in range(n):
        s1, s2 = int(split_pts[i, 0]), int(split_pts[i, 1])
        fe = float(F_e_v[i, 0])
        fc = float(F_c_v[i, 0])
        entry = {
            "user_id": int(user_ids[i]),
            "partition_s1": s1,
            "partition_s2": s2,
            "exit_thresholds": {
                exit_id: float(Y[i, boundary])
                for exit_id, boundary in zip(paras.exit_ids, paras.E)
            },
        }
        if paras.resource_mode == "fixed_worker_pool":
            entry["partition_boundary_1"] = s1
            entry["partition_boundary_2"] = s2
            entry["device_boundaries"] = [0, s1]
            entry["edge_boundaries"] = [s1, s2]
            entry["cloud_boundaries"] = [
                s2,
                int(paras.partition_manifest.final_boundary_id),
            ]
        else:
            entry.update(
                {
                    "device_layers": [0, s1],
                    "edge_layers": [s1, s2],
                    "cloud_layers": [s2, m],
                    "edge_compute_alloc": fe,
                    "cloud_compute_alloc": fc,
                    "edge_compute_quota": fe / f_e_max if f_e_max > 0 else 0.0,
                    "cloud_compute_quota": fc / f_c_max if f_c_max > 0 else 0.0,
                }
            )
        if include_debug_rows:
            entry["X_row"] = X[i].astype(float).tolist()
            entry["Y_row"] = Y[i].astype(float).tolist()
        users_out.append(entry)

    result = {
        "decision_id": decision_id or "unknown",
        "bundle_id": bundle_id,
        "num_users": n,
        "num_layers": m,
        "exit_ids": list(paras.exit_ids),
        "layer_index_base": 0,
        "slice_semantics": "python_left_closed_right_open",
        "users": users_out,
    }
    if paras.resource_mode == "fixed_worker_pool":
        result["manifest_id"] = paras.manifest_id
        result["model_hash"] = paras.partition_manifest.model_hash
        result["resource_mode"] = "fixed_worker_pool"
    return result
