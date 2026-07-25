"""Random-search optimizer for the current Phase 2 decision contract."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from Src.Phase2_Scheduler.Optimizer.baseline_common import evaluate_candidate
from Src.Phase2_Scheduler.Optimizer.baseline_common import threshold_grid
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Partitioning.split_actions import (
    encode_split_row,
    enumerate_deployment_pairs,
)


def _normalise_initial_solution(
    initial_solution: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    paras: Paras,
    valid_x_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X, Y, _F_e, _F_c = initial_solution
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    expected_shape = (int(paras.n), int(paras.m))
    if X.shape != expected_shape or Y.shape != expected_shape:
        raise ValueError(
            f"initial_solution X/Y must both have shape {expected_shape}; "
            f"got X={X.shape}, Y={Y.shape}"
        )

    legal_rows = {row.tobytes() for row in valid_x_rows}
    for user, row in enumerate(X):
        binary_row = (row > 0.5).astype(np.float32)
        if binary_row.tobytes() not in legal_rows:
            raise ValueError(
                f"initial_solution contains an invalid X row for user {user}"
            )
        X[user] = binary_row

    normalised_Y = np.ones(expected_shape, dtype=np.float32)
    for boundary in paras.E:
        normalised_Y[:, int(boundary)] = np.clip(
            Y[:, int(boundary)], 0.0, 1.0
        )
    return X.copy(), normalised_Y


def optimize_random(
    paras: Paras,
    *,
    iterations: int = 200,
    seed: int | None = 42,
    threshold_step: float | None = 0.05,
    allocate_resources_enabled: bool = True,
    initial_solution: (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    ) = None,
    verbose: bool = False,
) -> tuple[
    float,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    list[float],
    list[dict[str, Any]],
]:
    """Randomly optimize the same ``(X, Y, F_e, F_c)`` used by DSCI and GA.

    Each candidate independently samples a manifest-valid split and all
    early-exit thresholds for every user. Resources are then recomputed by the
    same closed-form allocator used by DSCI. ``threshold_step=None`` samples
    continuous thresholds; a positive step samples from a reproducible grid.
    """

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    pairs = enumerate_deployment_pairs(paras.partition_boundary_ids)
    if not pairs:
        raise ValueError("partition manifest does not define any deployment pairs")
    valid_x_rows = np.stack(
        [
            encode_split_row(first, second, paras.m, dtype=np.float32)
            for first, second in pairs
        ]
    )
    grid = None if threshold_step is None else threshold_grid(threshold_step)
    exit_boundaries = np.asarray([int(value) for value in paras.E], dtype=int)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    def sample_candidate() -> tuple[np.ndarray, np.ndarray]:
        row_indices = rng.integers(0, len(valid_x_rows), size=int(paras.n))
        X = valid_x_rows[row_indices].copy()
        Y = np.ones((paras.n, paras.m), dtype=np.float32)
        if exit_boundaries.size:
            shape = (int(paras.n), len(exit_boundaries))
            if grid is None:
                values = rng.random(shape)
            else:
                values = rng.choice(grid, size=shape)
            Y[:, exit_boundaries] = values
        return X, Y

    if initial_solution is None:
        # Keep the same deterministic all-device/no-early-exit anchor as GA.
        X = np.tile(valid_x_rows[0], (paras.n, 1))
        Y = np.ones((paras.n, paras.m), dtype=np.float32)
    else:
        X, Y = _normalise_initial_solution(
            initial_solution, paras, valid_x_rows
        )

    best_value, F_e, F_c = evaluate_candidate(
        paras,
        X,
        Y,
        allocate_resources_enabled=allocate_resources_enabled,
    )
    best_value = (
        float(best_value) if np.isfinite(best_value) else float("-inf")
    )
    best_sol = (
        X.copy(),
        Y.copy(),
        np.asarray(F_e, dtype=np.float32).copy(),
        np.asarray(F_c, dtype=np.float32).copy(),
    )
    history: list[float] = []
    metrics: list[dict[str, Any]] = []

    for step in range(iterations):
        if step == 0:
            current_value = best_value
        else:
            X, Y = sample_candidate()
            value, F_e, F_c = evaluate_candidate(
                paras,
                X,
                Y,
                allocate_resources_enabled=allocate_resources_enabled,
            )
            current_value = (
                float(value) if np.isfinite(value) else float("-inf")
            )
            if current_value > best_value:
                best_value = current_value
                best_sol = (
                    X.copy(),
                    Y.copy(),
                    np.asarray(F_e, dtype=np.float32).copy(),
                    np.asarray(F_c, dtype=np.float32).copy(),
                )

        history.append(float(best_value))
        metrics.append(
            {
                "algorithm": "Random",
                "step": int(step),
                "current_obj": float(current_value),
                "best_obj": float(best_value),
                "evaluations": int(step + 1),
                "elapsed_s": float(time.perf_counter() - started),
            }
        )
        if verbose:
            print(
                f"Random evaluation {step + 1}: "
                f"current={current_value:.8f}, best={best_value:.8f}"
            )

    return float(best_value), best_sol, history, metrics
