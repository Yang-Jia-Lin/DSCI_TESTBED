"""Greedy coordinate-search baseline with evaluation metrics."""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from Src.Phase2_Scheduler.Objective.objective import objective
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Partitioning.split_actions import encode_split_row, enumerate_deployment_pairs


def _resources(paras: Paras):
    if paras.resource_mode == "fixed_worker_pool":
        return np.zeros((paras.n, 1), dtype=np.float32), np.zeros((paras.n, 1), dtype=np.float32)
    return (
        np.full((paras.n, 1), paras.f_e_max / paras.n, dtype=np.float32),
        np.full((paras.n, 1), paras.f_c_max / paras.n, dtype=np.float32),
    )


def _threshold_rows(paras: Paras, threshold_step: float) -> list[np.ndarray]:
    grid = np.arange(0.0, 1.0 + threshold_step / 2.0, threshold_step)
    rows = []
    for values in product(grid, repeat=len(paras.E)):
        row = np.ones(paras.m, dtype=np.float32)
        for boundary, value in zip(paras.E, values):
            row[int(boundary)] = float(value)
        rows.append(row)
    return rows


def optimize_greedy(
    paras: Paras,
    *,
    passes: int = 2,
    threshold_step: float = 0.1,
    tol: float = 1e-6,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[float], list[dict]]:
    pairs = enumerate_deployment_pairs(paras.partition_boundary_ids)
    x_rows = [encode_split_row(first, second, paras.m, dtype=np.float32) for first, second in pairs]
    y_rows = _threshold_rows(paras, threshold_step)
    F_e, F_c = _resources(paras)
    X = np.tile(x_rows[0], (paras.n, 1))
    Y = np.ones((paras.n, paras.m), dtype=np.float32)
    best_value = float(objective(X, Y, F_e, F_c, paras))
    history = [best_value]
    metrics = [
        {
            "algorithm": "Greedy",
            "step": 0,
            "current_obj": best_value,
            "best_obj": best_value,
            "evaluations": 1,
            "elapsed_s": 0.0,
        }
    ]
    evaluations = 1
    step = 1
    started = time.perf_counter()
    for _ in range(int(passes)):
        improved = False
        for user in range(paras.n):
            local_best = best_value
            local_x, local_y = X[user].copy(), Y[user].copy()
            for x_row in x_rows:
                for y_row in y_rows:
                    candidate_x, candidate_y = X.copy(), Y.copy()
                    candidate_x[user], candidate_y[user] = x_row, y_row
                    value = float(objective(candidate_x, candidate_y, F_e, F_c, paras))
                    evaluations += 1
                    if value > local_best + tol:
                        local_best = value
                        local_x, local_y = x_row.copy(), y_row.copy()
            if local_best > best_value + tol:
                X[user], Y[user] = local_x, local_y
                best_value = local_best
                improved = True
            history.append(best_value)
            metrics.append(
                {
                    "algorithm": "Greedy",
                    "step": step,
                    "current_obj": best_value,
                    "best_obj": best_value,
                    "evaluations": evaluations,
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            step += 1
        if not improved:
            break
    return best_value, (X, Y, F_e, F_c), history, metrics

