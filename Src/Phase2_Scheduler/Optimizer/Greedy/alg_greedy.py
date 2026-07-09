"""Greedy coordinate-search baseline with evaluation metrics."""

from __future__ import annotations

import time

import numpy as np

from Src.Phase2_Scheduler.Optimizer.baseline_common import default_resources
from Src.Phase2_Scheduler.Optimizer.baseline_common import evaluate_candidate
from Src.Phase2_Scheduler.Optimizer.baseline_common import threshold_rows
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Partitioning.split_actions import encode_split_row, enumerate_deployment_pairs


def optimize_greedy(
    paras: Paras,
    *,
    passes: int = 2,
    threshold_step: float = 0.1,
    tol: float = 1e-6,
    allocate_resources_enabled: bool = True,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[float], list[dict]]:
    pairs = enumerate_deployment_pairs(paras.partition_boundary_ids)
    x_rows = [encode_split_row(first, second, paras.m, dtype=np.float32) for first, second in pairs]
    y_rows = threshold_rows(paras, threshold_step)
    F_e, F_c = default_resources(paras)
    X = np.tile(x_rows[0], (paras.n, 1))
    Y = np.ones((paras.n, paras.m), dtype=np.float32)
    best_value, F_e, F_c = evaluate_candidate(
        paras,
        X,
        Y,
        allocate_resources_enabled=allocate_resources_enabled,
    )
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
            local_F_e, local_F_c = F_e.copy(), F_c.copy()
            for x_row in x_rows:
                for y_row in y_rows:
                    candidate_x, candidate_y = X.copy(), Y.copy()
                    candidate_x[user], candidate_y[user] = x_row, y_row
                    value, candidate_F_e, candidate_F_c = evaluate_candidate(
                        paras,
                        candidate_x,
                        candidate_y,
                        allocate_resources_enabled=allocate_resources_enabled,
                    )
                    evaluations += 1
                    if value > local_best + tol:
                        local_best = value
                        local_x, local_y = x_row.copy(), y_row.copy()
                        local_F_e, local_F_c = candidate_F_e.copy(), candidate_F_c.copy()
            if local_best > best_value + tol:
                X[user], Y[user] = local_x, local_y
                F_e, F_c = local_F_e, local_F_c
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
