"""Generic coordinate brute-force baseline for any number of exits."""

from __future__ import annotations

import numpy as np
import time

from Src.Phase2_Scheduler.Optimizer.baseline_common import default_resources
from Src.Phase2_Scheduler.Optimizer.baseline_common import evaluate_candidate
from Src.Phase2_Scheduler.Optimizer.baseline_common import threshold_rows
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Partitioning.split_actions import encode_split_row, enumerate_deployment_pairs


def _candidates(paras: Paras):
    rows = []
    for first, second in enumerate_deployment_pairs(paras.partition_boundary_ids):
        rows.append(encode_split_row(first, second, paras.m, dtype=np.float64))
    return rows


def optimize_BF(
    paras: Paras | None = None,
    max_iter: int = 3,
    restarts: int = 1,
    threshold_step: float = 0.25,
    tol: float = 1e-6,
    verbose: bool = False,
    return_metrics: bool = False,
    **_ignored,
):
    if paras is None:
        raise ValueError("paras is required")
    x_rows = _candidates(paras)
    y_rows = threshold_rows(paras, threshold_step)

    n = paras.n
    X = np.stack([x_rows[0]] * n)
    Y = np.ones((n, paras.m), dtype=np.float64)
    F_e, F_c = default_resources(paras)
    best, F_e, F_c = evaluate_candidate(paras, X, Y)
    history = [best]
    metrics = [
        {
            "algorithm": "BF",
            "step": 0,
            "current_obj": float(best),
            "best_obj": float(best),
            "evaluations": 1,
            "elapsed_s": 0.0,
        }
    ]
    evaluations = 1
    started = time.perf_counter()
    step = 1
    for _ in range(max_iter):
        improved = False
        for user in range(n):
            for x_row in x_rows:
                for y_row in y_rows:
                    candidate_x, candidate_y = X.copy(), Y.copy()
                    candidate_x[user], candidate_y[user] = x_row, y_row
                    value, candidate_F_e, candidate_F_c = evaluate_candidate(
                        paras,
                        candidate_x,
                        candidate_y,
                    )
                    evaluations += 1
                    if value > best + tol:
                        X, Y = candidate_x, candidate_y
                        F_e, F_c = candidate_F_e, candidate_F_c
                        best, improved = value, True
        history.append(best)
        metrics.append(
            {
                "algorithm": "BF",
                "step": int(step),
                "current_obj": float(best),
                "best_obj": float(best),
                "evaluations": int(evaluations),
                "elapsed_s": float(time.perf_counter() - started),
            }
        )
        step += 1
        if verbose:
            print(f"BF objective={best:.6f}")
        if not improved:
            break
    if return_metrics:
        return best, (X, Y, F_e, F_c), history, metrics
    return best, (X, Y, F_e, F_c), history
