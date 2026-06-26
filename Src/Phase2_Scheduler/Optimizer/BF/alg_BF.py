"""Generic coordinate brute-force baseline for any number of exits."""

from itertools import combinations, product

import numpy as np

from Src.Phase2_Scheduler.Objective.objective import objective
from Src.Phase2_Scheduler.paras import Paras


def _candidates(paras: Paras):
    rows = []
    for first, second in combinations(paras.partition_boundary_ids, 2):
        row = np.zeros(paras.m, dtype=np.float64)
        row[first] = row[second] = 1.0
        rows.append(row)
    return rows


def optimize_BF(
    paras: Paras | None = None,
    max_iter: int = 3,
    restarts: int = 1,
    threshold_step: float = 0.25,
    tol: float = 1e-6,
    verbose: bool = False,
    **_ignored,
):
    if paras is None:
        raise ValueError("paras is required")
    x_rows = _candidates(paras)
    grid = np.arange(0.0, 1.0 + threshold_step / 2.0, threshold_step)
    threshold_rows = []
    for values in product(grid, repeat=len(paras.E)):
        row = np.ones(paras.m, dtype=np.float64)
        for boundary, value in zip(paras.E, values):
            row[boundary] = value
        threshold_rows.append(row)

    n = paras.n
    X = np.stack([x_rows[len(x_rows) // 2]] * n)
    Y = np.ones((n, paras.m), dtype=np.float64)
    F_e = np.zeros((n, 1)) if paras.resource_mode == "fixed_worker_pool" else np.full((n, 1), paras.f_e_max / n)
    F_c = np.zeros((n, 1)) if paras.resource_mode == "fixed_worker_pool" else np.full((n, 1), paras.f_c_max / n)
    best = float(objective(X, Y, F_e, F_c, paras))
    history = [best]
    for _ in range(max_iter):
        improved = False
        for user in range(n):
            for x_row in x_rows:
                for y_row in threshold_rows:
                    candidate_x, candidate_y = X.copy(), Y.copy()
                    candidate_x[user], candidate_y[user] = x_row, y_row
                    value = float(objective(candidate_x, candidate_y, F_e, F_c, paras))
                    if value > best + tol:
                        X, Y, best, improved = candidate_x, candidate_y, value, True
        history.append(best)
        if verbose:
            print(f"BF objective={best:.6f}")
        if not improved:
            break
    return best, (X, Y, F_e, F_c), history
