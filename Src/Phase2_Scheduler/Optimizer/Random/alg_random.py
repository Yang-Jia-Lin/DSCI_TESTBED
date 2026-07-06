"""Random-search baseline with metrics compatible with evaluation plots."""

from __future__ import annotations

import time

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


def optimize_random(
    paras: Paras,
    *,
    iterations: int = 200,
    seed: int = 42,
    threshold_step: float = 0.05,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[float], list[dict]]:
    rng = np.random.default_rng(seed)
    pairs = enumerate_deployment_pairs(paras.partition_boundary_ids)
    grid = np.arange(0.0, 1.0 + threshold_step / 2.0, threshold_step)
    F_e, F_c = _resources(paras)
    best_value = float("-inf")
    best_sol = None
    history: list[float] = []
    metrics: list[dict] = []
    started = time.perf_counter()
    for step in range(int(iterations)):
        X = np.zeros((paras.n, paras.m), dtype=np.float32)
        Y = np.ones((paras.n, paras.m), dtype=np.float32)
        for user in range(paras.n):
            first, second = pairs[int(rng.integers(0, len(pairs)))]
            X[user] = encode_split_row(first, second, paras.m, dtype=np.float32)
            for boundary in paras.E:
                Y[user, int(boundary)] = float(rng.choice(grid))
        value = float(objective(X, Y, F_e, F_c, paras))
        if value > best_value:
            best_value = value
            best_sol = (X.copy(), Y.copy(), F_e.copy(), F_c.copy())
        history.append(best_value)
        metrics.append(
            {
                "algorithm": "Random",
                "step": step,
                "current_obj": value,
                "best_obj": best_value,
                "evaluations": step + 1,
                "elapsed_s": time.perf_counter() - started,
            }
        )
    if best_sol is None:
        raise RuntimeError("Random search produced no solution")
    return best_value, best_sol, history, metrics

