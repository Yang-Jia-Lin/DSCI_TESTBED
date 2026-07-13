"""Random-search baseline with metrics compatible with evaluation plots."""

from __future__ import annotations

import time

import numpy as np

from Src.Phase2_Scheduler.Optimizer.baseline_common import evaluate_candidate
from Src.Phase2_Scheduler.Optimizer.baseline_common import threshold_grid
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Partitioning.split_actions import encode_split_row, enumerate_deployment_pairs


def optimize_random(
    paras: Paras,
    *,
    iterations: int = 200,
    seed: int = 42,
    threshold_step: float = 0.05,
    allocate_resources_enabled: bool = True,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list[float], list[dict]]:
    rng = np.random.default_rng(seed)
    pairs = enumerate_deployment_pairs(paras.partition_boundary_ids)
    grid = threshold_grid(threshold_step)
    X0 = np.tile(
        encode_split_row(pairs[0][0], pairs[0][1], paras.m, dtype=np.float32),
        (paras.n, 1),
    )
    Y0 = np.ones((paras.n, paras.m), dtype=np.float32)
    best_value, F_e0, F_c0 = evaluate_candidate(
        paras,
        X0,
        Y0,
        allocate_resources_enabled=allocate_resources_enabled,
    )
    best_sol = (X0.copy(), Y0.copy(), F_e0.copy(), F_c0.copy())
    history: list[float] = []
    metrics: list[dict] = []
    started = time.perf_counter()
    for step in range(int(iterations)):
        if step == 0:
            X, Y, F_e, F_c, value = X0, Y0, F_e0, F_c0, best_value
        else:
            X = np.zeros((paras.n, paras.m), dtype=np.float32)
            Y = np.ones((paras.n, paras.m), dtype=np.float32)
            for user in range(paras.n):
                first, second = pairs[int(rng.integers(0, len(pairs)))]
                X[user] = encode_split_row(first, second, paras.m, dtype=np.float32)
                for boundary in paras.E:
                    Y[user, int(boundary)] = float(rng.choice(grid))
            value, F_e, F_c = evaluate_candidate(
                paras,
                X,
                Y,
                allocate_resources_enabled=allocate_resources_enabled,
            )
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
    return best_value, best_sol, history, metrics
