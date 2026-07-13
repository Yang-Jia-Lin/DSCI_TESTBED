"""Shared helpers for classical scheduler baselines."""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np

from Src.Phase2_Scheduler.Objective.compute_P import compute_layer_exit_probs
from Src.Phase2_Scheduler.Objective.objective import objective
from Src.Phase2_Scheduler.paras import Paras


def default_resources(paras: Paras) -> tuple[np.ndarray, np.ndarray]:
    if paras.resource_mode == "fixed_worker_pool":
        return (
            np.zeros((paras.n, 1), dtype=np.float32),
            np.zeros((paras.n, 1), dtype=np.float32),
        )
    return (
        np.full((paras.n, 1), paras.f_e_max / paras.n, dtype=np.float32),
        np.full((paras.n, 1), paras.f_c_max / paras.n, dtype=np.float32),
    )


def allocate_resources_for_xy(
    paras: Paras,
    X: np.ndarray,
    Y: np.ndarray,
    F_e: np.ndarray | None = None,
    F_c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same closed-form resource allocation as DSCI for fixed X/Y."""
    if paras.resource_mode == "fixed_worker_pool":
        return default_resources(paras)
    if F_e is None or F_c is None:
        F_e, F_c = default_resources(paras)

    from Src.Phase2_Scheduler.Optimizer.DSCI.agent import (
        allocate_resources,
        compute_iota_kappa,
    )

    exit_prob = compute_layer_exit_probs(Y, paras)
    iota, kappa = compute_iota_kappa(X, paras.C_e, paras.C_c, exit_prob)
    f_e, f_c = allocate_resources(iota, kappa, paras.f_e_max, paras.f_c_max)
    return (
        f_e.reshape(paras.n, 1).astype(np.float32),
        f_c.reshape(paras.n, 1).astype(np.float32),
    )


def evaluate_candidate(
    paras: Paras,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    allocate_resources_enabled: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    F_e, F_c = (
        allocate_resources_for_xy(paras, X, Y)
        if allocate_resources_enabled
        else default_resources(paras)
    )
    return float(objective(X, Y, F_e, F_c, paras)), F_e, F_c


def threshold_grid(step: float) -> np.ndarray:
    step = float(step)
    if step <= 0.0 or step > 1.0:
        raise ValueError("threshold_step must be in (0, 1]")
    values = np.arange(0.0, 1.0 + step / 2.0, step, dtype=np.float32)
    return np.clip(np.unique(np.round(values, 6)), 0.0, 1.0)


def threshold_rows(
    paras: Paras,
    threshold_step: float,
    *,
    extra_values: Iterable[float] | None = None,
) -> list[np.ndarray]:
    grid = threshold_grid(threshold_step)
    if extra_values is not None:
        extra = np.asarray(list(extra_values), dtype=np.float32)
        if extra.size:
            grid = np.clip(np.unique(np.concatenate([grid, extra])), 0.0, 1.0)

    rows: list[np.ndarray] = []
    for values in product(grid, repeat=len(paras.E)):
        row = np.ones(paras.m, dtype=np.float32)
        for boundary, value in zip(paras.E, values):
            row[int(boundary)] = float(value)
        rows.append(row)
    return rows
