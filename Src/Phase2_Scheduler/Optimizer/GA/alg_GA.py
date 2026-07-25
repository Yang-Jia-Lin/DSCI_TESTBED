"""Genetic-algorithm optimizer for the current Phase 2 decision contract."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from Src.Phase2_Scheduler.Optimizer.baseline_common import evaluate_candidate
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Partitioning.split_actions import (
    encode_split_row,
    enumerate_deployment_pairs,
)


@dataclass
class _Individual:
    X: np.ndarray
    Y: np.ndarray
    value: float = float("-inf")
    F_e: np.ndarray | None = None
    F_c: np.ndarray | None = None

    def copy(self) -> "_Individual":
        return _Individual(
            X=self.X.copy(),
            Y=self.Y.copy(),
            value=float(self.value),
            F_e=None if self.F_e is None else self.F_e.copy(),
            F_c=None if self.F_c is None else self.F_c.copy(),
        )


def _validate_hyperparameters(
    *,
    population_size: int,
    generations: int,
    mutation_rate: float,
    crossover_rate: float,
    elite_size: int,
    tournament_size: int,
    threshold_mutation_std: float,
    patience: int | None,
    rel_tolerance: float,
) -> None:
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if generations < 0:
        raise ValueError("generations must be non-negative")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in [0, 1]")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1]")
    if not 1 <= elite_size < population_size:
        raise ValueError("elite_size must be in [1, population_size)")
    if not 2 <= tournament_size <= population_size:
        raise ValueError("tournament_size must be in [2, population_size]")
    if threshold_mutation_std < 0.0:
        raise ValueError("threshold_mutation_std must be non-negative")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be positive or None")
    if rel_tolerance < 0.0:
        raise ValueError("rel_tolerance must be non-negative")


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

    legal_rows = {row.tobytes() for row in valid_x_rows.astype(np.float32)}
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


def optimize_GA(
    paras: Paras,
    population_size: int = 50,
    generations: int = 150,
    mutation_rate: float = 0.1,
    return_metrics: bool = False,
    *,
    seed: int | None = 42,
    crossover_rate: float = 0.9,
    elite_size: int = 2,
    tournament_size: int = 3,
    threshold_mutation_std: float = 0.1,
    initial_solution: (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    ) = None,
    patience: int | None = None,
    rel_tolerance: float = 1e-6,
    verbose: bool = False,
):
    """Optimize the same ``(X, Y, F_e, F_c)`` decision used by DSCI/PPO.

    ``X`` and ``Y`` are optimized independently for every user. Every split
    chromosome is a manifest-valid deployment pair, while edge/cloud resources
    are recomputed with the same closed-form allocator used by DSCI.

    The first three optional arguments and ``return_metrics`` retain the legacy
    GA call contract. New keyword-only arguments provide reproducibility,
    warm-starting, elitism, and optional early stopping.
    """

    _validate_hyperparameters(
        population_size=int(population_size),
        generations=int(generations),
        mutation_rate=float(mutation_rate),
        crossover_rate=float(crossover_rate),
        elite_size=int(elite_size),
        tournament_size=int(tournament_size),
        threshold_mutation_std=float(threshold_mutation_std),
        patience=patience,
        rel_tolerance=float(rel_tolerance),
    )

    n, m = int(paras.n), int(paras.m)
    pairs = enumerate_deployment_pairs(paras.partition_boundary_ids)
    if not pairs:
        raise ValueError("partition manifest does not define any deployment pairs")
    valid_x_rows = np.stack(
        [
            encode_split_row(first, second, m, dtype=np.float32)
            for first, second in pairs
        ]
    )
    exit_boundaries = np.asarray([int(value) for value in paras.E], dtype=int)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    evaluations = 0

    def evaluate(individual: _Individual) -> None:
        nonlocal evaluations
        value, F_e, F_c = evaluate_candidate(paras, individual.X, individual.Y)
        individual.value = (
            float(value) if np.isfinite(value) else float("-inf")
        )
        individual.F_e = np.asarray(F_e, dtype=np.float32)
        individual.F_c = np.asarray(F_c, dtype=np.float32)
        evaluations += 1

    def random_individual() -> _Individual:
        row_indices = rng.integers(0, len(valid_x_rows), size=n)
        X = valid_x_rows[row_indices].copy()
        Y = np.ones((n, m), dtype=np.float32)
        if exit_boundaries.size:
            Y[:, exit_boundaries] = rng.random((n, len(exit_boundaries)))
        return _Individual(X=X, Y=Y)

    population: list[_Individual] = []
    if initial_solution is not None:
        initial_X, initial_Y = _normalise_initial_solution(
            initial_solution, paras, valid_x_rows
        )
        population.append(_Individual(X=initial_X, Y=initial_Y))
    else:
        # A deterministic feasible anchor makes tiny GA runs useful and keeps
        # the all-device/no-early-exit corner represented in every run.
        population.append(
            _Individual(
                X=np.tile(valid_x_rows[0], (n, 1)),
                Y=np.ones((n, m), dtype=np.float32),
            )
        )
    while len(population) < int(population_size):
        population.append(random_individual())
    for individual in population:
        evaluate(individual)

    population.sort(key=lambda item: item.value, reverse=True)
    best = population[0].copy()
    population_values = np.asarray(
        [individual.value for individual in population], dtype=np.float64
    )
    history = [float(best.value)]
    metrics: list[dict[str, Any]] = [
        {
            "algorithm": "GA",
            "step": 0,
            "current_obj": float(population[0].value),
            "best_obj": float(best.value),
            "batch_mean_obj": float(np.mean(population_values)),
            "batch_std_obj": float(np.std(population_values, ddof=0)),
            "batch_size": int(len(population)),
            "evaluations": int(evaluations),
            "elapsed_s": float(time.perf_counter() - started),
        }
    ]
    if verbose:
        print(f"GA generation 0: best={best.value:.8f}")

    stale_generations = 0

    def select_parent() -> _Individual:
        indices = rng.choice(
            len(population), size=int(tournament_size), replace=False
        )
        candidates = (population[int(index)] for index in indices)
        return max(candidates, key=lambda item: item.value)

    def crossover(
        parent_a: _Individual, parent_b: _Individual
    ) -> tuple[_Individual, _Individual]:
        if rng.random() >= float(crossover_rate):
            return parent_a.copy(), parent_b.copy()

        x_mask = rng.random(n) < 0.5
        X_a = np.where(x_mask[:, None], parent_a.X, parent_b.X).astype(np.float32)
        X_b = np.where(x_mask[:, None], parent_b.X, parent_a.X).astype(np.float32)
        Y_a, Y_b = parent_a.Y.copy(), parent_b.Y.copy()
        if exit_boundaries.size:
            y_mask = rng.random((n, len(exit_boundaries))) < 0.5
            a_values = parent_a.Y[:, exit_boundaries]
            b_values = parent_b.Y[:, exit_boundaries]
            Y_a[:, exit_boundaries] = np.where(y_mask, a_values, b_values)
            Y_b[:, exit_boundaries] = np.where(y_mask, b_values, a_values)
        return _Individual(X_a, Y_a), _Individual(X_b, Y_b)

    def mutate(individual: _Individual) -> None:
        for user in range(n):
            if rng.random() < float(mutation_rate):
                individual.X[user] = valid_x_rows[int(rng.integers(len(valid_x_rows)))]
        if exit_boundaries.size:
            mutation_mask = (
                rng.random((n, len(exit_boundaries))) < float(mutation_rate)
            )
            noise = rng.normal(
                0.0,
                float(threshold_mutation_std),
                size=(n, len(exit_boundaries)),
            )
            values = individual.Y[:, exit_boundaries]
            individual.Y[:, exit_boundaries] = np.clip(
                values + mutation_mask * noise, 0.0, 1.0
            )
        individual.value = float("-inf")
        individual.F_e = None
        individual.F_c = None

    for generation in range(1, int(generations) + 1):
        population.sort(key=lambda item: item.value, reverse=True)
        next_population = [
            population[index].copy() for index in range(int(elite_size))
        ]
        while len(next_population) < int(population_size):
            child_a, child_b = crossover(select_parent(), select_parent())
            mutate(child_a)
            evaluate(child_a)
            next_population.append(child_a)
            if len(next_population) < int(population_size):
                mutate(child_b)
                evaluate(child_b)
                next_population.append(child_b)

        population = next_population
        generation_best = max(population, key=lambda item: item.value)
        population_values = np.asarray(
            [individual.value for individual in population], dtype=np.float64
        )
        previous_best = float(best.value)
        if generation_best.value > best.value:
            best = generation_best.copy()

        scale = max(1.0, abs(previous_best))
        if best.value > previous_best + float(rel_tolerance) * scale:
            stale_generations = 0
        else:
            stale_generations += 1

        history.append(float(best.value))
        metrics.append(
            {
                "algorithm": "GA",
                "step": int(generation),
                "current_obj": float(generation_best.value),
                "best_obj": float(best.value),
                "batch_mean_obj": float(np.mean(population_values)),
                "batch_std_obj": float(np.std(population_values, ddof=0)),
                "batch_size": int(len(population)),
                "evaluations": int(evaluations),
                "elapsed_s": float(time.perf_counter() - started),
            }
        )
        if verbose:
            print(f"GA generation {generation}: best={best.value:.8f}")
        if patience is not None and stale_generations >= int(patience):
            break

    if best.F_e is None or best.F_c is None:
        raise RuntimeError("GA finished without a fully evaluated solution")
    best_solution = (
        best.X.astype(np.float32, copy=True),
        best.Y.astype(np.float32, copy=True),
        best.F_e.astype(np.float32, copy=True),
        best.F_c.astype(np.float32, copy=True),
    )
    result = (float(best.value), best_solution, history)
    if return_metrics:
        return (*result, metrics)
    return result
