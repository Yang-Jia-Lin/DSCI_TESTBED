"""Central objective evaluation, accounting, and normalized trace schema."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from Src.Phase2_Scheduler.Objective.compute_accuracy import compute_expected_accuracy
from Src.Phase2_Scheduler.Objective.compute_latency import compute_latency_breakdown
from Src.Phase2_Scheduler.Objective.compute_P import compute_layer_exit_probs


class EvaluationBudgetExceeded(RuntimeError):
    """Raised before an objective call would exceed the configured budget."""


@dataclass(frozen=True)
class ObjectiveBreakdown:
    expected_accuracy: np.ndarray
    expected_latency: np.ndarray
    expected_utility: np.ndarray
    latency_components: dict[str, np.ndarray]

    @property
    def utility_sum(self) -> float:
        return float(np.sum(self.expected_utility))

    @property
    def utility_mean(self) -> float:
        return float(np.mean(self.expected_utility))

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_accuracy": self.expected_accuracy.astype(float).tolist(),
            "expected_latency": self.expected_latency.astype(float).tolist(),
            "expected_utility": self.expected_utility.astype(float).tolist(),
            "utility_sum": self.utility_sum,
            "utility_mean": self.utility_mean,
            "latency_components": {
                key: np.asarray(value, dtype=float).tolist()
                for key, value in self.latency_components.items()
            },
        }


class ObjectiveEvaluator:
    """Single source of truth for objective values and evaluation counts."""

    def __init__(self, paras, *, evaluation_budget: int | None = None):
        self.paras = paras
        self.evaluation_budget = (
            None if evaluation_budget is None else int(evaluation_budget)
        )
        if self.evaluation_budget is not None and self.evaluation_budget <= 0:
            raise ValueError("evaluation_budget must be positive")
        self.evaluations = 0
        self.started_at = time.perf_counter()
        self.best_value = float("-inf")
        self.best_solution: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.last_breakdown: ObjectiveBreakdown | None = None
        self.trace: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int | None:
        if self.evaluation_budget is None:
            return None
        return max(0, self.evaluation_budget - self.evaluations)

    def breakdown(self, X, Y, F_e, F_c) -> ObjectiveBreakdown:
        P = compute_layer_exit_probs(np.asarray(Y), self.paras)
        latency_components = compute_latency_breakdown(
            np.asarray(X), P, self.paras, F_e, F_c
        )
        latency = np.asarray(latency_components["total"], dtype=np.float64)
        accuracy = compute_expected_accuracy(np.asarray(Y), P, self.paras)
        utility = float(self.paras.alpha) * accuracy - float(self.paras.beta) * latency
        return ObjectiveBreakdown(accuracy, latency, utility, latency_components)

    def evaluate(self, X, Y, F_e, F_c) -> float:
        if self.evaluation_budget is not None and self.evaluations >= self.evaluation_budget:
            raise EvaluationBudgetExceeded(
                f"objective evaluation budget {self.evaluation_budget} exhausted"
            )
        breakdown = self.breakdown(X, Y, F_e, F_c)
        self.evaluations += 1
        self.last_breakdown = breakdown
        value = breakdown.utility_sum
        if value > self.best_value:
            self.best_value = value
            self.best_solution = tuple(
                np.asarray(value_, dtype=np.float32).copy()
                for value_ in (X, Y, F_e, F_c)
            )
        return value

    def record(self, optimizer: str, step: int, current_utility: float, best_utility: float) -> dict[str, Any]:
        breakdown = self.last_breakdown
        record = {
            "optimizer": str(optimizer),
            "step": int(step),
            "objective_evaluations": int(self.evaluations),
            "current_utility": float(current_utility),
            "best_utility": float(best_utility),
            "expected_accuracy": (
                float(np.mean(breakdown.expected_accuracy)) if breakdown else None
            ),
            "expected_latency": (
                float(np.mean(breakdown.expected_latency)) if breakdown else None
            ),
            "elapsed_s": float(time.perf_counter() - self.started_at),
        }
        self.trace.append(record)
        return record
