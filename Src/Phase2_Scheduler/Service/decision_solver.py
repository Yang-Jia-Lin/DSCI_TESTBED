"""Synchronous, constrained decision solving for paper evaluation.

Unlike :mod:`algo_service`, this module never returns a stale/default cached
decision and never starts background work.  One call solves exactly the state
that was supplied.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import random
import time
from typing import Any, Literal

import numpy as np

from Src.Phase2_Scheduler.Objective.evaluator import (
    EvaluationBudgetExceeded,
    ObjectiveEvaluator,
)
from Src.Phase2_Scheduler.Optimizer.baseline_common import allocate_resources_for_xy
from Src.Phase2_Scheduler.Optimizer.DSCI.run_DSCI import _build_ppo_params
from Src.Phase2_Scheduler.Optimizer.DSCI.agent import PPOAgent
from Src.Phase2_Scheduler.Service.decision_codec import encode
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Partitioning.split_actions import (
    decode_split_row,
    encode_split_row,
    enumerate_deployment_pairs,
)


Coordination = Literal["joint", "independent"]
VariableRule = Literal["optimize", "fixed"]
ExitRule = Literal["optimize", "disabled", "fixed"]
Stages = Literal["joint", "split_then_exit"]
OptimizerName = Literal["ppo", "random", "ga", "static"]


@dataclass(frozen=True)
class DecisionSpec:
    coordination: Coordination = "joint"
    split_rule: VariableRule = "optimize"
    exit_rule: ExitRule = "optimize"
    allowed_split_pairs: tuple[tuple[int, int], ...] | None = None
    fixed_split: tuple[int, int] | tuple[tuple[int, int], ...] | None = None
    fixed_threshold: float | dict[str, float] | tuple[dict[str, float], ...] | None = None
    stages: Stages = "joint"
    optimizer: OptimizerName = "ppo"
    seed: int = 42
    evaluation_budget: int | None = 500
    threshold_step: float = 0.05
    optimizer_options: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.coordination not in {"joint", "independent"}:
            raise ValueError("coordination must be joint or independent")
        if self.split_rule not in {"optimize", "fixed"}:
            raise ValueError("split_rule must be optimize or fixed")
        if self.exit_rule not in {"optimize", "disabled", "fixed"}:
            raise ValueError("exit_rule must be optimize, disabled, or fixed")
        if self.stages not in {"joint", "split_then_exit"}:
            raise ValueError("stages must be joint or split_then_exit")
        if self.optimizer not in {"ppo", "random", "ga", "static"}:
            raise ValueError("unsupported optimizer")
        if self.optimizer == "static" and self.split_rule != "fixed":
            raise ValueError("static optimizer requires a fixed split")
        if self.optimizer == "static" and self.exit_rule == "optimize":
            raise ValueError("static optimizer requires fixed or disabled exits")
        if self.split_rule == "fixed" and self.fixed_split is None:
            raise ValueError("fixed split_rule requires fixed_split")
        if self.exit_rule == "fixed" and self.fixed_threshold is None:
            raise ValueError("fixed exit_rule requires fixed_threshold")
        if self.evaluation_budget is not None and int(self.evaluation_budget) <= 0:
            raise ValueError("evaluation_budget must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def split_pairs_for(self, n: int) -> list[tuple[int, int]]:
        value = self.fixed_split
        if value is None:
            raise ValueError("fixed_split is not configured")
        if len(value) == 2 and all(isinstance(item, (int, np.integer)) for item in value):
            pair = (int(value[0]), int(value[1]))
            return [pair] * int(n)
        pairs = [(int(item[0]), int(item[1])) for item in value]
        if len(pairs) != int(n):
            raise ValueError("per-user fixed_split count must match number of users")
        return pairs

    def threshold_rows_for(self, paras: Paras) -> np.ndarray:
        rows = np.ones((paras.n, paras.m), dtype=np.float32)
        if self.exit_rule == "disabled":
            return rows
        value = self.fixed_threshold
        if self.exit_rule != "fixed":
            return rows
        values = list(value) if isinstance(value, tuple) else [value] * paras.n
        if len(values) != paras.n:
            raise ValueError("per-user fixed_threshold count must match users")
        for user, item in enumerate(values):
            if isinstance(item, dict):
                for exit_id, boundary in zip(paras.exit_ids, paras.E):
                    if exit_id not in item:
                        raise ValueError(f"missing threshold for exit {exit_id!r}")
                    rows[user, int(boundary)] = float(item[exit_id])
            else:
                for boundary in paras.E:
                    rows[user, int(boundary)] = float(item)
        if np.any((rows < 0.0) | (rows > 1.0)):
            raise ValueError("exit thresholds must be in [0, 1]")
        return rows

    def for_user(self, user_index: int, total_users: int) -> "DecisionSpec":
        fixed_split = self.fixed_split
        if fixed_split is not None and not (
            len(fixed_split) == 2
            and all(isinstance(item, (int, np.integer)) for item in fixed_split)
        ):
            fixed_split = tuple(fixed_split)[user_index]
        fixed_threshold = self.fixed_threshold
        if isinstance(fixed_threshold, tuple):
            if len(fixed_threshold) != total_users:
                raise ValueError("per-user fixed_threshold count must match users")
            fixed_threshold = fixed_threshold[user_index]
        return replace(
            self,
            coordination="joint",
            fixed_split=fixed_split,
            fixed_threshold=fixed_threshold,
            seed=int(self.seed) + int(user_index),
        )


@dataclass
class SolveResult:
    X: np.ndarray
    Y: np.ndarray
    F_e: np.ndarray
    F_c: np.ndarray
    decision: dict[str, Any]
    expected: dict[str, Any]
    optimizer_trace: list[dict[str, Any]]
    objective_evaluations: int
    setup_s: float
    solve_s: float
    total_s: float
    spec: DecisionSpec
    state_signature: dict[str, Any]

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        result = {
            "decision": self.decision,
            "expected": self.expected,
            "optimizer_trace": self.optimizer_trace,
            "objective_evaluations": self.objective_evaluations,
            "timing": {
                "setup_s": self.setup_s,
                "solve_s": self.solve_s,
                "total_s": self.total_s,
            },
            "spec": self.spec.to_dict(),
            "state_signature": self.state_signature,
        }
        if include_arrays:
            result.update(
                {
                    "X": self.X.astype(float).tolist(),
                    "Y": self.Y.astype(float).tolist(),
                    "F_e": self.F_e.astype(float).tolist(),
                    "F_c": self.F_c.astype(float).tolist(),
                }
            )
        return result


def _allowed_pairs(paras: Paras, spec: DecisionSpec) -> list[tuple[int, int]]:
    legal = set(enumerate_deployment_pairs(paras.partition_boundary_ids))
    requested = list(spec.allowed_split_pairs or legal)
    pairs = [(int(first), int(second)) for first, second in requested]
    invalid = [pair for pair in pairs if pair not in legal]
    if invalid:
        raise ValueError(f"allowed_split_pairs contains illegal pairs: {invalid}")
    if spec.split_rule == "fixed":
        fixed = spec.split_pairs_for(paras.n)
        invalid = [pair for pair in fixed if pair not in legal]
        if invalid:
            raise ValueError(f"fixed_split contains illegal pairs: {invalid}")
        return sorted(set(fixed))
    if not pairs:
        raise ValueError("allowed_split_pairs must not be empty")
    return pairs


def _fixed_X(paras: Paras, spec: DecisionSpec) -> np.ndarray:
    pairs = spec.split_pairs_for(paras.n)
    return np.stack(
        [encode_split_row(*pair, paras.m, dtype=np.float32) for pair in pairs]
    )


def _candidate(
    paras: Paras,
    spec: DecisionSpec,
    rng: np.random.Generator,
    pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if spec.split_rule == "fixed":
        X = _fixed_X(paras, spec)
    else:
        X = np.stack(
            [
                encode_split_row(*pairs[int(rng.integers(len(pairs)))], paras.m)
                for _ in range(paras.n)
            ]
        ).astype(np.float32)
    if spec.exit_rule in {"disabled", "fixed"}:
        Y = spec.threshold_rows_for(paras)
    else:
        Y = np.ones((paras.n, paras.m), dtype=np.float32)
        grid = np.arange(0.0, 1.0 + spec.threshold_step / 2.0, spec.threshold_step)
        grid = np.clip(grid, 0.0, 1.0)
        for user in range(paras.n):
            for boundary in paras.E:
                Y[user, int(boundary)] = float(rng.choice(grid))
    F_e, F_c = allocate_resources_for_xy(paras, X, Y)
    return X, Y, F_e, F_c


def _evaluate(evaluator: ObjectiveEvaluator, solution):
    return evaluator.evaluate(*solution)


def _run_random(paras: Paras, spec: DecisionSpec, evaluator: ObjectiveEvaluator):
    rng = np.random.default_rng(spec.seed)
    pairs = _allowed_pairs(paras, spec)
    step = 0
    while evaluator.remaining is None or evaluator.remaining > 0:
        sol = _candidate(paras, spec, rng, pairs)
        value = _evaluate(evaluator, sol)
        evaluator.record("random", step, value, evaluator.best_value)
        step += 1
        if evaluator.evaluation_budget is None and step >= int(
            spec.optimizer_options.get("iterations", 200)
        ):
            break


def _enforce_candidate(paras: Paras, spec: DecisionSpec, X: np.ndarray, Y: np.ndarray):
    pairs = _allowed_pairs(paras, spec)
    if spec.split_rule == "fixed":
        X = _fixed_X(paras, spec)
    else:
        legal = set(pairs)
        for user in range(paras.n):
            pair = decode_split_row(X[user])
            if pair not in legal:
                X[user] = encode_split_row(*pairs[0], paras.m)
    if spec.exit_rule in {"disabled", "fixed"}:
        Y = spec.threshold_rows_for(paras)
    else:
        for boundary in range(paras.m):
            if boundary not in paras.E:
                Y[:, boundary] = 1.0
        Y[:, paras.E] = np.clip(Y[:, paras.E], 0.0, 1.0)
    return X.astype(np.float32), Y.astype(np.float32)


def _run_ga(paras: Paras, spec: DecisionSpec, evaluator: ObjectiveEvaluator):
    rng = np.random.default_rng(spec.seed)
    random.seed(spec.seed)
    pairs = _allowed_pairs(paras, spec)
    population_size = int(spec.optimizer_options.get("population_size", 20))
    population_size = max(2, population_size)
    mutation_rate = float(spec.optimizer_options.get("mutation_rate", 0.1))
    population = [_candidate(paras, spec, rng, pairs) for _ in range(population_size)]
    generation = 0
    while evaluator.remaining is None or evaluator.remaining > 0:
        scored = []
        for solution in population:
            if evaluator.remaining == 0:
                break
            value = _evaluate(evaluator, solution)
            scored.append((value, solution))
        if not scored:
            break
        scored.sort(key=lambda item: item[0], reverse=True)
        evaluator.last_breakdown = evaluator.breakdown(*scored[0][1])
        evaluator.record("ga", generation, scored[0][0], evaluator.best_value)
        generation += 1
        if evaluator.evaluation_budget is None and generation >= int(
            spec.optimizer_options.get("generations", 50)
        ):
            break
        elite = scored[0][1]
        parents = [item[1] for item in scored[: max(2, len(scored) // 2)]]
        next_population = [tuple(value.copy() for value in elite)]
        while len(next_population) < population_size:
            left = parents[int(rng.integers(len(parents)))]
            right = parents[int(rng.integers(len(parents)))]
            mask_x = rng.random((paras.n, 1)) < 0.5
            X = np.where(mask_x, left[0], right[0]).copy()
            mask_y = rng.random((paras.n, paras.m)) < 0.5
            Y = np.where(mask_y, left[1], right[1]).copy()
            if spec.split_rule == "optimize":
                for user in range(paras.n):
                    if rng.random() < mutation_rate:
                        X[user] = encode_split_row(
                            *pairs[int(rng.integers(len(pairs)))], paras.m
                        )
            if spec.exit_rule == "optimize":
                for user in range(paras.n):
                    for boundary in paras.E:
                        if rng.random() < mutation_rate:
                            Y[user, int(boundary)] = float(rng.random())
            X, Y = _enforce_candidate(paras, spec, X, Y)
            F_e, F_c = allocate_resources_for_xy(paras, X, Y)
            next_population.append((X, Y, F_e, F_c))
        population = next_population


def _run_ppo(paras: Paras, spec: DecisionSpec, evaluator: ObjectiveEvaluator):
    spec = replace(spec, allowed_split_pairs=tuple(_allowed_pairs(paras, spec)))
    np.random.seed(spec.seed)
    random.seed(spec.seed)
    try:
        import torch

        torch.manual_seed(spec.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(spec.seed)
    except ImportError:  # pragma: no cover
        pass
    params = _build_ppo_params(dict(spec.optimizer_options))
    agent = PPOAgent(paras, params, evaluator=evaluator, decision_spec=spec)
    try:
        agent.train()
    except EvaluationBudgetExceeded:
        pass
    if not evaluator.trace and evaluator.best_solution is not None:
        evaluator.last_breakdown = evaluator.breakdown(*evaluator.best_solution)
        evaluator.record("ppo", 0, evaluator.best_value, evaluator.best_value)


def _run_static(paras: Paras, spec: DecisionSpec, evaluator: ObjectiveEvaluator):
    rng = np.random.default_rng(spec.seed)
    solution = _candidate(paras, spec, rng, _allowed_pairs(paras, spec))
    value = _evaluate(evaluator, solution)
    evaluator.record("static", 0, value, value)


def _state_signature(state: dict, spec: DecisionSpec) -> dict[str, Any]:
    details = {
        "bundle_id": state.get("bundle_id"),
        "resource_mode": state.get("resource_mode"),
        "shared_resource_model": bool(state.get("shared_resource_model", False)),
        "user_ids": [int(user.get("user_id", index)) for index, user in enumerate(state["users"])],
        "execution_profile_ids": [
            str(owner.get("execution_profile_id", ""))
            for owner in [*state["users"], state["edge"], state["cloud"]]
        ],
        "bandwidth": {
            "d2e": [float(user["BW_d2e"]) for user in state["users"]],
            "e2c": float(state["cloud"]["BW_e2c"]),
        },
        "shared_resources": {
            "d2e_link_ids": [
                str(user.get("d2e_link_id", f"user-{index}"))
                for index, user in enumerate(state["users"])
            ],
            "d2e_capacity_mbps": state["edge"].get("d2e_capacity_mbps"),
            "e2c_link_id": str(state["cloud"].get("e2c_link_id", "edge-cloud")),
            "e2c_capacity_mbps": state["cloud"].get("e2c_capacity_mbps"),
            "edge_worker_count": state["edge"].get("worker_count"),
            "cloud_worker_count": state["cloud"].get("worker_count"),
        },
        "spec": spec.to_dict(),
    }
    canonical = json.dumps(
        details, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "details": details}


def _assemble_result(
    state: dict,
    paras: Paras,
    spec: DecisionSpec,
    evaluator: ObjectiveEvaluator,
    *,
    setup_s: float,
    solve_s: float,
    total_s: float,
) -> SolveResult:
    if evaluator.best_solution is None:
        raise RuntimeError("optimizer did not evaluate any candidate")
    X, Y, F_e, F_c = evaluator.best_solution
    expected = evaluator.breakdown(X, Y, F_e, F_c).to_dict()
    decision_id = str(state.get("round_id") or f"sync-{int(time.time() * 1000)}")
    decision = encode(
        X,
        Y,
        F_e,
        F_c,
        paras,
        decision_id=decision_id,
        bundle_id=paras.bundle_id,
        user_ids=[int(user.get("user_id", index)) for index, user in enumerate(state["users"])],
    )
    decision["objective"] = float(expected["utility_sum"])
    decision["decision_source"] = f"synchronous:{spec.optimizer}:{spec.coordination}"
    decision["expected"] = expected
    return SolveResult(
        X=X,
        Y=Y,
        F_e=F_e,
        F_c=F_c,
        decision=decision,
        expected=expected,
        optimizer_trace=list(evaluator.trace),
        objective_evaluations=int(evaluator.evaluations),
        setup_s=float(setup_s),
        solve_s=float(solve_s),
        total_s=float(total_s),
        spec=spec,
        state_signature=_state_signature(state, spec),
    )


def _solve_joint(state: dict, spec: DecisionSpec, total_started: float) -> SolveResult:
    setup_started = time.perf_counter()
    paras = Paras.from_state(state)
    setup_s = time.perf_counter() - setup_started

    if spec.stages == "split_then_exit":
        first = solve_decision(
            state,
            replace(spec, stages="joint", exit_rule="disabled", fixed_threshold=None),
        )
        split_pairs = tuple(decode_split_row(row) for row in first.X)
        second = solve_decision(
            state,
            replace(
                spec,
                stages="joint",
                split_rule="fixed",
                fixed_split=split_pairs,
                exit_rule="optimize",
            ),
        )
        second.optimizer_trace = [
            {**row, "stage": "split"} for row in first.optimizer_trace
        ] + [{**row, "stage": "exit"} for row in second.optimizer_trace]
        second.objective_evaluations += first.objective_evaluations
        second.setup_s += first.setup_s
        second.solve_s += first.solve_s
        second.total_s = time.perf_counter() - total_started
        second.spec = spec
        return second

    evaluator = ObjectiveEvaluator(paras, evaluation_budget=spec.evaluation_budget)
    solve_started = time.perf_counter()
    try:
        if spec.optimizer == "random":
            _run_random(paras, spec, evaluator)
        elif spec.optimizer == "ga":
            _run_ga(paras, spec, evaluator)
        elif spec.optimizer == "static":
            _run_static(paras, spec, evaluator)
        else:
            _run_ppo(paras, spec, evaluator)
    except EvaluationBudgetExceeded:
        pass
    solve_s = time.perf_counter() - solve_started
    return _assemble_result(
        state,
        paras,
        spec,
        evaluator,
        setup_s=setup_s,
        solve_s=solve_s,
        total_s=time.perf_counter() - total_started,
    )


def _solve_independent(state: dict, spec: DecisionSpec, total_started: float) -> SolveResult:
    per_user: list[SolveResult] = []
    total_users = len(state["users"])
    for index, user in enumerate(state["users"]):
        child_state = copy.deepcopy(state)
        child_state["users"] = [copy.deepcopy(user)]
        child_state["round_id"] = f"{state.get('round_id', 'sync')}-user-{user.get('user_id', index)}"
        per_user.append(
            solve_decision(child_state, spec.for_user(index, total_users))
        )

    setup_started = time.perf_counter()
    paras = Paras.from_state(state)
    setup_s = time.perf_counter() - setup_started
    X = np.concatenate([result.X for result in per_user], axis=0)
    Y = np.concatenate([result.Y for result in per_user], axis=0)
    F_e, F_c = allocate_resources_for_xy(paras, X, Y)
    evaluator = ObjectiveEvaluator(paras)
    evaluator.evaluate(X, Y, F_e, F_c)
    evaluator.record("independent", 0, evaluator.best_value, evaluator.best_value)
    result = _assemble_result(
        state,
        paras,
        spec,
        evaluator,
        setup_s=setup_s + sum(item.setup_s for item in per_user),
        solve_s=sum(item.solve_s for item in per_user),
        total_s=time.perf_counter() - total_started,
    )
    result.optimizer_trace = [
        {**row, "user_id": state["users"][index].get("user_id", index)}
        for index, item in enumerate(per_user)
        for row in item.optimizer_trace
    ] + [{**row, "stage": "merged_shared_evaluation"} for row in evaluator.trace]
    result.objective_evaluations = (
        sum(item.objective_evaluations for item in per_user) + evaluator.evaluations
    )
    result.X, result.Y, result.F_e, result.F_c = X, Y, F_e, F_c
    return result


def solve_decision(state: dict, spec: DecisionSpec | dict[str, Any] | None = None) -> SolveResult:
    """Synchronously solve one scheduling state under explicit constraints."""
    total_started = time.perf_counter()
    if spec is None:
        spec = DecisionSpec()
    elif isinstance(spec, dict):
        spec = DecisionSpec(**spec)
    spec.validate()
    if not isinstance(state, dict) or not state.get("users"):
        raise ValueError("state must contain at least one user")
    state = copy.deepcopy(state)
    if len(state["users"]) > 1:
        state.setdefault("shared_resource_model", True)
    if spec.coordination == "independent":
        return _solve_independent(state, spec, total_started)
    return _solve_joint(state, spec, total_started)
