"""Testbed algorithm service: cached decisions + background DSCI training."""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Src.Phase2_Scheduler.algo_config import DEFAULT as DEFAULT_ALGO_CONFIG
from Src.Phase2_Scheduler.Service.decision_codec import encode
from Src.Phase2_Scheduler.Service.reward_adapter import (
    RoundRewardResult,
    compute_round_reward,
)
from Src.Phase2_Scheduler.Service.state_adapter import to_paras
from Src.Phase2_Scheduler.Objective.compute_P import compute_layer_exit_probs
from Src.Phase2_Scheduler.Objective.objective import objective
from Src.Phase2_Scheduler.Optimizer.DSCI.agent import (
    PPOAgent,
    _init_feasible_XY,
    allocate_resources,
    compute_iota_kappa,
)
from Src.Phase2_Scheduler.Optimizer.DSCI.run_DSCI import _build_ppo_params
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Config.paths import SOLUTION_CACHE_DIR

INTERFACE_SOLUTION_DIR = SOLUTION_CACHE_DIR
LATEST_SOLUTION_PATH = INTERFACE_SOLUTION_DIR / "latest_solution.npz"
LATEST_META_PATH = INTERFACE_SOLUTION_DIR / "latest_solution_meta.json"
TRAINING_EVENTS_PATH = INTERFACE_SOLUTION_DIR / "training_events.jsonl"
DIRECT_REUSE_DISTANCE = 0.005
NEAR_WARM_START_DISTANCE = 0.05
MEDIUM_WARM_START_DISTANCE = 0.15

_PRESET_MODE_ALIASES = {
    "dsci": None,
    "cached": None,
    "auto": None,
    "device": ("device", False),
    "pure_device": ("device", False),
    "device_no_exit": ("device", False),
    "device_early_exit": ("device", True),
    "device_exit": ("device", True),
    "edge": ("edge", False),
    "pure_edge": ("edge", False),
    "edge_no_exit": ("edge", False),
    "edge_early_exit": ("edge", True),
    "edge_exit": ("edge", True),
    "cloud": ("cloud", False),
    "pure_cloud": ("cloud", False),
    "cloud_no_exit": ("cloud", False),
    "cloud_early_exit": ("cloud", True),
    "cloud_exit": ("cloud", True),
}


@dataclass
class AlgoServiceConfig:
    checkpoint_path: str | Path | None = None
    enable_training: bool = False
    deterministic: bool = True
    outer_ema: float = 1.0
    buffer_size: int = DEFAULT_ALGO_CONFIG.buffer_size
    custom_ppo_hyperparams: dict | None = None
    auto_train: bool = True
    latest_solution_path: str | Path = LATEST_SOLUTION_PATH
    latest_meta_path: str | Path = LATEST_META_PATH
    training_events_path: str | Path = TRAINING_EVENTS_PATH
    max_cached_solutions: int = 10
    fixed_split: Any = None
    fixed_threshold: Any = None


@dataclass
class CachedSolution:
    X: np.ndarray
    Y: np.ndarray
    F_e: np.ndarray
    F_c: np.ndarray
    objective: float
    state_signature: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    compat_key: dict[str, Any] | None = None
    state_vector: list[float] | None = None
    policy_path: str | None = None
    training_mode: str | None = None


@dataclass
class CacheMatch:
    solution: CachedSolution
    distance: float
    training_mode: str
    policy_path: str | None = None


@dataclass
class AlgoService:
    """Stateful coordinator for cached HTTP decisions and background DSCI runs."""

    config: AlgoServiceConfig = field(default_factory=AlgoServiceConfig)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _cached_solution: CachedSolution | None = field(
        default=None, init=False, repr=False
    )
    _training_status: str = field(default="idle", init=False, repr=False)
    _training_signature: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _training_thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _last_error: str | None = field(default=None, init=False, repr=False)
    _last_reward: RoundRewardResult | None = field(default=None, init=False, repr=False)
    _update_epoch: int = field(default=0, init=False, repr=False)
    _cache_entries: list[CachedSolution] = field(
        default_factory=list, init=False, repr=False
    )
    _last_reuse_distance: float | None = field(default=None, init=False, repr=False)
    _last_training_mode: str | None = field(default=None, init=False, repr=False)
    _last_warm_start_source: str | None = field(default=None, init=False, repr=False)
    _training_started_at: float | None = field(default=None, init=False, repr=False)
    _last_training_started_at: float | None = field(default=None, init=False, repr=False)
    _last_training_finished_at: float | None = field(default=None, init=False, repr=False)
    _last_training_duration_s: float | None = field(default=None, init=False, repr=False)
    _last_training_round_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.config.latest_solution_path = Path(self.config.latest_solution_path)
        self.config.latest_meta_path = Path(self.config.latest_meta_path)
        self.config.training_events_path = Path(self.config.training_events_path)
        self.config.fixed_split = self._parse_fixed_split(self.config.fixed_split)
        self.config.fixed_threshold = self._parse_fixed_threshold(
            self.config.fixed_threshold
        )
        self._load_latest_solution()
        self._load_archived_solutions()

    def _ppo_params(self) -> dict:
        return _build_ppo_params(self.config.custom_ppo_hyperparams)

    @staticmethod
    def _round_float(value) -> float:
        return round(float(value), 4)

    @classmethod
    def _state_signature(cls, state: dict, paras: Paras) -> dict[str, Any]:
        users = []
        f_u_values = np.asarray(paras.F_u, dtype=float).reshape(-1)
        bw_d2e_values = (
            np.asarray(paras.B_u, dtype=float).reshape(-1)
            if paras.B_u is not None
            else np.full(int(paras.n), 0.0)
        )
        for i, user in enumerate(state.get("users") or []):
            users.append(
                {
                    "user_id": int(user.get("user_id", i)),
                    "f_u": cls._round_float(f_u_values[i]),
                    "BW_d2e": cls._round_float(bw_d2e_values[i]),
                    "compute_profile_id": user.get("compute_profile_id"),
                    "execution_profile_id": user.get("execution_profile_id"),
                }
            )
        return {
            "model": {
                "bundle_id": paras.bundle_id,
                "m": int(paras.m),
                "exit_ids": list(paras.exit_ids),
            },
            "num_users": int(paras.n),
            "resource_mode": paras.resource_mode,
            "manifest_id": paras.manifest_id,
            "model_hash": (
                paras.partition_manifest.model_hash
                if paras.partition_manifest is not None
                else None
            ),
            "users": users,
            "edge": {
                "f_e_max": cls._round_float(paras.f_e_max),
                "compute_profile_id": (state.get("edge") or {}).get(
                    "compute_profile_id"
                ),
                "execution_profile_id": (state.get("edge") or {}).get(
                    "execution_profile_id"
                ),
                "worker_count": (state.get("edge") or {}).get("worker_count"),
            },
            "cloud": {
                "f_c_max": cls._round_float(paras.f_c_max),
                "BW_e2c": cls._round_float(paras.b_c),
                "compute_profile_id": (state.get("cloud") or {}).get(
                    "compute_profile_id"
                ),
                "execution_profile_id": (state.get("cloud") or {}).get(
                    "execution_profile_id"
                ),
                "worker_count": (state.get("cloud") or {}).get("worker_count"),
            },
        }

    @staticmethod
    def _profile_token(entry: dict[str, Any]) -> Any:
        return entry.get("execution_profile_id") or entry.get("compute_profile_id")

    @classmethod
    def _compat_key(cls, signature: dict[str, Any]) -> dict[str, Any]:
        users = signature.get("users") or []
        return {
            "model": copy.deepcopy(signature.get("model") or {}),
            "num_users": signature.get("num_users"),
            "resource_mode": signature.get("resource_mode"),
            "manifest_id": signature.get("manifest_id"),
            "model_hash": signature.get("model_hash"),
            "user_profiles": [
                cls._profile_token(user) for user in users
            ],
            "edge_profile": cls._profile_token(signature.get("edge") or {}),
            "cloud_profile": cls._profile_token(signature.get("cloud") or {}),
        }

    @staticmethod
    def _float_or_zero(value: Any) -> float:
        try:
            if value is None:
                return 0.0
            out = float(value)
        except (TypeError, ValueError):
            return 0.0
        return out if np.isfinite(out) else 0.0

    @classmethod
    def _state_vector(cls, signature: dict[str, Any]) -> list[float]:
        vector: list[float] = []
        for user in signature.get("users") or []:
            vector.append(cls._float_or_zero(user.get("BW_d2e")))
            vector.append(cls._float_or_zero(user.get("f_u")))

        edge = signature.get("edge") or {}
        cloud = signature.get("cloud") or {}
        vector.extend(
            [
                cls._float_or_zero(edge.get("f_e_max")),
                cls._float_or_zero(edge.get("worker_count")),
                cls._float_or_zero(cloud.get("f_c_max")),
                cls._float_or_zero(cloud.get("BW_e2c")),
                cls._float_or_zero(cloud.get("worker_count")),
            ]
        )
        return vector

    @staticmethod
    def _state_distance(
        left: list[float] | None, right: list[float] | None
    ) -> float:
        if left is None or right is None or len(left) != len(right):
            return float("inf")
        a = np.asarray(left, dtype=np.float64)
        b = np.asarray(right, dtype=np.float64)
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
        delta = (a - b) / scale
        return float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0

    @staticmethod
    def _training_mode_for_distance(distance: float) -> str:
        if distance <= DIRECT_REUSE_DISTANCE:
            return "reuse"
        if distance <= NEAR_WARM_START_DISTANCE:
            return "near"
        if distance <= MEDIUM_WARM_START_DISTANCE:
            return "medium"
        return "cold"

    @staticmethod
    def _extract_user_ids(state: dict, n: int) -> list[int]:
        users = state.get("users") or []
        if len(users) != n:
            return list(range(n))
        return [int(u.get("user_id", i)) for i, u in enumerate(users)]

    @staticmethod
    def _arrays_compatible(solution: CachedSolution, paras: Paras) -> bool:
        expected = (int(paras.n), int(paras.m))
        return (
            solution.X.shape == expected
            and solution.Y.shape == expected
            and np.asarray(solution.F_e).reshape(-1).shape[0] == int(paras.n)
            and np.asarray(solution.F_c).reshape(-1).shape[0] == int(paras.n)
        )

    def _normalise_cached_solution(
        self, solution: CachedSolution
    ) -> CachedSolution:
        if solution.compat_key is None:
            solution.compat_key = self._compat_key(solution.state_signature)
        if solution.state_vector is None:
            solution.state_vector = self._state_vector(solution.state_signature)
        return solution

    def _remember_cache_entry(self, solution: CachedSolution) -> None:
        solution = self._normalise_cached_solution(solution)
        self._cache_entries = [
            entry
            for entry in self._cache_entries
            if entry.policy_path != solution.policy_path
            or entry.state_signature != solution.state_signature
        ]
        self._cache_entries.append(solution)
        keep = max(1, int(self.config.max_cached_solutions))
        self._cache_entries = sorted(
            self._cache_entries, key=lambda item: item.created_at
        )[-keep:]

    def _best_cache_match(
        self, signature: dict[str, Any], paras: Paras
    ) -> CacheMatch | None:
        compat_key = self._compat_key(signature)
        vector = self._state_vector(signature)
        candidates: list[CacheMatch] = []
        entries = list(self._cache_entries)
        if self._cached_solution is not None:
            entries.append(self._cached_solution)

        seen: set[tuple[str | None, float]] = set()
        for entry in entries:
            entry = self._normalise_cached_solution(entry)
            key = (entry.policy_path, entry.created_at)
            if key in seen:
                continue
            seen.add(key)
            if not self._arrays_compatible(entry, paras):
                continue
            if entry.compat_key != compat_key:
                continue
            distance = self._state_distance(entry.state_vector, vector)
            candidates.append(
                CacheMatch(
                    solution=entry,
                    distance=distance,
                    training_mode=self._training_mode_for_distance(distance),
                    policy_path=entry.policy_path,
                )
            )
        if not candidates:
            return None
        return min(candidates, key=lambda match: (match.distance, -match.solution.created_at))

    @staticmethod
    def _allocate_resources_for_xy(
        X: np.ndarray, Y: np.ndarray, paras: Paras
    ) -> tuple[np.ndarray, np.ndarray]:
        if paras.resource_mode == "fixed_worker_pool":
            return (
                np.zeros((paras.n, 1), dtype=np.float32),
                np.zeros((paras.n, 1), dtype=np.float32),
            )
        exit_prob = compute_layer_exit_probs(Y, paras)
        iota, kappa = compute_iota_kappa(X, paras.C_e, paras.C_c, exit_prob)
        f_e, f_c = allocate_resources(iota, kappa, paras.f_e_max, paras.f_c_max)
        return (
            f_e.reshape(paras.n, 1).astype(np.float32),
            f_c.reshape(paras.n, 1).astype(np.float32),
        )

    def _default_solution(
        self, paras: Paras, signature: dict[str, Any]
    ) -> CachedSolution:
        X, Y = _init_feasible_XY(paras)
        F_e, F_c = self._allocate_resources_for_xy(X, Y, paras)
        obj = float(objective(X, Y, F_e, F_c, paras))
        return CachedSolution(
            X=X,
            Y=Y,
            F_e=F_e,
            F_c=F_c,
            objective=obj,
            state_signature=copy.deepcopy(signature),
            compat_key=self._compat_key(signature),
            state_vector=self._state_vector(signature),
        )

    def _revalue_cached_solution(
        self, solution: CachedSolution, paras: Paras, signature: dict[str, Any]
    ) -> CachedSolution:
        X = solution.X.astype(np.float32, copy=True)
        Y = solution.Y.astype(np.float32, copy=True)
        F_e, F_c = self._allocate_resources_for_xy(X, Y, paras)
        obj = float(objective(X, Y, F_e, F_c, paras))
        return CachedSolution(
            X=X,
            Y=Y,
            F_e=F_e,
            F_c=F_c,
            objective=obj,
            state_signature=copy.deepcopy(signature),
            compat_key=self._compat_key(signature),
            state_vector=self._state_vector(signature),
            policy_path=solution.policy_path,
            training_mode=solution.training_mode,
        )

    @staticmethod
    def _normalise_decision_mode(state: dict) -> tuple[str, bool] | None:
        mode = state.get("decision_mode") or state.get("decision_policy")
        if mode is None:
            return None

        if isinstance(mode, str):
            key = mode.strip().lower().replace("-", "_")
            if key not in _PRESET_MODE_ALIASES:
                raise ValueError(
                    "decision_mode must be one of: "
                    + ", ".join(sorted(k for k in _PRESET_MODE_ALIASES if k))
                )
            return _PRESET_MODE_ALIASES[key]

        if isinstance(mode, dict):
            placement = str(mode.get("placement", "dsci")).strip().lower()
            placement = placement.replace("pure_", "").replace("-", "_")
            if placement in ("dsci", "cached", "auto"):
                return None
            if placement not in ("device", "edge", "cloud"):
                raise ValueError(
                    "decision_mode.placement must be device, edge, or cloud"
                )
            return placement, bool(mode.get("early_exit", False))

        raise ValueError("decision_mode must be a string or object")

    @staticmethod
    def _parse_fixed_split(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None

        if isinstance(value, str):
            parts = [p for p in value.replace(",", " ").split() if p]
            if len(parts) != 2:
                raise ValueError("fixed_split string must contain exactly two integers")
            return int(parts[0]), int(parts[1])

        if isinstance(value, dict):
            s1 = value.get("partition_s1", value.get("s1"))
            s2 = value.get("partition_s2", value.get("s2"))
            if s1 is None or s2 is None:
                raise ValueError(
                    "fixed_split object must contain partition_s1/partition_s2"
                )
            return int(s1), int(s2)

        try:
            parts = list(value)
        except TypeError as exc:
            raise ValueError(
                "fixed_split must be a two-integer list, string, or object"
            ) from exc

        if len(parts) != 2:
            raise ValueError("fixed_split must contain exactly two integers")
        return int(parts[0]), int(parts[1])

    def _fixed_split_for_state(self, state: dict) -> tuple[int, int] | None:
        if "fixed_split" in state:
            return self._parse_fixed_split(state["fixed_split"])
        if "split_points" in state:
            return self._parse_fixed_split(state["split_points"])
        if "partition_s1" in state or "partition_s2" in state:
            return self._parse_fixed_split(state)
        return self.config.fixed_split

    @staticmethod
    def _validate_fixed_split(split: tuple[int, int], paras: Paras) -> tuple[int, int]:
        s1, s2 = int(split[0]), int(split[1])
        if paras.resource_mode == "fixed_worker_pool":
            paras.partition_manifest.validate_boundary_pair(s1, s2)
            return s1, s2
        m = int(paras.m)
        if not (0 <= s1 < s2 < m):
            raise ValueError(
                f"fixed_split requires 0 <= s1 < s2 < {m}, got ({s1}, {s2})"
            )
        return s1, s2

    @staticmethod
    def _parse_fixed_threshold(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get("value", value.get("threshold"))
        threshold = float(value)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"fixed_threshold requires 0.0 <= threshold <= 1.0, got {threshold}"
            )
        return threshold

    def _fixed_threshold_for_state(self, state: dict) -> float | None:
        if "fixed_threshold" in state:
            return self._parse_fixed_threshold(state["fixed_threshold"])
        if "exit_threshold" in state:
            return self._parse_fixed_threshold(state["exit_threshold"])
        return self.config.fixed_threshold

    @staticmethod
    def _make_resource_vector(total: float, n: int, enabled: bool) -> np.ndarray:
        if not enabled or total <= 0:
            return np.zeros((n, 1), dtype=np.float32)
        return (np.ones((n, 1), dtype=np.float32) * (float(total) / n)).astype(
            np.float32
        )

    def _preset_solution(
        self,
        paras: Paras,
        signature: dict[str, Any],
        placement: str,
        early_exit: bool,
    ) -> CachedSolution:
        n, m = int(paras.n), int(paras.m)
        last = m - 1
        penultimate = max(0, m - 2)

        if paras.resource_mode == "fixed_worker_pool":
            final = int(paras.partition_manifest.final_boundary_id)
            exit_1 = int(paras.E[0])
            exit_2 = int(paras.E[-1])
            if placement == "device":
                s1, s2 = (exit_1, final) if early_exit else (final - 1, final)
            elif placement == "edge":
                s1, s2 = (0, exit_2) if early_exit else (0, final - 1)
            else:
                s1, s2 = (0, 1)
        elif placement == "device":
            s1, s2 = (int(paras.E[0]), last) if early_exit and paras.E else (penultimate, last)
        elif placement == "edge":
            s1, s2 = (0, int(paras.E[-1])) if early_exit and paras.E else (0, penultimate)
        else:
            s1, s2 = (0, 4 if m > 5 else 1)

        if not (0 <= s1 < s2 < m):
            s1, s2 = max(0, m // 3), min(last, (2 * m) // 3)
            if s1 == s2:
                s2 = min(last, s1 + 1)

        X = np.zeros((n, m), dtype=np.float32)
        X[:, s1] = 1.0
        X[:, s2] = 1.0

        Y = np.ones((n, m), dtype=np.float32)
        if early_exit:
            if placement == "device" and paras.E:
                Y[:, int(paras.E[0])] = 0.0
            elif placement == "edge" and paras.E:
                Y[:, int(paras.E[-1])] = 0.0
            elif placement == "cloud":
                for layer in paras.E:
                    if 0 <= layer < m:
                        Y[:, layer] = 0.0

        fixed_workers = paras.resource_mode == "fixed_worker_pool"
        F_e = self._make_resource_vector(
            paras.f_e_max, n, placement == "edge" and not fixed_workers
        )
        F_c = self._make_resource_vector(
            paras.f_c_max, n, placement == "cloud" and not fixed_workers
        )

        preset_signature = copy.deepcopy(signature)
        preset_signature["decision_mode"] = {
            "placement": placement,
            "early_exit": bool(early_exit),
        }
        return CachedSolution(
            X=X,
            Y=Y,
            F_e=F_e,
            F_c=F_c,
            objective=0.0,
            state_signature=preset_signature,
        )

    def _fixed_split_solution(
        self,
        paras: Paras,
        signature: dict[str, Any],
        s1: int,
        s2: int,
    ) -> CachedSolution:
        n, m = int(paras.n), int(paras.m)
        X = np.zeros((n, m), dtype=np.float32)
        X[:, s1] = 1.0
        X[:, s2] = 1.0

        Y = np.ones((n, m), dtype=np.float32)
        F_e, F_c = self._allocate_resources_for_xy(X, Y, paras)
        obj = float(objective(X, Y, F_e, F_c, paras))

        fixed_signature = copy.deepcopy(signature)
        fixed_signature["fixed_split"] = {
            "partition_s1": int(s1),
            "partition_s2": int(s2),
        }
        return CachedSolution(
            X=X,
            Y=Y,
            F_e=F_e,
            F_c=F_c,
            objective=obj,
            state_signature=fixed_signature,
        )

    def _with_fixed_threshold(
        self,
        solution: CachedSolution,
        paras: Paras,
        threshold: float,
    ) -> CachedSolution:
        Y = solution.Y.astype(np.float64, copy=True)
        for layer in paras.E:
            layer_idx = int(layer)
            if 0 <= layer_idx < int(paras.m):
                Y[:, layer_idx] = float(threshold)

        F_e, F_c = self._allocate_resources_for_xy(solution.X, Y, paras)
        obj = float(objective(solution.X, Y, F_e, F_c, paras))

        threshold_signature = copy.deepcopy(solution.state_signature)
        threshold_signature["fixed_threshold"] = float(threshold)
        return CachedSolution(
            X=solution.X.astype(np.float32, copy=True),
            Y=Y,
            F_e=F_e,
            F_c=F_c,
            objective=obj,
            state_signature=threshold_signature,
            created_at=solution.created_at,
        )

    def _solution_for_response(
        self,
        paras: Paras,
        signature: dict[str, Any],
        match: CacheMatch | None = None,
    ) -> tuple[CachedSolution, str]:
        default = self._default_solution(paras, signature)
        if match is None:
            return default, "default"

        warm = self._revalue_cached_solution(match.solution, paras, signature)
        if match.solution.state_signature == signature:
            return warm, "cached_dsci:exact"
        if match.training_mode == "reuse":
            return warm, f"cached_dsci:reuse:{match.distance:.6f}"

        if warm.objective >= default.objective:
            return warm, f"cached_dsci:warm:{match.distance:.6f}"
        return default, "default"

    def _should_start_training(
        self, match: CacheMatch | None, paras: Paras
    ) -> bool:
        if not self.config.auto_train:
            return False
        if self._training_status == "running":
            return False
        if match is None or not self._arrays_compatible(match.solution, paras):
            return True
        return match.training_mode != "reuse"

    def _start_training_locked(
        self,
        state: dict,
        signature: dict[str, Any],
        match: CacheMatch | None = None,
    ) -> None:
        started_at = time.time()
        planned_mode = match.training_mode if match else "cold"
        if match is not None and planned_mode == "cold" and match.policy_path:
            planned_mode = "cold_warm"
        self._training_status = "running"
        self._training_signature = copy.deepcopy(signature)
        self._last_error = None
        self._last_training_mode = planned_mode
        self._last_warm_start_source = match.policy_path if match else None
        self._training_started_at = started_at
        self._last_training_round_id = str(state.get("round_id") or "")
        train_state = copy.deepcopy(state)
        train_signature = copy.deepcopy(signature)
        train_match = copy.deepcopy(match)
        thread = threading.Thread(
            target=self._train_background,
            args=(train_state, train_signature, train_match, started_at),
            daemon=True,
            name="DSCIBackgroundTraining",
        )
        self._training_thread = thread
        thread.start()

    def make_decision(self, state: dict) -> dict[str, Any]:
        """Return the current cached/default decision and train the next one in back."""
        paras = to_paras(state)
        signature = self._state_signature(state, paras)
        fixed_split = self._fixed_split_for_state(state)
        fixed_threshold = self._fixed_threshold_for_state(state)
        preset_mode = self._normalise_decision_mode(state)

        if fixed_split is not None:
            s1, s2 = self._validate_fixed_split(fixed_split, paras)
            solution = self._fixed_split_solution(paras, signature, s1, s2)
            decision_source = f"fixed_split:{s1}:{s2}"
        elif preset_mode is None:
            with self._lock:
                match = self._best_cache_match(signature, paras)
                solution, decision_source = self._solution_for_response(
                    paras, signature, match
                )
                self._last_reuse_distance = match.distance if match else None
                if self._should_start_training(match, paras):
                    self._start_training_locked(state, signature, match)
        else:
            placement, early_exit = preset_mode
            solution = self._preset_solution(paras, signature, placement, early_exit)
            decision_source = (
                f"preset:{placement}:{'early_exit' if early_exit else 'no_exit'}"
            )

        if fixed_threshold is not None:
            solution = self._with_fixed_threshold(solution, paras, fixed_threshold)
            decision_source = f"{decision_source}:threshold:{fixed_threshold:g}"

        decision_id = state.get("round_id") or f"round_{int(time.time() * 1000)}"
        decision = encode(
            solution.X,
            solution.Y,
            solution.F_e,
            solution.F_c,
            paras,
            decision_id=str(decision_id),
            bundle_id=paras.bundle_id,
            user_ids=self._extract_user_ids(state, paras.n),
        )
        decision["objective"] = float(solution.objective)
        decision["decision_source"] = decision_source
        return decision

    def _training_params(self, match: CacheMatch | None) -> dict[str, Any]:
        params = self._ppo_params()
        custom = self.config.custom_ppo_hyperparams or {}
        if "outer_ema" not in custom:
            params["outer_ema"] = float(self.config.outer_ema)

        mode = match.training_mode if match else "cold"
        if mode == "near":
            params.update(
                {
                    "max_epochs": 30,
                    "min_epochs": 8,
                    "target_steps": 400,
                    "k_epochs": 5,
                    "lr": 5e-5,
                    "entropy_coef": 0.003,
                    "outer_ema": 1.0,
                }
            )
        elif mode == "medium":
            params.update(
                {
                    "max_epochs": 80,
                    "min_epochs": 20,
                    "target_steps": 800,
                    "k_epochs": 8,
                    "lr": 8e-5,
                    "entropy_coef": 0.006,
                    "outer_ema": 0.5,
                }
            )
        return params

    def _load_warm_policy(
        self, agent: PPOAgent, match: CacheMatch | None
    ) -> tuple[str, str | None]:
        if match is not None and match.policy_path:
            try:
                agent.load_checkpoint(match.policy_path)
                mode = match.training_mode
                if mode == "cold":
                    mode = "cold_warm"
                return mode, match.policy_path
            except Exception as exc:
                self._last_error = f"Warm-start policy load failed: {exc}"

        if self.config.checkpoint_path:
            try:
                agent.load_checkpoint(self.config.checkpoint_path)
                return "checkpoint", str(self.config.checkpoint_path)
            except Exception as exc:
                self._last_error = f"Checkpoint policy load failed: {exc}"

        return "cold", None

    @staticmethod
    def _initial_solution_from_match(
        match: CacheMatch | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if match is None:
            return None
        sol = match.solution
        return (
            sol.X.astype(np.float32, copy=True),
            sol.Y.astype(np.float32, copy=True),
            sol.F_e.astype(np.float32, copy=True),
            sol.F_c.astype(np.float32, copy=True),
        )

    def _train_background(
        self,
        state: dict,
        signature: dict[str, Any],
        match: CacheMatch | None = None,
        started_at: float | None = None,
    ) -> None:
        started_at = float(started_at or time.time())
        round_id = str(state.get("round_id") or "")
        training_mode = match.training_mode if match else "cold"
        policy_source = match.policy_path if match else None
        try:
            paras = to_paras(state)
            params = self._training_params(match)
            agent = PPOAgent(paras, params)
            training_mode, policy_source = self._load_warm_policy(agent, match)
            initial_solution = self._initial_solution_from_match(match)
            best_val, best_sol, _history = agent.train(
                initial_solution=initial_solution
            )
            if best_sol is None:
                raise RuntimeError("DSCI training returned no solution")
            finished_at = time.time()
            duration_s = finished_at - started_at

            X, Y, F_e, F_c = best_sol
            solution = CachedSolution(
                X=np.asarray(X, dtype=np.float32),
                Y=np.asarray(Y, dtype=np.float32),
                F_e=np.asarray(F_e, dtype=np.float32).reshape(paras.n, 1),
                F_c=np.asarray(F_c, dtype=np.float32).reshape(paras.n, 1),
                objective=float(best_val),
                state_signature=copy.deepcopy(signature),
                compat_key=self._compat_key(signature),
                state_vector=self._state_vector(signature),
                training_mode=training_mode,
            )

            with self._lock:
                self._cached_solution = solution
                self._remember_cache_entry(solution)
                self._training_status = "idle"
                self._training_signature = None
                self._last_training_mode = training_mode
                self._last_warm_start_source = policy_source
                self._training_started_at = None
                self._last_training_started_at = started_at
                self._last_training_finished_at = finished_at
                self._last_training_duration_s = duration_s
                self._last_training_round_id = round_id
                self._update_epoch += 1

            try:
                policy_state = agent.best_policy_state_dict or {
                    k: v.detach().cpu().clone()
                    for k, v in agent.policy.state_dict().items()
                }
                self._save_latest_solution(
                    solution,
                    policy_state_dict=policy_state,
                    training_mode=training_mode,
                    training_started_at=started_at,
                    training_finished_at=finished_at,
                    training_duration_s=duration_s,
                )
                self._append_training_event(
                    {
                        "event": "training_complete",
                        "round_id": round_id,
                        "training_mode": training_mode,
                        "policy_source": policy_source,
                        "objective": float(best_val),
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "duration_s": duration_s,
                        "update_epoch": self._update_epoch,
                    }
                )
            except Exception as exc:  # pragma: no cover - filesystem dependent
                with self._lock:
                    self._training_status = "error"
                    self._last_error = f"Failed to persist latest solution: {exc}"
        except Exception as exc:  # pragma: no cover - long-running training path
            finished_at = time.time()
            duration_s = finished_at - started_at
            with self._lock:
                self._training_status = "error"
                self._training_signature = None
                self._training_started_at = None
                self._last_training_started_at = started_at
                self._last_training_finished_at = finished_at
                self._last_training_duration_s = duration_s
                self._last_training_round_id = round_id
                self._last_error = str(exc)
            self._append_training_event(
                {
                    "event": "training_error",
                    "round_id": round_id,
                    "training_mode": training_mode,
                    "policy_source": policy_source,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_s": duration_s,
                    "error": str(exc),
                }
            )

    def _append_training_event(self, event: dict[str, Any]) -> None:
        try:
            path = Path(self.config.training_events_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"logged_at": time.time(), **event}
            with path.open("a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        except Exception as exc:  # pragma: no cover - logging should not break service
            with self._lock:
                self._last_error = f"Failed to write training event: {exc}"

    def _save_latest_solution(
        self,
        solution: CachedSolution,
        *,
        policy_state_dict: dict[str, Any] | None = None,
        training_mode: str | None = None,
        training_started_at: float | None = None,
        training_finished_at: float | None = None,
        training_duration_s: float | None = None,
    ) -> None:
        solution_path = Path(self.config.latest_solution_path)
        meta_path = Path(self.config.latest_meta_path)
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        timestamp = f"{timestamp}{int((time.time() % 1) * 1000):03d}"
        archived_solution = solution_path.with_name(f"solution_{timestamp}.npz")
        archived_meta = meta_path.with_name(f"solution_{timestamp}_meta.json")
        archived_policy = solution_path.with_name(f"solution_{timestamp}_policy.pt")
        latest_policy = solution_path.with_name(f"{solution_path.stem}_policy.pt")

        solution.compat_key = self._compat_key(solution.state_signature)
        solution.state_vector = self._state_vector(solution.state_signature)
        solution.training_mode = training_mode or solution.training_mode
        if policy_state_dict is not None:
            torch.save(policy_state_dict, archived_policy)
            torch.save(policy_state_dict, latest_policy)
            solution.policy_path = str(archived_policy)

        meta = {
            "state_signature": solution.state_signature,
            "compat_key": solution.compat_key,
            "state_vector": solution.state_vector,
            "objective": float(solution.objective),
            "created_at": float(solution.created_at),
            "policy_path": str(archived_policy) if policy_state_dict is not None else solution.policy_path,
            "training_mode": solution.training_mode,
            "training_started_at": training_started_at,
            "training_finished_at": training_finished_at,
            "training_duration_s": training_duration_s,
            "saved_at": time.time(),
        }
        self._write_solution_pair(archived_solution, archived_meta, solution, meta)

        latest_meta = dict(meta)
        latest_meta["policy_path"] = (
            str(latest_policy) if policy_state_dict is not None else solution.policy_path
        )
        self._write_solution_pair(solution_path, meta_path, solution, latest_meta)
        self._prune_archived_solutions(solution_path.parent)

    @staticmethod
    def _write_solution_pair(
        solution_path: Path,
        meta_path: Path,
        solution: CachedSolution,
        meta: dict[str, Any],
    ) -> None:
        tmp_solution = solution_path.with_name(solution_path.name + ".tmp.npz")
        np.savez(
            tmp_solution,
            X=solution.X,
            Y=solution.Y,
            F_e=solution.F_e,
            F_c=solution.F_c,
            objective=np.array(solution.objective, dtype=np.float64),
        )
        tmp_solution.replace(solution_path)

        tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        tmp_meta.replace(meta_path)

    def _prune_archived_solutions(self, cache_dir: Path) -> None:
        keep = max(1, int(self.config.max_cached_solutions))
        archives = sorted(cache_dir.glob("solution_*.npz"), key=lambda p: p.name)
        stale = archives[:-keep]
        for solution_path in stale:
            stem = solution_path.stem
            meta_path = solution_path.with_name(f"{stem}_meta.json")
            policy_path = solution_path.with_name(f"{stem}_policy.pt")
            try:
                solution_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                policy_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_latest_solution(self) -> None:
        solution_path = Path(self.config.latest_solution_path)
        meta_path = Path(self.config.latest_meta_path)
        if not solution_path.exists() or not meta_path.exists():
            return

        try:
            solution = self._load_solution_pair(solution_path, meta_path)
            self._cached_solution = solution
            self._remember_cache_entry(solution)
        except Exception as exc:
            self._cached_solution = None
            self._training_status = "error"
            self._last_error = f"Failed to load latest solution: {exc}"

    def _load_archived_solutions(self) -> None:
        cache_dir = Path(self.config.latest_solution_path).parent
        if not cache_dir.exists():
            return
        for solution_path in sorted(cache_dir.glob("solution_*.npz")):
            meta_path = solution_path.with_name(f"{solution_path.stem}_meta.json")
            if not meta_path.exists():
                continue
            try:
                self._remember_cache_entry(
                    self._load_solution_pair(solution_path, meta_path)
                )
            except Exception:
                continue

    def _load_solution_pair(
        self, solution_path: Path, meta_path: Path
    ) -> CachedSolution:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        signature = meta["state_signature"]
        signature_model = signature.get("model", {})
        if "bundle_id" not in signature_model:
            raise ValueError("Legacy solution cache without bundle_id is not supported")
        data = np.load(solution_path, allow_pickle=False)
        policy_path = meta.get("policy_path")
        if policy_path and not Path(policy_path).exists():
            fallback = solution_path.with_name(f"{solution_path.stem}_policy.pt")
            policy_path = str(fallback) if fallback.exists() else None
        solution = CachedSolution(
            X=np.asarray(data["X"], dtype=np.float32),
            Y=np.asarray(data["Y"], dtype=np.float32),
            F_e=np.asarray(data["F_e"], dtype=np.float32),
            F_c=np.asarray(data["F_c"], dtype=np.float32),
            objective=float(meta.get("objective", data["objective"])),
            state_signature=signature,
            created_at=float(meta.get("created_at", time.time())),
            compat_key=meta.get("compat_key"),
            state_vector=meta.get("state_vector"),
            policy_path=policy_path,
            training_mode=meta.get("training_mode"),
        )
        return self._normalise_cached_solution(solution)

    def report_measurements(self, payload: dict) -> dict[str, Any]:
        """Ingest deploy measurements without online PPO buffer updates."""
        with self._lock:
            reward_result = compute_round_reward(payload)
            self._last_reward = reward_result
            return {
                "status": "ok",
                "decision_id": reward_result.decision_id,
                "round_reward": reward_result.round_reward,
                "per_user_rewards": reward_result.per_user_rewards,
                "policy_updated": False,
            }

    def health(self) -> dict[str, Any]:
        with self._lock:
            cached = self._cached_solution
            return {
                "status": "ok",
                "checkpoint": str(self.config.checkpoint_path)
                if self.config.checkpoint_path
                else None,
                "enable_training": self.config.enable_training,
                "auto_train": self.config.auto_train,
                "objective_alpha": float(DEFAULT_ALGO_CONFIG.alpha),
                "objective_beta": float(DEFAULT_ALGO_CONFIG.beta),
                "training_status": self._training_status,
                "training_state_signature": self._training_signature,
                "training_started_at": self._training_started_at,
                "last_training_started_at": self._last_training_started_at,
                "last_training_finished_at": self._last_training_finished_at,
                "last_training_duration_s": self._last_training_duration_s,
                "last_training_round_id": self._last_training_round_id,
                "training_events_path": str(self.config.training_events_path),
                "has_cached_solution": cached is not None,
                "cached_state_signature": cached.state_signature if cached else None,
                "cached_objective": float(cached.objective) if cached else None,
                "cache_entries": len(self._cache_entries),
                "last_reuse_distance": self._last_reuse_distance,
                "last_training_mode": self._last_training_mode,
                "last_warm_start_source": self._last_warm_start_source,
                "policy_cache_enabled": True,
                "fixed_split": self.config.fixed_split,
                "fixed_threshold": self.config.fixed_threshold,
                "last_error": self._last_error,
                "update_epochs": self._update_epoch,
            }


def make_decision(state: dict, service: AlgoService | None = None) -> dict:
    svc = service or AlgoService()
    return svc.make_decision(state)


def report_measurements(payload: dict, service: AlgoService | None = None) -> dict:
    svc = service or AlgoService()
    return svc.report_measurements(payload)


if __name__ == "__main__":
    import json as _json

    state = {
        "round_id": "round_0001",
        "bundle_id": "resnet50-cifar10-ee-v1",
        "users": [
            {
                "user_id": 0,
                "BW_d2e": 18.5,
                "f_u": 2e9,
                "compute_profile_id": "device-pytorch",
            },
            {
                "user_id": 1,
                "BW_d2e": 12.0,
                "f_u": 2e9,
                "compute_profile_id": "device-pytorch",
            },
        ],
        "edge": {
            "f_e_max": 20e9,
            "compute_profile_id": "edge-pytorch",
            "cpu_util": 0.6,
        },
        "cloud": {
            "BW_e2c": 120.0,
            "f_c_max": 50e9,
            "compute_profile_id": "cloud-pytorch",
            "cpu_util": 0.4,
        },
    }

    svc = AlgoService(config=AlgoServiceConfig(auto_train=False))
    decision = svc.make_decision(state)
    print("=== Decision JSON (excerpt) ===")
    print(_json.dumps({k: decision[k] for k in decision if k != "users"}, indent=2))
    print("user[0]:", _json.dumps(decision["users"][0], indent=2)[:500], "...")
    print("\n=== Health ===")
    print(_json.dumps(svc.health(), indent=2))
