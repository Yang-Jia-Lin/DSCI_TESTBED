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

from Src.Algorithm.Interface.decision_codec import encode
from Src.Algorithm.Interface.reward_adapter import (
    RoundRewardResult,
    compute_round_reward,
)
from Src.Algorithm.Interface.state_adapter import to_paras
from Src.Algorithm.Optimizer.DSCI.agent import (
    PPOAgent,
    _init_feasible_XY,
    allocate_resources,
    compute_iota_kappa,
)
from Src.Algorithm.Optimizer.DSCI.run_DSCI import _build_ppo_params
from Src.Configs.algo_config import DEFAULT as DEFAULT_ALGO_CONFIG
from Src.Configs.paras import Paras, RESULT_PPO_PATH
from Src.Objective.compute_P import compute_layer_exit_probs
from Src.Objective.objective import objective

LATEST_SOLUTION_PATH = Path(RESULT_PPO_PATH) / "latest_solution.npz"
LATEST_META_PATH = Path(RESULT_PPO_PATH) / "latest_solution_meta.json"


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


@dataclass
class CachedSolution:
    X: np.ndarray
    Y: np.ndarray
    F_e: np.ndarray
    F_c: np.ndarray
    objective: float
    state_signature: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass
class AlgoService:
    """Stateful coordinator for cached HTTP decisions and background DSCI runs."""

    config: AlgoServiceConfig = field(default_factory=AlgoServiceConfig)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _cached_solution: CachedSolution | None = field(default=None, init=False, repr=False)
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

    def __post_init__(self) -> None:
        self.config.latest_solution_path = Path(self.config.latest_solution_path)
        self.config.latest_meta_path = Path(self.config.latest_meta_path)
        self._load_latest_solution()

    def _ppo_params(self) -> dict:
        return _build_ppo_params(self.config.custom_ppo_hyperparams)

    @staticmethod
    def _round_float(value) -> float:
        return round(float(value), 4)

    @classmethod
    def _state_signature(cls, state: dict, paras: Paras) -> dict[str, Any]:
        cfg = paras.model_cfg
        model_name = state.get("model_name") or (cfg.name if cfg is not None else None)
        users = []
        for i, user in enumerate(state.get("users") or []):
            users.append(
                {
                    "user_id": int(user.get("user_id", i)),
                    "f_u": cls._round_float(user["f_u"]),
                    "BW_d2e": cls._round_float(user["BW_d2e"]),
                }
            )
        return {
            "model": {
                "name": model_name,
                "m": int(paras.m),
                "early_exit_layers": [int(x) for x in paras.E],
            },
            "num_users": int(paras.n),
            "users": users,
            "edge": {"f_e_max": cls._round_float(state["edge"]["f_e_max"])},
            "cloud": {
                "f_c_max": cls._round_float(state["cloud"]["f_c_max"]),
                "BW_e2c": cls._round_float(state["cloud"]["BW_e2c"]),
            },
        }

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

    @staticmethod
    def _allocate_resources_for_xy(
        X: np.ndarray, Y: np.ndarray, paras: Paras
    ) -> tuple[np.ndarray, np.ndarray]:
        exit_prob = compute_layer_exit_probs(Y, paras)
        iota, kappa = compute_iota_kappa(X, paras.C, exit_prob)
        f_e, f_c = allocate_resources(iota, kappa, paras.f_e_max, paras.f_c_max)
        return (
            f_e.reshape(paras.n, 1).astype(np.float32),
            f_c.reshape(paras.n, 1).astype(np.float32),
        )

    def _default_solution(self, paras: Paras, signature: dict[str, Any]) -> CachedSolution:
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
        )

    def _solution_for_response(
        self, paras: Paras, signature: dict[str, Any]
    ) -> CachedSolution:
        cached = self._cached_solution
        if cached is None or not self._arrays_compatible(cached, paras):
            return self._default_solution(paras, signature)

        X = cached.X.astype(np.float32, copy=True)
        Y = cached.Y.astype(np.float32, copy=True)
        F_e, F_c = self._allocate_resources_for_xy(X, Y, paras)
        obj = float(objective(X, Y, F_e, F_c, paras))
        return CachedSolution(
            X=X,
            Y=Y,
            F_e=F_e,
            F_c=F_c,
            objective=obj,
            state_signature=copy.deepcopy(signature),
        )

    def _should_start_training(self, signature: dict[str, Any], paras: Paras) -> bool:
        if not self.config.auto_train:
            return False
        if self._training_status == "running":
            return False
        cached = self._cached_solution
        if cached is None or not self._arrays_compatible(cached, paras):
            return True
        return cached.state_signature != signature

    def _start_training_locked(self, state: dict, signature: dict[str, Any]) -> None:
        self._training_status = "running"
        self._training_signature = copy.deepcopy(signature)
        self._last_error = None
        train_state = copy.deepcopy(state)
        train_signature = copy.deepcopy(signature)
        thread = threading.Thread(
            target=self._train_background,
            args=(train_state, train_signature),
            daemon=True,
            name="DSCIBackgroundTraining",
        )
        self._training_thread = thread
        thread.start()

    def make_decision(self, state: dict) -> dict[str, Any]:
        """Return the current cached/default decision and train the next one in back."""
        paras = to_paras(state)
        signature = self._state_signature(state, paras)

        with self._lock:
            solution = self._solution_for_response(paras, signature)
            if self._should_start_training(signature, paras):
                self._start_training_locked(state, signature)

        decision_id = state.get("round_id") or f"round_{int(time.time() * 1000)}"
        model_name = state.get("model_name")
        if model_name is None and paras.model_cfg is not None:
            model_name = paras.model_cfg.name

        decision = encode(
            solution.X,
            solution.Y,
            solution.F_e,
            solution.F_c,
            paras,
            decision_id=str(decision_id),
            model_name=model_name,
            user_ids=self._extract_user_ids(state, paras.n),
        )
        decision["objective"] = float(solution.objective)
        return decision

    def _train_background(self, state: dict, signature: dict[str, Any]) -> None:
        try:
            paras = to_paras(state)
            agent = PPOAgent(paras, self._ppo_params())
            best_val, best_sol, _history = agent.train()
            if best_sol is None:
                raise RuntimeError("DSCI training returned no solution")

            X, Y, F_e, F_c = best_sol
            solution = CachedSolution(
                X=np.asarray(X, dtype=np.float32),
                Y=np.asarray(Y, dtype=np.float32),
                F_e=np.asarray(F_e, dtype=np.float32).reshape(paras.n, 1),
                F_c=np.asarray(F_c, dtype=np.float32).reshape(paras.n, 1),
                objective=float(best_val),
                state_signature=copy.deepcopy(signature),
            )

            with self._lock:
                self._cached_solution = solution
                self._training_status = "idle"
                self._training_signature = None
                self._update_epoch += 1

            try:
                self._save_latest_solution(solution)
            except Exception as exc:  # pragma: no cover - filesystem dependent
                with self._lock:
                    self._training_status = "error"
                    self._last_error = f"Failed to persist latest solution: {exc}"
        except Exception as exc:  # pragma: no cover - long-running training path
            with self._lock:
                self._training_status = "error"
                self._training_signature = None
                self._last_error = str(exc)

    def _save_latest_solution(self, solution: CachedSolution) -> None:
        solution_path = Path(self.config.latest_solution_path)
        meta_path = Path(self.config.latest_meta_path)
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

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

        meta = {
            "state_signature": solution.state_signature,
            "objective": float(solution.objective),
            "created_at": float(solution.created_at),
            "saved_at": time.time(),
        }
        tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        tmp_meta.replace(meta_path)

    def _load_latest_solution(self) -> None:
        solution_path = Path(self.config.latest_solution_path)
        meta_path = Path(self.config.latest_meta_path)
        if not solution_path.exists() or not meta_path.exists():
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            data = np.load(solution_path, allow_pickle=False)
            self._cached_solution = CachedSolution(
                X=np.asarray(data["X"], dtype=np.float32),
                Y=np.asarray(data["Y"], dtype=np.float32),
                F_e=np.asarray(data["F_e"], dtype=np.float32),
                F_c=np.asarray(data["F_c"], dtype=np.float32),
                objective=float(meta.get("objective", data["objective"])),
                state_signature=meta["state_signature"],
                created_at=float(meta.get("created_at", time.time())),
            )
        except Exception as exc:
            self._cached_solution = None
            self._training_status = "error"
            self._last_error = f"Failed to load latest solution: {exc}"

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
                "training_status": self._training_status,
                "training_state_signature": self._training_signature,
                "has_cached_solution": cached is not None,
                "cached_state_signature": cached.state_signature if cached else None,
                "cached_objective": float(cached.objective) if cached else None,
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
        "model_name": "Resnet50",
        "users": [
            {"user_id": 0, "BW_d2e": 18.5, "f_u": 2.0},
            {"user_id": 1, "BW_d2e": 12.0, "f_u": 1.8},
        ],
        "edge": {"f_e_max": 20.0, "cpu_util": 0.6},
        "cloud": {"BW_e2c": 120.0, "f_c_max": 50.0, "cpu_util": 0.4},
    }

    svc = AlgoService(config=AlgoServiceConfig(auto_train=False))
    decision = svc.make_decision(state)
    print("=== Decision JSON (excerpt) ===")
    print(_json.dumps({k: decision[k] for k in decision if k != "users"}, indent=2))
    print("user[0]:", _json.dumps(decision["users"][0], indent=2)[:500], "...")
    print("\n=== Health ===")
    print(_json.dumps(svc.health(), indent=2))
