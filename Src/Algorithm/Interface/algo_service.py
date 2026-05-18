"""Testbed algorithm service: one decision round + measurement feedback."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Src.Algorithm.Interface.decision_codec import encode
from Src.Algorithm.Interface.reward_adapter import (
    RewardAdapterError,
    RoundRewardResult,
    apply_rewards_to_buffer,
    compute_round_reward,
)
from Src.Algorithm.Interface.state_adapter import to_paras
from Src.Algorithm.Optimizer.DSCI.agent import PPOAgent
from Src.Algorithm.Optimizer.DSCI.run_DSCI import _build_ppo_params, infer_one_round
from Src.Configs.algo_config import DEFAULT as DEFAULT_ALGO_CONFIG
from Src.Configs.paras import Paras


@dataclass
class AlgoServiceConfig:
    checkpoint_path: str | Path | None = None
    enable_training: bool = False
    deterministic: bool = True
    outer_ema: float = 1.0
    buffer_size: int = DEFAULT_ALGO_CONFIG.buffer_size
    custom_ppo_hyperparams: dict | None = None


@dataclass
class PendingRound:
    decision_id: str
    n_users: int
    buffer_start: int
    paras: Paras


@dataclass
class AlgoService:
    """Stateful coordinator for HTTP testbed rounds."""

    config: AlgoServiceConfig = field(default_factory=AlgoServiceConfig)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _agent: PPOAgent | None = field(default=None, init=False, repr=False)
    _agent_key: tuple | None = field(default=None, init=False, repr=False)
    _pending: PendingRound | None = field(default=None, init=False, repr=False)
    _update_epoch: int = field(default=0, init=False, repr=False)
    _last_reward: RoundRewardResult | None = field(default=None, init=False, repr=False)

    def _ppo_params(self) -> dict:
        return _build_ppo_params(self.config.custom_ppo_hyperparams)

    def _agent_cache_key(self, paras: Paras) -> tuple:
        cfg = paras.model_cfg
        name = cfg.name if cfg is not None else "default"
        return (int(paras.m), tuple(int(x) for x in paras.E), name)

    def _get_or_create_agent(self, paras: Paras) -> PPOAgent:
        key = self._agent_cache_key(paras)
        if self._agent is None or self._agent_key != key:
            self._agent = PPOAgent(paras, self._ppo_params())
            self._agent_key = key
            if self.config.checkpoint_path is not None:
                path = Path(self.config.checkpoint_path)
                if path.is_file():
                    self._agent.load_checkpoint(path)
        else:
            self._agent.paras = paras
        return self._agent

    @staticmethod
    def _extract_user_ids(state: dict, n: int) -> list[int]:
        users = state.get("users") or []
        if len(users) != n:
            return list(range(n))
        ids = []
        for i, u in enumerate(users):
            ids.append(int(u.get("user_id", i)))
        return ids

    def make_decision(self, state: dict) -> dict[str, Any]:
        """Run one decision round from deploy state JSON."""
        with self._lock:
            if self._pending is not None:
                raise RuntimeError(
                    f"Previous round {self._pending.decision_id!r} has no measurements yet"
                )

            paras = to_paras(state)
            agent = self._get_or_create_agent(paras)
            buffer_start = len(agent.buffer)

            record = bool(self.config.enable_training)
            _obj, (X, Y, F_e, F_c), paras = infer_one_round(
                paras,
                agent=agent,
                deterministic=self.config.deterministic,
                outer_ema=self.config.outer_ema,
                record_transitions=record,
            )

            decision_id = state.get("round_id") or f"round_{buffer_start:04d}"
            model_name = state.get("model_name")
            if model_name is None and paras.model_cfg is not None:
                model_name = paras.model_cfg.name

            decision = encode(
                X,
                Y,
                F_e,
                F_c,
                paras,
                decision_id=str(decision_id),
                model_name=model_name,
                user_ids=self._extract_user_ids(state, paras.n),
            )

            if record:
                self._pending = PendingRound(
                    decision_id=str(decision_id),
                    n_users=int(paras.n),
                    buffer_start=buffer_start,
                    paras=paras,
                )

            decision["objective"] = float(_obj)
            return decision

    def report_measurements(self, payload: dict) -> dict[str, Any]:
        """Ingest deploy measurements; optionally trigger PPO update."""
        with self._lock:
            pending = self._pending
            expected_id = pending.decision_id if pending is not None else None
            expected_n = pending.n_users if pending is not None else None
            paras = pending.paras if pending is not None else None

            reward_result = compute_round_reward(
                payload,
                paras=paras,
                expected_decision_id=expected_id,
                expected_num_users=expected_n,
            )
            self._last_reward = reward_result

            policy_updated = False
            if self.config.enable_training and self._agent is not None and pending is not None:
                apply_rewards_to_buffer(
                    self._agent.buffer,
                    reward_result.per_user_rewards,
                    buffer_start=pending.buffer_start,
                )
                if len(self._agent.buffer) >= self.config.buffer_size:
                    self._agent.update_policy(epoch=self._update_epoch)
                    self._update_epoch += 1
                    self._agent.buffer.clear()
                    policy_updated = True

            self._pending = None

            return {
                "status": "ok",
                "decision_id": reward_result.decision_id,
                "round_reward": reward_result.round_reward,
                "per_user_rewards": reward_result.per_user_rewards,
                "policy_updated": policy_updated,
            }

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "checkpoint": str(self.config.checkpoint_path)
                if self.config.checkpoint_path
                else None,
                "enable_training": self.config.enable_training,
                "pending_decision_id": (
                    self._pending.decision_id if self._pending else None
                ),
                "buffer_steps": len(self._agent.buffer) if self._agent else 0,
                "update_epochs": self._update_epoch,
            }


def make_decision(state: dict, service: AlgoService | None = None) -> dict:
    svc = service or AlgoService()
    return svc.make_decision(state)


def report_measurements(payload: dict, service: AlgoService | None = None) -> dict:
    svc = service or AlgoService()
    return svc.report_measurements(payload)
