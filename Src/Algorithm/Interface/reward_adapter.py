"""Convert deploy measurement JSON into PPO rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from Src.Configs.algo_config import DEFAULT as DEFAULT_ALGO_CONFIG
from Src.Configs.paras import Paras


class RewardAdapterError(ValueError):
    """Invalid measurement payload."""


@dataclass
class RoundRewardResult:
    decision_id: str
    per_user_rewards: list[float]
    round_reward: float
    alpha: float
    beta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "per_user_rewards": self.per_user_rewards,
            "round_reward": self.round_reward,
            "alpha": self.alpha,
            "beta": self.beta,
        }


def _user_index(measurement: dict, fallback: int) -> int:
    if "user_id" in measurement:
        return int(measurement["user_id"])
    return fallback


def validate_measurements(
    payload: dict,
    *,
    expected_decision_id: str | None = None,
    expected_num_users: int | None = None,
) -> list[dict]:
    """Validate measurement JSON (section 4.3) and return sorted per-user records."""
    if not isinstance(payload, dict):
        raise RewardAdapterError("Measurement payload must be a JSON object")

    decision_id = payload.get("decision_id")
    if not decision_id:
        raise RewardAdapterError("Missing required field: decision_id")

    if expected_decision_id is not None and decision_id != expected_decision_id:
        raise RewardAdapterError(
            f"decision_id mismatch: expected {expected_decision_id!r}, "
            f"got {decision_id!r}"
        )

    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or len(measurements) == 0:
        raise RewardAdapterError("measurements must be a non-empty list")

    if expected_num_users is not None and len(measurements) != expected_num_users:
        raise RewardAdapterError(
            f"Expected {expected_num_users} measurements, got {len(measurements)}"
        )

    required = ("T_total", "is_correct")
    parsed: list[tuple[int, dict]] = []
    for idx, m in enumerate(measurements):
        if not isinstance(m, dict):
            raise RewardAdapterError(f"measurements[{idx}] must be an object")
        for key in required:
            if key not in m:
                raise RewardAdapterError(f"measurements[{idx}] missing {key!r}")
        if m["T_total"] is None:
            raise RewardAdapterError(f"measurements[{idx}].T_total must not be null")
        if m["is_correct"] is None:
            raise RewardAdapterError(f"measurements[{idx}].is_correct must not be null")

        uid = _user_index(m, idx)
        parsed.append((uid, m))

    parsed.sort(key=lambda x: x[0])
    return [m for _, m in parsed]


def compute_user_reward(
    is_correct: bool,
    t_total: float,
    *,
    alpha: float,
    beta: float,
) -> float:
    """reward_i = alpha * is_correct_i - beta * T_total_i"""
    return float(alpha) * float(bool(is_correct)) - float(beta) * float(t_total)


def compute_round_reward(
    payload: dict,
    paras: Paras | None = None,
    *,
    alpha: float | None = None,
    beta: float | None = None,
    expected_decision_id: str | None = None,
    expected_num_users: int | None = None,
) -> RoundRewardResult:
    """Compute per-user and mean round reward from a measurement payload."""
    records = validate_measurements(
        payload,
        expected_decision_id=expected_decision_id,
        expected_num_users=expected_num_users,
    )

    if alpha is None or beta is None:
        if paras is not None:
            alpha = float(paras.alpha) if alpha is None else alpha
            beta = float(paras.beta) if beta is None else beta
        else:
            alpha = float(DEFAULT_ALGO_CONFIG.alpha) if alpha is None else alpha
            beta = float(DEFAULT_ALGO_CONFIG.beta) if beta is None else beta

    per_user: list[float] = []
    for m in records:
        per_user.append(
            compute_user_reward(
                bool(m["is_correct"]),
                float(m["T_total"]),
                alpha=float(alpha),
                beta=float(beta),
            )
        )

    round_reward = float(np.mean(per_user)) if per_user else 0.0
    return RoundRewardResult(
        decision_id=str(payload["decision_id"]),
        per_user_rewards=per_user,
        round_reward=round_reward,
        alpha=float(alpha),
        beta=float(beta),
    )


def apply_rewards_to_buffer(
    buffer,
    per_user_rewards: list[float],
    *,
    buffer_start: int,
) -> None:
    """
    Write per-user rewards into the tail of a ``RolloutBuffer``.

    The last ``len(per_user_rewards)`` entries (from ``buffer_start``) are updated.
    """
    n = len(per_user_rewards)
    if n == 0:
        return
    end = buffer_start + n
    if end > len(buffer.rewards):
        raise RewardAdapterError(
            f"Buffer too short: need indices [{buffer_start}, {end}), "
            f"len={len(buffer.rewards)}"
        )
    for offset, reward in enumerate(per_user_rewards):
        buffer.rewards[buffer_start + offset] = torch.tensor(
            float(reward), dtype=torch.float32
        )
