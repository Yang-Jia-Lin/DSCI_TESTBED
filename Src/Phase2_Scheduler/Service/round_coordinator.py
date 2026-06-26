"""Fixed-size multi-device round coordination for the v2 scheduler API."""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from Src.Phase2_Scheduler.Service.algo_service import AlgoService


class RoundCoordinatorError(ValueError):
    """Invalid round operation."""


class RoundConflictError(RuntimeError):
    """Operation conflicts with the active round."""


@dataclass
class DeviceRegistration:
    user_id: int
    device: dict
    last_heartbeat: float


@dataclass
class SchedulingRound:
    round_id: str
    expected_users: int
    created_at: float
    status: str = "waiting"
    registered_devices: dict[int, DeviceRegistration] = field(default_factory=dict)
    batch_decision: dict | None = None
    per_user_decisions: dict[int, dict] = field(default_factory=dict)
    measurements: dict[int, dict] = field(default_factory=dict)
    decision_version: int = 0
    error: str | None = None


class RoundCoordinator:
    """Coordinate one fixed-size scheduling round at a time."""

    def __init__(
        self,
        service: AlgoService,
        *,
        expected_users: int,
        node_state_provider: Callable[[], tuple[dict, dict]],
        heartbeat_timeout_s: float = 15.0,
        barrier_timeout_s: float = 60.0,
        clock: Callable[[], float] = time.time,
    ):
        if expected_users <= 0:
            raise ValueError("expected_users must be positive")
        self.service = service
        self.expected_users = int(expected_users)
        self.node_state_provider = node_state_provider
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.barrier_timeout_s = float(barrier_timeout_s)
        self.clock = clock
        self._lock = threading.RLock()
        self._round: SchedulingRound | None = None

    def register(self, round_id: str, payload: dict) -> dict:
        user_id, device = self._parse_registration(payload)
        now = self.clock()
        with self._lock:
            current = self._get_or_create_round(round_id, now)
            self._expire_if_needed(current, now)
            if current.status != "waiting":
                raise RoundConflictError(
                    f"Round {round_id!r} is {current.status}; registration is closed"
                )

            existing = current.registered_devices.get(user_id)
            if existing is not None and existing.device != device:
                raise RoundConflictError(
                    f"user_id {user_id} is already registered with different state"
                )
            current.registered_devices[user_id] = DeviceRegistration(
                user_id=user_id,
                device=copy.deepcopy(device),
                last_heartbeat=now,
            )
            if len(current.registered_devices) == current.expected_users:
                round_id, users = self._start_optimization_locked(current)
            else:
                return self._status_locked(current)
        self._optimize_round(round_id, users)
        with self._lock:
            return self._status_locked(self._require_round(round_id))

    def heartbeat(self, round_id: str, user_id: int) -> dict:
        now = self.clock()
        with self._lock:
            current = self._require_round(round_id)
            self._expire_if_needed(current, now)
            registration = current.registered_devices.get(int(user_id))
            if registration is None:
                raise RoundCoordinatorError(f"user_id {user_id} is not registered")
            if current.status in {"completed", "failed"}:
                raise RoundConflictError(f"Round {round_id!r} is {current.status}")
            registration.last_heartbeat = now
            return self._status_locked(current)

    def decision_for_user(self, round_id: str, user_id: int) -> dict | None:
        now = self.clock()
        with self._lock:
            current = self._require_round(round_id)
            self._expire_if_needed(current, now)
            if int(user_id) not in current.registered_devices:
                raise RoundCoordinatorError(f"user_id {user_id} is not registered")
            if current.status == "failed":
                raise RoundConflictError(current.error or "Round failed")
            decision = current.per_user_decisions.get(int(user_id))
            return copy.deepcopy(decision) if decision is not None else None

    def submit_measurements(self, round_id: str, user_id: int, payload: dict) -> dict:
        with self._lock:
            current = self._require_round(round_id)
            if current.status not in {"ready", "completed"}:
                raise RoundConflictError(
                    f"Round {round_id!r} is {current.status}; measurements are not accepted"
                )
            user_id = int(user_id)
            if user_id not in current.registered_devices:
                raise RoundCoordinatorError(f"user_id {user_id} is not registered")
            self._validate_measurement_payload(current, payload)
            normalized = self._normalize_user_measurements(user_id, payload)
            existing = current.measurements.get(user_id)
            if existing is not None:
                if existing != normalized:
                    raise RoundConflictError(
                        f"user_id {user_id} already submitted different measurements"
                    )
                if (
                    len(current.measurements) == current.expected_users
                    and current.status == "ready"
                ):
                    self._complete_locked(current)
                return self._status_locked(current)
            existing_request_ids = {
                str(record["request_id"])
                for submitted in current.measurements.values()
                for record in submitted["measurements"]
            }
            duplicate_ids = existing_request_ids.intersection(
                record["request_id"] for record in normalized["measurements"]
            )
            if duplicate_ids:
                raise RoundCoordinatorError(
                    f"request_id already submitted by another user: {sorted(duplicate_ids)}"
                )
            current.measurements[user_id] = normalized
            if len(current.measurements) == current.expected_users:
                self._complete_locked(current)
            return self._status_locked(current)

    def status(self, round_id: str) -> dict:
        now = self.clock()
        with self._lock:
            current = self._require_round(round_id)
            self._expire_if_needed(current, now)
            return self._status_locked(current)

    def _get_or_create_round(self, round_id: str, now: float) -> SchedulingRound:
        round_id = str(round_id).strip()
        if not round_id:
            raise RoundCoordinatorError("round_id must be non-empty")
        if (
            self._round is not None
            and self._round.round_id == round_id
            and self._round.status in {"completed", "failed"}
        ):
            raise RoundConflictError(f"round_id {round_id!r} cannot be reused")
        if self._round is None or self._round.status in {"completed", "failed"}:
            self._round = SchedulingRound(
                round_id=round_id,
                expected_users=self.expected_users,
                created_at=now,
            )
        elif self._round.round_id != round_id:
            raise RoundConflictError(
                f"Round {self._round.round_id!r} is still active"
            )
        return self._round

    def _require_round(self, round_id: str) -> SchedulingRound:
        if self._round is None or self._round.round_id != str(round_id):
            raise RoundCoordinatorError(f"Unknown round_id {round_id!r}")
        return self._round

    def _expire_if_needed(self, current: SchedulingRound, now: float) -> None:
        if current.status != "waiting":
            return
        if now - current.created_at > self.barrier_timeout_s:
            current.status = "failed"
            current.error = "Barrier wait timeout"
            return
        expired = [
            user_id
            for user_id, registration in current.registered_devices.items()
            if now - registration.last_heartbeat > self.heartbeat_timeout_s
        ]
        if expired:
            current.status = "failed"
            current.error = f"Heartbeat timeout for users {sorted(expired)}"

    def _start_optimization_locked(self, current: SchedulingRound) -> tuple[str, list[dict]]:
        current.status = "optimizing"
        users = [
            {
                **copy.deepcopy(current.registered_devices[user_id].device),
                "user_id": user_id,
            }
            for user_id in sorted(current.registered_devices)
        ]
        return current.round_id, users

    def _optimize_round(self, round_id: str, users: list[dict]) -> None:
        try:
            edge, cloud = self.node_state_provider()
            state = {
                "round_id": round_id,
                "bundle_id": users[0]["bundle_id"],
                "resource_mode": "fixed_worker_pool",
                "users": users,
                "edge": copy.deepcopy(edge),
                "cloud": copy.deepcopy(cloud),
            }
            self._validate_batch_state(state)
            decision = self.service.make_decision(state)
            per_user_decisions = {
                int(user["user_id"]): {
                    "round_id": round_id,
                    "decision_id": str(decision["decision_id"]),
                    "decision_version": 1,
                    "bundle_id": decision["bundle_id"],
                    "manifest_id": decision.get("manifest_id"),
                    "model_hash": decision.get("model_hash"),
                    "resource_mode": decision.get("resource_mode"),
                    "decision_source": decision.get("decision_source"),
                    "objective": decision.get("objective"),
                    "user": copy.deepcopy(user),
                }
                for user in decision["users"]
            }
            if set(per_user_decisions) != {int(user["user_id"]) for user in users}:
                raise RoundCoordinatorError(
                    "Batch decision does not cover every registered user"
                )
        except Exception as exc:
            with self._lock:
                current = self._require_round(round_id)
                current.status = "failed"
                current.error = str(exc)
            raise
        with self._lock:
            current = self._require_round(round_id)
            current.decision_version = 1
            current.batch_decision = copy.deepcopy(decision)
            current.per_user_decisions = copy.deepcopy(per_user_decisions)
            current.status = "ready"

    @staticmethod
    def _parse_registration(payload: dict) -> tuple[int, dict]:
        if not isinstance(payload, dict):
            raise RoundCoordinatorError("Registration payload must be an object")
        if "user_id" not in payload or "device" not in payload:
            raise RoundCoordinatorError("Registration requires user_id and device")
        user_id = int(payload["user_id"])
        if user_id < 0:
            raise RoundCoordinatorError("user_id must be non-negative")
        device = copy.deepcopy(payload["device"])
        if not isinstance(device, dict):
            raise RoundCoordinatorError("device must be an object")
        bundle_id = payload.get("bundle_id")
        if not bundle_id:
            raise RoundCoordinatorError("Registration requires bundle_id")
        if payload.get("resource_mode") != "fixed_worker_pool":
            raise RoundCoordinatorError("v2 rounds require fixed_worker_pool")
        device["bundle_id"] = str(bundle_id)
        device["resource_mode"] = "fixed_worker_pool"
        required = {
            "manifest_id",
            "model_hash",
            "execution_profile_id",
            "backend",
            "worker_count",
            "threads_per_worker",
            "BW_d2e",
        }
        missing = sorted(required - set(device))
        if missing:
            raise RoundCoordinatorError(f"device missing fields: {missing}")
        return user_id, device

    @staticmethod
    def _validate_batch_state(state: dict) -> None:
        owners = [*state["users"], state["edge"], state["cloud"]]
        bundle_ids = {owner.get("bundle_id") for owner in owners}
        manifest_ids = {owner.get("manifest_id") for owner in owners}
        model_hashes = {owner.get("model_hash") for owner in owners}
        if bundle_ids != {state["bundle_id"]}:
            raise RoundCoordinatorError("All nodes must use the same bundle_id")
        if len(manifest_ids) != 1 or None in manifest_ids:
            raise RoundCoordinatorError("All nodes must use the same manifest_id")
        if len(model_hashes) != 1 or None in model_hashes:
            raise RoundCoordinatorError("All nodes must use the same model_hash")
        if "BW_e2c" not in state["cloud"]:
            raise RoundCoordinatorError("cloud state requires BW_e2c")

    @staticmethod
    def _validate_measurement_payload(
        current: SchedulingRound, payload: dict
    ) -> None:
        if not isinstance(payload, dict):
            raise RoundCoordinatorError("Measurement payload must be an object")
        if payload.get("decision_id") != current.batch_decision["decision_id"]:
            raise RoundCoordinatorError("decision_id does not match the round")
        if int(payload.get("decision_version", -1)) != current.decision_version:
            raise RoundCoordinatorError("decision_version does not match the round")
        measurements = payload.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            raise RoundCoordinatorError("measurements must be a non-empty list")

    @staticmethod
    def _normalize_user_measurements(user_id: int, payload: dict) -> dict:
        seen: set[str] = set()
        records = []
        for record in payload["measurements"]:
            if not isinstance(record, dict):
                raise RoundCoordinatorError("Every measurement must be an object")
            request_id = str(record.get("request_id", "")).strip()
            if not request_id:
                raise RoundCoordinatorError("Every measurement requires request_id")
            if request_id in seen:
                raise RoundCoordinatorError(f"Duplicate request_id {request_id!r}")
            if "T_total" not in record or "is_correct" not in record:
                raise RoundCoordinatorError(
                    "Every measurement requires T_total and is_correct"
                )
            t_total = float(record["T_total"])
            accuracy = float(record["is_correct"])
            if not math.isfinite(t_total) or t_total < 0:
                raise RoundCoordinatorError("T_total must be non-negative")
            if not (0.0 <= accuracy <= 1.0):
                raise RoundCoordinatorError("is_correct/accuracy must be in [0, 1]")
            seen.add(request_id)
            records.append(
                {
                    **copy.deepcopy(record),
                    "request_id": request_id,
                    "T_total": t_total,
                    "is_correct": accuracy,
                    "user_id": user_id,
                }
            )
        return {
            "decision_id": str(payload["decision_id"]),
            "decision_version": int(payload["decision_version"]),
            "measurements": records,
        }

    def _complete_locked(self, current: SchedulingRound) -> None:
        per_user_records = []
        for user_id in sorted(current.measurements):
            records = current.measurements[user_id]["measurements"]
            per_user_records.append(
                {
                    "user_id": user_id,
                    "T_total": sum(float(item["T_total"]) for item in records)
                    / len(records),
                    "is_correct": sum(float(item["is_correct"]) for item in records)
                    / len(records),
                }
            )
        try:
            self.service.report_measurements(
                {
                    "decision_id": current.batch_decision["decision_id"],
                    "measurements": per_user_records,
                }
            )
            current.status = "completed"
        except Exception as exc:
            current.status = "failed"
            current.error = f"Failed to report round measurements: {exc}"
            raise

    @staticmethod
    def _status_locked(current: SchedulingRound) -> dict:
        return {
            "round_id": current.round_id,
            "status": current.status,
            "expected_users": current.expected_users,
            "registered_users": sorted(current.registered_devices),
            "registered_count": len(current.registered_devices),
            "decision_id": (
                current.batch_decision.get("decision_id")
                if current.batch_decision is not None
                else None
            ),
            "decision_version": current.decision_version,
            "measurement_users": sorted(current.measurements),
            "error": current.error,
        }
