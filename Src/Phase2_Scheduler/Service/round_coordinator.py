"""Fixed-size multi-device round coordination for the v2 scheduler API."""

from __future__ import annotations

import copy
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
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
    decision_mode: str | None = None
    bandwidth_observed_at: float | None = None
    bandwidth_sample: dict | None = None


@dataclass
class RequestBarrier:
    request_seq: int
    ready_users: set[int] = field(default_factory=set)
    ready_at: dict[int, float] = field(default_factory=dict)
    release_at: float | None = None
    decision_version: int | None = None
    bandwidth_calibration_started_at: float | None = None
    bandwidth_calibration_elapsed_s: float = 0.0


@dataclass
class BandwidthLease:
    user_id: int
    token: str
    issued_at: float
    expires_at: float
    attempt: int
    request_seq: int | None = None


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
    request_barriers: dict[int, RequestBarrier] = field(default_factory=dict)
    active_request_seq: int | None = None
    dynamic_bandwidth: bool = False
    pending_batch_decision: dict | None = None
    pending_per_user_decisions: dict[int, dict] = field(default_factory=dict)
    decision_history: dict[int, dict] = field(default_factory=dict)
    last_scheduled_bandwidth: dict[str, float] = field(default_factory=dict)
    last_optimization_at: float | None = None
    reoptimization_running: bool = False
    reoptimization_dirty: bool = False
    calibration_started_at: float | None = None
    calibration_queue: list[int] = field(default_factory=list)
    calibration_attempts: dict[int, int] = field(default_factory=dict)
    calibrated_users: set[int] = field(default_factory=set)
    active_bandwidth_lease: BandwidthLease | None = None
    periodic_calibration_user: int | None = None
    edge_bandwidth_sample: dict | None = None
    edge_bw_e2c: float | None = None
    edge_bandwidth_observed_at: float | None = None
    edge_calibration_in_progress: bool = False
    last_training_finished_at: float | None = None
    solution_refresh_pending: bool = False


class RoundCoordinator:
    """Coordinate one fixed-size scheduling round at a time."""

    def __init__(
        self,
        service: AlgoService,
        *,
        expected_users: int,
        node_state_provider: Callable[[], tuple[dict, dict]],
        edge_bandwidth_calibrator: Callable[[float], dict] | None = None,
        heartbeat_timeout_s: float = 15.0,
        barrier_timeout_s: float = 60.0,
        request_release_delay_s: float = 0.25,
        dynamic_bandwidth: bool = False,
        bandwidth_change_threshold: float = 0.20,
        bandwidth_min_reschedule_interval_s: float = 30.0,
        bandwidth_debounce_s: float = 2.0,
        bandwidth_stale_after_s: float = 300.0,
        iperf_calibration_duration_s: float = 3.0,
        iperf_calibration_timeout_s: float = 8.0,
        initial_calibration_timeout_s: float = 120.0,
        clock: Callable[[], float] = time.time,
    ):
        if expected_users <= 0:
            raise ValueError("expected_users must be positive")
        if float(bandwidth_change_threshold) <= 0.0:
            raise ValueError("bandwidth_change_threshold must be positive")
        if float(bandwidth_min_reschedule_interval_s) < 0.0:
            raise ValueError(
                "bandwidth_min_reschedule_interval_s must be non-negative"
            )
        if float(bandwidth_stale_after_s) <= 0.0:
            raise ValueError("bandwidth_stale_after_s must be positive")
        self.service = service
        self.expected_users = int(expected_users)
        self.node_state_provider = node_state_provider
        self.edge_bandwidth_calibrator = edge_bandwidth_calibrator
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.barrier_timeout_s = float(barrier_timeout_s)
        self.request_release_delay_s = float(request_release_delay_s)
        self.dynamic_bandwidth = bool(dynamic_bandwidth)
        self.bandwidth_change_threshold = float(bandwidth_change_threshold)
        self.bandwidth_min_reschedule_interval_s = float(
            bandwidth_min_reschedule_interval_s
        )
        self.bandwidth_debounce_s = float(bandwidth_debounce_s)
        self.bandwidth_stale_after_s = float(bandwidth_stale_after_s)
        self.iperf_calibration_duration_s = float(iperf_calibration_duration_s)
        self.iperf_calibration_timeout_s = float(iperf_calibration_timeout_s)
        self.initial_calibration_timeout_s = float(initial_calibration_timeout_s)
        self.clock = clock
        self._lock = threading.RLock()
        self._round: SchedulingRound | None = None
        self._reoptimization_timer: threading.Timer | None = None

    def register(self, round_id: str, payload: dict) -> dict:
        user_id, device, decision_mode, requested_dynamic = self._parse_registration(payload)
        now = self.clock()
        with self._lock:
            current = self._get_or_create_round(round_id, now)
            if not current.registered_devices:
                current.dynamic_bandwidth = self.dynamic_bandwidth
            if requested_dynamic != self.dynamic_bandwidth:
                raise RoundConflictError(
                    "Device and Scheduler dynamic_bandwidth modes must match"
                )
            self._expire_if_needed(current, now)
            if current.status != "waiting":
                raise RoundConflictError(
                    f"Round {round_id!r} is {current.status}; registration is closed"
                )

            existing = current.registered_devices.get(user_id)
            if existing is not None and (
                existing.device != device or existing.decision_mode != decision_mode
            ):
                raise RoundConflictError(
                    f"user_id {user_id} is already registered with different state"
                )
            current.registered_devices[user_id] = DeviceRegistration(
                user_id=user_id,
                device=copy.deepcopy(device),
                last_heartbeat=now,
                decision_mode=decision_mode,
            )
            if len(current.registered_devices) == current.expected_users:
                if current.dynamic_bandwidth:
                    current.status = "calibrating"
                    current.calibration_started_at = now
                    current.calibration_queue = sorted(current.registered_devices)
                    return self._status_locked(current)
                round_id, users = self._start_optimization_locked(current)
            else:
                return self._status_locked(current)
        self._optimize_round(round_id, users)
        with self._lock:
            return self._status_locked(self._require_round(round_id))

    def heartbeat(self, round_id: str, user_id: int) -> dict:
        now = self.clock()
        training_finished_at = None
        if self.dynamic_bandwidth:
            try:
                training_finished_at = self.service.health().get(
                    "last_training_finished_at"
                )
            except Exception:
                training_finished_at = None
        with self._lock:
            current = self._require_round(round_id)
            self._expire_if_needed(current, now)
            registration = current.registered_devices.get(int(user_id))
            if registration is None:
                raise RoundCoordinatorError(f"user_id {user_id} is not registered")
            if current.status in {"completed", "failed"}:
                raise RoundConflictError(f"Round {round_id!r} is {current.status}")
            registration.last_heartbeat = now
            if (
                training_finished_at is not None
                and (
                    current.last_training_finished_at is None
                    or float(training_finished_at)
                    > current.last_training_finished_at
                )
            ):
                current.last_training_finished_at = float(training_finished_at)
                current.solution_refresh_pending = True
                self._schedule_reoptimization_locked(current, now, force=True)
            return self._status_locked(current)

    def decision_for_user(self, round_id: str, user_id: int) -> dict | None:
        now = self.clock()
        optimization: tuple[str, list[dict]] | None = None
        with self._lock:
            current = self._require_round(round_id)
            self._expire_if_needed(current, now)
            if int(user_id) not in current.registered_devices:
                raise RoundCoordinatorError(f"user_id {user_id} is not registered")
            if current.status == "failed":
                raise RoundConflictError(current.error or "Round failed")
            if self._initial_calibration_complete_locked(current, now):
                optimization = self._start_optimization_locked(current)
            decision = current.per_user_decisions.get(int(user_id))
        if optimization is not None:
            self._optimize_round(*optimization, version=1, pending=False)
            with self._lock:
                current = self._require_round(round_id)
                decision = current.per_user_decisions.get(int(user_id))
        return copy.deepcopy(decision) if decision is not None else None

    def ready_request(self, round_id: str, user_id: int, request_seq: int) -> dict:
        """Join a per-request barrier and return its common release timestamp."""
        now = self.clock()
        run_edge_calibration = False
        with self._lock:
            current = self._require_round(round_id)
            self._expire_bandwidth_lease_locked(current, now)
            if current.status not in {"decision_ready", "request_barrier", "running"}:
                raise RoundConflictError(
                    f"Round {round_id!r} is {current.status}; request barriers are closed"
                )
            user_id = int(user_id)
            request_seq = int(request_seq)
            if user_id not in current.registered_devices:
                raise RoundCoordinatorError(f"user_id {user_id} is not registered")
            if request_seq < 0:
                raise RoundCoordinatorError("request_seq must be non-negative")
            expected_seq = 0 if current.active_request_seq is None else current.active_request_seq + 1
            if request_seq > expected_seq:
                raise RoundConflictError(
                    f"request_seq {request_seq} is ahead of expected {expected_seq}"
                )
            barrier = current.request_barriers.setdefault(
                request_seq, RequestBarrier(request_seq=request_seq)
            )
            barrier.ready_users.add(user_id)
            barrier.ready_at.setdefault(user_id, now)
            if barrier.release_at is None and len(barrier.ready_users) == current.expected_users:
                calibration = self._prepare_periodic_calibration_locked(
                    current, barrier, now
                )
                if calibration is not None:
                    current.status = "request_barrier"
                    run_edge_calibration = calibration == "edge_start"
                else:
                    self._release_barrier_locked(current, barrier, now)
            elif barrier.release_at is None:
                current.status = "request_barrier"
            result = self._barrier_status(current, barrier, user_id=user_id)
        if run_edge_calibration:
            self._run_edge_calibration(round_id, request_seq)
            with self._lock:
                current = self._require_round(round_id)
                barrier = current.request_barriers[request_seq]
                result = self._barrier_status(current, barrier, user_id=user_id)
        return result

    def acquire_bandwidth_lease(self, round_id: str, user_id: int) -> dict:
        now = self.clock()
        with self._lock:
            current = self._require_round(round_id)
            user_id = int(user_id)
            if not current.dynamic_bandwidth:
                raise RoundConflictError("Dynamic bandwidth mode is not enabled")
            if user_id not in current.registered_devices:
                raise RoundCoordinatorError(f"user_id {user_id} is not registered")
            self._expire_bandwidth_lease_locked(current, now)
            lease = current.active_bandwidth_lease
            if lease is not None:
                if lease.user_id == user_id:
                    return self._lease_status(lease, granted=True)
                return {
                    "status": "waiting",
                    "active_user_id": lease.user_id,
                    "retry_after_s": 0.5,
                }

            request_seq = None
            eligible_user = None
            if current.status == "calibrating" and current.calibration_queue:
                eligible_user = current.calibration_queue[0]
            elif current.periodic_calibration_user is not None:
                eligible_user = current.periodic_calibration_user
                request_seq = current.active_request_seq
                for seq, barrier in current.request_barriers.items():
                    if (
                        barrier.release_at is None
                        and barrier.bandwidth_calibration_started_at is not None
                    ):
                        request_seq = seq
                        break
            if eligible_user != user_id:
                return {"status": "idle", "retry_after_s": 0.5}

            attempt = current.calibration_attempts.get(user_id, 0) + 1
            current.calibration_attempts[user_id] = attempt
            lease = BandwidthLease(
                user_id=user_id,
                token=uuid.uuid4().hex,
                issued_at=now,
                expires_at=now + self.iperf_calibration_timeout_s,
                attempt=attempt,
                request_seq=request_seq,
            )
            current.active_bandwidth_lease = lease
            return self._lease_status(lease, granted=True)

    def report_device_bandwidth(
        self, round_id: str, user_id: int, payload: dict
    ) -> dict:
        now = self.clock()
        optimization: tuple[str, list[dict]] | None = None
        with self._lock:
            current = self._require_round(round_id)
            user_id = int(user_id)
            registration = current.registered_devices.get(user_id)
            if registration is None:
                raise RoundCoordinatorError(f"user_id {user_id} is not registered")
            sample = self._normalize_bandwidth_sample(payload, expected_link="d2e", now=now)
            source = sample["source"]
            if source == "iperf":
                self._validate_lease_token_locked(current, user_id, payload)
            if sample.get("status") == "failed":
                self._finish_bandwidth_lease_locked(current, success=False, now=now)
            else:
                effective = float(sample["filtered_bw_mbps"])
                registration.device["BW_d2e"] = effective
                registration.bandwidth_observed_at = now
                registration.bandwidth_sample = copy.deepcopy(sample)
                if source == "iperf":
                    self._finish_bandwidth_lease_locked(
                        current, success=True, now=now
                    )

            if self._initial_calibration_complete_locked(current, now):
                optimization = self._start_optimization_locked(current)
            elif current.decision_version > 0:
                self._schedule_reoptimization_locked(current, now)
            status = self._status_locked(current)
        if optimization is not None:
            self._optimize_round(*optimization, version=1, pending=False)
            with self._lock:
                status = self._status_locked(self._require_round(round_id))
        return status

    def report_edge_bandwidth(self, round_id: str, payload: dict) -> dict:
        now = self.clock()
        with self._lock:
            current = self._require_round(round_id)
            sample = self._normalize_bandwidth_sample(payload, expected_link="e2c", now=now)
            if sample.get("status") == "failed":
                return self._status_locked(current)
            current.edge_bandwidth_sample = copy.deepcopy(sample)
            current.edge_bw_e2c = float(sample["filtered_bw_mbps"])
            current.edge_bandwidth_observed_at = now
            if current.decision_version > 0:
                self._schedule_reoptimization_locked(current, now)
            return self._status_locked(current)

    def submit_measurements(self, round_id: str, user_id: int, payload: dict) -> dict:
        with self._lock:
            current = self._require_round(round_id)
            if current.status not in {
                "ready",
                "decision_ready",
                "request_barrier",
                "running",
                "completed",
            }:
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
                    and current.status != "completed"
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
                "decision_mode": current.registered_devices[user_id].decision_mode,
            }
            for user_id in sorted(current.registered_devices)
        ]
        return current.round_id, users

    def _optimize_round(
        self,
        round_id: str,
        users: list[dict],
        *,
        version: int = 1,
        pending: bool = False,
    ) -> None:
        try:
            edge, cloud = self.node_state_provider()
            with self._lock:
                current = self._require_round(round_id)
                decision_id = (
                    f"{round_id}:v{int(version)}"
                    if current.dynamic_bandwidth
                    else round_id
                )
                if current.edge_bw_e2c is None and edge.get("BW_e2c") is not None:
                    current.edge_bw_e2c = float(edge["BW_e2c"])
                    bandwidth_state = edge.get("bandwidth_e2c") or {}
                    current.edge_bandwidth_observed_at = (
                        float(bandwidth_state["iperf_observed_at"])
                        if bandwidth_state.get("iperf_observed_at") is not None
                        else self.clock()
                    )
                if current.edge_bw_e2c is not None:
                    edge["BW_e2c"] = float(current.edge_bw_e2c)
                    cloud["BW_e2c"] = float(current.edge_bw_e2c)
            state = {
                "round_id": decision_id,
                "bundle_id": users[0]["bundle_id"],
                "resource_mode": "fixed_worker_pool",
                "shared_resource_model": len(users) > 1,
                "users": users,
                "edge": copy.deepcopy(edge),
                "cloud": copy.deepcopy(cloud),
            }
            decision_modes = {
                str(user.get("decision_mode"))
                for user in users
                if user.get("decision_mode")
            }
            if len(decision_modes) > 1:
                raise RoundCoordinatorError(
                    f"All users in one round must use the same decision_mode: {sorted(decision_modes)}"
                )
            if decision_modes:
                state["decision_mode"] = decision_modes.pop()
            self._validate_batch_state(state)
            decision = self.service.make_decision(state)
            decision["round_id"] = round_id
            decision["decision_id"] = decision_id
            decision["decision_version"] = int(version)
            shared_decision_fields = {
                key: decision.get(key)
                for key in (
                    "objective_mode",
                    "objective_alpha",
                    "objective_beta",
                    "target_accuracy",
                    "accuracy_tolerance",
                    "expected_accuracy",
                    "expected_latency",
                    "constraint_status",
                    "constraint_satisfied",
                )
            }
            per_user_decisions = {
                int(user["user_id"]): {
                    "round_id": round_id,
                    "decision_id": str(decision["decision_id"]),
                    "decision_version": int(version),
                    "bundle_id": decision["bundle_id"],
                    "manifest_id": decision.get("manifest_id"),
                    "model_hash": decision.get("model_hash"),
                    "resource_mode": decision.get("resource_mode"),
                    "decision_source": decision.get("decision_source"),
                    "objective": decision.get("objective"),
                    **shared_decision_fields,
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
                current.reoptimization_running = False
                if pending and current.decision_version > 0:
                    current.error = f"Dynamic reoptimization failed: {exc}"
                else:
                    current.status = "failed"
                    current.error = str(exc)
            raise
        with self._lock:
            current = self._require_round(round_id)
            current.reoptimization_running = False
            if pending:
                current.pending_batch_decision = copy.deepcopy(decision)
                current.pending_per_user_decisions = copy.deepcopy(per_user_decisions)
                current.status = (
                    "request_barrier"
                    if current.status == "request_barrier"
                    else "decision_ready"
                )
                current.last_optimization_at = self.clock()
                if current.reoptimization_dirty:
                    current.reoptimization_dirty = False
                    self._reoptimization_timer = threading.Timer(
                        max(
                            self.bandwidth_debounce_s,
                            self.bandwidth_min_reschedule_interval_s,
                        ),
                        self._run_reoptimization,
                        args=(round_id,),
                    )
                    self._reoptimization_timer.daemon = True
                    self._reoptimization_timer.start()
            else:
                current.decision_version = int(version)
                current.batch_decision = copy.deepcopy(decision)
                current.per_user_decisions = copy.deepcopy(per_user_decisions)
                current.decision_history[int(version)] = copy.deepcopy(decision)
                current.last_scheduled_bandwidth = self._bandwidth_vector_locked(current)
                current.last_optimization_at = self.clock()
                current.status = "decision_ready"

    def _initial_calibration_complete_locked(
        self, current: SchedulingRound, now: float
    ) -> bool:
        if not current.dynamic_bandwidth or current.status != "calibrating":
            return False
        all_done = len(current.calibrated_users) >= current.expected_users
        timed_out = (
            current.calibration_started_at is not None
            and now - current.calibration_started_at
            >= self.initial_calibration_timeout_s
        )
        if timed_out:
            current.active_bandwidth_lease = None
            current.calibration_queue.clear()
        return all_done or timed_out

    @staticmethod
    def _lease_status(lease: BandwidthLease, *, granted: bool) -> dict:
        return {
            "status": "granted" if granted else "waiting",
            "lease_token": lease.token,
            "user_id": lease.user_id,
            "attempt": lease.attempt,
            "issued_at": lease.issued_at,
            "expires_at": lease.expires_at,
            "request_seq": lease.request_seq,
        }

    def _validate_lease_token_locked(
        self, current: SchedulingRound, user_id: int, payload: dict
    ) -> None:
        lease = current.active_bandwidth_lease
        token = str(payload.get("lease_token") or "")
        if lease is None or lease.user_id != user_id or lease.token != token:
            raise RoundConflictError("Invalid or expired bandwidth lease")

    def _finish_bandwidth_lease_locked(
        self, current: SchedulingRound, *, success: bool, now: float
    ) -> None:
        lease = current.active_bandwidth_lease
        if lease is None:
            return
        user_id = lease.user_id
        current.active_bandwidth_lease = None
        if current.status == "calibrating":
            if current.calibration_queue and current.calibration_queue[0] == user_id:
                current.calibration_queue.pop(0)
            if success or lease.attempt >= 3:
                current.calibrated_users.add(user_id)
            else:
                current.calibration_queue.append(user_id)
        elif current.periodic_calibration_user == user_id:
            current.periodic_calibration_user = None
            if lease.request_seq is not None:
                barrier = current.request_barriers.get(lease.request_seq)
                if barrier is not None and barrier.release_at is None:
                    barrier.bandwidth_calibration_elapsed_s = max(
                        0.0,
                        now - float(barrier.bandwidth_calibration_started_at or now),
                    )
                    self._release_barrier_locked(current, barrier, now)

    def _expire_bandwidth_lease_locked(
        self, current: SchedulingRound, now: float
    ) -> None:
        lease = current.active_bandwidth_lease
        if lease is not None and now >= lease.expires_at:
            self._finish_bandwidth_lease_locked(current, success=False, now=now)

    def _prepare_periodic_calibration_locked(
        self, current: SchedulingRound, barrier: RequestBarrier, now: float
    ) -> str | None:
        if not current.dynamic_bandwidth:
            return None
        if current.periodic_calibration_user is not None:
            return "wait"
        if current.edge_calibration_in_progress:
            return "wait"
        stale_users = [
            user_id
            for user_id, registration in sorted(current.registered_devices.items())
            if registration.bandwidth_observed_at is None
            or now - registration.bandwidth_observed_at >= self.bandwidth_stale_after_s
        ]
        if not stale_users:
            edge_stale = (
                current.edge_bandwidth_observed_at is None
                or now - current.edge_bandwidth_observed_at
                >= self.bandwidth_stale_after_s
            )
            if edge_stale and self.edge_bandwidth_calibrator is not None:
                current.edge_calibration_in_progress = True
                barrier.bandwidth_calibration_started_at = now
                return "edge_start"
            return None
        current.periodic_calibration_user = stale_users[0]
        current.calibration_attempts[stale_users[0]] = 0
        barrier.bandwidth_calibration_started_at = now
        return "device"

    def _run_edge_calibration(self, round_id: str, request_seq: int) -> None:
        sample = None
        try:
            sample = self.edge_bandwidth_calibrator(
                self.iperf_calibration_duration_s
            )
        except Exception as exc:
            sample = {"status": "failed", "message": str(exc)}
        now = self.clock()
        with self._lock:
            current = self._require_round(round_id)
            barrier = current.request_barriers[request_seq]
            current.edge_calibration_in_progress = False
            if isinstance(sample, dict) and sample.get("status") != "failed":
                normalized = self._normalize_bandwidth_sample(
                    sample, expected_link="e2c", now=now
                )
                current.edge_bandwidth_sample = copy.deepcopy(normalized)
                current.edge_bw_e2c = float(normalized["filtered_bw_mbps"])
                current.edge_bandwidth_observed_at = now
                self._schedule_reoptimization_locked(current, now)
            barrier.bandwidth_calibration_elapsed_s = max(
                0.0, now - float(barrier.bandwidth_calibration_started_at or now)
            )
            if barrier.release_at is None:
                self._release_barrier_locked(current, barrier, now)

    def _release_barrier_locked(
        self, current: SchedulingRound, barrier: RequestBarrier, now: float
    ) -> None:
        if current.pending_batch_decision is not None:
            current.decision_version = int(
                current.pending_batch_decision["decision_version"]
            )
            current.batch_decision = current.pending_batch_decision
            current.per_user_decisions = current.pending_per_user_decisions
            current.decision_history[current.decision_version] = copy.deepcopy(
                current.batch_decision
            )
            current.pending_batch_decision = None
            current.pending_per_user_decisions = {}
            current.last_scheduled_bandwidth = self._bandwidth_vector_locked(current)
            current.last_optimization_at = now
        barrier.decision_version = current.decision_version
        barrier.release_at = now + self.request_release_delay_s
        current.active_request_seq = barrier.request_seq
        current.status = "running"

    @staticmethod
    def _normalize_bandwidth_sample(
        payload: dict, *, expected_link: str, now: float
    ) -> dict:
        if not isinstance(payload, dict):
            raise RoundCoordinatorError("Bandwidth sample must be an object")
        source = str(payload.get("source") or "").strip().lower()
        if source not in {"passive", "iperf"}:
            raise RoundCoordinatorError("Bandwidth source must be passive or iperf")
        link = str(payload.get("link") or expected_link).strip().lower()
        if link != expected_link:
            raise RoundCoordinatorError(f"Expected {expected_link} bandwidth sample")
        if str(payload.get("status") or "ok").lower() == "failed":
            return {
                **copy.deepcopy(payload),
                "link": link,
                "source": source,
                "status": "failed",
                "received_at": float(now),
            }
        raw = float(payload.get("bw_mbps"))
        filtered = float(payload.get("filtered_bw_mbps", raw))
        if not math.isfinite(raw) or raw <= 0.0:
            raise RoundCoordinatorError("bw_mbps must be finite and positive")
        if not math.isfinite(filtered) or filtered <= 0.0:
            raise RoundCoordinatorError(
                "filtered_bw_mbps must be finite and positive"
            )
        return {
            **copy.deepcopy(payload),
            "sample_id": str(payload.get("sample_id") or uuid.uuid4().hex),
            "link": link,
            "source": source,
            "bw_mbps": raw,
            "filtered_bw_mbps": filtered,
            "payload_bytes": int(payload.get("payload_bytes") or 0),
            "elapsed_s": (
                float(payload["elapsed_s"])
                if payload.get("elapsed_s") is not None
                else None
            ),
            "decision_version": int(payload.get("decision_version") or 0),
            "received_at": float(now),
        }

    @staticmethod
    def _bandwidth_vector_locked(current: SchedulingRound) -> dict[str, float]:
        vector = {
            f"d2e:{user_id}": float(registration.device["BW_d2e"])
            for user_id, registration in current.registered_devices.items()
        }
        if current.edge_bw_e2c is not None:
            vector["e2c"] = float(current.edge_bw_e2c)
        elif current.batch_decision is not None:
            # The initial provider value is not part of the encoded decision.
            vector.setdefault("e2c", 0.0)
        return vector

    def _bandwidth_changed_locked(self, current: SchedulingRound) -> bool:
        latest = self._bandwidth_vector_locked(current)
        baseline = current.last_scheduled_bandwidth
        for key, value in latest.items():
            previous = baseline.get(key)
            if previous is None or previous <= 0.0:
                return True
            if abs(value - previous) / previous >= self.bandwidth_change_threshold:
                return True
        return False

    def _schedule_reoptimization_locked(
        self, current: SchedulingRound, now: float, *, force: bool = False
    ) -> None:
        if not current.dynamic_bandwidth or (
            not force and not self._bandwidth_changed_locked(current)
        ):
            return
        if current.reoptimization_running:
            current.reoptimization_dirty = True
            return
        delay = self.bandwidth_debounce_s
        if current.last_optimization_at is not None:
            delay = max(
                delay,
                current.last_optimization_at
                + self.bandwidth_min_reschedule_interval_s
                - now,
            )
        if self._reoptimization_timer is not None and self._reoptimization_timer.is_alive():
            current.reoptimization_dirty = True
            return
        round_id = current.round_id
        self._reoptimization_timer = threading.Timer(
            max(0.0, delay), self._run_reoptimization, args=(round_id,)
        )
        self._reoptimization_timer.daemon = True
        self._reoptimization_timer.start()

    def _run_reoptimization(self, round_id: str) -> None:
        with self._lock:
            try:
                current = self._require_round(round_id)
            except RoundCoordinatorError:
                return
            if current.status in {"completed", "failed", "waiting", "calibrating"}:
                return
            force = current.solution_refresh_pending
            if not force and not self._bandwidth_changed_locked(current):
                current.reoptimization_dirty = False
                return
            current.reoptimization_running = True
            current.reoptimization_dirty = False
            current.solution_refresh_pending = False
            users = [
                {
                    **copy.deepcopy(current.registered_devices[user_id].device),
                    "user_id": user_id,
                    "decision_mode": current.registered_devices[user_id].decision_mode,
                }
                for user_id in sorted(current.registered_devices)
            ]
            version = max(
                current.decision_version,
                int(
                    (current.pending_batch_decision or {}).get(
                        "decision_version", current.decision_version
                    )
                ),
            ) + 1
        try:
            self._optimize_round(round_id, users, version=version, pending=True)
        except Exception:
            # _optimize_round records the error while preserving the active decision.
            return

    @staticmethod
    def _parse_registration(payload: dict) -> tuple[int, dict, str | None, bool]:
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
        decision_mode = payload.get("decision_mode")
        if decision_mode is not None:
            decision_mode = str(decision_mode)
        return user_id, device, decision_mode, bool(payload.get("dynamic_bandwidth", False))

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
        measurements = payload.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            raise RoundCoordinatorError("measurements must be a non-empty list")
        if not current.dynamic_bandwidth:
            if payload.get("decision_id") != current.batch_decision["decision_id"]:
                raise RoundCoordinatorError("decision_id does not match the round")
            if int(payload.get("decision_version", -1)) != current.decision_version:
                raise RoundCoordinatorError("decision_version does not match the round")
            return
        for record in measurements:
            version = int(
                record.get("decision_version", payload.get("decision_version", -1))
            )
            decision = current.decision_history.get(version)
            if decision is None:
                raise RoundCoordinatorError(
                    f"Unknown decision_version {version} in measurement"
                )
            decision_id = record.get("decision_id", payload.get("decision_id"))
            if decision_id != decision.get("decision_id"):
                raise RoundCoordinatorError(
                    f"decision_id does not match decision_version {version}"
                )

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
                    "decision_id": str(
                        record.get("decision_id", payload.get("decision_id"))
                    ),
                    "decision_version": int(
                        record.get(
                            "decision_version", payload.get("decision_version", -1)
                        )
                    ),
                }
            )
        return {
            "decision_id": str(payload.get("decision_id") or "mixed"),
            "decision_version": int(payload.get("decision_version", -1)),
            "measurements": records,
        }

    def _complete_locked(self, current: SchedulingRound) -> None:
        try:
            versions = (
                sorted(current.decision_history)
                if current.dynamic_bandwidth
                else [current.decision_version]
            )
            for version in versions:
                decision = current.decision_history.get(version, current.batch_decision)
                per_user_records = []
                for user_id in sorted(current.measurements):
                    records = [
                        item
                        for item in current.measurements[user_id]["measurements"]
                        if int(item.get("decision_version", version)) == version
                    ]
                    if not records:
                        continue
                    per_user_records.append(
                        {
                            "user_id": user_id,
                            "T_total": sum(float(item["T_total"]) for item in records)
                            / len(records),
                            "is_correct": sum(
                                float(item["is_correct"]) for item in records
                            )
                            / len(records),
                        }
                    )
                if len(per_user_records) != current.expected_users:
                    continue
                self.service.report_measurements(
                    {
                        "decision_id": decision["decision_id"],
                        "objective_alpha": decision.get("objective_alpha"),
                        "objective_beta": decision.get("objective_beta"),
                        "measurements": per_user_records,
                    }
                )
            current.status = "completed"
            if self._reoptimization_timer is not None:
                self._reoptimization_timer.cancel()
        except Exception as exc:
            current.status = "failed"
            current.error = f"Failed to report round measurements: {exc}"
            raise

    @staticmethod
    def _status_locked(current: SchedulingRound) -> dict:
        active_barrier = (
            current.request_barriers.get(current.active_request_seq)
            if current.active_request_seq is not None
            else None
        )
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
            "dynamic_bandwidth": current.dynamic_bandwidth,
            "pending_decision_version": (
                current.pending_batch_decision.get("decision_version")
                if current.pending_batch_decision is not None
                else None
            ),
            "calibrated_users": sorted(current.calibrated_users),
            "calibration_queue": list(current.calibration_queue),
            "active_bandwidth_lease_user": (
                current.active_bandwidth_lease.user_id
                if current.active_bandwidth_lease is not None
                else None
            ),
            "measurement_users": sorted(current.measurements),
            "active_request_seq": current.active_request_seq,
            "request_release_at": (
                active_barrier.release_at if active_barrier is not None else None
            ),
            "error": current.error,
        }

    @staticmethod
    def _barrier_status(
        current: SchedulingRound,
        barrier: RequestBarrier,
        *,
        user_id: int | None = None,
    ) -> dict:
        payload = {
            "round_id": current.round_id,
            "status": "released" if barrier.release_at is not None else "waiting",
            "request_seq": barrier.request_seq,
            "ready_users": sorted(barrier.ready_users),
            "ready_at": {
                str(user_id): ready_at
                for user_id, ready_at in sorted(barrier.ready_at.items())
            },
            "expected_users": current.expected_users,
            "release_at": barrier.release_at,
            "decision_version": barrier.decision_version,
            "T_bandwidth_calibration": barrier.bandwidth_calibration_elapsed_s,
        }
        if (
            barrier.release_at is not None
            and user_id is not None
            and int(user_id) in current.per_user_decisions
        ):
            payload["decision"] = copy.deepcopy(
                current.per_user_decisions[int(user_id)]
            )
        return payload
