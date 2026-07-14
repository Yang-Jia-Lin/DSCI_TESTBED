"""Deterministic shared-resource model for synchronized inference batches.

The model intentionally mirrors the runtime primitives: links use processor
sharing with optional per-flow caps, while Edge/Cloud use fixed-size FCFS
worker pools.  Work values may already be probability weighted, which makes
the model a fast fluid approximation for early-exit routing.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np


@dataclass(frozen=True)
class LinkJob:
    arrival_s: float
    work_megabits: float
    max_rate_mbps: float


@dataclass(frozen=True)
class WorkerJob:
    arrival_s: float
    service_s: float


class LinkResource:
    """A work-conserving processor-sharing link."""

    def __init__(self, capacity_mbps: float):
        self.capacity_mbps = float(capacity_mbps)
        if not math.isfinite(self.capacity_mbps) or self.capacity_mbps <= 0:
            raise ValueError("link capacity_mbps must be positive and finite")

    @staticmethod
    def _fair_rates(active: list[int], jobs: list[LinkJob], capacity: float) -> dict[int, float]:
        remaining = set(active)
        rates: dict[int, float] = {}
        capacity_left = float(capacity)
        while remaining:
            share = capacity_left / len(remaining)
            capped = [idx for idx in remaining if jobs[idx].max_rate_mbps <= share]
            if not capped:
                rates.update({idx: share for idx in remaining})
                break
            for idx in capped:
                rate = max(0.0, float(jobs[idx].max_rate_mbps))
                rates[idx] = rate
                capacity_left -= rate
                remaining.remove(idx)
        return rates

    def schedule(self, jobs: list[LinkJob]) -> np.ndarray:
        count = len(jobs)
        completion = np.full(count, np.nan, dtype=np.float64)
        remaining = np.array([max(0.0, float(job.work_megabits)) for job in jobs])
        pending = sorted(range(count), key=lambda idx: (jobs[idx].arrival_s, idx))
        cursor = 0
        active: list[int] = []
        now = float(pending and jobs[pending[0]].arrival_s or 0.0)
        eps = 1e-12

        while cursor < count or active:
            if not active and cursor < count:
                now = max(now, float(jobs[pending[cursor]].arrival_s))
            while cursor < count and float(jobs[pending[cursor]].arrival_s) <= now + eps:
                idx = pending[cursor]
                cursor += 1
                if remaining[idx] <= eps:
                    completion[idx] = now
                else:
                    if jobs[idx].max_rate_mbps <= 0:
                        raise ValueError("link job max_rate_mbps must be positive")
                    active.append(idx)
            if not active:
                continue

            rates = self._fair_rates(active, jobs, self.capacity_mbps)
            finish_after = min(remaining[idx] / rates[idx] for idx in active)
            next_arrival = (
                float(jobs[pending[cursor]].arrival_s) if cursor < count else math.inf
            )
            delta = min(finish_after, max(0.0, next_arrival - now))
            if delta <= eps and next_arrival < math.inf:
                now = next_arrival
                continue
            for idx in active:
                remaining[idx] = max(0.0, remaining[idx] - rates[idx] * delta)
            now += delta
            finished = [idx for idx in active if remaining[idx] <= eps]
            for idx in finished:
                completion[idx] = now
                active.remove(idx)

        if np.isnan(completion).any():
            raise RuntimeError("link scheduling failed to complete every job")
        return completion


class WorkerPoolResource:
    """A fixed-size FCFS worker pool."""

    def __init__(self, worker_count: int):
        self.worker_count = int(worker_count)
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")

    def schedule(self, jobs: list[WorkerJob]) -> tuple[np.ndarray, np.ndarray]:
        starts = np.zeros(len(jobs), dtype=np.float64)
        completion = np.zeros(len(jobs), dtype=np.float64)
        available = [0.0] * self.worker_count
        heapq.heapify(available)
        for idx in sorted(range(len(jobs)), key=lambda value: (jobs[value].arrival_s, value)):
            worker_ready = heapq.heappop(available)
            start = max(float(jobs[idx].arrival_s), worker_ready)
            finish = start + max(0.0, float(jobs[idx].service_s))
            starts[idx] = start
            completion[idx] = finish
            heapq.heappush(available, finish)
        return starts, completion


@dataclass(frozen=True)
class SharedResourceBreakdown:
    device_compute: np.ndarray
    d2e_transfer: np.ndarray
    edge_queue: np.ndarray
    edge_compute: np.ndarray
    e2c_transfer: np.ndarray
    cloud_queue: np.ndarray
    cloud_compute: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return (
            self.device_compute
            + self.d2e_transfer
            + self.edge_queue
            + self.edge_compute
            + self.e2c_transfer
            + self.cloud_queue
            + self.cloud_compute
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "device_compute": self.device_compute,
            "d2e_transfer": self.d2e_transfer,
            "edge_queue": self.edge_queue,
            "edge_compute": self.edge_compute,
            "e2c_transfer": self.e2c_transfer,
            "cloud_queue": self.cloud_queue,
            "cloud_compute": self.cloud_compute,
            "total": self.total,
        }


class SharedResourceModel:
    """Evaluate one synchronized multi-user batch across links and workers."""

    @staticmethod
    def _schedule_link_groups(
        arrivals: np.ndarray,
        work_megabits: np.ndarray,
        max_rates: np.ndarray,
        link_ids: list[str],
        capacities: dict[str, float],
    ) -> np.ndarray:
        completion = np.asarray(arrivals, dtype=np.float64).copy()
        for link_id in sorted(set(link_ids)):
            indices = [idx for idx, value in enumerate(link_ids) if value == link_id]
            resource = LinkResource(float(capacities[link_id]))
            group_jobs = [
                LinkJob(
                    arrival_s=float(arrivals[idx]),
                    work_megabits=float(work_megabits[idx]),
                    max_rate_mbps=float(max_rates[idx]),
                )
                for idx in indices
            ]
            group_completion = resource.schedule(group_jobs)
            for local_idx, global_idx in enumerate(indices):
                completion[global_idx] = group_completion[local_idx]
        return completion

    @staticmethod
    def _schedule_workers(
        arrivals: np.ndarray,
        services: np.ndarray,
        worker_count: int,
        active: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        starts = np.asarray(arrivals, dtype=np.float64).copy()
        completion = starts.copy()
        indices = [idx for idx, enabled in enumerate(active) if bool(enabled)]
        if not indices:
            return starts, completion
        local_starts, local_completion = WorkerPoolResource(worker_count).schedule(
            [
                WorkerJob(float(arrivals[idx]), float(services[idx]))
                for idx in indices
            ]
        )
        for local_idx, global_idx in enumerate(indices):
            starts[global_idx] = local_starts[local_idx]
            completion[global_idx] = local_completion[local_idx]
        return starts, completion

    def evaluate(
        self,
        *,
        device_compute: np.ndarray,
        d2e_work_megabits: np.ndarray,
        d2e_max_rates_mbps: np.ndarray,
        d2e_link_ids: list[str],
        d2e_capacities_mbps: dict[str, float],
        d2e_overhead_s: np.ndarray,
        edge_service_s: np.ndarray,
        edge_worker_count: int,
        e2c_work_megabits: np.ndarray,
        e2c_max_rates_mbps: np.ndarray,
        e2c_link_ids: list[str],
        e2c_capacities_mbps: dict[str, float],
        e2c_overhead_s: np.ndarray,
        cloud_service_s: np.ndarray,
        cloud_worker_count: int,
    ) -> SharedResourceBreakdown:
        device_compute = np.asarray(device_compute, dtype=np.float64)
        d2e_done_raw = self._schedule_link_groups(
            device_compute,
            np.asarray(d2e_work_megabits, dtype=np.float64),
            np.asarray(d2e_max_rates_mbps, dtype=np.float64),
            list(d2e_link_ids),
            dict(d2e_capacities_mbps),
        )
        d2e_overhead_s = np.asarray(d2e_overhead_s, dtype=np.float64)
        edge_arrival = d2e_done_raw + d2e_overhead_s
        edge_service_s = np.asarray(edge_service_s, dtype=np.float64)
        edge_active = (
            np.asarray(d2e_work_megabits, dtype=np.float64) > 1e-12
        ) | (edge_service_s > 1e-12) | (
            np.asarray(e2c_work_megabits, dtype=np.float64) > 1e-12
        )
        edge_start, edge_done = self._schedule_workers(
            edge_arrival, edge_service_s, edge_worker_count, edge_active
        )

        e2c_done_raw = self._schedule_link_groups(
            edge_done,
            np.asarray(e2c_work_megabits, dtype=np.float64),
            np.asarray(e2c_max_rates_mbps, dtype=np.float64),
            list(e2c_link_ids),
            dict(e2c_capacities_mbps),
        )
        e2c_overhead_s = np.asarray(e2c_overhead_s, dtype=np.float64)
        cloud_arrival = e2c_done_raw + e2c_overhead_s
        cloud_service_s = np.asarray(cloud_service_s, dtype=np.float64)
        cloud_active = (
            np.asarray(e2c_work_megabits, dtype=np.float64) > 1e-12
        ) | (cloud_service_s > 1e-12)
        cloud_start, _cloud_done = self._schedule_workers(
            cloud_arrival, cloud_service_s, cloud_worker_count, cloud_active
        )
        return SharedResourceBreakdown(
            device_compute=device_compute,
            d2e_transfer=(d2e_done_raw - device_compute) + d2e_overhead_s,
            edge_queue=edge_start - edge_arrival,
            edge_compute=edge_service_s,
            e2c_transfer=(e2c_done_raw - edge_done) + e2c_overhead_s,
            cloud_queue=cloud_start - cloud_arrival,
            cloud_compute=cloud_service_s,
        )
