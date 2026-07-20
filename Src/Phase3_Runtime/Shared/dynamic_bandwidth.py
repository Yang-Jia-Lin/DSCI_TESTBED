"""Thread-safe bandwidth filtering for dynamic runtime scheduling."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class BandwidthSample:
    sample_id: str
    link: str
    source: str
    bw_mbps: float
    filtered_bw_mbps: float
    payload_bytes: int
    elapsed_s: float | None
    observed_at: float
    decision_version: int

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "link": self.link,
            "source": self.source,
            "bw_mbps": self.bw_mbps,
            "filtered_bw_mbps": self.filtered_bw_mbps,
            "payload_bytes": self.payload_bytes,
            "elapsed_s": self.elapsed_s,
            "observed_at": self.observed_at,
            "decision_version": self.decision_version,
        }


class BandwidthEstimator:
    """Keep separate passive and active estimates and expose one effective value."""

    def __init__(
        self,
        *,
        link: str,
        initial_mbps: float,
        alpha: float = 0.3,
        min_payload_bytes: int = 256 * 1024,
        passive_ready_samples: int = 3,
        stale_after_s: float = 300.0,
        clock=time.time,
    ):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if int(min_payload_bytes) < 0:
            raise ValueError("min_payload_bytes must be non-negative")
        if int(passive_ready_samples) <= 0:
            raise ValueError("passive_ready_samples must be positive")
        if float(stale_after_s) <= 0:
            raise ValueError("stale_after_s must be positive")
        self.link = str(link)
        self.alpha = float(alpha)
        self.min_payload_bytes = int(min_payload_bytes)
        self.passive_ready_samples = int(passive_ready_samples)
        self.stale_after_s = float(stale_after_s)
        self.clock = clock
        self._lock = threading.RLock()
        self._fallback_mbps = self._valid_rate(initial_mbps)
        self._passive_ewma: float | None = None
        self._passive_count = 0
        self._passive_at: float | None = None
        self._active_mbps: float | None = None
        self._active_at: float | None = None

    @staticmethod
    def _valid_rate(value: float) -> float:
        rate = float(value)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("bandwidth must be finite and positive")
        return rate

    def _is_fresh(self, observed_at: float | None, now: float) -> bool:
        return observed_at is not None and now - observed_at <= self.stale_after_s

    def effective_mbps(self, *, now: float | None = None) -> float:
        now = float(self.clock() if now is None else now)
        with self._lock:
            if (
                self._passive_count >= self.passive_ready_samples
                and self._passive_ewma is not None
                and self._is_fresh(self._passive_at, now)
            ):
                return float(self._passive_ewma)
            if self._active_mbps is not None and self._is_fresh(self._active_at, now):
                return float(self._active_mbps)
            return float(self._fallback_mbps)

    def needs_calibration(self, *, now: float | None = None) -> bool:
        now = float(self.clock() if now is None else now)
        with self._lock:
            passive_ready = (
                self._passive_count >= self.passive_ready_samples
                and self._is_fresh(self._passive_at, now)
            )
            active_ready = self._is_fresh(self._active_at, now)
            return not passive_ready and not active_ready

    def observe(
        self,
        bw_mbps: float,
        *,
        source: str,
        payload_bytes: int = 0,
        elapsed_s: float | None = None,
        decision_version: int = 0,
        observed_at: float | None = None,
        sample_id: str | None = None,
    ) -> BandwidthSample | None:
        rate = self._valid_rate(bw_mbps)
        source = str(source).strip().lower()
        if source not in {"passive", "iperf"}:
            raise ValueError("source must be passive or iperf")
        payload_bytes = int(payload_bytes or 0)
        if source == "passive" and payload_bytes < self.min_payload_bytes:
            return None
        if elapsed_s is not None:
            elapsed_s = float(elapsed_s)
            if not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
                raise ValueError("elapsed_s must be finite and positive")
        now = float(self.clock() if observed_at is None else observed_at)
        with self._lock:
            if source == "passive":
                if self._passive_ewma is None:
                    self._passive_ewma = rate
                else:
                    self._passive_ewma = (
                        self.alpha * rate + (1.0 - self.alpha) * self._passive_ewma
                    )
                self._passive_count += 1
                self._passive_at = now
            else:
                self._active_mbps = rate
                self._active_at = now
            filtered = self.effective_mbps(now=now)
        return BandwidthSample(
            sample_id=str(sample_id or uuid.uuid4().hex),
            link=self.link,
            source=source,
            bw_mbps=rate,
            filtered_bw_mbps=filtered,
            payload_bytes=payload_bytes,
            elapsed_s=elapsed_s,
            observed_at=now,
            decision_version=int(decision_version),
        )

    def snapshot(self, *, now: float | None = None) -> dict:
        now = float(self.clock() if now is None else now)
        with self._lock:
            return {
                "link": self.link,
                "effective_bw_mbps": self.effective_mbps(now=now),
                "passive_ewma_mbps": self._passive_ewma,
                "passive_samples": self._passive_count,
                "passive_observed_at": self._passive_at,
                "iperf_mbps": self._active_mbps,
                "iperf_observed_at": self._active_at,
                "needs_calibration": self.needs_calibration(now=now),
            }
