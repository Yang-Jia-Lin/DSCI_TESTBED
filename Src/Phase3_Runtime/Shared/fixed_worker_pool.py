"""Fixed-size process worker pool with fixed per-worker thread counts."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass


def _configure_worker(threads_per_worker: int, initializer, initargs):
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    try:
        import torch

        torch.set_num_threads(threads_per_worker)
    except ImportError:
        pass
    if initializer is not None:
        initializer(*initargs)


def _run_job(fn, args, kwargs):
    return fn(*args, **kwargs)


@dataclass(frozen=True)
class WorkerPoolConfig:
    worker_count: int
    threads_per_worker: int
    max_queue_size: int = 64

    def validate(self) -> None:
        cpu_count = os.cpu_count() or 1
        if self.worker_count <= 0 or self.threads_per_worker <= 0:
            raise ValueError("worker_count and threads_per_worker must be positive")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.worker_count * self.threads_per_worker > cpu_count:
            raise ValueError(
                "worker_count * threads_per_worker exceeds available logical CPUs"
            )


class FixedWorkerPool:
    def __init__(self, config: WorkerPoolConfig, fn, *, initializer=None, initargs=()):
        config.validate()
        self.config = config
        self._fn = fn
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(config.max_queue_size + config.worker_count)
        self._inflight = 0
        self._executor = ProcessPoolExecutor(
            max_workers=config.worker_count,
            initializer=_configure_worker,
            initargs=(config.threads_per_worker, initializer, initargs),
        )

    def submit(self, *args, **kwargs) -> Future:
        self._slots.acquire()
        with self._lock:
            self._inflight += 1

        future = self._executor.submit(_run_job, self._fn, args, kwargs)

        def done(_completed):
            with self._lock:
                self._inflight -= 1
            self._slots.release()

        future.add_done_callback(done)
        return future

    def state(self) -> dict:
        with self._lock:
            return {
                "worker_count": self.config.worker_count,
                "threads_per_worker": self.config.threads_per_worker,
                "inflight_requests": self._inflight,
            }

    def shutdown(self, *, wait: bool = True, terminate: bool = False) -> None:
        processes = []
        if terminate:
            processes = list((getattr(self._executor, "_processes", None) or {}).values())
        self._executor.shutdown(wait=wait, cancel_futures=True)
        if not terminate:
            return
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=1.0)
