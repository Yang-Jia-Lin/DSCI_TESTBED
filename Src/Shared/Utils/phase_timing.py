"""Small JSONL timing helper for experiment overhead accounting."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from Src.Shared.Config.paths import RUNTIME_DIR
from Src.Shared.Utils.utils_function import NumpyEncoder

PHASE1_OVERHEAD_LOG = RUNTIME_DIR / "ExperimentLogs" / "phase1_overhead.jsonl"


def append_timing_event(path: str | Path, event: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, cls=NumpyEncoder) + "\n")


@contextmanager
def timed_event(
    *,
    phase: str,
    step: str,
    bundle_id: str,
    path: str | Path = PHASE1_OVERHEAD_LOG,
    **extra,
) -> Iterator[None]:
    started = time.perf_counter()
    status = "success"
    error = None
    try:
        yield
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finished = time.perf_counter()
        event = {
            "logged_at": time.time(),
            "phase": phase,
            "step": step,
            "bundle_id": bundle_id,
            "status": status,
            "duration_s": finished - started,
            **extra,
        }
        if error:
            event["error"] = error
        append_timing_event(path, event)

