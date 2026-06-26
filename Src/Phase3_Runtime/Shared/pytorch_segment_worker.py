"""Process-local PyTorch segment worker entrypoints."""

from __future__ import annotations

import time

_EXECUTOR = None


def init_pytorch_worker(bundle_id: str):
    global _EXECUTOR
    from Src.Phase3_Runtime.Shared.model_loader import load_full_model
    from Src.Shared.Partitioning.manifest import load_partition_manifest
    from Src.Shared.Partitioning.pytorch_executor import PyTorchSegmentExecutor

    manifest = load_partition_manifest(bundle_id)
    _EXECUTOR = PyTorchSegmentExecutor(load_full_model(manifest), manifest)


def execute_pytorch_range(
    start_boundary: int,
    end_boundary: int,
    tensors: dict,
    exit_thresholds: dict[str, float] | None = None,
):
    if _EXECUTOR is None:
        raise RuntimeError("PyTorch worker is not initialized")
    started = time.perf_counter()
    result = _EXECUTOR.execute_range_with_exits(
        start_boundary, end_boundary, tensors, exit_thresholds or {}
    )
    return {
        **result,
        "T_compute_s": time.perf_counter() - started,
    }
