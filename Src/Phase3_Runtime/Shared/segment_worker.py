"""Process-local segment executor jobs for PyTorch and MNN workers."""

from __future__ import annotations

import time

_EXECUTOR = None
_MANIFEST = None


def init_pytorch_worker(manifest_id: str):
    global _EXECUTOR, _MANIFEST
    from Src.Phase3_Runtime.Shared.model_loader import load_full_model
    from Src.Shared.Partitioning.manifest import load_partition_manifest
    from Src.Shared.Partitioning.pytorch_executor import PyTorchSegmentExecutor

    _MANIFEST = load_partition_manifest(manifest_id)
    _EXECUTOR = PyTorchSegmentExecutor(load_full_model(_MANIFEST), _MANIFEST)


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


def init_mnn_worker(manifest_id: str):
    global _EXECUTOR, _MANIFEST
    from Src.Phase3_Runtime.Shared.mnn_segment_executor import MNNSegmentExecutor
    from Src.Shared.Partitioning.manifest import load_partition_manifest

    _MANIFEST = load_partition_manifest(manifest_id)
    _EXECUTOR = MNNSegmentExecutor(_MANIFEST)


def execute_mnn_range(
    start_boundary: int,
    end_boundary: int,
    tensors: dict,
    exit_thresholds: dict[str, float] | None = None,
):
    if _EXECUTOR is None:
        raise RuntimeError("MNN worker is not initialized")
    started = time.perf_counter()
    import numpy as np

    bundle = tensors
    executed_segments = []
    logits = None
    confidence = prediction = exit_boundary_id = exit_logical_layer = None
    for segment_id in range(start_boundary, end_boundary):
        bundle = _EXECUTOR.execute_segment(segment_id, bundle)
        executed_segments.append(segment_id)
        boundary_id = segment_id + 1
        candidate = _EXECUTOR.exit_logits(boundary_id, bundle)
        if candidate is None:
            continue
        item = next(
            (
                value
                for value in _MANIFEST.early_exits
                if int(value["boundary_id"]) == boundary_id
            ),
            None,
        )
        logical_layer = int(item["logical_layer"]) if item is not None else None
        flat = np.asarray(candidate, dtype=np.float64).reshape(-1)
        probabilities = np.exp(flat - np.max(flat))
        probabilities /= probabilities.sum()
        candidate_confidence = float(np.max(probabilities))
        threshold = (exit_thresholds or {}).get(str(logical_layer))
        if (
            boundary_id == _MANIFEST.final_boundary_id
            or threshold is not None
            and candidate_confidence >= float(threshold)
        ):
            logits = candidate
            confidence = candidate_confidence
            prediction = int(np.argmax(flat))
            exit_boundary_id = boundary_id
            exit_logical_layer = logical_layer
            break
    return {
        "tensors": bundle,
        "logits": logits,
        "confidence": confidence,
        "prediction": prediction,
        "exit_boundary_id": exit_boundary_id,
        "exit_logical_layer": exit_logical_layer,
        "T_compute_s": time.perf_counter() - started,
        "executed_segments": executed_segments,
    }
