"""Measured atomic-segment latency profiles for fixed worker pools."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Src.Shared.Config.paths import DATA_DIR
from Src.Shared.Partitioning.manifest import PartitionManifest, load_partition_manifest

DEFAULT_SEGMENT_PROFILE_ROOT = DATA_DIR / "SegmentProfiles"
SEGMENT_COLUMNS = (
    "segment_id",
    "raw_latency_mean_s",
    "raw_latency_median_s",
    "raw_latency_p95_s",
    "calibrated_latency_s",
    "input_bytes",
    "output_bytes",
)


class SegmentProfileError(ValueError):
    """Invalid, missing, or incompatible measured segment profile."""


@dataclass(frozen=True)
class SegmentProfile:
    profile_id: str
    profile_dir: Path
    metadata: dict
    segments: pd.DataFrame

    @property
    def manifest_id(self) -> str:
        return str(self.metadata["manifest_id"])

    @property
    def latencies(self) -> np.ndarray:
        return self.segments["calibrated_latency_s"].to_numpy(dtype=np.float64)

    @property
    def total_latency_s(self) -> float:
        return float(self.metadata["total_model_latency_s"])

    @property
    def worker_count(self) -> int:
        return int(self.metadata["worker_count"])

    @property
    def threads_per_worker(self) -> int:
        return int(self.metadata["threads_per_worker"])

    @property
    def exit_head_latencies(self) -> dict[int, float]:
        return {
            int(layer): float(latency)
            for layer, latency in self.metadata["exit_head_latencies_s"].items()
        }


def _root(profile_root: str | Path | None = None) -> Path:
    return Path(
        profile_root
        or os.environ.get("DSCI_SEGMENT_PROFILE_ROOT")
        or DEFAULT_SEGMENT_PROFILE_ROOT
    )


def validate_segment_profile(
    metadata: dict,
    segments: pd.DataFrame,
    *,
    manifest: PartitionManifest | None = None,
) -> None:
    missing = sorted(set(SEGMENT_COLUMNS) - set(segments.columns))
    if missing:
        raise SegmentProfileError(f"segments.csv missing columns: {missing}")
    required = {
        "profile_id",
        "manifest_id",
        "model_name",
        "model_hash",
        "backend",
        "worker_count",
        "threads_per_worker",
        "total_model_latency_s",
        "num_segments",
        "exit_head_latencies_s",
    }
    missing_meta = sorted(required - set(metadata))
    if missing_meta:
        raise SegmentProfileError(f"metadata.json missing fields: {missing_meta}")
    count = int(metadata["num_segments"])
    if len(segments) != count:
        raise SegmentProfileError(f"Segment row count {len(segments)} != {count}")
    ids = segments["segment_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, np.arange(count, dtype=np.int64)):
        raise SegmentProfileError("segment_id must be contiguous from zero")
    numeric = segments[list(SEGMENT_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0):
        raise SegmentProfileError("Segment profile values must be finite and non-negative")
    total = float(metadata["total_model_latency_s"])
    if total <= 0 or not np.isclose(
        float(segments["calibrated_latency_s"].sum()), total, rtol=1e-6, atol=1e-9
    ):
        raise SegmentProfileError("Calibrated segment latency sum must equal total latency")
    if int(metadata["worker_count"]) <= 0 or int(metadata["threads_per_worker"]) <= 0:
        raise SegmentProfileError("Worker configuration must be positive")
    exit_head_latencies = metadata["exit_head_latencies_s"]
    if not isinstance(exit_head_latencies, dict):
        raise SegmentProfileError("exit_head_latencies_s must be an object")
    if any(
        not np.isfinite(float(value)) or float(value) < 0
        for value in exit_head_latencies.values()
    ):
        raise SegmentProfileError("Exit-head latencies must be finite and non-negative")
    if manifest is not None:
        if str(metadata["manifest_id"]) != manifest.manifest_id:
            raise SegmentProfileError("Profile manifest_id does not match partition manifest")
        if str(metadata["model_name"]) != manifest.model_name:
            raise SegmentProfileError("Profile model_name does not match partition manifest")
        if str(metadata["model_hash"]) != manifest.model_hash:
            raise SegmentProfileError("Profile model_hash does not match partition manifest")
        if count != len(manifest.segments):
            raise SegmentProfileError("Profile does not cover every manifest segment")
        expected_exits = {
            str(int(item["logical_layer"]))
            for item in manifest.early_exits
            if int(item["boundary_id"]) != manifest.final_boundary_id
        }
        if set(exit_head_latencies) != expected_exits:
            raise SegmentProfileError("Profile does not cover every non-final exit head")


def write_segment_profile(
    *,
    profile_id: str,
    manifest: PartitionManifest,
    backend: str,
    worker_count: int,
    threads_per_worker: int,
    samples_s: list[list[float]],
    total_model_latency_s: float,
    exit_head_samples_s: dict[int, list[float]] | None = None,
    profile_root: str | Path | None = None,
    overwrite: bool = False,
) -> SegmentProfile:
    if len(samples_s) != len(manifest.segments) or any(not row for row in samples_s):
        raise SegmentProfileError("samples_s must contain non-empty samples for every segment")
    expected_exit_layers = {
        int(item["logical_layer"])
        for item in manifest.early_exits
        if int(item["boundary_id"]) != manifest.final_boundary_id
    }
    if (
        exit_head_samples_s is None
        or set(exit_head_samples_s) != expected_exit_layers
        or any(not values for values in exit_head_samples_s.values())
    ):
        raise SegmentProfileError(
            "exit_head_samples_s must contain non-empty samples for every exit head"
        )
    mean = np.array([np.mean(row) for row in samples_s], dtype=np.float64)
    median = np.array([np.median(row) for row in samples_s], dtype=np.float64)
    p95 = np.array([np.percentile(row, 95) for row in samples_s], dtype=np.float64)
    total_model_latency_s = float(total_model_latency_s)
    if total_model_latency_s <= 0 or float(mean.sum()) <= 0:
        raise SegmentProfileError("Measured latencies must be positive")
    calibrated = mean * (total_model_latency_s / float(mean.sum()))
    boundary_bytes = manifest.boundary_bytes
    segments = pd.DataFrame(
        {
            "segment_id": np.arange(len(manifest.segments), dtype=np.int64),
            "raw_latency_mean_s": mean,
            "raw_latency_median_s": median,
            "raw_latency_p95_s": p95,
            "calibrated_latency_s": calibrated,
            "input_bytes": boundary_bytes[:-1],
            "output_bytes": boundary_bytes[1:],
        }
    )
    metadata = {
        "profile_id": profile_id,
        "manifest_id": manifest.manifest_id,
        "model_name": manifest.model_name,
        "model_hash": manifest.model_hash,
        "backend": str(backend),
        "worker_count": int(worker_count),
        "threads_per_worker": int(threads_per_worker),
        "total_model_latency_s": total_model_latency_s,
        "num_segments": len(manifest.segments),
        "exit_head_latencies_s": {
            str(int(item["logical_layer"])): float(np.mean(exit_head_samples_s[int(item["logical_layer"])]))
            for item in manifest.early_exits
            if int(item["boundary_id"]) != manifest.final_boundary_id
        },
    }
    validate_segment_profile(metadata, segments, manifest=manifest)
    directory = _root(profile_root) / profile_id
    if directory.exists() and not overwrite:
        raise SegmentProfileError(f"Segment profile already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    segments.to_csv(directory / "segments.csv", index=False)
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return SegmentProfile(profile_id, directory, metadata, segments)


def load_segment_profile(
    profile_id: str,
    *,
    manifest: PartitionManifest | None = None,
    expected_backend: str | None = None,
    profile_root: str | Path | None = None,
) -> SegmentProfile:
    directory = _root(profile_root) / profile_id
    with (directory / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    segments = pd.read_csv(directory / "segments.csv")
    if str(metadata.get("profile_id")) != profile_id:
        raise SegmentProfileError("Profile id mismatch")
    if expected_backend and str(metadata.get("backend")) != expected_backend:
        raise SegmentProfileError("Profile backend mismatch")
    validate_segment_profile(metadata, segments, manifest=manifest)
    return SegmentProfile(profile_id, directory, metadata, segments)


def segment_profile_state(role: str, backend: str) -> dict:
    key = f"DSCI_{role.upper()}_{backend.upper()}_SEGMENT_PROFILE_ID"
    common = f"DSCI_{role.upper()}_SEGMENT_PROFILE_ID"
    profile_id = os.environ.get(key) or os.environ.get(common)
    if not profile_id:
        raise SegmentProfileError(f"Set {key} or {common}")
    profile = load_segment_profile(profile_id, expected_backend=backend)
    validate_segment_profile(
        profile.metadata,
        profile.segments,
        manifest=load_partition_manifest(profile.manifest_id),
    )
    overhead_key = f"DSCI_{role.upper()}_PROTOCOL_OVERHEAD_S"
    return {
        "resource_mode": "fixed_worker_pool",
        "manifest_id": profile.manifest_id,
        "model_name": str(profile.metadata["model_name"]),
        "model_hash": str(profile.metadata["model_hash"]),
        "execution_profile_id": profile.profile_id,
        "backend": backend,
        "worker_count": profile.worker_count,
        "threads_per_worker": profile.threads_per_worker,
        "protocol_overhead_s": float(os.environ.get(overhead_key, "0")),
    }
