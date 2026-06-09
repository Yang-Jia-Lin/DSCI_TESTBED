"""Read, validate, and write calibrated device compute profiles."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_ROOT = PROJECT_ROOT / "Data" / "ComputeProfiles"

PROFILE_COLUMNS = (
    "layer_index",
    "layer",
    "theoretical_flops",
    "raw_latency_s",
    "calibrated_latency_s",
    "equivalent_flops",
)


class ComputeProfileError(ValueError):
    """Raised when a compute profile is missing or internally inconsistent."""


@dataclass(frozen=True)
class ComputeProfile:
    profile_id: str
    profile_dir: Path
    metadata: dict
    layers: pd.DataFrame

    @property
    def theta(self) -> float:
        return float(self.metadata["theta_flops_per_s"])

    @property
    def total_latency_s(self) -> float:
        return float(self.metadata["total_latency_s"])

    @property
    def equivalent_flops(self) -> np.ndarray:
        return self.layers["equivalent_flops"].to_numpy(dtype=np.float64)


def _profile_dir(profile_id: str, profile_root: str | Path | None = None) -> Path:
    if not profile_id or Path(profile_id).name != profile_id:
        raise ComputeProfileError(f"Invalid compute_profile_id: {profile_id!r}")
    root = (
        profile_root
        or os.environ.get("DSCI_COMPUTE_PROFILE_ROOT")
        or DEFAULT_PROFILE_ROOT
    )
    return Path(root) / profile_id


def validate_compute_profile(
    metadata: dict,
    layers: pd.DataFrame,
    *,
    expected_layers: Iterable[str] | None = None,
    expected_theoretical_flops: Iterable[float] | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> None:
    # Build a plain list of missing columns to avoid mypy/pyright Index typing issues
    missing_columns = sorted([c for c in PROFILE_COLUMNS if c not in layers.columns])
    if missing_columns:
        raise ComputeProfileError(f"layers.csv missing columns: {missing_columns}")

    required_meta = {
        "profile_id",
        "model_name",
        "backend",
        "total_latency_s",
        "theta_flops_per_s",
        "num_layers",
    }
    missing_meta = sorted(required_meta - set(metadata))
    if missing_meta:
        raise ComputeProfileError(f"metadata.json missing fields: {missing_meta}")

    num_layers = int(metadata["num_layers"])
    if len(layers) != num_layers:
        raise ComputeProfileError(
            f"Profile row count {len(layers)} != metadata num_layers {num_layers}"
        )

    indices = layers["layer_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(num_layers, dtype=np.int64)):
        raise ComputeProfileError("layer_index must be contiguous and start at 0")

    if expected_layers is not None:
        expected = list(expected_layers)
        actual = layers["layer"].astype(str).tolist()
        if actual != expected:
            raise ComputeProfileError("Profile layer names/order do not match layer stats")
    if expected_theoretical_flops is not None:
        expected_flops = np.asarray(
            list(expected_theoretical_flops), dtype=np.float64
        )
        actual_flops = layers["theoretical_flops"].to_numpy(dtype=np.float64)
        if expected_flops.shape != actual_flops.shape or not np.allclose(
            actual_flops, expected_flops, rtol=0.0, atol=0.0
        ):
            raise ComputeProfileError(
                "Profile theoretical FLOPs do not match canonical layer stats"
            )

    numeric_columns = [
        "theoretical_flops",
        "raw_latency_s",
        "calibrated_latency_s",
        "equivalent_flops",
    ]
    numeric = layers[numeric_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0):
        raise ComputeProfileError("Profile numeric values must be finite and non-negative")

    total_latency = float(metadata["total_latency_s"])
    theta = float(metadata["theta_flops_per_s"])
    if not np.isfinite(total_latency) or total_latency <= 0:
        raise ComputeProfileError("total_latency_s must be finite and positive")
    if not np.isfinite(theta) or theta <= 0:
        raise ComputeProfileError("theta_flops_per_s must be finite and positive")

    theoretical_total = float(layers["theoretical_flops"].sum())
    calibrated_total = float(layers["calibrated_latency_s"].sum())
    equivalent_total = float(layers["equivalent_flops"].sum())
    if theoretical_total <= 0:
        raise ComputeProfileError("Total theoretical FLOPs must be positive")
    if not np.isclose(calibrated_total, total_latency, rtol=rtol, atol=atol):
        raise ComputeProfileError(
            f"Calibrated latency sum {calibrated_total} != total {total_latency}"
        )
    if not np.isclose(equivalent_total, theoretical_total, rtol=rtol, atol=atol):
        raise ComputeProfileError(
            f"Equivalent FLOPs sum {equivalent_total} != theoretical total {theoretical_total}"
        )
    expected_theta = theoretical_total / total_latency
    if not np.isclose(theta, expected_theta, rtol=rtol, atol=atol):
        raise ComputeProfileError(f"Theta {theta} != expected {expected_theta}")


def load_compute_profile(
    profile_id: str,
    *,
    profile_root: str | Path | None = None,
    expected_layers: Iterable[str] | None = None,
    expected_theoretical_flops: Iterable[float] | None = None,
    expected_backend: str | None = None,
    expected_model: str | None = None,
) -> ComputeProfile:
    directory = _profile_dir(profile_id, profile_root)
    metadata_path = directory / "metadata.json"
    layers_path = directory / "layers.csv"
    if not metadata_path.is_file() or not layers_path.is_file():
        raise ComputeProfileError(
            f"Compute profile {profile_id!r} is incomplete under {directory}"
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    layers = pd.read_csv(layers_path)
    if str(metadata.get("profile_id")) != profile_id:
        raise ComputeProfileError(
            f"metadata profile_id {metadata.get('profile_id')!r} != {profile_id!r}"
        )
    if expected_backend and str(metadata.get("backend")) != expected_backend:
        raise ComputeProfileError(
            f"Profile backend {metadata.get('backend')!r} != {expected_backend!r}"
        )
    if expected_model and str(metadata.get("model_name")) != expected_model:
        raise ComputeProfileError(
            f"Profile model {metadata.get('model_name')!r} != {expected_model!r}"
        )
    validate_compute_profile(
        metadata,
        layers,
        expected_layers=expected_layers,
        expected_theoretical_flops=expected_theoretical_flops,
    )
    return ComputeProfile(profile_id, directory, metadata, layers)


def write_compute_profile(
    *,
    profile_id: str,
    layer_names: Iterable[str],
    theoretical_flops: Iterable[float],
    raw_latencies_s: Iterable[float],
    total_latency_s: float,
    model_name: str,
    backend: str,
    profile_root: str | Path | None = None,
    metadata_extra: dict | None = None,
    overwrite: bool = False,
) -> ComputeProfile:
    names = list(layer_names)
    flops = np.asarray(list(theoretical_flops), dtype=np.float64)
    raw = np.asarray(list(raw_latencies_s), dtype=np.float64)
    if len(names) != len(flops) or len(names) != len(raw):
        raise ComputeProfileError("Layer names, FLOPs, and latency arrays must align")
    if np.any(raw < 0) or not np.all(np.isfinite(raw)) or float(raw.sum()) <= 0:
        raise ComputeProfileError("Raw layer latencies must be finite and have positive sum")
    if np.any(flops < 0) or not np.all(np.isfinite(flops)) or float(flops.sum()) <= 0:
        raise ComputeProfileError("Theoretical FLOPs must be finite and have positive sum")

    total_latency_s = float(total_latency_s)
    if not np.isfinite(total_latency_s) or total_latency_s <= 0:
        raise ComputeProfileError("total_latency_s must be finite and positive")
    calibrated = raw * (total_latency_s / float(raw.sum()))
    theta = float(flops.sum()) / total_latency_s
    equivalent = calibrated * theta

    layers = pd.DataFrame(
        {
            "layer_index": np.arange(len(names), dtype=np.int64),
            "layer": names,
            "theoretical_flops": flops,
            "raw_latency_s": raw,
            "calibrated_latency_s": calibrated,
            "equivalent_flops": equivalent,
        }
    )
    metadata = {
        "profile_id": profile_id,
        "model_name": model_name,
        "backend": backend,
        "num_layers": len(names),
        "total_latency_s": total_latency_s,
        "theta_flops_per_s": theta,
    }
    metadata.update(metadata_extra or {})
    validate_compute_profile(metadata, layers, expected_layers=names)

    directory = _profile_dir(profile_id, profile_root)
    if directory.exists() and not overwrite:
        raise ComputeProfileError(
            f"Compute profile directory already exists: {directory}. "
            "Use a new profile ID or explicitly allow overwrite."
        )
    directory.mkdir(parents=True, exist_ok=True)
    layers.to_csv(directory / "layers.csv", index=False)
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return ComputeProfile(profile_id, directory, metadata, layers)


def compute_profile_state(role: str, backend: str) -> dict:
    """Return the measured-state fields advertised by one deployment node."""
    role = role.lower()
    backend = backend.lower()
    if role not in {"device", "edge", "cloud"}:
        raise ComputeProfileError(f"Unknown compute-profile role: {role!r}")
    backend_key = f"DSCI_{role.upper()}_{backend.upper()}_COMPUTE_PROFILE_ID"
    common_key = f"DSCI_{role.upper()}_COMPUTE_PROFILE_ID"
    profile_id = os.environ.get(backend_key) or os.environ.get(common_key)
    if not profile_id:
        raise ComputeProfileError(
            f"Set {backend_key} or {common_key} to a calibrated compute profile ID"
        )
    profile = load_compute_profile(profile_id, expected_backend=backend)
    capacity_key = {
        "device": "f_u",
        "edge": "f_e_max",
        "cloud": "f_c_max",
    }[role]
    return {
        "compute_profile_id": profile.profile_id,
        capacity_key: profile.theta,
    }
