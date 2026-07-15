"""Readiness checks for evaluation artifacts.

The checks are intentionally non-blocking: missing datasets, bundles, or
hardware profiles are reported as ``pending`` so the paper scripts can be
prepared before the true-device measurements exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from Scripts.EvaluationCommon.config import EXPECTED_BUNDLES
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import SEGMENT_PROFILE_DIR, bundle_paths
from Src.Shared.Partitioning.manifest import load_partition_manifest


@dataclass(frozen=True)
class BundleReadiness:
    bundle_id: str
    status: str
    registered: bool
    weights: bool
    manifest: bool
    exit_curves: bool
    dataset_root: bool
    device_profiles: int
    device_nx_profiles: int
    device_nano_profiles: int
    device_pi5_profiles: int
    edge_profiles: int
    cloud_profiles: int
    notes: str

    def to_row(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "status": self.status,
            "registered": self.registered,
            "weights": self.weights,
            "manifest": self.manifest,
            "exit_curves": self.exit_curves,
            "dataset_root": self.dataset_root,
            "device_profiles": self.device_profiles,
            "device_nx_profiles": self.device_nx_profiles,
            "device_nano_profiles": self.device_nano_profiles,
            "device_pi5_profiles": self.device_pi5_profiles,
            "edge_profiles": self.edge_profiles,
            "cloud_profiles": self.cloud_profiles,
            "notes": self.notes,
        }


def _profile_counts(bundle_id: str) -> tuple[int, int, int, int, int, int]:
    device = nx = nano = pi5 = edge = cloud = 0
    for metadata_path in SEGMENT_PROFILE_DIR.glob("*/metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("bundle_id") != bundle_id:
            continue
        profile_id = str(metadata.get("profile_id") or metadata_path.parent.name)
        if profile_id.startswith("device-"):
            device += 1
            if profile_id.startswith("device-nx"):
                nx += 1
            if profile_id.startswith("device-nano"):
                nano += 1
            if profile_id.startswith("device-pi5"):
                pi5 += 1
        elif profile_id.startswith("edge-"):
            edge += 1
        elif profile_id.startswith("cloud-"):
            cloud += 1
    return device, nx, nano, pi5, edge, cloud


def check_bundle_readiness(bundle_id: str) -> BundleReadiness:
    try:
        bundle = get_bundle(bundle_id)
        paths = bundle_paths(bundle.bundle_id)
        registered = True
    except Exception as exc:
        return BundleReadiness(
            bundle_id=bundle_id,
            status="pending",
            registered=False,
            weights=False,
            manifest=False,
            exit_curves=False,
            dataset_root=False,
            device_profiles=0,
            device_nx_profiles=0,
            device_nano_profiles=0,
            device_pi5_profiles=0,
            edge_profiles=0,
            cloud_profiles=0,
            notes=f"not registered in Src.Shared.Config.model_config: {exc}",
        )

    weights = paths.weight_path.is_file()
    manifest = paths.manifest_path.is_file()
    exit_curves = paths.offline_table_path.is_file()
    dataset_root = paths.dataset_root.exists()
    notes: list[str] = []
    if manifest:
        try:
            load_partition_manifest(bundle.bundle_id)
        except Exception as exc:
            manifest = False
            notes.append(f"manifest invalid: {exc}")
    (
        device_profiles,
        device_nx_profiles,
        device_nano_profiles,
        device_pi5_profiles,
        edge_profiles,
        cloud_profiles,
    ) = _profile_counts(bundle.bundle_id)

    required = (weights, manifest, exit_curves, dataset_root)
    profile_ready = all(
        count > 0
        for count in (
            device_nx_profiles,
            device_nano_profiles,
            device_pi5_profiles,
            edge_profiles,
            cloud_profiles,
        )
    )
    status = "ready" if all(required) and profile_ready else "pending"
    if not weights:
        notes.append("missing weights.pth")
    if not manifest:
        notes.append("missing or invalid manifest.json")
    if not exit_curves:
        notes.append("missing exit_curves.csv")
    if not dataset_root:
        notes.append(f"missing dataset root: {paths.dataset_root}")
    if not profile_ready:
        notes.append("missing one or more device/edge/cloud segment profiles")
    if device_nx_profiles == 0:
        notes.append("missing NX device profile")
    if device_nano_profiles == 0:
        notes.append("missing Nano device profile")
    if device_pi5_profiles == 0:
        notes.append("missing Pi5 device profile")

    return BundleReadiness(
        bundle_id=bundle.bundle_id,
        status=status,
        registered=registered,
        weights=weights,
        manifest=manifest,
        exit_curves=exit_curves,
        dataset_root=dataset_root,
        device_profiles=device_profiles,
        device_nx_profiles=device_nx_profiles,
        device_nano_profiles=device_nano_profiles,
        device_pi5_profiles=device_pi5_profiles,
        edge_profiles=edge_profiles,
        cloud_profiles=cloud_profiles,
        notes="; ".join(notes) if notes else "ok",
    )


def write_readiness_report(
    output_dir: str | Path,
    *,
    bundle_ids: tuple[str, ...] = EXPECTED_BUNDLES,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [check_bundle_readiness(bundle_id).to_row() for bundle_id in bundle_ids]
    path = output_dir / "artifact_readiness.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path
