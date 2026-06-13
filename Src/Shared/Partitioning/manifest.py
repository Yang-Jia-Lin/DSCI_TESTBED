"""Executable partition manifests generated from a model bundle."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Src.Shared.Config.model_config import ModelBundleSpec, get_bundle
from Src.Shared.Config.paths import bundle_paths


class PartitionManifestError(ValueError):
    pass


@dataclass(frozen=True)
class PartitionManifest:
    manifest_id: str
    bundle_id: str
    dataset_id: str
    model_hash: str
    input_shape: tuple[int, int, int]
    boundaries: tuple[dict[str, Any], ...]
    segments: tuple[dict[str, Any], ...]
    early_exits: tuple[dict[str, Any], ...]
    path: Path | None = None

    @property
    def boundary_ids(self) -> tuple[int, ...]:
        return tuple(int(item["boundary_id"]) for item in self.boundaries)

    @property
    def segment_ids(self) -> tuple[int, ...]:
        return tuple(int(item["segment_id"]) for item in self.segments)

    @property
    def final_boundary_id(self) -> int:
        return int(self.boundaries[-1]["boundary_id"])

    @property
    def exit_ids(self) -> tuple[str, ...]:
        return tuple(str(item["exit_id"]) for item in self.early_exits if not item.get("final"))

    @property
    def exit_boundary_ids(self) -> tuple[int, ...]:
        return tuple(int(item["boundary_id"]) for item in self.early_exits if not item.get("final"))

    @property
    def boundary_bytes(self) -> tuple[int, ...]:
        return tuple(int(item["serialized_num_bytes"]) for item in self.boundaries)

    def exit_for_boundary(self, boundary_id: int) -> dict[str, Any] | None:
        return next(
            (item for item in self.early_exits if int(item["boundary_id"]) == int(boundary_id)),
            None,
        )

    def boundary_for_exit(self, exit_id: str) -> int:
        item = next((item for item in self.early_exits if item["exit_id"] == exit_id), None)
        if item is None:
            raise PartitionManifestError(f"Unknown exit_id {exit_id!r}")
        return int(item["boundary_id"])

    def validate_exit_thresholds(self, thresholds: dict[str, float]) -> None:
        if set(thresholds) != set(self.exit_ids):
            raise PartitionManifestError(
                f"exit_thresholds keys must be {list(self.exit_ids)}, "
                f"got {sorted(thresholds)}"
            )
        if any(not 0.0 <= float(value) <= 1.0 for value in thresholds.values()):
            raise PartitionManifestError("Every exit threshold must be in [0, 1]")

    def validate_boundary_pair(self, first: int, second: int) -> None:
        valid = set(self.boundary_ids)
        if first not in valid or second not in valid or not 0 <= first < second <= self.final_boundary_id:
            raise PartitionManifestError(
                f"Invalid partition boundaries ({first}, {second}) for {self.manifest_id}"
            )

    def validate_range(self, start: int, end: int) -> None:
        if start not in self.boundary_ids or end not in self.boundary_ids or start > end:
            raise PartitionManifestError(f"Invalid range ({start}, {end}) for {self.manifest_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "bundle_id": self.bundle_id,
            "dataset_id": self.dataset_id,
            "model_hash": self.model_hash,
            "input_shape": list(self.input_shape),
            "boundaries": list(self.boundaries),
            "segments": list(self.segments),
            "early_exits": list(self.early_exits),
        }


def model_file_hash(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        return "weights-unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_file(manifest: PartitionManifest, path: str | Path) -> None:
    actual = model_file_hash(path)
    if actual == "weights-unavailable" or manifest.model_hash == "weights-unavailable":
        raise PartitionManifestError(
            f"Bundle weights are unavailable at {Path(path)}; train or migrate weights "
            "and regenerate the partition manifest"
        )
    if actual != manifest.model_hash:
        raise PartitionManifestError(
            f"Model hash {actual!r} does not match manifest {manifest.model_hash!r}"
        )


def build_partition_manifest(
    bundle: ModelBundleSpec | str,
    *,
    manifest_id: str | None = None,
) -> PartitionManifest:
    import torch

    from Src.Shared.Models.ModelNet.MultiExitResNet import build_model

    bundle = get_bundle(bundle) if isinstance(bundle, str) else bundle
    model = build_model(bundle).eval()
    names = ["stem"]
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        names.extend(f"{layer_name}.{index}" for index in range(len(getattr(model, layer_name))))
    names.extend(("final_pool", "final_classifier"))
    modules = {"stem": None, "final_pool": None, "final_classifier": None}
    for name in names:
        if "." in name:
            layer, index = name.split(".")
            modules[name] = getattr(model, layer)[int(index)]

    selected_manifest_id = manifest_id or bundle.manifest_id
    x = torch.zeros((1, *bundle.input_shape), dtype=torch.float32)
    boundaries: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    attach_boundaries: dict[str, int] = {}

    def boundary_record(boundary_id: int, label: str, output_name: str, value):
        tensors = {output_name: value.detach().cpu()}
        meta = {
            "name": output_name,
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "num_bytes": int(value.numel() * value.element_size()),
        }
        return {
            "boundary_id": boundary_id,
            "label": label,
            "live_tensors": [meta],
            "serialized_num_bytes": len(
                pickle.dumps(
                    {
                        "manifest_id": selected_manifest_id,
                        "boundary_id": boundary_id,
                        "tensors": tensors,
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            ),
        }

    boundaries.append(boundary_record(0, "input", "main", x))
    with torch.no_grad():
        for segment_id, name in enumerate(names):
            if name == "stem":
                x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
                output_name = "main"
            elif name == "final_pool":
                x = torch.flatten(model.avgpool(x), 1)
                output_name = "main"
            elif name == "final_classifier":
                x = model.fc(x)
                output_name = "logits"
            else:
                x = modules[name](x)
                output_name = "main"
            end = segment_id + 1
            segments.append(
                {
                    "segment_id": segment_id,
                    "name": name,
                    "start_boundary": segment_id,
                    "end_boundary": end,
                    "input_names": ["main"],
                    "output_names": [output_name],
                }
            )
            boundaries.append(boundary_record(end, name, output_name, x))
            layer_name = name.split(".")[0]
            if "." in name and name == f"{layer_name}.{len(getattr(model, layer_name)) - 1}":
                attach_boundaries[layer_name] = end

    exits = tuple(
        {
            "exit_id": item.exit_id,
            "attach_point": item.attach_point,
            "boundary_id": attach_boundaries[item.attach_point],
            "head_name": f"exit_heads.{item.exit_id}",
            "final": False,
        }
        for item in bundle.exits
    ) + (
        {
            "exit_id": "final",
            "attach_point": "final_classifier",
            "boundary_id": len(segments),
            "head_name": "fc",
            "final": True,
        },
    )
    manifest = PartitionManifest(
        manifest_id=selected_manifest_id,
        bundle_id=bundle.bundle_id,
        dataset_id=bundle.dataset_id,
        model_hash=model_file_hash(bundle_paths(bundle.bundle_id).weight_path),
        input_shape=bundle.input_shape,
        boundaries=tuple(boundaries),
        segments=tuple(segments),
        early_exits=exits,
    )
    validate_partition_manifest(manifest)
    return manifest


def validate_partition_manifest(manifest: PartitionManifest) -> None:
    if manifest.boundary_ids != tuple(range(len(manifest.boundaries))):
        raise PartitionManifestError("boundary_id values must be contiguous from zero")
    if manifest.segment_ids != tuple(range(len(manifest.segments))):
        raise PartitionManifestError("segment_id values must be contiguous from zero")
    if len(manifest.boundaries) != len(manifest.segments) + 1:
        raise PartitionManifestError("Manifest must have one more boundary than segment")
    exit_ids = [item["exit_id"] for item in manifest.early_exits]
    if len(exit_ids) != len(set(exit_ids)) or exit_ids[-1] != "final":
        raise PartitionManifestError("Manifest exits must be unique and end with final")


def write_partition_manifest(manifest: PartitionManifest, path=None, *, overwrite=False) -> Path:
    target = Path(path or bundle_paths(manifest.bundle_id).manifest_path)
    if target.exists() and not overwrite:
        raise PartitionManifestError(f"Manifest already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target


def load_partition_manifest(bundle_id: str | None = None, *, path=None) -> PartitionManifest:
    bundle = get_bundle(bundle_id)
    target = Path(path or bundle_paths(bundle.bundle_id).manifest_path)
    if not target.is_file():
        raise PartitionManifestError(f"Partition manifest not found: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    manifest = PartitionManifest(
        manifest_id=str(data["manifest_id"]),
        bundle_id=str(data["bundle_id"]),
        dataset_id=str(data["dataset_id"]),
        model_hash=str(data["model_hash"]),
        input_shape=tuple(int(x) for x in data["input_shape"]),
        boundaries=tuple(data["boundaries"]),
        segments=tuple(data["segments"]),
        early_exits=tuple(data["early_exits"]),
        path=target,
    )
    if manifest.bundle_id != bundle.bundle_id:
        raise PartitionManifestError("Manifest bundle_id mismatch")
    validate_partition_manifest(manifest)
    return manifest
