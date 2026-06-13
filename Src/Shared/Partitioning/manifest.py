"""Executable partition manifest shared by profiling, scheduling, and runtime."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Src.Shared.Config.paths import DATA_DIR, RESNET50_PATHS

DEFAULT_MANIFEST_ROOT = DATA_DIR / "PartitionManifests"
DEFAULT_MANIFEST_ID = "resnet50-cifar10-partition-v1"


class PartitionManifestError(ValueError):
    """Invalid or incompatible executable partition manifest."""


@dataclass(frozen=True)
class PartitionManifest:
    manifest_id: str
    model_name: str
    model_hash: str
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
    def boundary_bytes(self) -> tuple[int, ...]:
        return tuple(
            int(
                item.get(
                    "serialized_num_bytes",
                    sum(int(t["num_bytes"]) for t in item["live_tensors"]),
                )
            )
            for item in self.boundaries
        )

    def validate_boundary_pair(self, first: int, second: int) -> None:
        valid = set(self.boundary_ids)
        if first not in valid or second not in valid:
            raise PartitionManifestError(
                f"Partition boundaries ({first}, {second}) are not in manifest "
                f"{self.manifest_id!r}"
            )
        if not 0 <= first < second <= self.final_boundary_id:
            raise PartitionManifestError(
                f"Require 0 <= first < second <= {self.final_boundary_id}, "
                f"got ({first}, {second})"
            )

    def validate_range(self, start: int, end: int) -> None:
        valid = set(self.boundary_ids)
        if start not in valid or end not in valid or start > end:
            raise PartitionManifestError(
                f"Invalid executable boundary range ({start}, {end}) for "
                f"manifest {self.manifest_id!r}"
            )

    def exit_boundary_for_logical_layer(self, layer: int) -> int:
        for item in self.early_exits:
            if int(item["logical_layer"]) == int(layer):
                return int(item["boundary_id"])
        if int(layer) >= 127:
            return self.final_boundary_id
        raise PartitionManifestError(f"No exit boundary for logical layer {layer}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "boundary_semantics": "segments_before_boundary_have_executed",
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
    if actual != manifest.model_hash:
        raise PartitionManifestError(
            f"Model hash {actual!r} does not match manifest {manifest.model_hash!r}"
        )


def build_resnet50_manifest(
    *,
    manifest_id: str = DEFAULT_MANIFEST_ID,
    sample_shape: tuple[int, int, int, int] = (1, 3, 227, 227),
) -> PartitionManifest:
    """Build the common PyTorch/MNN boundary set at residual-block boundaries."""
    import torch

    from Src.Shared.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50

    model = MultiEEResNet50(
        Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True
    ).eval()
    fx_graph = torch.fx.symbolic_trace(model, concrete_args={"stage": "final"})
    fx_nodes = list(fx_graph.graph.nodes)
    fx_index = {node: index for index, node in enumerate(fx_nodes)}
    names = ["stem"]
    names.extend(f"layer1.{i}" for i in range(3))
    names.extend(f"layer2.{i}" for i in range(4))
    names.extend(f"layer3.{i}" for i in range(6))
    names.extend(f"layer4.{i}" for i in range(3))
    names.extend(("final_pool", "final_classifier"))

    modules = [None]
    modules.extend(model.layer1)
    modules.extend(model.layer2)
    modules.extend(model.layer3)
    modules.extend(model.layer4)
    modules.extend((None, None))

    boundaries: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    x = torch.zeros(sample_shape, dtype=torch.float32)

    def tensor_meta(name: str, value: torch.Tensor) -> dict[str, Any]:
        return {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "num_bytes": int(value.numel() * value.element_size()),
        }

    def fx_cut_node(label):
        if label == "input":
            return next(node for node in fx_nodes if node.op == "placeholder")
        if label == "stem":
            target = "maxpool"
        elif label == "final_pool":
            return next(node for node in reversed(fx_nodes) if node.name.startswith("flatten"))
        elif label == "final_classifier":
            target = "fc"
        else:
            target = f"{label}.relu"
        return next(
            node
            for node in reversed(fx_nodes)
            if node.op == "call_module" and str(node.target) == target
        )

    def boundary_record(boundary_id, logical_layer_after, label, name, value):
        tensors = {name: value.detach().cpu()}
        cut = fx_cut_node(label)
        live_values = [
            node.name
            for node in fx_nodes[: fx_index[cut] + 1]
            if any(fx_index[user] > fx_index[cut] for user in node.users)
        ]
        return {
            "boundary_id": boundary_id,
            "logical_layer_after": logical_layer_after,
            "label": label,
            "live_tensors": [tensor_meta(name, value)],
            "fx_node_after": cut.name,
            "fx_live_values": live_values,
            "serialized_num_bytes": len(
                pickle.dumps(
                    {
                        "manifest_id": manifest_id,
                        "boundary_id": boundary_id,
                        "tensors": tensors,
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            ),
        }

    boundaries.append(
        boundary_record(0, None, "input", "main", x)
    )

    logical_after = [
        3, 12, 19, 26, 35, 42, 49, 56, 67, 74, 81, 88, 95, 102, 111, 118, 125,
        126, 127,
    ]
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
                x = modules[segment_id](x)
                output_name = "main"
            end_boundary = segment_id + 1
            segments.append(
                {
                    "segment_id": segment_id,
                    "name": name,
                    "start_boundary": segment_id,
                    "end_boundary": end_boundary,
                    "input_names": ["main"],
                    "output_names": [output_name],
                }
            )
            boundaries.append(
                boundary_record(
                    end_boundary,
                    logical_after[segment_id],
                    name,
                    output_name,
                    x,
                )
            )

    early_exits = (
        {"logical_layer": 57, "boundary_id": 8, "head_segment_id": "exit_57"},
        {"logical_layer": 103, "boundary_id": 14, "head_segment_id": "exit_103"},
        {"logical_layer": 127, "boundary_id": 19, "head_segment_id": "final"},
    )
    manifest = PartitionManifest(
        manifest_id=manifest_id,
        model_name="Resnet50",
        model_hash=model_file_hash(RESNET50_PATHS.resolve_weight_path()),
        boundaries=tuple(boundaries),
        segments=tuple(segments),
        early_exits=early_exits,
    )
    validate_partition_manifest(manifest)
    return manifest


def validate_partition_manifest(manifest: PartitionManifest) -> None:
    if not manifest.model_name or not manifest.model_hash:
        raise PartitionManifestError("Manifest requires model_name and model_hash")
    boundary_ids = manifest.boundary_ids
    if boundary_ids != tuple(range(len(boundary_ids))):
        raise PartitionManifestError("boundary_id values must be contiguous from zero")
    if manifest.segment_ids != tuple(range(len(manifest.segments))):
        raise PartitionManifestError("segment_id values must be contiguous from zero")
    if len(manifest.boundaries) != len(manifest.segments) + 1:
        raise PartitionManifestError("Manifest must have exactly one more boundary than segment")
    for segment in manifest.segments:
        sid = int(segment["segment_id"])
        if int(segment["start_boundary"]) != sid or int(segment["end_boundary"]) != sid + 1:
            raise PartitionManifestError("Segments must connect adjacent boundaries")


def write_partition_manifest(
    manifest: PartitionManifest,
    path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    target = Path(path or DEFAULT_MANIFEST_ROOT / f"{manifest.manifest_id}.json")
    if target.exists() and not overwrite:
        raise PartitionManifestError(f"Manifest already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
        handle.write("\n")
    return target


def load_partition_manifest(
    manifest_id: str = DEFAULT_MANIFEST_ID,
    *,
    path: str | Path | None = None,
) -> PartitionManifest:
    target = Path(path or DEFAULT_MANIFEST_ROOT / f"{manifest_id}.json")
    if not target.is_file():
        raise PartitionManifestError(f"Partition manifest not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    manifest = PartitionManifest(
        manifest_id=str(data["manifest_id"]),
        model_name=str(data["model_name"]),
        model_hash=str(data["model_hash"]),
        boundaries=tuple(data["boundaries"]),
        segments=tuple(data["segments"]),
        early_exits=tuple(data["early_exits"]),
        path=target,
    )
    if manifest.manifest_id != manifest_id:
        raise PartitionManifestError(
            f"Manifest id {manifest.manifest_id!r} != expected {manifest_id!r}"
        )
    validate_partition_manifest(manifest)
    return manifest
