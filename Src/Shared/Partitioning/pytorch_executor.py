"""PyTorch executor for atomic segments defined by a partition manifest."""

from __future__ import annotations

from typing import Any

import torch

from Src.Shared.Partitioning.manifest import PartitionManifest


class SegmentExecutionError(ValueError):
    """Invalid segment range or tensor bundle."""


class PyTorchSegmentExecutor:
    def __init__(self, model: torch.nn.Module, manifest: PartitionManifest):
        self.model = model.eval()
        self.manifest = manifest
        self._segments = [
            self._resolve_segment(str(segment["name"])) for segment in manifest.segments
        ]

    def _resolve_segment(self, name: str):
        if name == "stem":
            return lambda x: self.model.maxpool(
                self.model.relu(self.model.bn1(self.model.conv1(x)))
            )
        if name == "final_pool":
            return lambda x: torch.flatten(self.model.avgpool(x), 1)
        if name == "final_classifier":
            return self.model.fc
        module: Any = self.model
        for part in name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    def execute_segment(
        self, segment_id: int, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if segment_id not in self.manifest.segment_ids:
            raise SegmentExecutionError(f"Unknown segment_id: {segment_id}")
        segment = self.manifest.segments[segment_id]
        input_names = list(segment["input_names"])
        missing = [name for name in input_names if name not in tensors]
        if missing:
            raise SegmentExecutionError(f"Segment {segment_id} missing inputs: {missing}")
        if input_names != ["main"]:
            raise SegmentExecutionError(
                "Current common manifest only permits residual-block boundaries"
            )
        with torch.no_grad():
            output = self._segments[segment_id](tensors["main"])
        return {str(segment["output_names"][0]): output}

    def execute_range(
        self,
        start_boundary: int,
        end_boundary: int,
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        self.manifest.validate_range(start_boundary, end_boundary)
        bundle = tensors
        for segment_id in range(start_boundary, end_boundary):
            if "main" not in bundle and "logits" in bundle:
                bundle = {"main": bundle["logits"]}
            bundle = self.execute_segment(segment_id, bundle)
        return bundle

    def exit_logits(
        self, boundary_id: int, tensors: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        value = tensors.get("main")
        if value is None:
            value = tensors.get("logits")
        if value is None:
            return None
        with torch.no_grad():
            if boundary_id == 8:
                return self.model.fc2(torch.flatten(self.model.avgpool(value), 1))
            if boundary_id == 14:
                return self.model.fc3(torch.flatten(self.model.avgpool(value), 1))
            if boundary_id == self.manifest.final_boundary_id:
                return tensors.get("logits")
        return None

    def execute_range_with_exits(
        self,
        start_boundary: int,
        end_boundary: int,
        tensors: dict[str, torch.Tensor],
        exit_thresholds: dict[str, float],
    ) -> dict:
        self.manifest.validate_range(start_boundary, end_boundary)
        bundle = tensors
        executed_segments = []
        for segment_id in range(start_boundary, end_boundary):
            if "main" not in bundle and "logits" in bundle:
                bundle = {"main": bundle["logits"]}
            bundle = self.execute_segment(segment_id, bundle)
            executed_segments.append(segment_id)
            boundary_id = segment_id + 1
            logits = self.exit_logits(boundary_id, bundle)
            if logits is None:
                continue
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            exit_item = next(
                (
                    item
                    for item in self.manifest.early_exits
                    if int(item["boundary_id"]) == boundary_id
                ),
                None,
            )
            logical_layer = (
                int(exit_item["logical_layer"]) if exit_item is not None else None
            )
            threshold = exit_thresholds.get(str(logical_layer))
            should_exit = (
                boundary_id == self.manifest.final_boundary_id
                or (
                    threshold is not None
                    and float(confidence.item()) >= float(threshold)
                )
            )
            if should_exit:
                return {
                    "tensors": bundle,
                    "logits": logits,
                    "confidence": float(confidence.item()),
                    "prediction": int(prediction.item()),
                    "exit_boundary_id": boundary_id,
                    "exit_logical_layer": logical_layer,
                    "executed_segments": executed_segments,
                }
        return {
            "tensors": bundle,
            "logits": None,
            "confidence": None,
            "prediction": None,
            "exit_boundary_id": None,
            "exit_logical_layer": None,
            "executed_segments": executed_segments,
        }
