"""Generic PyTorch executor for manifest-defined atomic segments."""

from __future__ import annotations

import torch

from Src.Shared.Partitioning.manifest import PartitionManifest


class SegmentExecutionError(ValueError):
    pass


class PyTorchSegmentExecutor:
    def __init__(self, model: torch.nn.Module, manifest: PartitionManifest):
        self.model = model.eval()
        self.manifest = manifest
        self._segments = [self._resolve(str(item["name"])) for item in manifest.segments]

    def _resolve(self, name):
        if name == "stem":
            return lambda x: self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        if name == "final_pool":
            return lambda x: torch.flatten(self.model.avgpool(x), 1)
        if name == "final_classifier":
            return self.model.fc
        module = self.model
        for part in name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    def execute_segment(self, segment_id: int, tensors: dict[str, torch.Tensor]):
        if segment_id not in self.manifest.segment_ids:
            raise SegmentExecutionError(f"Unknown segment_id: {segment_id}")
        segment = self.manifest.segments[segment_id]
        with torch.no_grad():
            output = self._segments[segment_id](tensors["main"])
        return {str(segment["output_names"][0]): output}

    def execute_range(self, start_boundary: int, end_boundary: int, tensors: dict):
        self.manifest.validate_range(start_boundary, end_boundary)
        bundle = tensors
        for segment_id in range(start_boundary, end_boundary):
            bundle = self.execute_segment(segment_id, bundle)
        return bundle

    def exit_logits(self, boundary_id: int, tensors: dict):
        item = self.manifest.exit_for_boundary(boundary_id)
        if item is None:
            return None
        if item.get("final"):
            return tensors.get("logits")
        with torch.no_grad():
            return self.model.classify_exit(str(item["exit_id"]), tensors["main"])

    def execute_range_with_exits(self, start_boundary, end_boundary, tensors, exit_thresholds):
        self.manifest.validate_range(start_boundary, end_boundary)
        self.manifest.validate_exit_thresholds(exit_thresholds)
        bundle = tensors
        executed = []
        for segment_id in range(start_boundary, end_boundary):
            bundle = self.execute_segment(segment_id, bundle)
            executed.append(segment_id)
            boundary_id = segment_id + 1
            item = self.manifest.exit_for_boundary(boundary_id)
            if item is None:
                continue
            logits = self.exit_logits(boundary_id, bundle)
            if logits is not None:
                logits = logits.detach()
            confidence, prediction = torch.softmax(logits, dim=1).max(dim=1)
            threshold = exit_thresholds.get(str(item["exit_id"]))
            if item.get("final") or threshold is not None and confidence.item() >= float(threshold):
                return {
                    "tensors": bundle,
                    "logits": logits,
                    "confidence": float(confidence.item()),
                    "prediction": int(prediction.item()),
                    "exit_boundary_id": boundary_id,
                    "exit_id": str(item["exit_id"]),
                    "executed_segments": executed,
                }
        return {
            "tensors": bundle,
            "logits": None,
            "confidence": None,
            "prediction": None,
            "exit_boundary_id": None,
            "exit_id": None,
            "executed_segments": executed,
        }
