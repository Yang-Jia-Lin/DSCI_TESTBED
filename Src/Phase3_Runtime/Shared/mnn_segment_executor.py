"""Generic multi-input/multi-output MNN atomic segment executor."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Partitioning.manifest import PartitionManifest


class MNNSegmentExecutor:
    def __init__(self, manifest: PartitionManifest, model_root: str | Path | None = None):
        try:
            import MNN  # type: ignore
        except ImportError as exc:
            raise ImportError("MNN Python package is required for the MNN backend") from exc
        self.MNN = MNN
        self.manifest = manifest
        self.model_root = Path(
            model_root or bundle_paths(manifest.bundle_id).mnn_root
        )
        self._cache = {}
        missing = [
            self.model_root / f"segment_{sid}.mnn"
            for sid in manifest.segment_ids
            if not (self.model_root / f"segment_{sid}.mnn").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"MNN manifest is incomplete; missing {len(missing)} segment models"
            )
        missing_heads = [
            self.model_root / f"exit_{item['exit_id']}.mnn"
            for item in manifest.early_exits
            if int(item["boundary_id"]) != manifest.final_boundary_id
            and not (self.model_root / f"exit_{item['exit_id']}.mnn").is_file()
        ]
        if missing_heads:
            raise FileNotFoundError("MNN manifest is missing early-exit head models")

    def _load(self, segment_id: int):
        if segment_id not in self._cache:
            interpreter = self.MNN.Interpreter(
                str(self.model_root / f"segment_{segment_id}.mnn")
            )
            session = interpreter.createSession(
                {"numThread": int(os.environ.get("OMP_NUM_THREADS", "1"))}
            )
            self._cache[segment_id] = (interpreter, session)
        return self._cache[segment_id]

    def execute_segment(self, segment_id: int, tensors: dict) -> dict:
        segment = self.manifest.segments[segment_id]
        interpreter, session = self._load(segment_id)
        for name in segment["input_names"]:
            array = tensors[name]
            if hasattr(array, "detach"):
                array = array.detach().cpu().numpy()
            array = np.ascontiguousarray(array, dtype=np.float32)
            target = interpreter.getSessionInput(session, name)
            source = self.MNN.Tensor(
                array.shape,
                self.MNN.Halide_Type_Float,
                array,
                self.MNN.Tensor_DimensionType_Caffe,
            )
            target.copyFrom(source)
        interpreter.runSession(session)
        outputs = interpreter.getSessionOutputAll(session)
        output_meta = {
            item["name"]: item
            for item in self.manifest.boundaries[int(segment["end_boundary"])][
                "live_tensors"
            ]
        }
        return {
            name: np.asarray(outputs[name].getData(), dtype=np.float32)
            .reshape(tuple(int(value) for value in output_meta[name]["shape"]))
            .copy()
            for name in segment["output_names"]
        }

    def execute_range(self, start_boundary: int, end_boundary: int, tensors: dict) -> dict:
        self.manifest.validate_range(start_boundary, end_boundary)
        bundle = tensors
        for segment_id in range(start_boundary, end_boundary):
            if "main" not in bundle and "logits" in bundle:
                bundle = {"main": bundle["logits"]}
            bundle = self.execute_segment(segment_id, bundle)
        return bundle

    def exit_logits(self, boundary_id: int, tensors: dict):
        item = next(
            (
                value
                for value in self.manifest.early_exits
                if int(value["boundary_id"]) == int(boundary_id)
            ),
            None,
        )
        if item is None:
            return None
        if int(boundary_id) == self.manifest.final_boundary_id:
            return tensors.get("logits")
        key = f"exit_{item['exit_id']}"
        if key not in self._cache:
            interpreter = self.MNN.Interpreter(str(self.model_root / f"{key}.mnn"))
            self._cache[key] = (
                interpreter,
                interpreter.createSession(
                    {"numThread": int(os.environ.get("OMP_NUM_THREADS", "1"))}
                ),
            )
        interpreter, session = self._cache[key]
        array = tensors["main"]
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        array = np.ascontiguousarray(array, dtype=np.float32)
        target = interpreter.getSessionInput(session, "main")
        target.copyFrom(
            self.MNN.Tensor(
                array.shape,
                self.MNN.Halide_Type_Float,
                array,
                self.MNN.Tensor_DimensionType_Caffe,
            )
        )
        interpreter.runSession(session)
        output = interpreter.getSessionOutput(session, "logits")
        return np.asarray(output.getData(), dtype=np.float32).reshape(1, -1).copy()
