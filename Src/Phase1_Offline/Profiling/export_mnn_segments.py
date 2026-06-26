"""Export every segment and exit head of a model bundle to ONNX/MNN."""

import argparse
import shutil
import subprocess

import torch
import torch.nn as nn

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Models.ModelNet.MultiExitResNet import build_model
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Partitioning.pytorch_executor import PyTorchSegmentExecutor


class ExitHead(nn.Module):
    def __init__(self, model, exit_id):
        super().__init__()
        self.model = model
        self.exit_id = exit_id

    def forward(self, main):
        return self.model.classify_exit(self.exit_id, main)


class SegmentModule(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, main):
        return self.function(main)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--mnnconvert", default="MNNConvert")
    parser.add_argument("--onnx-only", action="store_true")
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    paths = bundle_paths(bundle.bundle_id)
    manifest = load_partition_manifest(bundle.bundle_id)
    model = build_model(bundle)
    model.load_state_dict(torch.load(paths.weight_path, map_location="cpu", weights_only=True))
    model.eval()
    executor = PyTorchSegmentExecutor(model, manifest)
    paths.mnn_root.mkdir(parents=True, exist_ok=True)
    converter = shutil.which(args.mnnconvert)
    if not args.onnx_only and converter is None:
        raise FileNotFoundError(f"MNN converter not found: {args.mnnconvert}")
    bundle_data = {"main": torch.randn((1, *bundle.input_shape))}
    for segment in manifest.segments:
        segment_id = int(segment["segment_id"])
        resolved = executor._segments[segment_id]
        module = resolved if isinstance(resolved, nn.Module) else SegmentModule(resolved)
        sample = bundle_data["main"]
        output_name = str(segment["output_names"][0])
        onnx = paths.mnn_root / f"segment_{segment_id}.onnx"
        torch.onnx.export(module, (sample,), onnx, input_names=["main"], output_names=[output_name], opset_version=17)
        bundle_data = executor.execute_segment(segment_id, bundle_data)
        if converter and not args.onnx_only:
            subprocess.run([converter, "-f", "ONNX", "--modelFile", str(onnx), "--MNNModel", str(paths.mnn_root / f"segment_{segment_id}.mnn")], check=True)
    for item in manifest.early_exits:
        if item.get("final"):
            continue
        exit_id = str(item["exit_id"])
        meta = manifest.boundaries[int(item["boundary_id"])]["live_tensors"][0]
        sample = torch.randn(tuple(meta["shape"]))
        onnx = paths.mnn_root / f"exit_{exit_id}.onnx"
        torch.onnx.export(ExitHead(model, exit_id), (sample,), onnx, input_names=["main"], output_names=["logits"], opset_version=17)
        if converter and not args.onnx_only:
            subprocess.run([converter, "-f", "ONNX", "--modelFile", str(onnx), "--MNNModel", str(paths.mnn_root / f"exit_{exit_id}.mnn")], check=True)


if __name__ == "__main__":
    main()
