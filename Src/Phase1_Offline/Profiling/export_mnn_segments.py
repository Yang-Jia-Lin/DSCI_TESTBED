"""Export every manifest segment to ONNX and optionally convert it to MNN."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

from Src.Shared.Config.paths import DATA_DIR, RESNET50_PATHS
from Src.Shared.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50
from Src.Shared.Partitioning.manifest import load_partition_manifest


class StemSegment(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.conv1, self.bn1, self.relu, self.maxpool = (
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

    def forward(self, main):
        return self.maxpool(self.relu(self.bn1(self.conv1(main))))


class FinalPoolSegment(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.avgpool = model.avgpool

    def forward(self, main):
        return torch.flatten(self.avgpool(main), 1)


class ExitHeadSegment(nn.Module):
    def __init__(self, model, head_name):
        super().__init__()
        self.avgpool = model.avgpool
        self.head = getattr(model, head_name)

    def forward(self, main):
        return self.head(torch.flatten(self.avgpool(main), 1))


def _resolve_module(model, name):
    if name == "stem":
        return StemSegment(model)
    if name == "final_pool":
        return FinalPoolSegment(model)
    if name == "final_classifier":
        return model.fc
    module = model
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-id", default="resnet50-cifar10-partition-v1")
    parser.add_argument("--output-root")
    parser.add_argument("--mnnconvert", default="MNNConvert")
    parser.add_argument("--onnx-only", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_partition_manifest(args.manifest_id)
    root = Path(
        args.output_root
        or DATA_DIR / "Weights" / "mnn_segments" / manifest.manifest_id
    )
    root.mkdir(parents=True, exist_ok=True)
    model = MultiEEResNet50(
        Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True
    )
    model.load_state_dict(
        torch.load(RESNET50_PATHS.resolve_weight_path(), map_location="cpu", weights_only=True)
    )
    model.eval()
    bundle = {"main": torch.randn(1, 3, 227, 227)}
    converter = shutil.which(args.mnnconvert)
    if not args.onnx_only and converter is None:
        raise FileNotFoundError(f"MNN converter not found: {args.mnnconvert}")

    for segment in manifest.segments:
        sid = int(segment["segment_id"])
        module = _resolve_module(model, str(segment["name"])).eval()
        input_name = str(segment["input_names"][0])
        output_name = str(segment["output_names"][0])
        sample = bundle[input_name]
        onnx_path = root / f"segment_{sid}.onnx"
        torch.onnx.export(
            module,
            (sample,),
            onnx_path,
            input_names=[input_name],
            output_names=[output_name],
            opset_version=17,
            dynamo=False,
        )
        with torch.no_grad():
            bundle = {output_name: module(sample)}
            if output_name == "logits" and sid + 1 < manifest.final_boundary_id:
                bundle = {"main": bundle["logits"]}
        if converter is not None and not args.onnx_only:
            subprocess.run(
                [
                    converter,
                    "-f",
                    "ONNX",
                    "--modelFile",
                    str(onnx_path),
                    "--MNNModel",
                    str(root / f"segment_{sid}.mnn"),
                    "--bizCode",
                    manifest.manifest_id,
                ],
                check=True,
            )
    for item in manifest.early_exits:
        boundary_id = int(item["boundary_id"])
        if boundary_id == manifest.final_boundary_id:
            continue
        logical_layer = int(item["logical_layer"])
        head = ExitHeadSegment(model, "fc2" if logical_layer == 57 else "fc3").eval()
        boundary = manifest.boundaries[boundary_id]
        main_meta = next(
            tensor for tensor in boundary["live_tensors"] if tensor["name"] == "main"
        )
        sample = torch.randn(tuple(int(value) for value in main_meta["shape"]))
        onnx_path = root / f"exit_{logical_layer}.onnx"
        torch.onnx.export(
            head,
            (sample,),
            onnx_path,
            input_names=["main"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )
        if converter is not None and not args.onnx_only:
            subprocess.run(
                [
                    converter,
                    "-f",
                    "ONNX",
                    "--modelFile",
                    str(onnx_path),
                    "--MNNModel",
                    str(root / f"exit_{logical_layer}.mnn"),
                    "--bizCode",
                    manifest.manifest_id,
                ],
                check=True,
            )
    print(f"Exported {len(manifest.segments)} segments under {root}")


if __name__ == "__main__":
    main()
