"""
Scripts/Exp1_Offline/profile_nn_meter_layer_stats.py

nn-Meter 离线 Profiling 脚本 — 数据驱动映射

Usage:
  cd DSCI_testbed
  set PYTHONPATH=D:\\Coding\\Python\\DSCI_testbed
  python Scripts/Exp1_Offline/profile_nn_meter_layer_stats.py
"""

import argparse
import json
import logging
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nn_meter import load_latency_predictor
from nn_meter.ir_converter import model_to_graph
from nn_meter.predictor.prediction.extract_feature import get_predict_features
from nn_meter.predictor.prediction.predict_by_kernel import merge_conv_kernels
from nn_meter.predictor.prediction.utils import get_kernel_name

from Src.Models.ModelNet.Resnet50 import MultiEEResNet50, Bottleneck

logging.getLogger("nn-Meter").setLevel(logging.WARNING)


# ================================================================
#  逐 Kernel 时延提取
# ================================================================
def predict_per_kernel(predictors, kernel_units):
    features_dict = get_predict_features(kernel_units)
    results = []
    for layer_idx in sorted(features_dict.keys()):
        op = list(features_dict[layer_idx].keys())[0]
        feats = features_dict[layer_idx][op]
        rkernel = merge_conv_kernels(op)
        kernelname = get_kernel_name(rkernel)
        lat = 0.0
        if kernelname in predictors:
            pred = predictors[kernelname]
            pys = pred.predict([feats])
            if len(pys) > 0:
                lat = float(pys[0])
        ku = kernel_units[layer_idx] if layer_idx < len(kernel_units) else {}
        results.append({
            "kernel_idx": layer_idx,
            "op": op,
            "name": ku.get("name", f"kernel#{layer_idx}"),
            "kernel_predictor": kernelname,
            "cin": ku.get("cin"),
            "cout": ku.get("cout"),
            "inputh": ku.get("inputh"),
            "latency_ms": lat,
        })
    return results


# ================================================================
#  分段模型
# ================================================================
class MainBackbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv1, self.bn1, self.relu, self.maxpool = m.conv1, m.bn1, m.relu, m.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.avgpool, self.fc = m.avgpool, m.fc

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class EarlyExitBranch2(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = m.fc2

    def forward(self, x):
        return self.fc2(torch.flatten(self.avgpool(x), 1))


class EarlyExitBranch3(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = m.fc3

    def forward(self, x):
        return self.fc3(torch.flatten(self.avgpool(x), 1))


# ================================================================
#  Kernel → DSCI Layer 映射（数据驱动）
# ================================================================
def map_kernels_to_dsci_layers(kernel_lats, csv_layer_names):
    """
    数据驱动映射：遍历 nn-Meter 的 kernel 列表，按 ResNet50 Bottleneck
    结构顺序与 CSV 层名对应。

    nn-Meter 对 ResNet50 Bottleneck 的 kernel 模式（32x32 输入下观察到的）：
      Stem:
        [0] conv-bn-relu  → conv1 (+bn1,relu fused)
        [1] maxpool       → maxpool

      每个 Bottleneck Block:
        如果有 downsample:
          [k] conv-bn-relu  → downsample.0 (+downsample.1 fused)
        [k+1] conv-bn-relu  → conv1 (+bn1 fused)
        [k+2] conv-bn-relu  → conv2 (+bn2 fused, 含relu)
        [k+3] conv-bn       → conv3 (+bn3 fused, 无relu)
        [k+4] add-relu      → relu (residual add+relu)

      Tail:
        [n-2] global-avgpool → (不在 CSV 主层中，合并到 fc)
        [n-1] fc             → fc
    """
    latency_map = OrderedDict((ln, 0.0) for ln in csv_layer_names)
    k_idx = 0

    def safe_get_lat():
        nonlocal k_idx
        if k_idx < len(kernel_lats):
            lat = kernel_lats[k_idx]["latency_ms"]
            k_idx += 1
            return lat
        return 0.0

    # Stem
    latency_map["conv1"] = safe_get_lat()          # conv-bn-relu → conv1(+bn1+relu)
    latency_map["maxpool"] = safe_get_lat()          # maxpool

    # Helper for one Bottleneck
    def map_block(prefix, has_ds):
        if has_ds:
            latency_map[f"{prefix}.downsample.0"] = safe_get_lat()  # conv-bn-relu
        latency_map[f"{prefix}.conv1"] = safe_get_lat()              # conv-bn-relu (1x1)
        latency_map[f"{prefix}.conv2"] = safe_get_lat()              # conv-bn-relu (3x3)
        latency_map[f"{prefix}.conv3"] = safe_get_lat()              # conv-bn     (1x1)
        latency_map[f"{prefix}.relu"] = safe_get_lat()               # add-relu

    # Layer1: [3, ...], first has downsample
    for i in range(3):
        map_block(f"layer1.{i}", has_ds=(i == 0))

    # Layer2: [_, 4, ...], first has downsample
    for i in range(4):
        map_block(f"layer2.{i}", has_ds=(i == 0))

    # Layer3: [_, _, 6, ...], first has downsample
    for i in range(6):
        map_block(f"layer3.{i}", has_ds=(i == 0))

    # Layer4: [_, _, _, 3], first has downsample
    for i in range(3):
        map_block(f"layer4.{i}", has_ds=(i == 0))

    # Tail: global-avgpool + fc
    avgpool_lat = safe_get_lat()  # global-avgpool
    fc_lat = safe_get_lat()       # fc
    latency_map["fc"] = avgpool_lat + fc_lat

    return latency_map, k_idx


# ================================================================
#  Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", default="cortexA76cpu_tflite21")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 32, 32])
    parser.add_argument("--output_dir", default="Data/OfflineTables")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_shape = tuple(args.input_shape)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Predictor:   {args.predictor}")
    print(f"Input shape: {input_shape}")

    # Step 1: Load predictor
    predictor = load_latency_predictor(args.predictor)

    # Step 2: Create model
    full_model = MultiEEResNet50(Bottleneck, [3, 4, 6, 3], num_classes=10)
    full_model.eval()

    # Step 3: Profile backbone
    print("\nProfiling backbone...")
    backbone = MainBackbone(full_model)
    backbone.eval()
    graph = model_to_graph(backbone, "torch", input_shape=input_shape)
    predictor.kd.load_graph(graph)
    kernel_units = predictor.kd.get_kernels()
    backbone_kernels = predict_per_kernel(predictor.kernel_predictors, kernel_units)

    total_backbone = sum(k["latency_ms"] for k in backbone_kernels)
    print(f"  {len(backbone_kernels)} kernels, total {total_backbone:.4f} ms")

    # Print kernel list
    for k in backbone_kernels:
        print(f"  [{k['kernel_idx']:3d}] {k['op']:<30s}  {k['latency_ms']:.4f} ms  "
              f"cin={k['cin']} cout={k['cout']} inputh={k['inputh']}")

    # Step 4: Profile early exit branches
    print("\nProfiling early exit branches...")
    with torch.no_grad():
        dummy = torch.randn(*input_shape)
        x = full_model.maxpool(full_model.relu(full_model.bn1(full_model.conv1(dummy))))
        x = full_model.layer1(x)
        x2 = full_model.layer2(x)
        x3 = full_model.layer3(x2)
        branch2_shape = tuple(x2.shape)
        branch3_shape = tuple(x3.shape)

    branch2_lat = predictor.predict(EarlyExitBranch2(full_model).eval(), "torch", input_shape=branch2_shape)
    branch3_lat = predictor.predict(EarlyExitBranch3(full_model).eval(), "torch", input_shape=branch3_shape)
    print(f"  Branch2 ({branch2_shape}): {branch2_lat:.4f} ms")
    print(f"  Branch3 ({branch3_shape}): {branch3_lat:.4f} ms")

    # Step 5: Read CSV
    csv_path = output_dir / "resnet50_cifar10_layer_stats.csv"
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["layer"] = df["layer"].str.strip()
    layer_names = df["layer"].tolist()
    print(f"\nCSV has {len(layer_names)} layers")

    # Step 6: Map kernels
    latency_map, consumed = map_kernels_to_dsci_layers(backbone_kernels, layer_names)
    print(f"Consumed {consumed}/{len(backbone_kernels)} kernels")

    # Add early exit branches
    if "avgpool" in latency_map:
        latency_map["avgpool"] = branch2_lat * 0.5
    if "fc2" in latency_map:
        latency_map["fc2"] = branch2_lat * 0.5
    if "fc3" in latency_map:
        latency_map["fc3"] = branch3_lat

    # Step 7: Build latency column
    nn_lat_col = [latency_map.get(ln, 0.0) for ln in layer_names]
    df["nn_meter_latency_ms"] = nn_lat_col

    total_mapped = sum(nn_lat_col)
    nonzero = sum(1 for v in nn_lat_col if v > 0)
    print(f"\nTotal mapped latency: {total_mapped:.4f} ms")
    print(f"Non-zero layers: {nonzero}/{len(nn_lat_col)}")

    # Step 8: Save
    if csv_path.exists() and not args.overwrite:
        backup = csv_path.with_suffix(f".backup_{datetime.now():%Y%m%d%H%M%S}.csv")
        shutil.copy2(csv_path, backup)
        print(f"Backed up to: {backup}")

    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Step 9: Audit JSON
    audit_path = output_dir / "resnet50_cifar10_kernel_profile.json"
    audit = {
        "predictor": args.predictor,
        "input_shape": list(input_shape),
        "timestamp": datetime.now().isoformat(),
        "backbone_total_ms": total_backbone,
        "branch2_total_ms": branch2_lat,
        "branch3_total_ms": branch3_lat,
        "total_mapped_ms": total_mapped,
        "kernels": [{
            "idx": k["kernel_idx"], "op": k["op"], "name": k["name"],
            "cin": k["cin"], "cout": k["cout"], "inputh": k["inputh"],
            "latency_ms": round(k["latency_ms"], 6),
        } for k in backbone_kernels],
        "layer_latencies": {ln: round(v, 6) for ln, v in latency_map.items() if v > 0},
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"Saved audit: {audit_path}")

    # Step 10: Comparison table
    print(f"\n{'='*80}")
    print(f"{'Layer':<25s} {'FLOPs':>12s} {'nn-Meter(ms)':>14s}")
    print(f"{'='*80}")
    for _, row in df.iterrows():
        ln = row["layer"]
        flops = int(row["approx_flops"])
        nn_lat = row["nn_meter_latency_ms"]
        marker = "" if nn_lat > 0 else "  (fused→0)"
        print(f"{ln:<25s} {flops:>12d} {nn_lat:>14.4f}{marker}")

    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()
