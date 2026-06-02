"""
Scripts/Exp1_Offline/explore_nn_meter.py

探索 nn-Meter API：
  1) 验证安装
  2) 测试整体预测
  3) 提取逐 kernel 时延（核心）
  4) 映射 kernel → DSCI 层索引
"""

import sys
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

# ── nn-Meter imports ─────────────────────────────────────────────
from nn_meter import load_latency_predictor, list_latency_predictors
from nn_meter.ir_converter import model_to_graph
from nn_meter.kernel_detector import KernelDetector
from nn_meter.predictor.prediction.predict_by_kernel import nn_predict
from nn_meter.predictor.prediction.extract_feature import get_predict_features
from nn_meter.predictor.prediction.utils import get_kernel_name

# ── 项目 imports ─────────────────────────────────────────────────
from Src.Models.ModelNet.Resnet50 import MultiEEResNet50, Bottleneck

# suppress excessive logging
logging.getLogger("nn-Meter").setLevel(logging.WARNING)


# ================================================================
#  Part 0: 辅助 — 拆出 kernel-level 时延
# ================================================================
def predict_per_kernel(predictors, kernel_units):
    """
    与 nn_predict() 等价，但返回逐 kernel 的时延列表而不是总和。

    Returns:
        list of dict:  [{
            "kernel_name": str,       # e.g. "conv-bn-relu#3"
            "op":          str,       # 融合后的算子类型 e.g. "conv-bn-relu"
            "features":    list,
            "latency_ms":  float,
        }, ...]
    """
    from nn_meter.predictor.prediction.predict_by_kernel import merge_conv_kernels

    features_dict = get_predict_features(kernel_units)
    results = []

    # features_dict: {layer_idx: {op: features_list}}
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

        results.append({
            "layer_idx": layer_idx,
            "op": op,
            "kernel_predictor": kernelname,
            "features": feats,
            "latency_ms": lat,
        })

    return results


# ================================================================
#  Part 1: 构建分段模型 wrapper（处理 early exit）
# ================================================================
class MainBackbone(nn.Module):
    """去掉早退出分支的纯主干 ResNet50"""
    def __init__(self, full_model):
        super().__init__()
        self.conv1 = full_model.conv1
        self.bn1 = full_model.bn1
        self.relu = full_model.relu
        self.maxpool = full_model.maxpool
        self.layer1 = full_model.layer1
        self.layer2 = full_model.layer2
        self.layer3 = full_model.layer3
        self.layer4 = full_model.layer4
        self.avgpool = full_model.avgpool
        self.fc = full_model.fc

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class EarlyExitBranch2(nn.Module):
    """layer2 后的早退出分支：avgpool + fc2"""
    def __init__(self, full_model):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc2 = full_model.fc2

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x


class EarlyExitBranch3(nn.Module):
    """layer3 后的早退出分支：avgpool + fc3"""
    def __init__(self, full_model):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = full_model.fc3

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc3(x)
        return x


# ================================================================
#  Main
# ================================================================
def main():
    predictor_name = "cortexA76cpu_tflite21"
    input_shape = (1, 3, 227, 227)

    # ── Step 1: 列出可用 predictor ──
    print("=" * 60)
    print("Step 1: Available predictors")
    print("=" * 60)
    preds = list_latency_predictors()
    for p in preds:
        print(f"  {p['name']}  (v{p.get('version', '?')})")

    # ── Step 2: 加载 predictor ──
    print(f"\nLoading predictor: {predictor_name}")
    predictor = load_latency_predictor(predictor_name)
    print(f"  Kernel predictors: {list(predictor.kernel_predictors.keys())}")

    # ── Step 3: 创建模型 ──
    full_model = MultiEEResNet50(Bottleneck, [3, 4, 6, 3], num_classes=10)
    full_model.eval()

    # ── Step 4: 整体预测（主干） ──
    print("\n" + "=" * 60)
    print("Step 2: Total model prediction (backbone only)")
    print("=" * 60)
    backbone = MainBackbone(full_model)
    backbone.eval()
    total_lat = predictor.predict(backbone, model_type="torch", input_shape=input_shape)
    print(f"  Backbone total latency: {total_lat:.4f} ms")

    # ── Step 5: 提取逐 kernel 时延（核心实验） ──
    print("\n" + "=" * 60)
    print("Step 3: Per-kernel latency breakdown (backbone)")
    print("=" * 60)

    # 手动执行内部流程
    graph = model_to_graph(backbone, "torch", input_shape=input_shape)
    predictor.kd.load_graph(graph)
    kernel_units = predictor.kd.get_kernels()

    print(f"\n  Total kernel units detected: {len(kernel_units)}")
    print(f"\n  Raw kernel units:")
    for i, ku in enumerate(kernel_units):
        print(f"    [{i:3d}] op={ku['op']:<30s}  name={ku['name']}")
        if 'cin' in ku:
            print(f"          cin={ku.get('cin')}  cout={ku.get('cout')}  "
                  f"inputh={ku.get('inputh')}  ks={ku.get('ks')}")

    # 逐 kernel 预测
    kernel_lats = predict_per_kernel(predictor.kernel_predictors, kernel_units)

    total_from_kernels = 0.0
    print(f"\n  Per-kernel latency predictions:")
    for kl in kernel_lats:
        total_from_kernels += kl["latency_ms"]
        print(f"    [{kl['layer_idx']:3d}] {kl['op']:<30s} → "
              f"{kl['kernel_predictor']:<20s}  {kl['latency_ms']:.4f} ms")

    print(f"\n  Sum of kernel latencies: {total_from_kernels:.4f} ms")
    print(f"  Total from predict():    {total_lat:.4f} ms")
    print(f"  Difference:              {abs(total_from_kernels - total_lat):.4f} ms")

    # ── Step 6: 早退出分支 ──
    print("\n" + "=" * 60)
    print("Step 4: Early exit branches")
    print("=" * 60)

    # Branch 2 input shape: layer2 output = (1, 512, 4, 4) for CIFAR
    branch2 = EarlyExitBranch2(full_model)
    branch2.eval()
    branch2_shape = (1, 512, 4, 4)
    try:
        branch2_lat = predictor.predict(branch2, model_type="torch", input_shape=branch2_shape)
        print(f"  Branch2 (avgpool+fc2) latency: {branch2_lat:.4f} ms")
    except Exception as e:
        print(f"  Branch2 prediction failed: {e}")
        print("  Falling back to manual per-layer estimation...")
        # avgpool + fc 时延可以手动估计
        branch2_lat = 0.0

    # Branch 3 input shape: layer3 output = (1, 1024, 2, 2)
    branch3 = EarlyExitBranch3(full_model)
    branch3.eval()
    branch3_shape = (1, 1024, 2, 2)
    try:
        branch3_lat = predictor.predict(branch3, model_type="torch", input_shape=branch3_shape)
        print(f"  Branch3 (avgpool+fc3) latency: {branch3_lat:.4f} ms")
    except Exception as e:
        print(f"  Branch3 prediction failed: {e}")
        branch3_lat = 0.0

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Backbone latency:   {total_lat:.4f} ms")
    print(f"  Branch2 latency:    {branch2_lat:.4f} ms")
    print(f"  Branch3 latency:    {branch3_lat:.4f} ms")
    print(f"  Kernel count:       {len(kernel_lats)}")


if __name__ == "__main__":
    main()
