"""
Scripts/Exp0_Motivation/exp1_decoupling_failure/simulator.py
端到端推理时延模拟器（计算 + 传输），对应论文 E[V(τ)] 耦合公式。
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Scripts.Exp0_Motivation.utils.config import DEVICE_GFLOPS, EDGE_GFLOPS, RTT_MS
from Scripts.Exp0_Motivation.utils.network_sim import (
    compute_latency_ms,
    transmission_latency_ms,
)


def _transmission_ratio(
    split_layer: int, exit_rates_per_head: list[float], exit_layer_indices: list[int]
) -> float:
    """
    有效传输比例 = 1 - Σ r_i，其中早退头层索引 <= split_layer。
    """
    ratio = 1.0
    for rate, layer in zip(exit_rates_per_head, exit_layer_indices):
        if layer <= split_layer:
            ratio -= float(rate)
    return max(0.0, ratio)


def simulate_latency(
    config: dict,
    exit_rates_per_head: list[float],
    model_info: dict,
    device_gflops: float = DEVICE_GFLOPS,
    edge_gflops: float = EDGE_GFLOPS,
    rtt_ms: float = RTT_MS,
) -> tuple[float, float]:
    """
    模拟端到端期望时延（ms）与传输比例。

    Args:
        config: 含 split_layer, bandwidth_mbps
        exit_rates_per_head: 各早退头退出率 (0~1)
        model_info: ResNet50EEWrapper.get_model_info() 返回值

    Returns:
        (avg_latency_ms, transmission_ratio)
    """
    split_layer = int(config.get("split_layer", 0))
    bandwidth_mbps = float(config["bandwidth_mbps"])
    exit_layers = model_info["exit_layer_indices"]
    total_flops = float(model_info["total_flops"])

    # Local：split_layer=0，全量本地推理
    if split_layer <= 0:
        lat = compute_latency_ms(total_flops, device_gflops)
        return lat, 0.0

    flops_up_to = float(model_info["flops_cumulative"][split_layer])
    device_latency_ms = compute_latency_ms(flops_up_to, device_gflops)

    tx_ratio = _transmission_ratio(split_layer, exit_rates_per_head, exit_layers)

    feature_bytes = int(model_info["feature_bytes"][split_layer])
    tx_ms = transmission_latency_ms(feature_bytes, bandwidth_mbps, rtt_ms)
    transmission_latency_ms_val = tx_ms * tx_ratio

    remaining_flops = max(0.0, total_flops - flops_up_to)
    edge_latency_ms = (
        compute_latency_ms(remaining_flops, edge_gflops) * tx_ratio
    )

    end_to_end = (
        device_latency_ms + transmission_latency_ms_val + edge_latency_ms
    )
    return float(end_to_end), float(tx_ratio)
