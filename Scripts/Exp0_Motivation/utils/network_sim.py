"""
Scripts/Exp0_Motivation/utils/network_sim.py
带宽/RTT 时延辅助函数。
"""


def transmission_latency_ms(
    feature_bytes: int, bandwidth_mbps: float, rtt_ms: float
) -> float:
    """
    计算传输时延（ms）= 传播时延 + 往返时延。

    Args:
        feature_bytes: 特征数据量 (bytes)
        bandwidth_mbps: 带宽 (Mbps)
        rtt_ms: 往返时延 (ms)

    Returns:
        传输时延 (ms)
    """
    if bandwidth_mbps <= 0:
        return float("inf")
    bandwidth_bps = bandwidth_mbps * 1e6 / 8.0
    prop_ms = feature_bytes / bandwidth_bps * 1000.0
    return prop_ms + rtt_ms


def compute_latency_ms(flops: float, gflops: float) -> float:
    """
    计算纯计算时延（ms）。

    Args:
        flops: 运算量 (MACs / FLOPs)
        gflops: 算力 (GFLOPS)

    Returns:
        计算时延 (ms)
    """
    if gflops <= 0:
        return float("inf")
    return flops / (gflops * 1e9) * 1000.0
