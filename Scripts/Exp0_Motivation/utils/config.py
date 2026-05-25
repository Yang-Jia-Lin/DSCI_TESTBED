"""
Scripts/Exp0_Motivation/utils/config.py
共享超参数（硬件算力、网络、调度、实验种子）。
"""

# 硬件算力参数（动机图强调对比时可略极端，主实验 testbed 仍用实测标定）
DEVICE_GFLOPS = 0.16  # 端侧偏弱 → Local 基线升高，深切分+高传输更易倒置
EDGE_GFLOPS = 8.0  # 边缘算力 (GFLOPS)

# 网络参数
RTT_MS = 20.0  # 固定往返时延 (ms)

# 调度参数
DECISION_LATENCY_MS = 2.0  # DRL 单次前向推理时延 (ms)
OPTIMIZATION_LATENCY_MS = 500.0  # PPO+凸优化单周期时延 (ms)
SCHEDULE_PERIOD_S = 30.0  # 准静态调度周期 (s)

# 实验参数
N_SAMPLES = 1000
RANDOM_SEED = 42

# ResNet50 拓扑（来自 Src/Configs/model_config.py）
NUM_LAYERS = 128
EARLY_EXIT_LAYERS = [57, 103]
FINAL_LAYER = 127

# 解耦 Step1 在窄带下易选过深切分点（{57,103,127} 中偏向末层），放大 E[V]=V_max 失效
DECOUPLED_NARROW_BW_MBPS = 2.0
