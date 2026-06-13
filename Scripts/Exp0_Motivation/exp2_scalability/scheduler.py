"""
Scripts/Exp0_Motivation/exp2_scalability/scheduler.py
每请求调度 vs. DSCI 准静态调度的解析开销模型。
"""

from __future__ import annotations

import numpy as np

from Scripts.Exp0_Motivation.utils.config import RANDOM_SEED


class PerRequestScheduler:
    """
    每请求触发一次控制决策的调度模型。
    对应 MEOCI (TPDS 2026) / I-SplitEE (ICC 2024) 类方法。
    """

    def __init__(
        self,
        state_vector_dim: int = 20,
        config_bytes: int = 16,
        decision_latency_ms: float = 2.0,
        bandwidth_mbps: float = 4.0,
        rtt_ms: float = 20.0,
    ):
        self.state_vector_dim = state_vector_dim
        self.config_bytes = config_bytes
        self.decision_latency_ms = decision_latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.rtt_ms = rtt_ms

    def _bandwidth_bps(self) -> float:
        return self.bandwidth_mbps * 1e6 / 8.0

    def compute_scheduling_overhead_ms(self, n_users: int) -> float:
        """
        N 个并发用户时，单次调度轮次的总开销（ms）。

        = max(state_collection_latency_per_user)
          + n_users * decision_latency_ms
          + dispatch_latency_per_user * n_users
        """
        n_users = max(1, int(n_users))
        bps = self._bandwidth_bps()
        state_bytes = self.state_vector_dim * 4
        state_collection_ms = state_bytes / bps * 1000.0 + self.rtt_ms / 2.0
        dispatch_ms = self.config_bytes / bps * 1000.0 + self.rtt_ms / 2.0
        return (
            state_collection_ms
            + n_users * self.decision_latency_ms
            + n_users * dispatch_ms
        )

    def compute_config_drift(
        self,
        n_users: int,
        n_requests_per_user: int = 20,
        seed: int = RANDOM_SEED,
    ) -> float:
        """
        模拟 N 用户 × K 请求的配置漂移（归一化 0~1）。

        跨用户方差随 N 增大而上升（O(N) 漂移）；每用户内 K 次请求的抖动亦计入。
        """
        n_users = max(1, int(n_users))
        split_opt, tau_opt = 0.45, 0.80
        w1, w2 = 0.5, 0.5

        user_mean_splits: list[float] = []
        user_mean_taus: list[float] = []
        within_splits: list[float] = []
        within_taus: list[float] = []

        # 并发越高，控制器负载越大，逐请求决策噪声越大
        tau_sigma = 0.03 + 0.004 * n_users
        split_jump_p = min(0.55, 0.22 + 0.012 * n_users)

        for u in range(n_users):
            rng = np.random.default_rng(seed + u * 9973)
            splits_u: list[float] = []
            taus_u: list[float] = []
            for _ in range(n_requests_per_user):
                if rng.random() < split_jump_p:
                    delta = rng.choice([-0.10, -0.05, 0.0, 0.05, 0.10])
                else:
                    delta = 0.0
                splits_u.append(float(np.clip(split_opt + delta, 0.0, 1.0)))
                taus_u.append(
                    float(np.clip(tau_opt + rng.normal(0, tau_sigma), 0.0, 1.0))
                )
            user_mean_splits.append(float(np.mean(splits_u)))
            user_mean_taus.append(float(np.mean(taus_u)))
            if len(splits_u) > 1:
                within_splits.append(float(np.var(splits_u)))
                within_taus.append(float(np.var(taus_u)))

        cross_split = float(np.var(user_mean_splits)) if n_users > 1 else 0.0
        cross_tau = float(np.var(user_mean_taus)) if n_users > 1 else 0.0
        mean_within_split = float(np.mean(within_splits)) if within_splits else 0.0
        mean_within_tau = float(np.mean(within_taus)) if within_taus else 0.0

        drift = (
            w1 * (cross_split + 0.35 * mean_within_split)
            + w2 * (cross_tau + 0.35 * mean_within_tau) / (0.25**2)
        )
        return float(min(1.0, drift))

    def compute_effective_throughput(
        self,
        n_users: int,
        inference_latency_ms: float = 50.0,
        simulation_duration_s: float = 60.0,
    ) -> float:
        """
        扣除调度开销后的有效推理吞吐量（requests/s）。
        """
        n_users = max(1, int(n_users))
        overhead_round = self.compute_scheduling_overhead_ms(n_users)
        overhead_per_req = overhead_round / n_users
        per_req_total = inference_latency_ms + overhead_per_req
        if per_req_total <= 0:
            return 0.0
        return n_users / per_req_total * 1000.0


class QuasiStaticScheduler:
    """
    DSCI 准静态调度：每周期一次优化 + 广播，开销摊销到周期内请求。
    """

    def __init__(
        self,
        optimization_latency_ms: float = 500.0,
        config_bytes: int = 16,
        schedule_period_s: float = 30.0,
        bandwidth_mbps: float = 4.0,
        rtt_ms: float = 20.0,
    ):
        self.optimization_latency_ms = optimization_latency_ms
        self.config_bytes = config_bytes
        self.schedule_period_s = schedule_period_s
        self.bandwidth_mbps = bandwidth_mbps
        self.rtt_ms = rtt_ms

    def _bandwidth_bps(self) -> float:
        return self.bandwidth_mbps * 1e6 / 8.0

    def compute_scheduling_overhead_ms(
        self, n_users: int, inference_latency_ms: float = 50.0
    ) -> float:
        """
        单次调度周期总开销（ms）；摊销到每请求见 overhead_per_request_ms。
        """
        n_users = max(1, int(n_users))
        bps = self._bandwidth_bps()
        broadcast_ms = self.config_bytes / bps * 1000.0 + self.rtt_ms / 2.0
        return self.optimization_latency_ms + broadcast_ms

    def compute_config_drift(self, n_users: int, **kwargs) -> float:
        """周期内配置一致，drift = 0。"""
        return 0.0

    def overhead_per_request_ms(
        self, n_users: int, inference_latency_ms: float
    ) -> float:
        """周期内摊销到单请求的调度开销（ms/req）。"""
        n_users = max(1, int(n_users))
        period_overhead = self.compute_scheduling_overhead_ms(n_users)
        req_per_period = n_users * (
            self.schedule_period_s * 1000.0 / max(inference_latency_ms, 1e-6)
        )
        req_per_period = max(req_per_period, 1.0)
        return period_overhead / req_per_period

    def compute_effective_throughput(
        self,
        n_users: int,
        inference_latency_ms: float = 50.0,
        simulation_duration_s: float = 60.0,
    ) -> float:
        """近似线性扩展：调度开销摊销后接近纯推理吞吐。"""
        n_users = max(1, int(n_users))
        overhead = self.overhead_per_request_ms(n_users, inference_latency_ms)
        per_req = inference_latency_ms + overhead
        if per_req <= 0:
            return 0.0
        return n_users / per_req * 1000.0
