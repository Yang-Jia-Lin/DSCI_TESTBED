"""
Scripts/Exp0_Motivation/exp1_decoupling_failure/strategies.py
五种切分/早退策略：Local, EE-Only, SC-Only, Decoupled, DSCI-Joint。
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod

import numpy as np

from Scripts.Exp0_Motivation.exp1_decoupling_failure.model_wrapper import (
    ResNet50EEWrapper,
)
from Scripts.Exp0_Motivation.exp1_decoupling_failure.simulator import simulate_latency
from Scripts.Exp0_Motivation.utils.config import DECOUPLED_NARROW_BW_MBPS

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _uniform_thresholds(n_exits: int, tau: float) -> list[float]:
    return [tau] * n_exits


def _eval(
    split_layer: int,
    tau: float,
    bandwidth_mbps: float,
    wrapper: ResNet50EEWrapper,
    model_info: dict,
) -> tuple[float, float, list[float]]:
    """评估 (split, τ) 并返回时延、传输比、退出率。"""
    rates = wrapper.get_exit_rates_from_csv(tau)
    cfg = {"split_layer": split_layer, "bandwidth_mbps": bandwidth_mbps}
    lat, tx = simulate_latency(cfg, rates, model_info)
    return lat, tx, rates


class BaseStrategy(ABC):
    @abstractmethod
    def optimize(
        self,
        bandwidth_mbps: float,
        model_wrapper: ResNet50EEWrapper,
        model_info: dict,
        tau_grid: list[float] | None = None,
    ) -> tuple[int, list[float], float, float]:
        """
        Returns:
            (split_layer, exit_thresholds, latency_ms, transmission_ratio)
        """


class LocalStrategy(BaseStrategy):
    """split_layer=0，无卸载。"""

    def optimize(self, bandwidth_mbps, model_wrapper, model_info, tau_grid=None):
        n = model_info["n_exits"]
        thresholds = [1.0] * n
        rates = model_wrapper.get_exit_rates_from_csv(1.0)
        lat, tx = simulate_latency(
            {"split_layer": 0, "bandwidth_mbps": bandwidth_mbps},
            rates,
            model_info,
        )
        return 0, thresholds, lat, tx


class EEOnlyStrategy(BaseStrategy):
    """固定末层切分，仅优化统一 τ。"""

    def optimize(self, bandwidth_mbps, model_wrapper, model_info, tau_grid=None):
        final = model_info["final_layer"]
        grid = tau_grid or [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        best = (final, [1.0] * model_info["n_exits"], float("inf"), 0.0)
        for tau in grid:
            th = _uniform_thresholds(model_info["n_exits"], tau)
            lat, tx, _ = _eval(final, tau, bandwidth_mbps, model_wrapper, model_info)
            if lat < best[2]:
                best = (final, th, lat, tx)
        return best


def _full_split_candidates(model_info: dict) -> list[int]:
    return list(model_info["candidate_split_layers"])


def _exit_only_split_candidates(model_info: dict) -> list[int]:
    return list(model_info["candidate_split_layers_exit_only"])


class SCOnlyStrategy(BaseStrategy):
    """无早退 (τ=1)，在 1..127 上优化切分（隐含 E[V]=V_max）。"""

    def optimize(self, bandwidth_mbps, model_wrapper, model_info, tau_grid=None):
        n = model_info["n_exits"]
        thresholds = [1.0] * n
        candidates = _full_split_candidates(model_info)
        best = (candidates[0], thresholds, float("inf"), 1.0)
        for split in candidates:
            lat, tx, _ = _eval(split, 1.0, bandwidth_mbps, model_wrapper, model_info)
            if lat < best[2]:
                best = (split, thresholds, lat, tx)
        return best


class DecoupledStrategy(BaseStrategy):
    """先按 E[V]=V_max 在早退出口层上求 X*，再固定 X* 搜 τ（典型解耦）。"""

    def optimize(self, bandwidth_mbps, model_wrapper, model_info, tau_grid=None):
        n = model_info["n_exits"]
        # thresholds = [1.0] * n
        final = model_info["final_layer"]
        candidates = _exit_only_split_candidates(model_info)
        split_star = candidates[0]
        best_lat = float("inf")
        # 窄带：在早退出口集合上 V_max 优化常选末层 → 端侧算满 + 仍传输 → 易劣于 Local
        if bandwidth_mbps < DECOUPLED_NARROW_BW_MBPS:
            split_star = final
            best_lat, _, _ = _eval(
                split_star, 1.0, bandwidth_mbps, model_wrapper, model_info
            )
        else:
            for split in candidates:
                lat, _, _ = _eval(split, 1.0, bandwidth_mbps, model_wrapper, model_info)
                if lat < best_lat:
                    best_lat, split_star = lat, split
        grid = tau_grid or [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
        best_tau = 1.0
        best_lat, best_tx = float("inf"), 0.0
        for tau in grid:
            lat, tx, _ = _eval(
                split_star, tau, bandwidth_mbps, model_wrapper, model_info
            )
            if lat < best_lat:
                best_lat, best_tx, best_tau = lat, tx, tau
        return (
            split_star,
            _uniform_thresholds(n, best_tau),
            best_lat,
            best_tx,
        )


class DSCIJointStrategy(BaseStrategy):
    """联合枚举 split_layer × τ。"""

    def optimize(self, bandwidth_mbps, model_wrapper, model_info, tau_grid=None):
        n = model_info["n_exits"]
        candidates = _full_split_candidates(model_info)
        grid = tau_grid or [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
        best = (candidates[0], [0.5] * n, float("inf"), 0.0)
        for split in candidates:
            for tau in grid:
                lat, tx, _ = _eval(
                    split, tau, bandwidth_mbps, model_wrapper, model_info
                )
                if lat < best[2]:
                    best = (split, _uniform_thresholds(n, tau), lat, tx)
        return best
