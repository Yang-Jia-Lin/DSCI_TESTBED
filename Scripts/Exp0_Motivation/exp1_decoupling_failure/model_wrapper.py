"""
Scripts/Exp0_Motivation/exp1_decoupling_failure/model_wrapper.py

关键常量来源（实现前自项目文件读取）：
- 模型类: MultiEEResNet50 (Src/Models/ModelNet/Resnet50.py)
- 早退头层索引（仅此处设 τ）: E = [57, 103]；末层 127 为最终分类出口
- 切分点 X: 可在 1..127 任意层枚举；0 表示 Local
- 总层数 m = 128
- 层 FLOPs / 特征字节: Data/OfflineTables/Resnet50_layer_stats.csv → approx_flops, num_bytes
- 早退率: Data/OfflineTables/Resnet50_rates.csv → exit1_rate, exit2_rate (%, 阈值列 threshold)
- set_ieee_style(mode): single=4.0×2.8in, double=7.0×4.5in (Src/Utils/plot_utils.py)
- save_fig_for_ieee(save_path: Path, fig=None) → .pdf + .png
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Scripts.Exp0_Motivation.utils.config import (  # noqa: E402
    EARLY_EXIT_LAYERS,
    FINAL_LAYER,
    NUM_LAYERS,
)
from Src.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50  # noqa: E402

logger = logging.getLogger(__name__)

# 权重与 CSV 默认路径（相对项目根）
DEFAULT_WEIGHTS = Path("Data/Weights/ResNet50_multi_EE_model.pth")
LAYER_STATS_CSV = Path("Data/OfflineTables/Resnet50_layer_stats.csv")
RATES_CSV = Path("Data/OfflineTables/Resnet50_rates.csv")

# 加载失败时的硬编码回退（ResNet-50 @ CIFAR-10 典型量级）
_FALLBACK_EXIT_LAYERS = [57, 103]
_FALLBACK_FEATURE_BYTES = {57: 32768, 103: 16384, 127: 8192}
_FALLBACK_TOTAL_FLOPS = 4.1e9


class ResNet50EEWrapper:
    """
    封装项目 ResNet50 多早退头模型，提供实验所需的元数据与 CSV 查询接口。
    权重路径：Data/Weights/ResNet50_multi_EE_model.pth
    """

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        device: str = "cpu",
        project_root: Path | None = None,
    ):
        self.project_root = Path(project_root or PROJECT_ROOT)
        self.device = device
        self.weights_path = self.project_root / weights_path

        self._exit_layers = list(EARLY_EXIT_LAYERS)
        self._final_layer = FINAL_LAYER
        self._n_layers = NUM_LAYERS

        self._load_layer_stats()
        self._load_rates_csv()
        self._load_model()

    def _load_layer_stats(self) -> None:
        """从 Resnet50_layer_stats.csv 读取各层 FLOPs 与特征字节。"""
        csv_path = self.project_root / LAYER_STATS_CSV
        try:
            df = pd.read_csv(csv_path)
            flops = df["approx_flops"].astype(np.float64).tolist()
            bytes_ = df["num_bytes"].astype(int).tolist()
            n = min(len(flops), self._n_layers)
            self._per_layer_flops = flops[:n]
            self._per_layer_bytes = bytes_[:n]
            if n < self._n_layers:
                pad = self._n_layers - n
                self._per_layer_flops.extend([0.0] * pad)
                self._per_layer_bytes.extend([self._per_layer_bytes[-1]] * pad)
            self._flops_cumulative = np.cumsum(self._per_layer_flops).tolist()
            self._total_flops = float(self._flops_cumulative[-1])
        except Exception as exc:
            logger.warning(
                "无法读取 %s (%s)，使用硬编码 FLOPs/特征尺寸回退",
                csv_path,
                exc,
            )
            self._per_layer_flops = [_FALLBACK_TOTAL_FLOPS / self._n_layers] * (
                self._n_layers
            )
            self._flops_cumulative = np.cumsum(self._per_layer_flops).tolist()
            self._total_flops = _FALLBACK_TOTAL_FLOPS
            self._per_layer_bytes = [8192] * self._n_layers
            for layer, b in _FALLBACK_FEATURE_BYTES.items():
                if layer < self._n_layers:
                    self._per_layer_bytes[layer] = b

    def _load_rates_csv(self) -> None:
        """加载早退率表；缺列时使用 Beta 分布补充（见 get_exit_rates_from_csv）。"""
        csv_path = self.project_root / RATES_CSV
        try:
            self._rates_df = pd.read_csv(csv_path)
            if "threshold" not in self._rates_df.columns:
                raise ValueError("缺少 threshold 列")
            self._rate_thresholds = self._rates_df["threshold"].to_numpy(dtype=float)
            self._rate_cols = [
                c
                for c in self._rates_df.columns
                if c != "threshold" and "rate" in c.lower()
            ]
            if len(self._rate_cols) < len(self._exit_layers):
                raise ValueError(
                    f"早退率列不足: 需要 {len(self._exit_layers)} 列, 仅有 {self._rate_cols}"
                )
            self._rates_from_csv = True
            logger.info(
                "已加载 %s: %d 个阈值点, 列 %s",
                csv_path.name,
                len(self._rate_thresholds),
                self._rate_cols,
            )
        except Exception as exc:
            logger.warning(
                "无法读取 %s (%s)，早退率将用 Beta 分布合成",
                csv_path,
                exc,
            )
            self._rates_from_csv = False
            self._rate_thresholds = np.linspace(0.0, 1.0, 101)
            self._rate_cols = []

    def _load_model(self) -> None:
        """实例化 MultiEEResNet50 并加载权重（失败时仅保留元数据）。"""
        blocks_num = [3, 4, 6, 3]
        self.model = MultiEEResNet50(
            block=Bottleneck,
            blocks_num=blocks_num,
            num_classes=10,
            include_top=True,
        )
        try:
            import torch

            map_loc = self.device if self.device != "cpu" else "cpu"
            state = torch.load(
                self.weights_path, map_location=map_loc, weights_only=True
            )
            if isinstance(state, dict):
                logger.info(
                    "权重 state_dict keys 示例: %s ... (共 %d)",
                    list(state.keys())[:5],
                    len(state),
                )
            self.model.load_state_dict(state)
            self.model.to(map_loc)
            self.model.eval()
            self._weights_loaded = True
        except Exception as exc:
            logger.warning(
                "无法加载权重 %s (%s)，实验将仅使用 CSV 元数据",
                self.weights_path,
                exc,
            )
            self._weights_loaded = False

    def get_exit_layer_indices(self) -> list[int]:
        """返回各早退头对应的层编号列表。"""
        return list(self._exit_layers)

    def get_feature_size_bytes(self, layer_idx: int) -> int:
        """
        返回第 layer_idx 层输出特征图的字节数（float32 等效存储）。

        优先从 Data/OfflineTables/Resnet50_layer_stats.csv 的 num_bytes 列读取。
        """
        if 0 <= layer_idx < len(self._per_layer_bytes):
            return int(self._per_layer_bytes[layer_idx])
        return int(self._per_layer_bytes[-1])

    def get_flops_up_to(self, layer_idx: int) -> float:
        """
        返回从输入到第 layer_idx 层（含）的累积 FLOPs（MACs）。
        """
        if layer_idx < 0:
            return 0.0
        if layer_idx >= len(self._flops_cumulative):
            return float(self._flops_cumulative[-1])
        return float(self._flops_cumulative[layer_idx])

    def get_total_flops(self) -> float:
        """全模型累积 FLOPs。"""
        return float(self._total_flops)

    def get_exit_rates_from_csv(self, threshold: float) -> list[float]:
        """
        从 Data/OfflineTables/Resnet50_rates.csv 读取给定阈值下各早退头的退出率（0~1）。

        若 CSV 无精确匹配，对 threshold 做线性插值。
        若 CSV 不可用，用 Beta(2,5) 形状随 τ 单调上升的合成率（注明于日志）。
        """
        threshold = float(np.clip(threshold, 0.0, 1.0))

        if self._rates_from_csv:
            rates_pct = []
            for col in self._rate_cols[: len(self._exit_layers)]:
                col_vals = self._rates_df[col].to_numpy(dtype=float)
                rates_pct.append(
                    float(np.interp(threshold, self._rate_thresholds, col_vals))
                )
            return [r / 100.0 for r in rates_pct]

        # Beta 分布补充（仅 CSV 缺失时）
        from scipy.stats import beta

        a, b = 2.0, 5.0
        base = beta.cdf(threshold, a, b)
        return [
            min(0.99, base * (0.6 + 0.4 * i)) for i, _ in enumerate(self._exit_layers)
        ]

    def get_model_info(self) -> dict:
        """
        返回实验所需的模型元信息字典。
        """
        exit_layers = self.get_exit_layer_indices()
        # 切分：全层 1..127；解耦类方法 Step1 常在早退出口附近选点（工程常见）
        candidate_splits_full = list(range(1, self._n_layers))
        candidate_splits_exit = list(exit_layers) + [self._final_layer]
        return {
            "n_exits": len(exit_layers),
            "exit_layer_indices": exit_layers,
            "final_layer": self._final_layer,
            "total_layers": self._n_layers,
            "candidate_split_layers": candidate_splits_full,
            "candidate_split_layers_exit_only": candidate_splits_exit,
            "flops_cumulative": list(self._flops_cumulative),
            "flops_per_layer": list(self._per_layer_flops),
            "feature_bytes": list(self._per_layer_bytes),
            "feature_sizes": {
                layer: self.get_feature_size_bytes(layer)
                for layer in candidate_splits_full
            },
            "total_flops": self._total_flops,
            "weights_loaded": self._weights_loaded,
            "rates_from_csv": self._rates_from_csv,
        }
