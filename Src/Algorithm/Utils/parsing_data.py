"""
Src/Utils/parsing_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from Src.Configs.model_config import RESNET50
from Src.Configs.paras import Paras


def _resolve_rate_acc_paths(
    paras: Paras,
    rate_path: str | Path | None = None,
    acc_path: str | Path | None = None,
) -> tuple[Path, Path]:
    cfg = getattr(paras, "model_cfg", None) or RESNET50
    return Path(rate_path or cfg.rate_csv), Path(acc_path or cfg.acc_csv)


def parsing_rate_and_acc(
    paras: Paras,
    rate_path: str | Path | None = None,
    acc_path: str | Path | None = None,
):
    rate_csv, acc_csv = _resolve_rate_acc_paths(paras, rate_path, acc_path)
    df_acc = pd.read_csv(acc_csv)
    df_rate = pd.read_csv(rate_csv)

    m = paras.m
    exit_layer: list[int] = paras.E
    num_thresholds = len(df_acc) + 1
    num_exits = len(exit_layer)
    rate_matrix = np.zeros((num_thresholds, m))
    acc_matrix = np.zeros((num_thresholds, m))

    # 提取 Rate
    for row_idx, threshold in enumerate(df_rate.itertuples(index=False)):
        exit_rates = threshold[1 : num_exits + 1]  # 提取
        for i, layer in enumerate(exit_layer):  # 填充
            rate_matrix[row_idx, layer] = exit_rates[i]

    # 提取 Acc
    for row_idx, threshold in enumerate(df_acc.itertuples(index=False)):
        exit_accuracies = threshold[1 : num_exits + 1]
        # print(f"threshold.values is {threshold[num_exits + 1]}")
        for i, layer in enumerate(exit_layer):
            acc_matrix[row_idx, layer] = exit_accuracies[i]
        acc_matrix[:, m - 1] = threshold[num_exits + 1]
    return rate_matrix, acc_matrix


def _decode_split_points(x_row: np.ndarray) -> Tuple[int, int]:
    ones = np.flatnonzero(x_row)
    m = len(x_row)
    if len(ones) == 0:
        return -1, m - 1
    if len(ones) == 1:
        return int(ones[0]), m - 1
    return int(ones[0]), int(ones[1])


def split_points_matrix(X: np.ndarray) -> np.ndarray:
    return np.array([_decode_split_points(r) for r in X], dtype=int)


if __name__ == "__main__":
    paras = Paras()
    assert paras.accs is not None, "Paras.accs should be initialized"
    print(paras.accs[:, 57])
    print(paras.accs[:, 127])
