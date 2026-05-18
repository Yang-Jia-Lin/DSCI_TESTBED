"""
Src/Utils/parsing_data.py
"""

import numpy as np
import pandas as pd
from Src.paras import Paras, ACC_CSV_PATH, RATE_CSV_PATH
from typing import Tuple


def parsing_rate_and_acc(paras):
    df_acc = pd.read_csv(ACC_CSV_PATH)
    df_rate = pd.read_csv(RATE_CSV_PATH)

    m = paras.m
    exit_layer = paras.E
    num_thresholds = len(df_acc) + 1
    num_exits = len(exit_layer)
    rate_matrix = np.zeros((num_thresholds, m))
    acc_matrix = np.zeros((num_thresholds, m))

    # 提取 Rate
    for idx, threshold in df_rate.iterrows():
        exit_rates = threshold[1 : num_exits + 1].values  # 提取
        for i, layer in enumerate(exit_layer):  # 填充
            rate_matrix[idx, layer] = exit_rates[i]

    # 提取 Acc
    for idx, threshold in df_acc.iterrows():
        exit_accuracies = threshold[1 : num_exits + 1].values
        # print(f"threshold.values is {threshold[num_exits + 1]}")
        for i, layer in enumerate(exit_layer):
            acc_matrix[idx, layer] = exit_accuracies[i]
        acc_matrix[:, m - 1] = threshold.iloc[num_exits + 1]
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
    print(paras.accs[:, 57])
    print(paras.accs[:, 127])
