"""
Src/Experiments/Exp5_EE_Model/plot_combine_resnet.py
"""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Src.Algorithm.Utils.plot_utils import save_fig_for_ieee, set_ieee_style
from Src.Models.model_config import RESNET50 as MODEL_CFG
from Src.paras import COLORS, OFFLINE_TABLE_DIR, RESULT_EE_MODEL_PATH


def plot_expectation_vs_threshold(
    rate_data_dir: Path, acc_data_dir: Path, save_dir: Path = Path(RESULT_EE_MODEL_PATH)
):
    """
    结合计算延迟期望(E_t)和精度期望(E_acc)绘制双轴曲线。
    """
    # 数据
    m = 8
    exit_layer = [3, 6]
    csv_rate_path = Path(rate_data_dir)
    csv_acc_path = Path(acc_data_dir)
    if not csv_rate_path.exists() or not csv_acc_path.exists():
        print("错误: 请检查输入的 CSV 路径是否存在。")
        return
    df_rate = pd.read_csv(csv_rate_path)
    df_acc = pd.read_csv(csv_acc_path)
    num_thresholds = len(df_rate)
    num_exits = len(exit_layer)

    # 计算 E_t (延迟期望)
    rate_matrix = np.zeros((num_thresholds, m))
    for idx, row in enumerate(df_rate.itertuples(index=False)):
        exit_rates = np.asarray(row[1 : num_exits + 1])
        for i, layer in enumerate(exit_layer):
            rate_matrix[idx, layer] = exit_rates[i]
    rate_matrix = rate_matrix * 0.01
    P = np.zeros((num_thresholds, m))
    for i in range(num_thresholds):
        for j in range(m):
            if j == 0:
                P[i, j] = rate_matrix[i, j]
            else:
                P[i, j] = rate_matrix[i, j] * np.prod(1 - rate_matrix[i, :j])
        P[i, -1] = 1 - np.sum(P[i, :])
    t_matrix = np.arange(1, m + 1).reshape(1, -1).repeat(num_thresholds, axis=0)
    E_t = np.sum(P * t_matrix, axis=1)

    # 计算 E_acc (精度期望)
    acc_matrix = np.zeros((num_thresholds, m))
    for idx, row in enumerate(df_acc.itertuples(index=False)):
        exit_accuracies = np.asarray(row[1 : num_exits + 2])
        for i, layer in enumerate(exit_layer):
            acc_matrix[idx, layer] = exit_accuracies[i]
        acc_matrix[idx, m - 1] = exit_accuracies[-1]
    E_acc = np.sum(P * acc_matrix, axis=1)

    # 绘图
    set_ieee_style(mode="single")

    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(
        df_rate["threshold"],
        E_t,
        color=COLORS["green"],
        marker="o",
        markevery=3,
        label="Latency Expect",
    )
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Latency Expectation")
    ax1.tick_params(axis="y", colors=COLORS["green"])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(3.5, 8.5)

    ax2 = ax1.twinx()
    lns2 = ax2.plot(
        df_acc["threshold"],
        E_acc,
        color=COLORS["blue"],
        marker="^",
        markevery=3,
        label="Accuracy Expect",
    )
    ax2.set_ylabel("Accuracy Expectation (%)")
    ax2.tick_params(axis="y", colors=COLORS["blue"])
    ax2.set_ylim(58, 100)
    ax2.grid(False)  # 双轴图通常关闭右轴网格，防止画面太乱

    lns = lns1 + lns2
    labs = cast(list[str], [line.get_label() for line in lns])
    ax1.legend(
        lns,
        labs,
        loc="lower right",
        bbox_to_anchor=(1, 0),
        ncol=1,
        frameon=True,
        fontsize=10,
    )
    plt.tight_layout(pad=0.15)

    # 保存
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"{csv_rate_path.stem}_combined_expectation")
    plt.show()


if __name__ == "__main__":
    rate_csv = MODEL_CFG.resolve_rate_csv()
    acc_csv = MODEL_CFG.resolve_acc_csv()
    save_dir = Path(RESULT_EE_MODEL_PATH)
    plot_expectation_vs_threshold(rate_csv, acc_csv, save_dir)
