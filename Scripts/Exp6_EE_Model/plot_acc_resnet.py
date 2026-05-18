"""
Src/Experiments/Exp5_EE_Model/plot_acc_resnet.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from Src.Algorithm.Utils.plot_utils import save_fig_for_ieee, set_ieee_style

from Src.paras import ACC_CSV_PATH, COLORS, RESULT_EE_MODEL_PATH


def plot_accuracy_vs_threshold(
    data_dir: Path, save_dir: Path = Path(RESULT_EE_MODEL_PATH)
):
    """
    绘制模型 Early Exit 精度随阈值变化的曲线图
    """
    # 数据
    constant_value = 87.56
    if not data_dir.exists():
        print(f"错误: 找不到文件 {data_dir}")
        return
    df = pd.read_csv(data_dir)

    # 绘图
    set_ieee_style(mode="single")
    plt.figure()
    plt.plot(
        df["threshold"],
        df["exit1_accuracy"],
        color=COLORS["blue"],
        marker="o",
        label="Early Exit 1",
        markevery=3,
    )
    plt.plot(
        df["threshold"],
        df["exit2_accuracy"],
        color=COLORS["green"],
        marker="s",
        label="Early Exit 2",
        markevery=3,
    )
    plt.plot(
        df["threshold"],
        df["full_accuracy"],
        color=COLORS["red"],
        marker="^",
        label="Overall",
        markevery=3,
    )
    plt.axhline(y=constant_value, color="black", linestyle="--", label="Main Exit")
    plt.text(
        x=0.05,
        y=constant_value + 1.0,
        s=f"Main Exit: {constant_value:.2f}%",
        color="black",
        fontweight="bold",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy (%)")
    plt.xlim(0, 1.0)
    plt.ylim(60, 100)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout(pad=0.15)

    # 保存
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"{data_dir.stem}_accuracy_threshold")
    plt.show()


if __name__ == "__main__":
    csv_path = ACC_CSV_PATH
    save_dir = Path(RESULT_EE_MODEL_PATH)
    plot_accuracy_vs_threshold(csv_path, save_dir)
