"""
Src/Exp2_Dynamic/plot_decision.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Src.paras import COLORS, NUM_LAYERS
from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style


def plot_resource_trend(csv_path: Path, save_dir: Path):
    """
    读取动态实验 CSV 并生成三色空间划分趋势图
    自动识别第一列作为横坐标
    """
    # 数据
    total_layers = NUM_LAYERS
    if not csv_path.exists():
        print(f"[Error] CSV file not found at {csv_path}")
        return
    df = pd.read_csv(csv_path)
    x_col_name = df.columns[0]  # 获取第一列的列名
    x_values = np.asarray(df.iloc[:, 0])  # 获取第一列的数据
    cut_ee = np.asarray(df["avg_end_edge"])
    cut_ec = np.asarray(df["avg_edge_cloud"])
    utility = np.asarray(df["total_utility"])
    label_map = {
        "H_u": "Channel Gain $H_u$",
        "F_u": "User Computing Power $F_u$ (GHz)",
        "b_e": "Edge Bandwidth $B_e$ (MHz)",
        "b_c": "Cloud Bandwidth $B_c$ (MHz)",
        "BANDWIDTH_EDGE": "Bandwidth $B_e$ (MHz)",
    }
    display_label = label_map.get(
        x_col_name, x_col_name
    )  # 如果找不到映射则直接显示原列名

    # 绘图
    set_ieee_style(mode="single")
    # plt.rcParams['figure.figsize'] = (4.0, 3.3)
    fig, ax1 = plt.subplots()
    ax1.fill_between(
        x_values, 0, cut_ee, color=COLORS["green"], alpha=0.3, label="Local"
    )
    ax1.fill_between(
        x_values, cut_ee, cut_ec, color=COLORS["red"], alpha=0.3, label="Edge"
    )
    ax1.fill_between(
        x_values, cut_ec, total_layers, color=COLORS["blue"], alpha=0.3, label="Cloud"
    )
    ax1.plot(x_values, cut_ee, color=COLORS["brown"], linestyle="-", marker="o")
    ax1.plot(x_values, cut_ec, color=COLORS["purple"], linestyle="-", marker="s")
    ax1.set_xlabel(display_label)  # 动态设置横坐标标签
    ax1.set_ylabel("DNN Layer Index")
    ax1.set_ylim(0, total_layers)
    ax1.set_xlim(x_values.min(), x_values.max())
    ax2 = ax1.twinx()
    ax2.plot(
        x_values, utility, color=COLORS["black"], linestyle="--", label="Total Utility"
    )
    ax2.set_ylabel("Total System Utility")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center',
    #            bbox_to_anchor=(0.5, 1.01), fontsize='small', frameon=True, ncol=2)
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="lower center",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=True,
        columnspacing=0.8,
        handletextpad=0.2,  # 缩小图标与文字之间的距离
        handlelength=1.5,
    )
    plt.tight_layout(pad=0.15)

    # 保存
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(
        save_dir
        / f"resource_trend_analysis({x_col_name})_{datetime.now().strftime('%m%d_%H%M')}"
    )
    plt.show()


if __name__ == "__main__":
    # 路径
    from Src.paras import RESULT_TEST_PATH

    test_dir = Path(RESULT_TEST_PATH) / "Test_Resource_Trend"
    test_csv = test_dir / "test_dynamic_data.csv"

    # 模拟数据
    f_u_range = np.arange(0.5, 8.5, 0.5)
    n_steps = len(f_u_range)
    avg_end_edge = 10 + 8 * f_u_range + np.random.normal(0, 2, n_steps)
    avg_edge_cloud = 90 + 3 * f_u_range + np.random.normal(0, 2, n_steps)
    total_utility = 500 + 200 * np.log1p(f_u_range)
    avg_end_edge = np.clip(avg_end_edge, 0, 120)
    avg_edge_cloud = np.clip(avg_edge_cloud, avg_end_edge + 5, 127)
    df = pd.DataFrame(
        {
            "F_u": f_u_range,
            "avg_end_edge": avg_end_edge,
            "avg_edge_cloud": avg_edge_cloud,
            "total_utility": total_utility,
        }
    )
    test_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_csv, index=False)
    print(f"[Test] Mock CSV generated at: {test_csv}")

    # 测试
    plot_resource_trend(csv_path=test_csv, save_dir=test_dir)
