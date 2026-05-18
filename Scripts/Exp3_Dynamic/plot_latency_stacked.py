"""
Src/Exp2_Dynamic/plot_latency_stacked.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style

from Src.Configs.paras import COLORS


def plot_latency_stacked(user_labels, T_parts, save_dir: Path):
    """
    绘制延迟组成的堆叠柱状图
    """
    set_ieee_style(mode="single")
    # plt.rcParams['figure.figsize'] = (4.0, 3.5)
    fig, ax = plt.subplots()
    x = np.arange(len(user_labels))
    bottom = np.zeros_like(x, dtype=float)
    labels = ["Local", "U$\\to$E", "Edge", "E$\\to$C", "Cloud"]
    colors = [
        COLORS["green"],  # Local
        COLORS["grey"],  # U->E
        COLORS["red"],  # Edge
        COLORS["purple"],  # E->C
        COLORS["blue"],  # Cloud
    ]
    for comp, lab, col in zip(T_parts, labels, colors):
        ax.bar(
            x,
            comp,
            bottom=bottom,
            label=lab,
            color=col,
            edgecolor="white",
            linewidth=0.5,
            width=0.7,
        )
        bottom += comp
    ax.set_xticks(x)
    ax.set_xticklabels(user_labels)
    ax.set_xlabel("User Index")
    ax.set_ylabel("One User Inference Latency")
    # ax.set_title("Latency Composition")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        frameon=True,
        columnspacing=0.8,
        handletextpad=0.2,  # 缩小图标与文字之间的距离
        handlelength=1.5,
    )
    plt.tight_layout(pad=0.15)

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(
        save_dir / f"Latency_Stacked_{datetime.now().strftime('%m%d_%H%M')}"
    )
    plt.show()


if __name__ == "__main__":
    users = [f"U{i + 1}" for i in range(5)]
    T = [np.random.rand(5) * 0.2 for _ in range(5)]

    from Src.Configs.paras import COLORS, RESULT_TEST_PATH

    plot_latency_stacked(users, T, save_dir=Path(RESULT_TEST_PATH))
