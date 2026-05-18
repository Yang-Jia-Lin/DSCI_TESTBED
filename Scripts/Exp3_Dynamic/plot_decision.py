"""
Src/Exp2_Dynamic/plot_decision.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style

from Src.paras import COLORS, Paras


def plot_X(X_opt, EE_layers, save_dir: Path):
    """
    绘制决策变量 X (模型划分) 的热力图
    """
    n, m = X_opt.shape

    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    ax.matshow(X_opt, cmap="Blues", aspect="auto")
    ax.set_xlabel("DNN Layers")
    ax.set_ylabel("Users")
    # ax.set_title("Partitioning Decision ($\mathbf{X}$)")
    ax.set_xticks(np.arange(0, m, max(1, m // 10)))
    ax.set_xticklabels([i + 1 for i in range(0, m, max(1, m // 10))])
    ax.set_yticks(np.arange(0, n, max(1, n // 5)))
    ax.set_yticklabels([i + 1 for i in range(0, n, max(1, n // 5))])
    for l in EE_layers:
        # 早退分割线
        ax.axvline(x=l, color=COLORS["red"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.xaxis.set_ticks_position("bottom")
    plt.tight_layout(pad=0.15)

    # 保存
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"Decision_X_{datetime.now().strftime('%m%d_%H%M')}")
    plt.show()


def plot_Y(Y_opt, EE_layers, save_dir: Path):
    """
    绘制决策变量 Y (早退阈值) 的热力图
    """
    n = Y_opt.shape[0]
    Y_E = Y_opt[:, EE_layers]

    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    im = ax.imshow(Y_E, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Early Exit Layers")
    ax.set_ylabel("Users")
    # ax.set_title("Threshold Decision ($\mathbf{Y}$)")
    ax.set_xticks(range(len(EE_layers)))
    ax.set_xticklabels(EE_layers)
    ax.set_yticks(np.arange(0, n, max(1, n // 5)))
    ax.set_yticklabels([i + 1 for i in range(0, n, max(1, n // 5))])
    ax.xaxis.set_ticks_position("bottom")
    ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Threshold $\epsilon$")
    plt.tight_layout(pad=0.15)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"Decision_Y_{datetime.now().strftime('%m%d_%H%M')}")
    plt.show()


if __name__ == "__main__":
    paras = Paras()
    X = np.zeros([paras.n, paras.m])
    X[0, [10, 20]] = 1
    X[1, [40, 60]] = 1
    Y = np.ones([paras.n, paras.m])
    Y[2, 57] = 0.6
    Y[4, 103] = 0.8
    Y[5, 57] = 0.5
    Y[5, 103] = 0.9
    Y[7, 57] = 0.1
    Y[9, 103] = 0.5

    from Src.paras import RESULT_TEST_PATH

    plot_X(X, paras.E, save_dir=Path(RESULT_TEST_PATH))
    plot_Y(Y, paras.E, save_dir=Path(RESULT_TEST_PATH))
