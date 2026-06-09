"""
Src/Exp3_DSCI_Convergency/plot_convergency.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from Src.Shared.Utils.plot_utils import save_fig_for_ieee, set_ieee_style
from Src.Shared.Config.paths import RESULT_CONVERGENCE_PATH
from Src.Shared.Config.visualization import COLORS


def plot_convergence(
    utility, save_dir: Path = Path(RESULT_CONVERGENCE_PATH), alg_name: str = "DSCI"
):
    """
    Plot the convergence curve include DSCI、BF、GA
    """
    utility = [1.0 * x for x in utility]

    # 绘图
    set_ieee_style(mode="single")
    plt.figure()
    plt.plot(utility, color=COLORS["blue"], label=alg_name, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Utility")
    # plt.title(f'{alg_name} Convergence')
    plt.tight_layout(pad=0.15)

    # 保存
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(
        save_dir
        / f"{alg_name}_utility_convergence_{datetime.now().strftime('%m%d_%H%M')}"
    )
    plt.show()


def plot_entropy(entropy_X, entropy_Y, save_dir=Path(RESULT_CONVERGENCE_PATH)):
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    ax.plot(
        entropy_X, color=COLORS["red"], label=r"Entropy $\mathbf{X}$", linewidth=1.5
    )
    ax.plot(
        entropy_Y,
        color=COLORS["green"],
        label=r"Entropy $\mathbf{Y}$",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy")
    ax.legend(
        loc="lower center",
        fontsize=10,
        bbox_to_anchor=(0.5, 0.98),  # 1.02 代表悬浮在绘图区上方
        ncol=2,
        frameon=True,
        columnspacing=1,
        handletextpad=0.4,  # 缩小图标与文字之间的距离
        handlelength=2,
    )
    plt.tight_layout(pad=0.15)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(
        save_dir / f"entropy_convergence_{datetime.now().strftime('%m%d_%H%M')}"
    )
    plt.show()


def plot_lan_and_acc(latency, acc, save_dir=Path(RESULT_CONVERGENCE_PATH)):
    set_ieee_style(mode="single")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    (l1,) = ax1.plot(latency, "--", color=COLORS["red"], label="Latency", linewidth=1.2)
    ax1.set_ylabel("Latency (s)")
    ax1.tick_params(axis="y")

    (l2,) = ax2.plot(acc, "-", color=COLORS["green"], label="Accuracy", linewidth=1.2)
    ax2.set_ylabel("Accuracy")
    ax2.tick_params(axis="y")

    ax1.set_xlabel("Epoch")
    lines = [l1, l2]
    ax1.legend(
        lines,
        [str(line.get_label()) for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
    )
    ax1.legend(
        lines,
        [str(line.get_label()) for line in lines],
        loc="lower center",
        fontsize=10,
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        columnspacing=1,
        handletextpad=0.4,  # 缩小图标与文字之间的距离
        handlelength=2,
    )
    plt.tight_layout(pad=0.15)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(
        save_dir / f"perf_tradeoff_{datetime.now().strftime('%m%d_%H%M')}"
    )
    plt.show()


if __name__ == "__main__":
    from Src.Phase2_Scheduler.Utils.log_function import load_and_analyze_results
    from Src.Shared.Config.paths import RESULT_DIR, RESULT_TEST_PATH

    PPO_path = RESULT_DIR / "Optimize" / "PPO" / "PPO_20260129_202802"
    X_opt, Y_opt, F_e, F_c, history, paras = load_and_analyze_results(
        exp_dir=PPO_path, analysis=False
    )

    plot_convergence(history, save_dir=Path(RESULT_TEST_PATH))
