"""Plot the measured ResNet-50/CIFAR-10 single-device ablation results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.EvaluationCommon.paper_figure_style import (  # noqa: E402
    ISPLITEE_COLOR,
    RESULTS_ROOT,
    SEAM_COLOR,
    apply_large_single_panel_style,
    bold_tick_labels,
    save_pdf,
)


VARIANTS = ["Cloud\nonly", "End\nonly", "Split\nonly", "EE\nonly", "Ours"]
LATENCY = np.array([360.29, 248.79, 130.40, 219.56, 119.74])
ACCURACY = np.array([95.86, 95.86, 95.86, 95.58, 95.59])
ACCURACY_FLOOR = 90.0


def _annotate_bars(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    offset: float,
    suffix: str = "",
    x_offset: float = 0.0,
    ha: str = "center",
    coordinate_ax: plt.Axes | None = None,
) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_offset,
            value + offset,
            f"{value:.2f}{suffix}",
            ha=ha,
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            transform=(ax if coordinate_ax is None else coordinate_ax).transData,
            zorder=10,
        )


def plot(output: Path) -> Path:
    apply_large_single_panel_style()
    fig, ax_lat = plt.subplots(figsize=(280 / 72, 2.6318))
    ax_acc = ax_lat.twinx()

    x = np.arange(len(VARIANTS))
    width = 0.34
    latency_bars = ax_lat.bar(
        x - width / 2,
        LATENCY,
        width=width,
        color=SEAM_COLOR,
        edgecolor="black",
        label="E2E latency",
        zorder=3,
    )
    accuracy_bars = ax_acc.bar(
        x + width / 2,
        ACCURACY - ACCURACY_FLOOR,
        bottom=ACCURACY_FLOOR,
        width=width,
        color=ISPLITEE_COLOR,
        edgecolor="black",
        hatch="//",
        label="Full-test accuracy",
        zorder=3,
    )

    _annotate_bars(
        ax_acc,
        latency_bars,
        LATENCY,
        offset=10.0,
        x_offset=width * 0.25,
        ha="right",
        coordinate_ax=ax_lat,
    )
    _annotate_bars(ax_acc, accuracy_bars, ACCURACY, offset=0.15)

    ax_lat.set_ylabel("E2E latency (ms)", color=SEAM_COLOR, labelpad=1.5)
    ax_acc.set_ylabel("Full-test acc. (%)", color=ISPLITEE_COLOR, labelpad=2.0)
    ax_lat.tick_params(axis="y", colors=SEAM_COLOR)
    ax_acc.tick_params(axis="y", colors=ISPLITEE_COLOR)
    ax_lat.spines["left"].set_color(SEAM_COLOR)
    ax_acc.spines["right"].set_color(ISPLITEE_COLOR)
    ax_lat.set_ylim(0, 450)
    ax_acc.set_ylim(ACCURACY_FLOOR, 97)
    ax_lat.set_yticks([0, 100, 200, 300, 400])
    ax_acc.set_yticks([90, 92, 94, 96])
    ax_lat.set_xticks(x, VARIANTS)
    bold_tick_labels(ax_lat, ax_acc)
    ax_lat.set_axisbelow(True)
    ax_lat.grid(axis="x", visible=False)
    ax_acc.grid(False)

    handles_lat, labels_lat = ax_lat.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_lat.legend(
        handles_lat + handles_acc,
        labels_lat + labels_acc,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        borderaxespad=0.0,
        borderpad=0.15,
        columnspacing=0.8,
        handlelength=1.3,
        handletextpad=0.4,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
    )
    fig.subplots_adjust(left=0.16, right=0.84, top=0.95, bottom=0.18)
    return save_pdf(fig, output, fixed_canvas=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "Exp4_Ablation" / "ablation.pdf",
    )
    args = parser.parse_args()
    print(plot(args.output))


if __name__ == "__main__":
    main()
