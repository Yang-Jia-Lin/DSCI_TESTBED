"""Plot multi-device scalability results from the paper draft."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.EvaluationCommon.paper_figure_style import (  # noqa: E402
    COMPARISON_FIGSIZE,
    COMPARISON_SUBPLOTS,
    ISPLITEE_COLOR,
    ISPLITEE_MARKER,
    NEUTRAL_COLOR,
    RESULTS_ROOT,
    SEAM_COLOR,
    SEAM_MARKER,
    add_comparison_legend,
    apply_comparison_figure_style,
    bold_tick_labels,
    save_pdf,
)


N = np.array([1, 2, 3, 4])
SEAM_MEAN = np.array([157.24, 219.33, 346.18, 292.02])
SEAM_MEAN_STD = np.array([5.95, 5.45, 1.24, 1.94])
SEAM_P95 = np.array([200.19, 314.50, 478.95, 492.94])
SEAM_P95_STD = np.array([1.63, 18.81, 7.32, 1.81])
SEAM_WORST = np.array([157.24, 244.64, 372.24, 385.91])
SEAM_WORST_STD = np.array([5.95, 8.22, 2.65, 9.32])
ISPLITEE_N = np.array([1, 2, 3])
ISPLITEE_WORST = np.array([282.91, 685.28, 826.02])
SEAM_ACCURACY = np.array([95.60, 95.56, 95.52, 95.21])
ISPLITEE_ACCURACY = np.array([94.20, 97.55, 94.12])


def plot(output: Path) -> Path:
    apply_comparison_figure_style()
    fig, (ax_lat, ax_acc) = plt.subplots(
        1,
        2,
        figsize=COMPARISON_FIGSIZE,
        sharex=True,
        gridspec_kw={"wspace": 0.32},
    )

    ax_lat.errorbar(
        N,
        SEAM_MEAN,
        yerr=SEAM_MEAN_STD,
        color=SEAM_COLOR,
        marker=SEAM_MARKER,
        linestyle="-",
        capsize=1.5,
        label="SEAS mean",
    )
    ax_lat.errorbar(
        N,
        SEAM_P95,
        yerr=SEAM_P95_STD,
        color=SEAM_COLOR,
        marker="^",
        linestyle="--",
        capsize=1.5,
        label="SEAS P95",
    )
    ax_lat.errorbar(
        N,
        SEAM_WORST,
        yerr=SEAM_WORST_STD,
        color=SEAM_COLOR,
        marker="D",
        linestyle=":",
        capsize=1.5,
        label="SEAS worst-dev.",
    )
    ax_lat.plot(
        ISPLITEE_N,
        ISPLITEE_WORST,
        color=ISPLITEE_COLOR,
        marker="D",
        linestyle=":",
        label="I-SplitEE worst-dev.",
    )
    ax_lat.set_ylabel("Latency (ms)", labelpad=1.5)
    ax_lat.set_ylim(0, 900)
    ax_lat.set_yticks([0, 400, 800])
    ax_acc.plot(
        N,
        SEAM_ACCURACY,
        color=SEAM_COLOR,
        marker=ISPLITEE_MARKER,
        linestyle="-.",
        label="SEAS acc.",
    )
    ax_acc.plot(
        ISPLITEE_N,
        ISPLITEE_ACCURACY,
        color=ISPLITEE_COLOR,
        marker=ISPLITEE_MARKER,
        linestyle="-.",
        label="I-SplitEE acc.",
    )
    ax_acc.set_ylabel("Top-1 acc. (%)", labelpad=1.5)
    ax_acc.set_ylim(0, 105)
    ax_acc.set_yticks([0, 50, 100])
    ax_acc.set_xticks(N)
    ax_acc.text(
        0.97,
        0.69,
        "I-SplitEE N=4\nnot measured",
        transform=ax_acc.transAxes,
        ha="right",
        va="center",
        color=NEUTRAL_COLOR,
        fontsize=6.0,
        fontweight="bold",
    )

    for ax in (ax_lat, ax_acc):
        ax.set_xlim(0.8, 4.2)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
    legend_specs = (
        ("SEAS Mean", SEAM_COLOR, SEAM_MARKER, "-"),
        ("I-SplitEE Mean", ISPLITEE_COLOR, SEAM_MARKER, "-"),
        ("P95", SEAM_COLOR, "^", "--"),
        ("P95", ISPLITEE_COLOR, "^", "--"),
        ("Worst-dev.", SEAM_COLOR, "D", ":"),
        ("Worst-dev.", ISPLITEE_COLOR, "D", ":"),
        ("Acc.", SEAM_COLOR, ISPLITEE_MARKER, "-."),
        ("Acc.", ISPLITEE_COLOR, ISPLITEE_MARKER, "-."),
    )
    legend_handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
        for _, color, marker, linestyle in legend_specs
    ]
    legend_labels = [label for label, _, _, _ in legend_specs]
    add_comparison_legend(fig, legend_handles, legend_labels, ncol=4)
    fig.supxlabel("Number of devices", x=0.5, y=0.02)
    bold_tick_labels(ax_lat, ax_acc)
    fig.subplots_adjust(**{**COMPARISON_SUBPLOTS, "bottom": 0.18})
    return save_pdf(fig, output, fixed_canvas=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "Exp3_Scalable" / "scalability.pdf",
    )
    args = parser.parse_args()
    print(plot(args.output))


if __name__ == "__main__":
    main()
