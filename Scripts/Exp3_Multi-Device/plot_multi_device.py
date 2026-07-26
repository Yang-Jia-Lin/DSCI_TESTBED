"""Plot the multi-device experiment results from the paper draft."""

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
    GA_COLOR,
    ISPLITEE_COLOR,
    ISPLITEE_MARKER,
    RANDOM_COLOR,
    SEAM_COLOR,
    SEAM_MARKER,
    add_comparison_legend,
    apply_comparison_figure_style,
    bold_tick_labels,
    save_pdf,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULT_FIGURE_DIR = EXPERIMENT_ROOT / "result_figure"

N = np.array([1, 2, 3, 4])
SEAS_MEAN = np.array([119.74, 219.33, 346.18, 292.02])
SEAS_P95 = np.array([177.97, 314.50, 478.95, 492.94])
SEAS_WORST = np.array([119.74, 244.64, 372.24, 385.91])
SEAS_ACCURACY = np.array([95.59, 95.56, 95.52, 95.21])
ISPLITEE_MEAN = np.array([294.31, 480.47, 730.77, 653.76])
ISPLITEE_P95 = np.array([391.05, 854.64, 1948.38, 1962.36])
ISPLITEE_WORST = np.array([294.31, 659.89, 1238.34, 1277.88])
ISPLITEE_ACCURACY = np.array([94.87, 94.90, 94.69, 94.37])


def plot(output: Path) -> Path:
    apply_comparison_figure_style()
    fig, ax_lat = plt.subplots(figsize=(255 / 72, 190 / 72))
    ax_acc = ax_lat.twinx()

    ax_lat.plot(
        N,
        SEAS_MEAN,
        color=SEAM_COLOR,
        marker=SEAM_MARKER,
        linestyle="-",
        label="SEAS mean",
    )
    ax_lat.plot(
        N,
        SEAS_P95,
        color=SEAM_COLOR,
        marker="^",
        linestyle="--",
        label="SEAS P95",
    )
    ax_lat.plot(
        N,
        SEAS_WORST,
        color=SEAM_COLOR,
        marker="D",
        linestyle=":",
        label="SEAS worst-dev.",
    )
    ax_lat.plot(
        N,
        ISPLITEE_MEAN,
        color=ISPLITEE_COLOR,
        marker=SEAM_MARKER,
        linestyle="-",
        label="I-SplitEE mean",
    )
    ax_lat.plot(
        N,
        ISPLITEE_P95,
        color=ISPLITEE_COLOR,
        marker="^",
        linestyle="--",
        label="I-SplitEE P95",
    )
    ax_lat.plot(
        N,
        ISPLITEE_WORST,
        color=ISPLITEE_COLOR,
        marker="D",
        linestyle=":",
        label="I-SplitEE worst-dev.",
    )
    ax_lat.set_ylabel("Latency (ms)", labelpad=1.5)
    ax_lat.set_ylim(0, 2050)
    ax_lat.set_yticks([0, 1000, 2000])
    ax_acc.plot(
        N,
        SEAS_ACCURACY,
        color=RANDOM_COLOR,
        marker=ISPLITEE_MARKER,
        linestyle="-.",
        label="SEAS acc.",
    )
    ax_acc.plot(
        N,
        ISPLITEE_ACCURACY,
        color=GA_COLOR,
        marker=ISPLITEE_MARKER,
        linestyle="-.",
        label="I-SplitEE acc.",
    )
    ax_acc.set_ylabel("Top-1 acc. (%)", labelpad=1.5)
    ax_acc.set_ylim(90, 97)
    ax_acc.set_yticks([90, 92, 94, 96])
    ax_acc.spines["right"].set_color(RANDOM_COLOR)
    ax_acc.tick_params(axis="y", colors=RANDOM_COLOR)
    ax_acc.yaxis.label.set_color(RANDOM_COLOR)

    ax_lat.set_xlim(0.8, 4.2)
    ax_lat.set_xticks(N)
    ax_lat.set_xlabel("Number of devices", labelpad=1.5)
    ax_lat.set_axisbelow(True)
    ax_lat.grid(axis="x", visible=False)
    ax_acc.grid(False)
    legend_specs = (
        ("SEAS Mean", SEAM_COLOR, SEAM_MARKER, "-"),
        ("I-SplitEE Mean", ISPLITEE_COLOR, SEAM_MARKER, "-"),
        ("P95", SEAM_COLOR, "^", "--"),
        ("P95", ISPLITEE_COLOR, "^", "--"),
        ("Worst-dev.", SEAM_COLOR, "D", ":"),
        ("Worst-dev.", ISPLITEE_COLOR, "D", ":"),
        ("Acc.", RANDOM_COLOR, ISPLITEE_MARKER, "-."),
        ("Acc.", GA_COLOR, ISPLITEE_MARKER, "-."),
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
    bold_tick_labels(ax_lat, ax_acc)
    fig.subplots_adjust(**{**COMPARISON_SUBPLOTS, "right": 0.86, "bottom": 0.18})
    return save_pdf(fig, output, fixed_canvas=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_FIGURE_DIR / "multi_device.pdf",
    )
    args = parser.parse_args()
    print(plot(args.output))


if __name__ == "__main__":
    main()
