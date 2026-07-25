"""Plot cross-architecture and cross-dataset results from the paper draft."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.EvaluationCommon.paper_figure_style import (  # noqa: E402
    COMPARISON_FIGSIZE,
    COMPARISON_SUBPLOTS,
    ISPLITEE_COLOR,
    RESULTS_ROOT,
    SEAM_COLOR,
    add_comparison_legend,
    apply_comparison_figure_style,
    bold_tick_labels,
    save_pdf,
)


LABELS = ["R-C10", "R-IN100", "R-NEU", "V-C10", "V-IN100", "V-NEU"]
SEAM_ACCURACY = np.array([95.60, 93.22, 98.52, 98.57, 94.10, 98.15])
ISPLITEE_ACCURACY = np.array([94.53, 77.93, 98.61, 95.53, 79.47, 99.72])
SEAM_LATENCY = np.array([157.24, 240.82, 201.76, 209.73, 208.48, 184.98])
ISPLITEE_LATENCY = np.array([321.28, 334.62, 307.07, 310.40, 318.28, 225.86])


def plot(output: Path) -> Path:
    apply_comparison_figure_style()
    fig, (ax_acc, ax_lat) = plt.subplots(
        1,
        2,
        figsize=COMPARISON_FIGSIZE,
        sharex=True,
        gridspec_kw={"wspace": 0.30},
    )
    x = np.arange(len(LABELS))
    width = 0.36

    for ax, seam, baseline, ylabel in (
        (ax_acc, SEAM_ACCURACY, ISPLITEE_ACCURACY, "Top-1 acc. (%)"),
        (ax_lat, SEAM_LATENCY, ISPLITEE_LATENCY, "E2E latency (ms)"),
    ):
        ax.bar(
            x - width / 2,
            seam,
            width,
            color=SEAM_COLOR,
            edgecolor="black",
            label="SEAS",
        )
        ax.bar(
            x + width / 2,
            baseline,
            width,
            color=ISPLITEE_COLOR,
            edgecolor="black",
            hatch="///",
            label="I-SplitEE",
        )
        ax.set_ylabel(ylabel, labelpad=1.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)

    # Start at zero so small accuracy differences are not visually exaggerated.
    ax_acc.set_ylim(0, 105)
    ax_acc.set_yticks([0, 50, 100])
    handles, legend_labels = ax_acc.get_legend_handles_labels()
    add_comparison_legend(fig, handles, legend_labels, anchor_y=0.88)
    ax_lat.set_ylim(0, 360)
    ax_lat.set_yticks([0, 150, 300])
    ax_lat.set_xticks(x, LABELS, rotation=38, ha="right", rotation_mode="anchor")
    for ax in (ax_acc, ax_lat):
        ax.set_xticks(x, LABELS, rotation=38, ha="right", rotation_mode="anchor")
    bold_tick_labels(ax_acc, ax_lat)
    fig.subplots_adjust(**COMPARISON_SUBPLOTS)
    return save_pdf(fig, output, fixed_canvas=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "Exp1_SEAM" / "generalization.pdf",
    )
    args = parser.parse_args()
    print(plot(args.output))


if __name__ == "__main__":
    main()
