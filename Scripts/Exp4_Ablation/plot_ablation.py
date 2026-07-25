"""Plot a combined ablation preview with explicitly marked placeholder data."""

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
    apply_compact_ieee_style,
    save_pdf,
)


VARIANTS = ["End\nonly", "Split\nonly", "EE\nonly", "SEAM"]
# Preview-only placeholders. Replace both arrays with measured values before use.
PLACEHOLDER_LATENCY = np.array([286.40, 264.70, 255.80, 240.82])
PLACEHOLDER_ACCURACY = np.array([91.48, 92.06, 92.74, 93.22])


def _annotate_bars(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    offset: float,
    suffix: str = "",
) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            fontweight="bold",
        )


def plot(output: Path) -> Path:
    apply_compact_ieee_style()
    # Match the physical axes ratio of each panel in plot_generalization.py.
    fig, ax_lat = plt.subplots(figsize=(3.55, 3.10))
    ax_acc = ax_lat.twinx()

    x = np.arange(len(VARIANTS))
    width = 0.34
    latency_bars = ax_lat.bar(
        x - width / 2,
        PLACEHOLDER_LATENCY,
        width=width,
        color=SEAM_COLOR,
        edgecolor="black",
        label="E2E latency",
        zorder=3,
    )
    accuracy_bars = ax_acc.bar(
        x + width / 2,
        PLACEHOLDER_ACCURACY,
        width=width,
        color=ISPLITEE_COLOR,
        edgecolor="black",
        hatch="//",
        label="Top-1 accuracy",
        zorder=3,
    )

    _annotate_bars(ax_lat, latency_bars, PLACEHOLDER_LATENCY, offset=5.0)
    _annotate_bars(ax_acc, accuracy_bars, PLACEHOLDER_ACCURACY, offset=1.2)

    ax_lat.set_ylabel("E2E latency (ms)", color=SEAM_COLOR, labelpad=1.5)
    ax_acc.set_ylabel("Top-1 accuracy (%)", color=ISPLITEE_COLOR, labelpad=2.0)
    ax_lat.tick_params(axis="y", colors=SEAM_COLOR)
    ax_acc.tick_params(axis="y", colors=ISPLITEE_COLOR)
    ax_lat.spines["left"].set_color(SEAM_COLOR)
    ax_acc.spines["right"].set_color(ISPLITEE_COLOR)
    ax_lat.set_ylim(0, 320)
    ax_acc.set_ylim(0, 105)
    ax_lat.set_yticks([0, 100, 200, 300])
    ax_acc.set_yticks([0, 25, 50, 75, 100])
    ax_lat.set_xticks(x, VARIANTS)
    ax_lat.set_axisbelow(True)
    ax_lat.grid(axis="x", visible=False)
    ax_acc.grid(False)

    handles_lat, labels_lat = ax_lat.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_lat.legend(
        handles_lat + handles_acc,
        labels_lat + labels_acc,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        columnspacing=1.2,
        handlelength=1.6,
    )
    fig.subplots_adjust(left=0.13, right=0.88, top=0.82, bottom=0.17)
    return save_pdf(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "Exp4_Ablation" / "ablation_combined_preview.pdf",
    )
    args = parser.parse_args()
    print(plot(args.output))


if __name__ == "__main__":
    main()
