"""Plot the split latency and accuracy views of the generalization results."""

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


LABELS = ["R-C10", "R-IN100", "R-NEU", "V-C10", "V-IN100", "V-NEU"]
SEAS_ACCURACY = np.array([95.59, 93.22, 98.52, 98.57, 94.10, 98.15])
ISPLITEE_ACCURACY = np.array([94.87, 77.93, 98.61, 95.53, 79.47, 99.72])
SEAS_LATENCY = np.array([119.74, 240.82, 201.76, 209.73, 208.48, 184.98])
ISPLITEE_LATENCY = np.array([294.31, 334.62, 307.07, 310.40, 318.28, 225.86])
FIGSIZE = (255 / 72, 190 / 72)


def _base_axes() -> tuple[plt.Figure, plt.Axes]:
    apply_large_single_panel_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xticks(np.arange(len(LABELS)), LABELS, rotation=38, ha="right")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    bold_tick_labels(ax)
    return fig, ax


def _add_bars(ax: plt.Axes, seas: np.ndarray, isplitee: np.ndarray) -> None:
    x = np.arange(len(LABELS))
    width = 0.36
    ax.bar(
        x - width / 2,
        seas,
        width,
        color=SEAM_COLOR,
        edgecolor="black",
        label="SEAS",
    )
    ax.bar(
        x + width / 2,
        isplitee,
        width,
        color=ISPLITEE_COLOR,
        edgecolor="black",
        hatch="///",
        label="I-SplitEE",
    )


def plot(output: Path) -> tuple[Path, Path]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    latency_output = output.with_name(f"{output.stem}_latency.pdf")
    accuracy_output = output.with_name(f"{output.stem}_accuracy.pdf")

    fig, ax = _base_axes()
    _add_bars(ax, SEAS_LATENCY, ISPLITEE_LATENCY)
    ax.set_ylabel("E2E latency (ms)", labelpad=1.5)
    ax.set_ylim(0, 380)
    ax.set_yticks([0, 150, 300])
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
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
    fig.subplots_adjust(left=0.15, right=0.98, top=0.79, bottom=0.27)
    latency_path = save_pdf(fig, latency_output, fixed_canvas=True)

    fig, ax = _base_axes()
    _add_bars(ax, SEAS_ACCURACY, ISPLITEE_ACCURACY)
    ax.set_ylabel("Top-1 acc. (%)", labelpad=1.5)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 50, 100])
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
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
    fig.subplots_adjust(left=0.15, right=0.98, top=0.79, bottom=0.27)
    accuracy_path = save_pdf(fig, accuracy_output, fixed_canvas=True)
    return latency_path, accuracy_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "Exp1_SEAM" / "generalization.pdf",
    )
    args = parser.parse_args()
    for path in plot(args.output):
        print(path)


if __name__ == "__main__":
    main()
