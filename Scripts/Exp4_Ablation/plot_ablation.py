"""Plot the incomplete ablation draft without replacing missing values by zero."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.EvaluationCommon.paper_figure_style import (  # noqa: E402
    NEUTRAL_COLOR,
    RESULTS_ROOT,
    SEAM_COLOR,
    apply_compact_ieee_style,
    save_pdf,
)


VARIANTS = ["End\nonly", "Split\nonly", "EE\nonly", "SEAM"]
LATENCY = np.array([np.nan, np.nan, np.nan, 240.82])
ACCURACY = np.array([np.nan, np.nan, np.nan, 93.22])


def _plot_metric(ax: plt.Axes, values: np.ndarray, ylabel: str, ylim: float) -> None:
    x = np.arange(len(VARIANTS))
    available = np.isfinite(values)
    ax.bar(
        x[available],
        values[available],
        width=0.58,
        color=SEAM_COLOR,
        edgecolor="black",
    )
    for xpos in x[~available]:
        ax.text(
            xpos,
            ylim * 0.05,
            "pending",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=6.0,
            fontweight="bold",
            color=NEUTRAL_COLOR,
        )
    for xpos, value in zip(x[available], values[available]):
        ax.text(
            xpos,
            value + ylim * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            fontweight="bold",
        )
    ax.set_ylabel(ylabel, labelpad=1.5)
    ax.set_ylim(0, ylim)
    ax.set_xticks(x, VARIANTS)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


def plot(output: Path) -> Path:
    apply_compact_ieee_style()
    fig, (ax_lat, ax_acc) = plt.subplots(
        1,
        2,
        figsize=(3.55, 1.82),
        gridspec_kw={"wspace": 0.32},
    )
    _plot_metric(ax_lat, LATENCY, "E2E latency (ms)", 300)
    _plot_metric(ax_acc, ACCURACY, "Top-1 acc. (%)", 105)
    ax_lat.set_yticks([0, 150, 300])
    ax_acc.set_yticks([0, 50, 100])
    fig.subplots_adjust(left=0.105, right=0.99, top=0.96, bottom=0.24)
    return save_pdf(fig, output)


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
