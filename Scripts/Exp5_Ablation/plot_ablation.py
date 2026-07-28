"""Plot the measured ResNet-50/CIFAR-10 single-device ablation results."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.EvaluationCommon.paper_figure_style import (  # noqa: E402
    PAPER_FIGURE_DIR,
    THREE_PANEL_FIGSIZE,
    apply_three_panel_style,
    bold_tick_labels,
    save_pdf,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PLOT_DATA_PATH = EXPERIMENT_ROOT / "result_data" / "ablation_plot_data.csv"

ACCURACY_FLOOR = 90.0
LATENCY_COLOR = "#009E73"
ACCURACY_COLOR = "#8E63B0"
ANNOTATION_SIZE = 5.4


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
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            transform=(ax if coordinate_ax is None else coordinate_ax).transData,
            zorder=10,
        )


def _load_plot_data(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(
            csv.DictReader(handle),
            key=lambda row: int(row["order"]),
        )
    if not rows:
        raise ValueError(f"No plot data found in {path}")
    variants = [
        (
            "SEAS\n(ours)"
            if row["variant"] == "Ours"
            else row["variant"].replace(" only", "\nonly")
        )
        for row in rows
    ]
    latency = np.asarray([float(row["latency_ms"]) for row in rows], dtype=float)
    accuracy = np.asarray(
        [float(row["full_test_accuracy_pct"]) for row in rows],
        dtype=float,
    )
    return variants, latency, accuracy


def plot(data_path: Path, output: Path) -> Path:
    variants, latency, accuracy = _load_plot_data(data_path)
    apply_three_panel_style()
    fig, ax_lat = plt.subplots(figsize=THREE_PANEL_FIGSIZE)
    ax_acc = ax_lat.twinx()

    x = np.arange(len(variants))
    width = 0.32
    latency_bars = ax_lat.bar(
        x - width / 2,
        latency,
        width=width,
        color=LATENCY_COLOR,
        edgecolor="black",
        label="E2E latency",
        zorder=3,
    )
    accuracy_bars = ax_acc.bar(
        x + width / 2,
        accuracy - ACCURACY_FLOOR,
        bottom=ACCURACY_FLOOR,
        width=width,
        color=ACCURACY_COLOR,
        edgecolor="black",
        hatch="//",
        label="Full-test accuracy",
        zorder=3,
    )

    _annotate_bars(
        ax_acc,
        latency_bars[-1:],
        latency[-1:],
        offset=10.0,
        x_offset=width * 0.25,
        ha="right",
        coordinate_ax=ax_lat,
    )
    _annotate_bars(
        ax_acc,
        accuracy_bars[-1:],
        accuracy[-1:],
        offset=0.15,
    )

    ax_lat.set_ylabel("E2E latency (ms)", color=LATENCY_COLOR, labelpad=1.5)
    ax_acc.set_ylabel("Top-1 acc. (%)", color=ACCURACY_COLOR, labelpad=1.5)
    ax_lat.tick_params(axis="y", colors=LATENCY_COLOR)
    ax_acc.tick_params(axis="y", colors=ACCURACY_COLOR)
    ax_lat.spines["left"].set_color(LATENCY_COLOR)
    ax_acc.spines["right"].set_color(ACCURACY_COLOR)
    ax_lat.set_ylim(0, 450)
    ax_acc.set_ylim(ACCURACY_FLOOR, 97)
    ax_lat.set_yticks([0, 100, 200, 300, 400])
    ax_acc.set_yticks([90, 92, 94, 96])
    ax_lat.set_xticks(x, variants)
    bold_tick_labels(ax_lat, ax_acc)
    ax_lat.set_axisbelow(True)
    ax_lat.grid(axis="x", visible=False)
    ax_acc.grid(False)

    handles_lat, labels_lat = ax_lat.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_lat.legend(
        handles_lat + handles_acc,
        labels_lat + labels_acc,
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
    fig.subplots_adjust(left=0.20, right=0.80, top=0.835, bottom=0.28)
    return save_pdf(fig, output, fixed_canvas=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PLOT_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURE_DIR / "5_ablation.pdf",
    )
    args = parser.parse_args()
    print(plot(args.data, args.output))


if __name__ == "__main__":
    main()
