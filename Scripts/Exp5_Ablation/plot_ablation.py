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
    ISPLITEE_COLOR,
    REFERENCE_FIGSIZE,
    SEAM_COLOR,
    apply_large_single_panel_style,
    bold_tick_labels,
    match_reference_plot_area,
    save_pdf,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PLOT_DATA_PATH = EXPERIMENT_ROOT / "result_data" / "ablation_plot_data.csv"
RESULT_FIGURE_DIR = EXPERIMENT_ROOT / "result_figure"

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


def _load_plot_data(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(
            csv.DictReader(handle),
            key=lambda row: int(row["order"]),
        )
    if not rows:
        raise ValueError(f"No plot data found in {path}")
    variants = [
        row["variant"].replace(" only", "\nonly") for row in rows
    ]
    latency = np.asarray([float(row["latency_ms"]) for row in rows], dtype=float)
    accuracy = np.asarray(
        [float(row["full_test_accuracy_pct"]) for row in rows],
        dtype=float,
    )
    return variants, latency, accuracy


def plot(data_path: Path, output: Path) -> Path:
    variants, latency, accuracy = _load_plot_data(data_path)
    apply_large_single_panel_style()
    fig, ax_lat = plt.subplots(figsize=REFERENCE_FIGSIZE)
    ax_acc = ax_lat.twinx()
    match_reference_plot_area(ax_lat, ax_acc)

    x = np.arange(len(variants))
    width = 0.34
    latency_bars = ax_lat.bar(
        x - width / 2,
        latency,
        width=width,
        color=SEAM_COLOR,
        edgecolor="black",
        label="E2E latency",
        zorder=3,
    )
    accuracy_bars = ax_acc.bar(
        x + width / 2,
        accuracy - ACCURACY_FLOOR,
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
        latency,
        offset=10.0,
        x_offset=width * 0.25,
        ha="right",
        coordinate_ax=ax_lat,
    )
    _annotate_bars(ax_acc, accuracy_bars, accuracy, offset=0.15)

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
    return save_pdf(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PLOT_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_FIGURE_DIR / "5_ablation.pdf",
    )
    args = parser.parse_args()
    print(plot(args.data, args.output))


if __name__ == "__main__":
    main()
