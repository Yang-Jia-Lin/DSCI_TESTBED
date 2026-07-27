"""Plot the multi-device experiment results from the paper draft."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.EvaluationCommon.paper_figure_style import (  # noqa: E402
    ISPLITEE_COLOR,
    NEUTRAL_COLOR,
    PAPER_FIGURE_DIR,
    SEAM_COLOR,
    SEAM_MARKER,
    THREE_PANEL_FIGSIZE,
    add_comparison_legend,
    apply_three_panel_style,
    bold_tick_labels,
    save_pdf,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PLOT_DATA_PATH = EXPERIMENT_ROOT / "result_data" / "multi_device_plot_data.csv"

AXIS_LABEL_SIZE = 7.9
TICK_LABEL_SIZE = 6.9
LEGEND_FONT_SIZE = 4.8
LINE_WIDTH = 1.3
MARKER_SIZE = 4.0
MARKER_EDGE_WIDTH = 1.0
MS_PER_S = 1000.0


def _load_plot_data(path: Path) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No plot data found in {path}")
    methods = ("SEAS", "I-SplitEE")
    fields = (
        "mean_latency_ms",
        "p95_latency_ms",
        "worst_device_mean_latency_ms",
        "accuracy_pct",
    )
    devices = np.asarray(
        sorted({int(row["n_devices"]) for row in rows}),
        dtype=int,
    )
    data: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        method_rows = {
            int(row["n_devices"]): row
            for row in rows
            if row["method"] == method
        }
        missing = [int(n) for n in devices if int(n) not in method_rows]
        if missing:
            raise ValueError(f"{method} is missing device counts: {missing}")
        data[method] = {
            field: np.asarray(
                [float(method_rows[int(n)][field]) for n in devices],
                dtype=float,
            )
            for field in fields
        }
    return devices, data


def plot(data_path: Path, output: Path) -> Path:
    devices, data = _load_plot_data(data_path)
    seas = data["SEAS"]
    isplitee = data["I-SplitEE"]
    apply_three_panel_style()
    mpl.rcParams.update(
        {
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
        }
    )
    fig, ax_lat = plt.subplots(figsize=THREE_PANEL_FIGSIZE)
    ax_acc = ax_lat.twinx()

    ax_lat.plot(
        devices,
        seas["mean_latency_ms"] / MS_PER_S,
        color=SEAM_COLOR,
        marker=SEAM_MARKER,
        linestyle="-",
        label="SEAS (Ours) mean",
    )
    ax_lat.plot(
        devices,
        seas["p95_latency_ms"] / MS_PER_S,
        color=SEAM_COLOR,
        marker="^",
        linestyle="--",
        label="SEAS P95",
    )
    ax_lat.plot(
        devices,
        seas["worst_device_mean_latency_ms"] / MS_PER_S,
        color=SEAM_COLOR,
        marker="D",
        linestyle=":",
        label="SEAS worst-dev.",
    )
    ax_lat.plot(
        devices,
        isplitee["mean_latency_ms"] / MS_PER_S,
        color=ISPLITEE_COLOR,
        marker=SEAM_MARKER,
        linestyle="-",
        label="I-SplitEE mean",
    )
    ax_lat.plot(
        devices,
        isplitee["p95_latency_ms"] / MS_PER_S,
        color=ISPLITEE_COLOR,
        marker="^",
        linestyle="--",
        label="I-SplitEE P95",
    )
    ax_lat.plot(
        devices,
        isplitee["worst_device_mean_latency_ms"] / MS_PER_S,
        color=ISPLITEE_COLOR,
        marker="D",
        linestyle=":",
        label="I-SplitEE worst-dev.",
    )
    ax_lat.set_ylabel(
        "Latency (s)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=1.5,
    )
    ax_lat.set_ylim(0, 2.05)
    ax_lat.set_yticks([0, 1, 2])
    ax_acc.plot(
        devices,
        seas["accuracy_pct"],
        color=SEAM_COLOR,
        marker="s",
        markerfacecolor="white",
        markeredgewidth=MARKER_EDGE_WIDTH,
        linestyle="-.",
        label="SEAS (Ours) acc.",
    )
    ax_acc.plot(
        devices,
        isplitee["accuracy_pct"],
        color=ISPLITEE_COLOR,
        marker="s",
        markerfacecolor="white",
        markeredgewidth=MARKER_EDGE_WIDTH,
        linestyle="-.",
        label="I-SplitEE acc.",
    )
    ax_acc.set_ylabel(
        "Top-1 acc. (%)",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=1.5,
    )
    ax_acc.set_ylim(90, 97)
    ax_acc.set_yticks([90, 92, 94, 96])
    ax_acc.spines["right"].set_color(NEUTRAL_COLOR)
    ax_acc.tick_params(axis="y", colors=NEUTRAL_COLOR)
    ax_acc.yaxis.label.set_color(NEUTRAL_COLOR)

    ax_lat.set_xlim(0.8, 4.2)
    ax_lat.set_xticks(devices)
    ax_lat.set_xlabel(
        "Number of devices",
        fontsize=AXIS_LABEL_SIZE,
        labelpad=1.5,
    )
    ax_lat.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax_acc.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax_lat.set_axisbelow(True)
    ax_lat.grid(axis="x", visible=False)
    ax_acc.grid(False)
    legend_specs = (
        ("SEAS (Ours)", SEAM_COLOR, None, "-"),
        ("I-SplitEE", ISPLITEE_COLOR, None, "-"),
        ("Mean", NEUTRAL_COLOR, SEAM_MARKER, "-"),
        ("P95", NEUTRAL_COLOR, "^", "--"),
        ("Worst-dev.", NEUTRAL_COLOR, "D", ":"),
        ("Acc. (right)", NEUTRAL_COLOR, "s", "-."),
    )
    legend_handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor=(
                "white" if label == "Acc. (right)" else color
            ),
            markeredgewidth=(
                MARKER_EDGE_WIDTH if label == "Acc. (right)" else None
            ),
        )
        for label, color, marker, linestyle in legend_specs
    ]
    legend_labels = [label for label, _, _, _ in legend_specs]
    add_comparison_legend(
        fig,
        legend_handles,
        legend_labels,
        ncol=3,
        anchor_y=0.815,
        font_size=LEGEND_FONT_SIZE,
    )
    bold_tick_labels(ax_lat, ax_acc)
    fig.subplots_adjust(left=0.19, right=0.83, top=0.783, bottom=0.238)
    return save_pdf(fig, output, fixed_canvas=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PLOT_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURE_DIR / "3_multi_device.pdf",
    )
    args = parser.parse_args()
    print(plot(args.data, args.output))


if __name__ == "__main__":
    main()
