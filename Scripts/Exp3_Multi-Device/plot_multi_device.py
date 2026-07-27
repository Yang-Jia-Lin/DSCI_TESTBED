"""Plot the multi-device experiment results from the paper draft."""

from __future__ import annotations

import argparse
import csv
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
    REFERENCE_FIGSIZE,
    SEAM_COLOR,
    SEAM_MARKER,
    add_comparison_legend,
    apply_comparison_figure_style,
    bold_tick_labels,
    match_reference_plot_area,
    save_pdf,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PLOT_DATA_PATH = EXPERIMENT_ROOT / "result_data" / "multi_device_plot_data.csv"
RESULT_FIGURE_DIR = EXPERIMENT_ROOT / "result_figure"

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
    apply_comparison_figure_style()
    fig, ax_lat = plt.subplots(figsize=REFERENCE_FIGSIZE)
    ax_acc = ax_lat.twinx()
    match_reference_plot_area(ax_lat, ax_acc)

    ax_lat.plot(
        devices,
        seas["mean_latency_ms"],
        color=SEAM_COLOR,
        marker=SEAM_MARKER,
        linestyle="-",
        label="SEAS mean",
    )
    ax_lat.plot(
        devices,
        seas["p95_latency_ms"],
        color=SEAM_COLOR,
        marker="^",
        linestyle="--",
        label="SEAS P95",
    )
    ax_lat.plot(
        devices,
        seas["worst_device_mean_latency_ms"],
        color=SEAM_COLOR,
        marker="D",
        linestyle=":",
        label="SEAS worst-dev.",
    )
    ax_lat.plot(
        devices,
        isplitee["mean_latency_ms"],
        color=ISPLITEE_COLOR,
        marker=SEAM_MARKER,
        linestyle="-",
        label="I-SplitEE mean",
    )
    ax_lat.plot(
        devices,
        isplitee["p95_latency_ms"],
        color=ISPLITEE_COLOR,
        marker="^",
        linestyle="--",
        label="I-SplitEE P95",
    )
    ax_lat.plot(
        devices,
        isplitee["worst_device_mean_latency_ms"],
        color=ISPLITEE_COLOR,
        marker="D",
        linestyle=":",
        label="I-SplitEE worst-dev.",
    )
    ax_lat.set_ylabel("Latency (ms)", labelpad=1.5)
    ax_lat.set_ylim(0, 2050)
    ax_lat.set_yticks([0, 1000, 2000])
    ax_acc.plot(
        devices,
        seas["accuracy_pct"],
        color=RANDOM_COLOR,
        marker=ISPLITEE_MARKER,
        linestyle="-.",
        label="SEAS acc.",
    )
    ax_acc.plot(
        devices,
        isplitee["accuracy_pct"],
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
    ax_lat.set_xticks(devices)
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
    return save_pdf(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PLOT_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_FIGURE_DIR / "3_multi_device.pdf",
    )
    args = parser.parse_args()
    print(plot(args.data, args.output))


if __name__ == "__main__":
    main()
