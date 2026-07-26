"""Plot the split latency and accuracy views of the generalization results."""

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
    SEAM_COLOR,
    apply_large_single_panel_style,
    bold_tick_labels,
    save_pdf,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PLOT_DATA_PATH = EXPERIMENT_ROOT / "result_data" / "cross_arch_dataset_plot_data.csv"
RESULT_FIGURE_DIR = EXPERIMENT_ROOT / "result_figure"

FIGSIZE = (255 / 72, 190 / 72)


def _load_plot_data(path: Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No plot data found in {path}")
    methods = ("SEAS", "I-SplitEE")
    orders = sorted({int(row["order"]) for row in rows})
    labels: list[str] = []
    series: dict[str, dict[str, list[float]]] = {
        method: {"latency_ms": [], "accuracy_pct": []} for method in methods
    }
    for order in orders:
        group = [row for row in rows if int(row["order"]) == order]
        label_values = {row["label"] for row in group}
        if len(label_values) != 1:
            raise ValueError(f"Order {order} has inconsistent labels")
        labels.append(label_values.pop())
        by_method = {row["method"]: row for row in group}
        missing = [method for method in methods if method not in by_method]
        if missing:
            raise ValueError(f"Order {order} is missing methods: {missing}")
        for method in methods:
            series[method]["latency_ms"].append(
                float(by_method[method]["latency_ms"])
            )
            series[method]["accuracy_pct"].append(
                float(by_method[method]["accuracy_pct"])
            )
    arrays = {
        method: {
            field: np.asarray(values, dtype=float)
            for field, values in fields.items()
        }
        for method, fields in series.items()
    }
    return labels, arrays


def _base_axes(labels: list[str]) -> tuple[plt.Figure, plt.Axes]:
    apply_large_single_panel_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=38, ha="right")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    bold_tick_labels(ax)
    return fig, ax


def _add_bars(ax: plt.Axes, seas: np.ndarray, isplitee: np.ndarray) -> None:
    x = np.arange(len(seas))
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


def plot(data_path: Path, output: Path) -> tuple[Path, Path]:
    labels, data = _load_plot_data(data_path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    latency_output = output.with_name(f"{output.stem}_latency.pdf")
    accuracy_output = output.with_name(f"{output.stem}_accuracy.pdf")

    fig, ax = _base_axes(labels)
    _add_bars(
        ax,
        data["SEAS"]["latency_ms"],
        data["I-SplitEE"]["latency_ms"],
    )
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

    fig, ax = _base_axes(labels)
    _add_bars(
        ax,
        data["SEAS"]["accuracy_pct"],
        data["I-SplitEE"]["accuracy_pct"],
    )
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
    parser.add_argument("--data", type=Path, default=PLOT_DATA_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_FIGURE_DIR / "cross_arch_dataset.pdf",
    )
    args = parser.parse_args()
    for path in plot(args.data, args.output):
        print(path)


if __name__ == "__main__":
    main()
