"""
Scripts/Exp0_Motivation/exp1_decoupling_failure/plot_exp1.py
实验1 图表：时延-带宽主图 + hard 组切分点漂移辅图。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style  # noqa: E402

from Scripts.Exp0_Motivation.utils.output_paths import resolve_output_dir

STYLES = {
    "Local": {"color": "#888888", "linestyle": "--", "marker": "s"},
    "EE-Only": {"color": "#2196F3", "marker": "^"},
    "SC-Only": {"color": "#FF9800", "marker": "D"},
    "Decoupled": {"color": "#F44336", "marker": "v"},
    "DSCI": {"color": "#4CAF50", "marker": "o", "linewidth": 2.5},
}
STRATEGY_ORDER = ["Local", "EE-Only", "SC-Only", "Decoupled", "DSCI"]


def _load_results(output_dir: Path) -> dict:
    path = output_dir / "Data" / "exp1_results.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _series(results: list[dict], group: str, strategy: str) -> tuple[list, list]:
    rows = sorted(
        [r for r in results if r["group"] == group and r["strategy"] == strategy],
        key=lambda x: x["bandwidth_mbps"],
    )
    bw = [r["bandwidth_mbps"] for r in rows]
    lat = [r["avg_latency_ms"] for r in rows]
    return bw, lat


def _find_inversion_point(bw, decoupled, local) -> int | None:
    for i in range(1, len(bw)):
        if decoupled[i] > local[i] and decoupled[i - 1] <= local[i - 1]:
            return i
    for i in range(len(bw)):
        if decoupled[i] > local[i]:
            return i
    return None


def plot_main_figure(payload: dict, output_dir: Path) -> None:
    set_ieee_style(mode="double")
    fig, axes = plt.subplots(1, 2)
    titles = {
        "easy": "(a) Easy Samples (High Exit Rate)",
        "hard": "(b) Hard Samples (Low Exit Rate)",
    }
    results = payload["results"]

    for ax, group in zip(axes, ("easy", "hard")):
        curves = {}
        for name in STRATEGY_ORDER:
            bw, lat = _series(results, group, name)
            kw = {**STYLES[name]}
            lw = kw.pop("linewidth", None)
            line, = ax.plot(bw, lat, label=name, **kw)
            if lw:
                line.set_linewidth(lw)
            curves[name] = (bw, lat)

        ax.invert_xaxis()
        ax.set_xlabel("Bandwidth (Mbps)")
        ax.set_ylabel("Average End-to-End Latency (ms)")
        ax.set_title(titles[group])

        if "Decoupled" in curves and "Local" in curves:
            idx = _find_inversion_point(
                curves["Decoupled"][0],
                curves["Decoupled"][1],
                curves["Local"][1],
            )
            if idx is not None:
                bx, by = curves["Decoupled"][0][idx], curves["Decoupled"][1][idx]
                ax.plot(
                    bx,
                    by,
                    marker="o",
                    markersize=8,
                    markerfacecolor="none",
                    markeredgecolor="red",
                    linestyle="none",
                    zorder=5,
                )
                ylo, yhi = ax.get_ylim()
                ax.annotate(
                    "Performance\nInversion",
                    xy=(bx, by),
                    xytext=(bx, ylo + 0.15 * (yhi - ylo)),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
                    fontsize=8,
                    color="red",
                    ha="center",
                )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=5,
        frameon=True,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    plt.tight_layout(pad=0.15)
    fig.subplots_adjust(top=0.72, wspace=0.28)
    save_fig_for_ieee(output_dir / "Figures" / "exp1_main", fig=fig)
    plt.close(fig)


def plot_split_drift(payload: dict, output_dir: Path) -> None:
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    results = payload["results"]
    exit_layers = payload["model_info"]["exit_layer_indices"]
    final_layer = payload["model_info"]["final_layer"]

    for name in ("SC-Only", "Decoupled", "DSCI"):
        rows = sorted(
            [r for r in results if r["group"] == "hard" and r["strategy"] == name],
            key=lambda x: x["bandwidth_mbps"],
        )
        bw = [r["bandwidth_mbps"] for r in rows]
        splits = [r["split_layer_chosen"] for r in rows]
        style = {k: v for k, v in STYLES[name].items() if k != "linewidth"}
        if name == "Decoupled":
            style["linestyle"] = "--"
        ax.plot(bw, splits, label=name, **style)

    ax.invert_xaxis()
    for layer in exit_layers:
        ax.axhline(layer, color="#bbbbbb", linestyle=":", linewidth=0.9, alpha=0.85)
    ax.axhline(final_layer, color="#cccccc", linestyle="--", linewidth=0.6, alpha=0.7)

    xmin, xmax = ax.get_xlim()
    x_text = xmin + 0.03 * (xmax - xmin)
    ax.text(x_text, exit_layers[0] + 1.5, f"L{exit_layers[0]}", fontsize=7, color="gray")
    ax.text(x_text, exit_layers[1] + 1.5, f"L{exit_layers[1]}", fontsize=7, color="gray")
    ax.text(x_text, final_layer + 1.0, f"L{final_layer}", fontsize=7, color="gray")

    ax.set_xlabel("Bandwidth (Mbps)")
    ax.set_ylabel("Chosen Split Layer X")
    ax.set_title("Split-Point Drift (Hard Group)")
    ax.set_ylim(-2, final_layer + 8)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout(pad=0.15)
    save_fig_for_ieee(output_dir / "Figures" / "exp1_split_drift", fig=fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, default=None)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    output_dir = resolve_output_dir(Path(PROJECT_ROOT), args.timestamp)
    payload = _load_results(output_dir)
    plot_main_figure(payload, output_dir)
    plot_split_drift(payload, output_dir)
    print(f"[plot_exp1] 图表已写入 {output_dir / 'Figures'}")


if __name__ == "__main__":
    main()
