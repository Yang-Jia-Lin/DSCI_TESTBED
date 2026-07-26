"""Plot all Exp0 motivation figures from prepared CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.Exp0_Motivation.run.config import (  # noqa: E402
    data_dir,
    figure_dir,
    prepare_result_dirs,
    update_paper_numbers,
)
from Src.Shared.Config.visualization import COLORS  # noqa: E402
from Src.Shared.Utils.plot_utils import save_fig_for_ieee, set_ieee_style  # noqa: E402


MARKERS = ("o", "s", "^", "D", "v", "P", "X")
FIG1_COLORS = {
    "ee1": "#1f77b4",
    "ee2": "#2ca02c",
    "policy": "#d62728",
    "remaining": "#ff7f0e",
    "baseline": "#000000",
}
PALETTE = {
    "Local-full": COLORS["grey"],
    "Cloud-full": COLORS["purple"],
    "Split-only": COLORS["green"],
    "EE-only": COLORS["purple"],
    "Decoupled": COLORS["red"],
    "Joint": COLORS["blue"],
}

AXIS_LABEL_FONTSIZE = 14
AXIS_TICK_FONTSIZE = 13


def _set_axis_font(ax) -> None:
    """Make all x/y axis labels and tick labels prominent and consistent."""
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontweight("bold")
    plt.setp(ax.get_xticklabels(), fontsize=AXIS_TICK_FONTSIZE, fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontsize=AXIS_TICK_FONTSIZE, fontweight="bold")
    ax.xaxis.get_offset_text().set_fontsize(AXIS_TICK_FONTSIZE)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontsize(AXIS_TICK_FONTSIZE)
    ax.yaxis.get_offset_text().set_fontweight("bold")


def _save(fig, run_dir: Path, name: str) -> dict:
    base = figure_dir(run_dir) / name
    save_fig_for_ieee(base, fig=fig)
    plt.close(fig)
    return {
        "pdf": str(base.with_suffix(".pdf")),
        "png": str(base.with_suffix(".png")),
    }


def _plot_figure1(run_dir: Path) -> dict:
    path = data_dir(run_dir) / "exp1_selection_effect.csv"
    frame = pd.read_csv(path)
    early_accuracy_frame = frame[frame["threshold"] < 1.0].copy()
    set_ieee_style(mode="single")

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(
        early_accuracy_frame["threshold"],
        early_accuracy_frame["after_layer2_conditional_accuracy_pct"],
        marker="o",
        markevery=12,
        color=FIG1_COLORS["ee1"],
        linewidth=2.2,
        alpha=0.95,
        markeredgewidth=1.0,
        label="Early Exit 1",
        zorder=3,
    )
    ax.plot(
        early_accuracy_frame["threshold"],
        early_accuracy_frame["after_layer3_conditional_accuracy_pct"],
        marker="s",
        markevery=12,
        color=FIG1_COLORS["ee2"],
        linewidth=2.2,
        alpha=0.95,
        markeredgewidth=1.0,
        label="Early Exit 2",
        zorder=3,
    )
    ax.plot(
        frame["threshold"],
        frame["overall_policy_accuracy_pct"],
        marker="^",
        markevery=12,
        color=FIG1_COLORS["policy"],
        linewidth=2.8,
        markeredgewidth=1.0,
        label="Overall Policy",
        zorder=4,
    )
    ax.axhline(
        frame["main_exit_accuracy_pct"].iloc[0],
        color=FIG1_COLORS["baseline"],
        linestyle="--",
        linewidth=1.8,
        label="Main Exit",
        zorder=2,
    )
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.text(
        0.04,
        frame["main_exit_accuracy_pct"].iloc[0] + 1.0,
        f"Main Exit: {frame['main_exit_accuracy_pct'].iloc[0]:.2f}%",
        color=FIG1_COLORS["baseline"],
        fontsize=9,
        fontweight="bold",
    )
    ax.set_ylim(60.0, 100.0)
    ax.legend(loc="lower right", frameon=True, ncol=1, fontsize=9)
    ax.set_title("Exit and Policy Accuracy")
    _set_axis_font(ax)
    fig.tight_layout(pad=0.2)
    accuracy_paths = _save(fig, run_dir, "fig1a_accuracy_expectation")

    set_ieee_style(mode="single")
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(
        frame["threshold"],
        frame["after_layer2_early_exit_probability_pct"],
        marker="o",
        markevery=12,
        color=FIG1_COLORS["ee1"],
        linewidth=2.4,
        markeredgewidth=1.0,
        label="Early Exit 1",
    )
    ax.plot(
        frame["threshold"],
        frame["after_layer3_early_exit_probability_pct"],
        marker="s",
        markevery=12,
        color=FIG1_COLORS["ee2"],
        linewidth=2.4,
        markeredgewidth=1.0,
        label="Early Exit 2",
    )
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Early Exit Probability (%)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 105.0)
    ax.legend(loc="lower left", frameon=True)
    ax.set_title("Early Exit Probability")
    _set_axis_font(ax)

    fig.tight_layout(pad=0.2)
    probability_paths = _save(fig, run_dir, "fig1b_early_exit_probability")
    return {
        "accuracy_expectation": accuracy_paths,
        "early_exit_probability": probability_paths,
    }


def _plot_figure2(run_dir: Path) -> dict:
    path = data_dir(run_dir) / "exp2_coupling_failure.csv"
    frame = pd.read_csv(path)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    strategies = ["Local-full", "Split-only", "EE-only", "Decoupled", "Joint"]
    for index, strategy in enumerate(strategies):
        group = frame[frame["strategy"] == strategy].sort_values("bandwidth_d2e_mbps")
        if strategy == "Decoupled":
            changes = (
                group[["b1", "b2", "tau"]]
                .ne(group[["b1", "b2", "tau"]].shift())
                .any(axis=1)
            )
            segment_ids = changes.cumsum()
            first_segment = True
            for _, segment in group.groupby(segment_ids):
                ax.plot(
                    segment["bandwidth_d2e_mbps"],
                    segment["latency_ms"],
                    marker=MARKERS[index % len(MARKERS)],
                    markevery=max(1, len(segment) // 5),
                    color=PALETTE[strategy],
                    label=strategy if first_segment else None,
                )
                first_segment = False
        else:
            ax.plot(
                group["bandwidth_d2e_mbps"],
                group["latency_ms"],
                marker=MARKERS[index % len(MARKERS)],
                markevery=max(1, len(group) // 8),
                color=PALETTE[strategy],
                label=strategy,
            )
    pivot = frame.pivot(
        index="bandwidth_d2e_mbps", columns="strategy", values="latency_ms"
    )
    joint_gain = pivot[pivot["Joint"] + 1e-9 < pivot["Decoupled"]]
    if not joint_gain.empty:
        ax.axvspan(
            float(joint_gain.index.min()) - 0.5,
            float(joint_gain.index.max()) + 0.5,
            color=COLORS["blue"],
            alpha=0.05,
            zorder=0,
        )
    ax.set_xlabel("Device-Edge Bandwidth (Mbps)")
    ax.set_ylabel("Expected Latency (ms)")
    ax.set_xticks([60, 80, 100, 120, 150])
    ax.legend(
        loc="upper right",
        frameon=True,
        ncol=1,
        fontsize=8,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.25,
    )
    ax.set_title("Coupled Decision Latency")
    _set_axis_font(ax)

    fig.tight_layout(pad=0.2)
    return _save(fig, run_dir, "fig2_coupling_failure")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    run_dir = prepare_result_dirs()
    paths = {
        "figure1": _plot_figure1(run_dir),
        "figure2": _plot_figure2(run_dir),
    }
    update_paper_numbers(run_dir, "figures", paths)
    print(f"Figures saved under: {figure_dir(run_dir)}")


if __name__ == "__main__":
    main()
