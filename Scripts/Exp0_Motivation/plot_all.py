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
from Scripts.EvaluationCommon.paper_figure_style import (  # noqa: E402
    THREE_PANEL_FIGSIZE,
    apply_three_panel_style,
    bold_tick_labels,
    save_pdf,
)


MARKERS = ("o", "s", "^", "D", "v", "P", "X")
FIG1_COLORS = {
    "ee1": "#0072B2",
    "ee2": "#009E73",
    "policy": "#D55E00",
    "remaining": "#8E63B0",
    "baseline": "#6B6B6B",
}
PALETTE = {
    "Local-full": "#6B6B6B",
    "Cloud-full": "#B8B8B8",
    "Split-only": "#009E73",
    "EE-only": "#8E63B0",
    "Decoupled": "#0072B2",
    "Joint": "#D55E00",
}


def _set_axis_font(ax) -> None:
    """Keep Exp0 axes consistent with the shared paper-figure style."""
    bold_tick_labels(ax)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontweight("bold")


def _save(fig, run_dir: Path, name: str) -> dict:
    base = figure_dir(run_dir) / name
    save_pdf(fig, base.with_suffix(".pdf"), fixed_canvas=True)
    return {
        "pdf": str(base.with_suffix(".pdf")),
        "png": str(base.with_suffix(".png")),
    }


def _plot_figure1(run_dir: Path) -> dict:
    path = data_dir(run_dir) / "exp1_selection_effect.csv"
    frame = pd.read_csv(path)
    early_accuracy_frame = frame[frame["threshold"] < 1.0].copy()
    apply_three_panel_style()

    fig, ax = plt.subplots(figsize=THREE_PANEL_FIGSIZE)
    ax.plot(
        early_accuracy_frame["threshold"],
        early_accuracy_frame["after_layer2_conditional_accuracy_pct"],
        marker="o",
        markevery=12,
        color=FIG1_COLORS["ee1"],
        linewidth=1.5,
        markersize=3.8,
        alpha=0.95,
        markeredgewidth=0.8,
        label="Early Exit 1",
        zorder=3,
    )
    ax.plot(
        early_accuracy_frame["threshold"],
        early_accuracy_frame["after_layer3_conditional_accuracy_pct"],
        marker="s",
        markevery=12,
        color=FIG1_COLORS["ee2"],
        linewidth=1.5,
        markersize=3.8,
        alpha=0.95,
        markeredgewidth=0.8,
        label="Early Exit 2",
        zorder=3,
    )
    ax.plot(
        frame["threshold"],
        frame["overall_policy_accuracy_pct"],
        marker="^",
        markevery=12,
        color=FIG1_COLORS["policy"],
        linewidth=1.7,
        markersize=3.8,
        markeredgewidth=0.8,
        label="Overall Policy",
        zorder=4,
    )
    ax.axhline(
        frame["main_exit_accuracy_pct"].iloc[0],
        color=FIG1_COLORS["baseline"],
        linestyle="--",
        linewidth=1.4,
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
        fontsize=5.0,
        fontweight="bold",
    )
    ax.set_ylim(60.0, 100.0)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="none",
        ncol=1,
        fontsize=5.7,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )
    _set_axis_font(ax)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.21, right=0.98, top=0.94, bottom=0.24)
    accuracy_paths = _save(fig, run_dir, "0-1a_accuracy_expectation")

    apply_three_panel_style()
    fig, ax = plt.subplots(figsize=THREE_PANEL_FIGSIZE)
    ax.plot(
        frame["threshold"],
        frame["after_layer2_early_exit_probability_pct"],
        marker="o",
        markevery=12,
        color=FIG1_COLORS["ee1"],
        linewidth=1.5,
        markersize=3.8,
        markeredgewidth=0.8,
        label="Early Exit 1",
    )
    ax.plot(
        frame["threshold"],
        frame["after_layer3_early_exit_probability_pct"],
        marker="s",
        markevery=12,
        color=FIG1_COLORS["ee2"],
        linewidth=1.5,
        markersize=3.8,
        markeredgewidth=0.8,
        label="Early Exit 2",
    )
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Early-exit prob. (%)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0.0, 105.0)
    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="none",
        ncol=1,
        fontsize=6.5,
        handlelength=1.6,
        handletextpad=0.45,
        borderpad=0.3,
        labelspacing=0.3,
    )
    _set_axis_font(ax)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.21, right=0.98, top=0.94, bottom=0.24)
    probability_paths = _save(fig, run_dir, "0-1b_early_exit_probability")
    return {
        "accuracy_expectation": accuracy_paths,
        "early_exit_probability": probability_paths,
    }


def _plot_figure2(run_dir: Path) -> dict:
    path = data_dir(run_dir) / "exp2_coupling_failure.csv"
    frame = pd.read_csv(path)
    apply_three_panel_style()
    fig, ax = plt.subplots(figsize=THREE_PANEL_FIGSIZE)

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
                    markersize=3.8,
                    markeredgewidth=0.8,
                    linewidth=1.4,
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
                markersize=3.8,
                markeredgewidth=0.8,
                linewidth=1.4,
                color=PALETTE[strategy],
                label="Joint (Ours)" if strategy == "Joint" else strategy,
            )
    pivot = frame.pivot(
        index="bandwidth_d2e_mbps", columns="strategy", values="latency_ms"
    )
    joint_gain = pivot[pivot["Joint"] + 1e-9 < pivot["Decoupled"]]
    if not joint_gain.empty:
        ax.axvspan(
            float(joint_gain.index.min()) - 0.5,
            float(joint_gain.index.max()) + 0.5,
            color=PALETTE["Joint"],
            alpha=0.05,
            zorder=0,
        )
    ax.set_xlabel("D2E bandwidth (Mbps)")
    ax.set_ylabel("Expected Latency (ms)")
    ax.set_xticks([60, 80, 100, 120, 150])
    ax.legend(
        loc="center left",
        frameon=True,
        framealpha=0.72,
        facecolor="white",
        edgecolor="none",
        ncol=1,
        fontsize=5.3,
        handlelength=1.45,
        columnspacing=0.55,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.3,
    )
    _set_axis_font(ax)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.21, right=0.98, top=0.94, bottom=0.24)
    return _save(fig, run_dir, "0-2_coupling_failure")


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
