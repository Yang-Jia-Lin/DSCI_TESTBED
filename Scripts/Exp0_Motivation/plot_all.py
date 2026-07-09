"""Plot all Exp0 motivation figures from prepared CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.Exp0_Motivation.config import (  # noqa: E402
    DEFAULT_CONFIG,
    data_dir,
    figure_dir,
    prepare_run_dir,
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
    "Per-request joint decision": COLORS["red"],
    "Slow-timeslot joint decision": COLORS["blue"],
}


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
    pivot = frame.pivot(index="bandwidth_d2e_mbps", columns="strategy", values="latency_ms")
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

    fig.tight_layout(pad=0.2)
    return _save(fig, run_dir, "fig2_coupling_failure")


def _plot_figure3(run_dir: Path) -> dict:
    path = data_dir(run_dir) / "exp3_decision_overhead.csv"
    frame = pd.read_csv(path)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    for index, scheduler in enumerate(
        ["Per-request joint decision", "Slow-timeslot joint decision"]
    ):
        group = frame[frame["scheduler"] == scheduler].sort_values("n_users")
        ax.plot(
            group["n_users"],
            group["scheduling_overhead_ms_per_request"],
            marker=MARKERS[index],
            color=PALETTE[scheduler],
            label=scheduler,
        )
    ax.set_xlabel("Concurrent Users")
    ax.set_ylabel("Scheduling Overhead (ms/request)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(DEFAULT_CONFIG.exp3_users)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_yticks([1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(1.5, 150.0)
    per_request = (
        frame[frame["scheduler"] == "Per-request joint decision"]
        .sort_values("n_users")
        .reset_index(drop=True)
    )
    slow_timeslot = (
        frame[frame["scheduler"] == "Slow-timeslot joint decision"]
        .sort_values("n_users")
        .reset_index(drop=True)
    )
    per_request_growth = (
        per_request["scheduling_overhead_ms_per_request"].iloc[-1]
        / per_request["scheduling_overhead_ms_per_request"].iloc[0]
    )
    slow_drop = (
        slow_timeslot["scheduling_overhead_ms_per_request"].iloc[0]
        / slow_timeslot["scheduling_overhead_ms_per_request"].iloc[-1]
    )
    final_gap = (
        per_request["scheduling_overhead_ms_per_request"].iloc[-1]
        / slow_timeslot["scheduling_overhead_ms_per_request"].iloc[-1]
    )
    ax.annotate(
        f"{per_request_growth:.1f}x growth",
        xy=(32, per_request["scheduling_overhead_ms_per_request"].iloc[-1]),
        xytext=(10, 65),
        arrowprops={"arrowstyle": "->", "color": PALETTE["Per-request joint decision"], "lw": 1.0},
        color=PALETTE["Per-request joint decision"],
        fontsize=8,
        fontweight="bold",
    )
    ax.annotate(
        f"{slow_drop:.1f}x amortized",
        xy=(8, slow_timeslot["scheduling_overhead_ms_per_request"].iloc[3]),
        xytext=(2.4, 3.1),
        arrowprops={"arrowstyle": "->", "color": PALETTE["Slow-timeslot joint decision"], "lw": 1.0},
        color=PALETTE["Slow-timeslot joint decision"],
        fontsize=8,
        fontweight="bold",
    )
    ax.annotate(
        f"{final_gap:.0f}x lower overhead",
        xy=(32, slow_timeslot["scheduling_overhead_ms_per_request"].iloc[-1]),
        xytext=(10, 4.2),
        arrowprops={"arrowstyle": "->", "color": COLORS["grey"], "lw": 1.0},
        color=COLORS["grey"],
        fontsize=8,
        fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout(pad=0.2)
    return _save(fig, run_dir, "fig3_decision_overhead")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    run_dir = prepare_run_dir(args.run_id, prefer_latest=True)
    paths = {
        "figure1": _plot_figure1(run_dir),
        "figure2": _plot_figure2(run_dir),
        "figure3": _plot_figure3(run_dir),
    }
    update_paper_numbers(run_dir, "figures", paths)
    print(f"Figures saved under: {figure_dir(run_dir)}")


if __name__ == "__main__":
    main()
