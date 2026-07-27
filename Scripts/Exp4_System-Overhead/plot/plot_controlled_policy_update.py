"""Plot standalone controlled PPO convergence and policy-update figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Scripts.EvaluationCommon.paper_figure_style import (
    PAPER_FIGURE_DIR,
    THREE_PANEL_FIGSIZE,
    apply_three_panel_style,
    bold_tick_labels,
    save_pdf,
)

MODE_ORDER = ["cold", "medium", "near", "reuse"]
MODE_LABELS = ["Cold", "Medium", "Near", "Reuse"]
COLORS = ["#009E73", "#3FA58F", "#6F8FAD", "#8E63B0"]
OBJECTIVE_COLOR = "#D55E00"
LATENCY_COLOR = "#009E73"
ACCURACY_COLOR = "#8E63B0"


def _band(
    axis,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    axis.plot(x, mean, color=color, linestyle=linestyle, label=label)
    axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


def _new_single_axis() -> tuple[plt.Figure, plt.Axes]:
    apply_three_panel_style()
    return plt.subplots(figsize=THREE_PANEL_FIGSIZE)


def _new_convergence_axis() -> tuple[plt.Figure, plt.Axes]:
    apply_three_panel_style()
    return plt.subplots(figsize=THREE_PANEL_FIGSIZE)


def _set_right_aligned_xlabel(
    axis: plt.Axes,
    label: str,
    *,
    x_position: float = 0.82,
) -> None:
    axis.set_xlabel(
        label,
        ha="left",
        fontsize=plt.rcParams["xtick.labelsize"],
    )
    axis.xaxis.set_label_coords(x_position, -0.065)


def _save(output_dir: Path, stem: str, fig: plt.Figure) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / stem
    save_pdf(fig, output.with_suffix(".pdf"), fixed_canvas=True)
    return output


def _save_convergence(output_dir: Path, stem: str, fig: plt.Figure) -> Path:
    return _save(output_dir, stem, fig)


def plot_convergence(summary_dir: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(summary_dir / "convergence_summary.csv")
    x = frame["epoch"].to_numpy(dtype=float)

    objective_fig, objective_axis = _new_convergence_axis()
    _band(
        objective_axis,
        x,
        frame["outer_objective_mean"].to_numpy(dtype=float),
        frame["outer_objective_std"].to_numpy(dtype=float),
        color=OBJECTIVE_COLOR,
        label="Scheduling objective",
    )
    _set_right_aligned_xlabel(objective_axis, "Epoch")
    objective_axis.set_ylabel("Objective")
    objective_axis.set_axisbelow(True)
    objective_axis.grid(axis="x", visible=False)
    bold_tick_labels(objective_axis)
    objective_fig.subplots_adjust(
        left=0.21,
        right=0.98,
        top=0.94,
        bottom=0.16,
    )

    metrics_fig, accuracy_axis = _new_convergence_axis()
    latency_axis = accuracy_axis.twinx()
    _band(
        accuracy_axis,
        x,
        100.0 * frame["expected_accuracy_mean_mean"].to_numpy(dtype=float),
        100.0 * frame["expected_accuracy_mean_std"].to_numpy(dtype=float),
        color=ACCURACY_COLOR,
        label="Expected accuracy",
    )
    _band(
        latency_axis,
        x,
        frame["expected_latency_mean_s_mean"].to_numpy(dtype=float),
        frame["expected_latency_mean_s_std"].to_numpy(dtype=float),
        color=LATENCY_COLOR,
        label="Expected latency",
        linestyle="--",
    )
    _set_right_aligned_xlabel(accuracy_axis, "Epoch", x_position=0.86)
    accuracy_axis.set_ylabel("Expected acc. (%)", color=ACCURACY_COLOR)
    latency_axis.set_ylabel("Expected latency (s)", color=LATENCY_COLOR)
    accuracy_axis.tick_params(axis="y", colors=ACCURACY_COLOR)
    latency_axis.tick_params(axis="y", colors=LATENCY_COLOR)
    accuracy_axis.spines["left"].set_color(ACCURACY_COLOR)
    latency_axis.spines["right"].set_color(LATENCY_COLOR)
    accuracy_axis.set_axisbelow(True)
    accuracy_axis.grid(axis="x", visible=False)
    latency_axis.grid(False)
    lines = accuracy_axis.lines + latency_axis.lines
    accuracy_axis.legend(
        lines,
        [line.get_label() for line in lines],
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="none",
        loc="center right",
        ncol=1,
        borderaxespad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
    )
    bold_tick_labels(accuracy_axis, latency_axis)
    metrics_fig.subplots_adjust(
        left=0.20,
        right=0.80,
        top=0.94,
        bottom=0.16,
    )

    return [
        _save_convergence(
            output_dir,
            "4_1a_ppo_objective",
            objective_fig,
        ),
        _save_convergence(
            output_dir,
            "4_1b_expected_system_metrics",
            metrics_fig,
        ),
    ]


def _ordered_policy_frame(summary_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_dir / "policy_update_summary.csv")
    order = {mode: index for index, mode in enumerate(MODE_ORDER)}
    frame["_order"] = frame["actual_training_mode"].map(order)
    if frame["_order"].isna().any():
        unknown = frame.loc[frame["_order"].isna(), "actual_training_mode"].tolist()
        raise ValueError(f"Unknown training modes: {unknown}")
    return frame.sort_values("_order").reset_index(drop=True)


def plot_policy_update(summary_dir: Path, output_dir: Path) -> list[Path]:
    frame = _ordered_policy_frame(summary_dir)
    x = np.arange(len(frame))
    labels = [
        MODE_LABELS[MODE_ORDER.index(mode)] for mode in frame["actual_training_mode"]
    ]

    panels = (
        (
            "foreground_service_ms_mean",
            "foreground_service_ms_std",
            "Response (ms)",
            "4_2a_foreground_response",
            1.0,
        ),
        (
            "background_update_s_mean",
            "background_update_s_std",
            "Update (s)",
            "4_2b_background_update",
            1.0,
        ),
        (
            "utility_retention_mean",
            "utility_retention_std",
            "Utility retention (%)",
            "4_2c_immediate_utility_retention",
            100.0,
        ),
    )
    outputs: list[Path] = []
    for mean_col, std_col, ylabel, stem, scale in panels:
        fig, axis = _new_single_axis()
        means = scale * frame[mean_col].to_numpy(dtype=float)
        stds = scale * frame[std_col].fillna(0.0).to_numpy(dtype=float)
        is_retention = stem == "4_2c_immediate_utility_retention"
        bars = axis.bar(
            x,
            means,
            width=0.62,
            yerr=None if is_retention else stds,
            capsize=0 if is_retention else 2,
            color=COLORS[: len(frame)],
            edgecolor="black",
            linewidth=0.7,
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        if is_retention:
            axis.set_ylim(90.0, 101.0)
            value_labels = [
                f"{np.floor(value * 100.0 + 1e-9) / 100.0:.2f}"
                for value in means
            ]
            axis.bar_label(bars, labels=value_labels, padding=2, fontsize=6.0)
        axis.margins(x=0.08)
        axis.set_axisbelow(True)
        axis.grid(axis="x", visible=False)
        bold_tick_labels(axis)
        fig.subplots_adjust(
            left=0.21,
            right=0.98,
            top=0.94,
            bottom=0.24,
        )
        outputs.append(_save(output_dir, stem, fig))
    return outputs


def plot_all(experiment_dir: Path, output_dir: Path) -> list[Path]:
    summary_dir = experiment_dir.resolve() / "summary"
    required = (
        summary_dir / "convergence_summary.csv",
        summary_dir / "policy_update_summary.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run summarize_controlled_policy_update first; missing: "
            + ", ".join(missing)
        )
    return [
        *plot_convergence(summary_dir, output_dir),
        *plot_policy_update(summary_dir, output_dir),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_FIGURE_DIR,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = plot_all(args.experiment_dir, args.output_dir)
    for output in outputs:
        print(f"Wrote {output}.png and {output}.pdf")


if __name__ == "__main__":
    main()
