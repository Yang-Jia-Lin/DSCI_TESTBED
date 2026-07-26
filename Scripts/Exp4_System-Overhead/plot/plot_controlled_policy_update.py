"""Plot controlled PPO convergence and policy-update summaries."""

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

from Src.Shared.Utils.plot_utils import save_fig_for_ieee, set_ieee_style


MODE_ORDER = ["cold", "medium", "near", "reuse"]
MODE_LABELS = ["Cold", "Medium", "Near", "Reuse"]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B8B8B8"]


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


def plot_convergence(summary_dir: Path, output_dir: Path) -> Path:
    frame = pd.read_csv(summary_dir / "convergence_summary.csv")
    x = frame["epoch"].to_numpy(dtype=float)
    set_ieee_style("double")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    _band(
        axes[0],
        x,
        frame["outer_objective_mean"].to_numpy(dtype=float),
        frame["outer_objective_std"].to_numpy(dtype=float),
        color="#4C78A8",
        label="Scheduling objective",
    )
    axes[0].set_xlabel("PPO epoch")
    axes[0].set_ylabel("Objective")
    axes[0].set_title("(a) Scheduling objective")
    axes[0].legend(frameon=False)

    accuracy_axis = axes[1]
    latency_axis = accuracy_axis.twinx()
    _band(
        accuracy_axis,
        x,
        frame["expected_accuracy_mean_mean"].to_numpy(dtype=float),
        frame["expected_accuracy_mean_std"].to_numpy(dtype=float),
        color="#54A24B",
        label="Expected accuracy",
    )
    _band(
        latency_axis,
        x,
        1000.0 * frame["expected_latency_mean_s_mean"].to_numpy(dtype=float),
        1000.0 * frame["expected_latency_mean_s_std"].to_numpy(dtype=float),
        color="#E45756",
        label="Expected latency",
        linestyle="--",
    )
    accuracy_axis.set_xlabel("PPO epoch")
    accuracy_axis.set_ylabel("Expected accuracy", color="#54A24B")
    latency_axis.set_ylabel("Expected latency (ms)", color="#E45756")
    accuracy_axis.set_title("(b) Expected system metrics")
    lines = accuracy_axis.lines + latency_axis.lines
    accuracy_axis.legend(
        lines,
        [line.get_label() for line in lines],
        frameon=False,
        loc="best",
    )

    _band(
        axes[2],
        x,
        frame["split_entropy_HX_mean"].to_numpy(dtype=float),
        frame["split_entropy_HX_std"].to_numpy(dtype=float),
        color="#B279A2",
        label=r"$H_X$",
    )
    _band(
        axes[2],
        x,
        frame["exit_entropy_HY_mean"].to_numpy(dtype=float),
        frame["exit_entropy_HY_std"].to_numpy(dtype=float),
        color="#FF9DA6",
        label=r"$H_Y$",
        linestyle="--",
    )
    axes[2].set_xlabel("PPO epoch")
    axes[2].set_ylabel("Policy entropy")
    axes[2].set_title("(c) Policy entropy")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "ppo_convergence_multi_seed"
    save_fig_for_ieee(output, fig=fig)
    plt.close(fig)
    return output


def _ordered_policy_frame(summary_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_dir / "policy_update_summary.csv")
    order = {mode: index for index, mode in enumerate(MODE_ORDER)}
    frame["_order"] = frame["actual_training_mode"].map(order)
    if frame["_order"].isna().any():
        unknown = frame.loc[frame["_order"].isna(), "actual_training_mode"].tolist()
        raise ValueError(f"Unknown training modes: {unknown}")
    return frame.sort_values("_order").reset_index(drop=True)


def plot_policy_update(summary_dir: Path, output_dir: Path) -> Path:
    frame = _ordered_policy_frame(summary_dir)
    set_ieee_style("double")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    x = np.arange(len(frame))
    labels = [
        MODE_LABELS[MODE_ORDER.index(mode)]
        for mode in frame["actual_training_mode"]
    ]

    panels = (
        (
            "foreground_service_ms_mean",
            "foreground_service_ms_std",
            "Response (ms)",
            "(a) Foreground response",
            1.0,
        ),
        (
            "background_update_s_mean",
            "background_update_s_std",
            "Update (s)",
            "(b) Background update",
            1.0,
        ),
        (
            "utility_retention_mean",
            "utility_retention_std",
            "Utility retention (%)",
            "(c) Immediate quality",
            100.0,
        ),
    )
    for axis, (mean_col, std_col, ylabel, title, scale) in zip(axes, panels):
        means = scale * frame[mean_col].to_numpy(dtype=float)
        stds = scale * frame[std_col].fillna(0.0).to_numpy(dtype=float)
        axis.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=COLORS[: len(frame)],
            edgecolor="black",
            linewidth=0.6,
        )
        axis.set_xticks(x, labels, rotation=15)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="x", visible=False)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "controlled_policy_update_overhead"
    save_fig_for_ieee(output, fig=fig)
    plt.close(fig)
    return output


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
        plot_convergence(summary_dir, output_dir),
        plot_policy_update(summary_dir, output_dir),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "result_figure",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = plot_all(args.experiment_dir, args.output_dir)
    for output in outputs:
        print(f"Wrote {output}.png and {output}.pdf")


if __name__ == "__main__":
    main()
