"""Plot standalone controlled PPO convergence and policy-update figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Src.Shared.Utils.plot_utils import save_fig_for_ieee, set_ieee_style

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def _new_single_axis() -> tuple[plt.Figure, plt.Axes]:
    set_ieee_style("single")
    return plt.subplots(figsize=(3.45, 2.55))


def _save(output_dir: Path, stem: str, fig: plt.Figure) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / stem
    fig.tight_layout()
    save_fig_for_ieee(output, fig=fig)
    plt.close(fig)
    return output


def plot_convergence(summary_dir: Path, output_dir: Path) -> list[Path]:
    frame = pd.read_csv(summary_dir / "convergence_summary.csv")
    x = frame["epoch"].to_numpy(dtype=float)

    objective_fig, objective_axis = _new_single_axis()
    _band(
        objective_axis,
        x,
        frame["outer_objective_mean"].to_numpy(dtype=float),
        frame["outer_objective_std"].to_numpy(dtype=float),
        color="#4C78A8",
        label="Scheduling objective",
    )
    objective_axis.set_xlabel("PPO epoch")
    objective_axis.set_ylabel("Objective")

    metrics_fig, accuracy_axis = _new_single_axis()
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
    lines = accuracy_axis.lines + latency_axis.lines
    accuracy_axis.legend(
        lines,
        [line.get_label() for line in lines],
        frameon=False,
        loc="center right",
        fontsize=8,
        handlelength=2.2,
    )

    return [
        _save(output_dir, "4_1a_ppo_objective", objective_fig),
        _save(output_dir, "4_1b_expected_system_metrics", metrics_fig),
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
        axis.grid(axis="x", visible=False)
        fig.subplots_adjust(bottom=0.24)
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
