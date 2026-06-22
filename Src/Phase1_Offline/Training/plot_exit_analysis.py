"""Plot training convergence and early-exit threshold curves for one bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Config.visualization import COLORS
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Utils.plot_utils import save_fig_for_ieee, set_ieee_style


MARKERS = ("o", "s", "^", "D", "v", "P", "X")
PALETTE = (
    COLORS["red"],
    COLORS["blue"],
    COLORS["green"],
    COLORS["purple"],
    COLORS["brown"],
    COLORS["grey"],
    COLORS["black"],
)


def _label(exit_id: str) -> str:
    return exit_id.replace("_", " ")


def _save_current(save_dir: Path, name: str, *, show: bool) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / name)
    if show:
        plt.show()
    plt.close()


def _plot_training_convergence(bundle_id: str, save_dir: Path, *, show: bool) -> None:
    paths = bundle_paths(bundle_id)
    files = [
        paths.analysis_root / "train_model_log.csv",
        paths.analysis_root / "finetune_exits_log.csv",
    ]
    frames = [pd.read_csv(path) for path in files if path.is_file()]
    if not frames:
        raise FileNotFoundError(
            f"No training logs found under {paths.analysis_root}; run train_model and finetune_exits first"
        )
    frame = pd.concat(frames, ignore_index=True)
    set_ieee_style(mode="single")
    plt.figure()
    for index, (stage, group) in enumerate(frame.groupby("stage", sort=False)):
        plt.plot(
            group["epoch"],
            group["val_acc"],
            color=PALETTE[index % len(PALETTE)],
            marker=MARKERS[index % len(MARKERS)],
            markevery=max(len(group) // 12, 1),
            label=_label(str(stage)),
        )
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy (%)")
    plt.xlim(left=0)
    plt.ylim(0, 100)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout(pad=0.15)
    _save_current(save_dir, f"{bundle_id}_training_convergence", show=show)


def _plot_exit_probability(bundle, curves: pd.DataFrame, save_dir: Path, *, show: bool) -> None:
    set_ieee_style(mode="single")
    plt.figure()
    for index, exit_spec in enumerate(bundle.exits):
        column = f"{exit_spec.exit_id}_sequential_rate"
        if column not in curves.columns:
            column = f"{exit_spec.exit_id}_rate"
        plt.plot(
            curves["threshold"],
            curves[column],
            color=PALETTE[(index + 1) % len(PALETTE)],
            marker=MARKERS[index % len(MARKERS)],
            markevery=max(len(curves) // 20, 1),
            label=_label(exit_spec.exit_id),
        )
    plt.xlabel("Threshold")
    plt.ylabel("Early Exit Probability (%)")
    plt.xlim(0, 1)
    plt.ylim(0, 105)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout(pad=0.15)
    _save_current(save_dir, f"{bundle.bundle_id}_exit_probability", show=show)


def _plot_accuracy(bundle, curves: pd.DataFrame, save_dir: Path, *, show: bool) -> None:
    set_ieee_style(mode="single")
    plt.figure()
    for index, exit_spec in enumerate(bundle.exits):
        column = f"{exit_spec.exit_id}_sequential_accuracy"
        if column not in curves.columns:
            column = f"{exit_spec.exit_id}_accuracy"
        plt.plot(
            curves["threshold"],
            curves[column],
            color=PALETTE[(index + 1) % len(PALETTE)],
            marker=MARKERS[index % len(MARKERS)],
            markevery=max(len(curves) // 20, 1),
            label=_label(exit_spec.exit_id),
        )
    overall_column = "overall_accuracy" if "overall_accuracy" in curves.columns else "final_accuracy"
    plt.plot(
        curves["threshold"],
        curves[overall_column],
        color=COLORS["red"],
        marker="^",
        markevery=max(len(curves) // 20, 1),
        label="overall",
    )
    final_acc = float(curves["final_accuracy"].iloc[0])
    plt.axhline(y=final_acc, color=COLORS["black"], linestyle="--", label="final")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy (%)")
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout(pad=0.15)
    _save_current(save_dir, f"{bundle.bundle_id}_accuracy_threshold", show=show)


def _plot_combined(bundle, curves: pd.DataFrame, save_dir: Path, *, show: bool) -> None:
    manifest = load_partition_manifest(bundle.bundle_id)
    expected_boundary = curves["final_rate"] * 0.01 * manifest.final_boundary_id
    for exit_spec in bundle.exits:
        boundary_id = manifest.boundary_for_exit(exit_spec.exit_id)
        column = f"{exit_spec.exit_id}_sequential_rate"
        if column not in curves.columns:
            column = f"{exit_spec.exit_id}_rate"
        expected_boundary += curves[column] * 0.01 * boundary_id

    accuracy_column = "overall_accuracy" if "overall_accuracy" in curves.columns else "final_accuracy"

    set_ieee_style(mode="single")
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(
        curves["threshold"],
        expected_boundary,
        color=COLORS["green"],
        marker="o",
        markevery=max(len(curves) // 20, 1),
        label="Expected Exit Boundary",
    )
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Expected Exit Boundary")
    ax1.tick_params(axis="y", colors=COLORS["green"])
    ax1.set_xlim(0, 1)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        curves["threshold"],
        curves[accuracy_column],
        color=COLORS["blue"],
        marker="^",
        markevery=max(len(curves) // 20, 1),
        label="Overall Accuracy",
    )
    ax2.set_ylabel("Overall Accuracy (%)")
    ax2.tick_params(axis="y", colors=COLORS["blue"])
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    lines = line1 + line2
    ax1.legend(lines, [line.get_label() for line in lines], loc="lower right", frameon=True)
    plt.tight_layout(pad=0.15)
    _save_current(save_dir, f"{bundle.bundle_id}_combined_expectation", show=show)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--curves-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    bundle = get_bundle(args.bundle_id)
    paths = bundle_paths(bundle.bundle_id)
    curves_path = Path(args.curves_csv or paths.analysis_root / "threshold_curves.csv")
    save_dir = Path(args.output_dir or paths.analysis_root)
    if not curves_path.is_file():
        raise FileNotFoundError(f"Threshold curve CSV not found: {curves_path}")
    curves = pd.read_csv(curves_path)

    _plot_training_convergence(bundle.bundle_id, save_dir, show=args.show)
    _plot_exit_probability(bundle, curves, save_dir, show=args.show)
    _plot_accuracy(bundle, curves, save_dir, show=args.show)
    _plot_combined(bundle, curves, save_dir, show=args.show)
    print(f"Saved plots under: {save_dir}")


if __name__ == "__main__":
    main()
