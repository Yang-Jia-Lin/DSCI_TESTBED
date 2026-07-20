"""Visualize DSCI/PPO startup-mode overhead from cached training events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Scripts.EvaluationCommon.config import DEFAULT_TRAINING_EVENTS, EXP3_RESULT_DIR
from Src.Shared.Utils.plot_utils import save_fig_for_ieee, set_ieee_style


STARTUP_MODE_ORDER = ["reuse", "near", "medium", "cold_warm", "cold"]
SEQUENCE_MODE_ORDER = ["cold", "medium", "near"]
MODE_LABELS = {
    "reuse": "Cache reuse",
    "near": "Near warm-start",
    "medium": "Medium warm-start",
    "cold_warm": "Cold-warm refinement",
    "cold": "Cold start",
}
MODE_COLORS = {
    "reuse": "#4C78A8",
    "near": "#54A24B",
    "medium": "#F58518",
    "cold_warm": "#9A9A9A",
    "cold": "#B54A4A",
}


def _mode_rank(mode: str, order: list[str]) -> int:
    try:
        return order.index(str(mode))
    except ValueError:
        return len(order)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def load_startup_events(path: str | Path) -> pd.DataFrame:
    """Load measured DSCI/PPO training-complete events only."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "round_id",
                "training_mode",
                "objective",
                "duration_s",
                "duration_min",
                "has_policy_source",
                "update_epoch",
                "started_at",
                "finished_at",
            ]
        )

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") != "training_complete":
                continue
            duration_s = float(record.get("duration_s", 0.0))
            rows.append(
                {
                    "round_id": record.get("round_id"),
                    "training_mode": record.get("training_mode"),
                    "objective": record.get("objective"),
                    "duration_s": duration_s,
                    "duration_min": duration_s / 60.0,
                    "has_policy_source": bool(record.get("policy_source")),
                    "update_epoch": record.get("update_epoch"),
                    "started_at": record.get("started_at"),
                    "finished_at": record.get("finished_at"),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["mode_rank"] = frame["training_mode"].map(
        lambda value: _mode_rank(str(value), STARTUP_MODE_ORDER)
    )
    frame = frame.sort_values(["mode_rank", "started_at", "round_id"], na_position="last")
    return frame.drop(columns=["mode_rank"])


def summarize_startup_modes(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "training_mode",
                "runs",
                "duration_mean_s",
                "duration_min_s",
                "duration_max_s",
                "duration_mean_min",
                "objective_mean",
                "speedup_vs_cold_mean",
            ]
        )

    summary = (
        events.groupby("training_mode", dropna=False)
        .agg(
            runs=("duration_s", "count"),
            duration_mean_s=("duration_s", "mean"),
            duration_min_s=("duration_s", "min"),
            duration_max_s=("duration_s", "max"),
            objective_mean=("objective", "mean"),
        )
        .reset_index()
    )
    summary["duration_mean_min"] = summary["duration_mean_s"] / 60.0
    cold = summary.loc[summary["training_mode"] == "cold", "duration_mean_s"]
    cold_mean = float(cold.iloc[0]) if not cold.empty and cold.iloc[0] > 0 else np.nan
    summary["speedup_vs_cold_mean"] = cold_mean / summary["duration_mean_s"]
    summary["mode_rank"] = summary["training_mode"].map(
        lambda value: _mode_rank(str(value), STARTUP_MODE_ORDER)
    )
    return summary.sort_values(["mode_rank", "training_mode"]).drop(columns=["mode_rank"])


def plot_mode_overhead(summary: pd.DataFrame, output_dir: str | Path) -> Path | None:
    if summary.empty:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = summary.copy()
    frame["mode_rank"] = frame["training_mode"].map(
        lambda value: _mode_rank(str(value), STARTUP_MODE_ORDER)
    )
    frame = frame.sort_values(["mode_rank", "training_mode"])

    set_ieee_style(mode="single")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    labels = [MODE_LABELS.get(str(mode), str(mode)) for mode in frame["training_mode"]]
    means = frame["duration_mean_s"].astype(float).to_numpy()
    mins = frame["duration_min_s"].astype(float).to_numpy()
    maxs = frame["duration_max_s"].astype(float).to_numpy()
    lower = np.maximum(means - mins, 0.0)
    upper = np.maximum(maxs - means, 0.0)
    colors = [MODE_COLORS.get(str(mode), "#4C78A8") for mode in frame["training_mode"]]
    positions = np.arange(len(frame))

    ax.barh(
        positions,
        means,
        xerr=np.vstack([lower, upper]),
        color=colors,
        alpha=0.9,
        capsize=3,
    )
    ax.set_yticks(positions, labels)
    ax.set_xlabel("PPO training duration (s)")
    ax.set_ylabel("Startup mode")

    xmax = float(max(maxs.max(), means.max()) * 1.45)
    ax.set_xlim(0.0, xmax)
    for pos, row in zip(positions, frame.itertuples(index=False)):
        mode = str(row.training_mode)
        mean_s = float(row.duration_mean_s)
        if mode == "cold":
            note = f"{mean_s:.1f}s"
        elif mode == "cold_warm":
            note = f"{mean_s:.1f}s, full refinement"
        else:
            note = f"{mean_s:.1f}s, {float(row.speedup_vs_cold_mean):.1f}x"
        label_x = max(float(row.duration_max_s), mean_s) + xmax * 0.012
        ax.text(label_x, pos, note, va="center", fontsize=7)

    ax.text(
        0.01,
        0.02,
        "Reuse: cached policy, 0s retraining.",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    plt.tight_layout(pad=0.2)
    target = output_dir / "dsci_startup_mode_overhead"
    save_fig_for_ieee(target, fig=fig)
    plt.close(fig)
    return target


def _canonical_sequence(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events

    fixedp = events[events["round_id"].astype(str).str.startswith("dsci-fixedp-")]
    source = fixedp if set(SEQUENCE_MODE_ORDER).issubset(set(fixedp["training_mode"])) else events
    rows = []
    for mode in SEQUENCE_MODE_ORDER:
        candidates = source[source["training_mode"] == mode].sort_values(
            ["started_at", "round_id"], na_position="last"
        )
        if candidates.empty:
            continue
        rows.append(candidates.iloc[0])
    return pd.DataFrame(rows)


def plot_sequence_speedup(events: pd.DataFrame, output_dir: str | Path) -> Path | None:
    sequence = _canonical_sequence(events)
    if len(sequence) < 2:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = sequence.copy()
    sequence["sequence_rank"] = sequence["training_mode"].map(
        lambda value: _mode_rank(str(value), SEQUENCE_MODE_ORDER)
    )
    sequence = sequence.sort_values("sequence_rank")

    set_ieee_style(mode="single")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    x = np.arange(len(sequence))
    y = sequence["duration_s"].astype(float).to_numpy()
    labels = [MODE_LABELS.get(str(mode), str(mode)) for mode in sequence["training_mode"]]
    ax.plot(x, y, marker="o", color="#4C78A8", linewidth=2.0)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("Training duration (s)")
    ax.set_xlabel("Startup progression")
    ax.set_ylim(0.0, float(y.max() * 1.18))

    first = float(y[0])
    for xi, yi in zip(x, y):
        speedup = first / float(yi) if yi > 0 else np.nan
        ax.annotate(
            f"{yi:.1f}s\n{speedup:.1f}x",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )

    if len(y) >= 3:
        ax.text(
            0.02,
            0.05,
            f"Canonical sequence speedup: {first / float(y[-1]):.1f}x.",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
        )

    plt.tight_layout(pad=0.2)
    target = output_dir / "dsci_startup_sequence_speedup"
    save_fig_for_ieee(target, fig=fig)
    plt.close(fig)
    return target


def write_startup_summary(
    *,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    cold = summary.loc[summary["training_mode"] == "cold"]
    medium = summary.loc[summary["training_mode"] == "medium"]
    near = summary.loc[summary["training_mode"] == "near"]
    sequence = _canonical_sequence(events)

    lines = [
        "# DSCI Startup-Mode Overhead Summary",
        "",
        "This summary reuses existing SolutionCache training events only; no PPO run is re-executed.",
        "",
        "## Main finding",
    ]
    if not cold.empty:
        cold_mean = float(cold["duration_mean_s"].iloc[0])
        lines.append(f"- Cold-start PPO training costs {cold_mean:.2f} s on average.")
        if not medium.empty:
            row = medium.iloc[0]
            lines.append(
                f"- Medium warm-start reduces mean training time to {float(row.duration_mean_s):.2f} s "
                f"({float(row.speedup_vs_cold_mean):.2f}x faster than cold mean)."
            )
        if not near.empty:
            row = near.iloc[0]
            lines.append(
                f"- Near warm-start reduces mean training time to {float(row.duration_mean_s):.2f} s "
                f"({float(row.speedup_vs_cold_mean):.2f}x faster than cold mean)."
            )
    else:
        lines.append("- No cold-start training event was found, so speedup versus cold cannot be computed.")
    lines.append("- Cache reuse has 0 s PPO retraining overhead because it returns the cached policy directly.")

    if len(sequence) >= 2:
        sequence = sequence.copy()
        sequence["sequence_rank"] = sequence["training_mode"].map(
            lambda value: _mode_rank(str(value), SEQUENCE_MODE_ORDER)
        )
        sequence = sequence.sort_values("sequence_rank")
        durations = sequence["duration_s"].astype(float).tolist()
        modes = sequence["training_mode"].astype(str).tolist()
        duration_text = " -> ".join(f"{value:.2f}s" for value in durations)
        lines.extend(
            [
                "",
                "## Canonical sequence",
                f"- {' -> '.join(modes)}: {duration_text}.",
                f"- End-to-end startup speedup in this sequence is {durations[0] / durations[-1]:.2f}x.",
            ]
        )

    lines.extend(["", "## Measured modes"])
    if summary.empty:
        lines.append("- No measured training-complete events were found.")
    else:
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.training_mode}: runs {int(row.runs)}, mean {_fmt(row.duration_mean_s)} s, "
                f"min {_fmt(row.duration_min_s)} s, max {_fmt(row.duration_max_s)} s, "
                f"speedup vs cold {_fmt(row.speedup_vs_cold_mean)}x."
            )
    target = output_dir / "dsci_startup_overhead_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def generate_startup_overhead_artifacts(
    *,
    training_events: str | Path = DEFAULT_TRAINING_EVENTS,
    output_dir: str | Path = EXP3_RESULT_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_startup_events(training_events)
    events_path = output_dir / "dsci_startup_modes.csv"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")

    summary = summarize_startup_modes(events)
    summary_path = output_dir / "dsci_startup_mode_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    outputs: dict[str, Path] = {
        "dsci_startup_modes": events_path,
        "dsci_startup_mode_summary": summary_path,
    }
    mode_fig = plot_mode_overhead(summary, output_dir)
    if mode_fig is not None:
        outputs["dsci_startup_mode_overhead_fig"] = mode_fig
    sequence_fig = plot_sequence_speedup(events, output_dir)
    if sequence_fig is not None:
        outputs["dsci_startup_sequence_speedup_fig"] = sequence_fig

    summary_md = write_startup_summary(
        events=events,
        summary=summary,
        output_dir=output_dir,
    )
    outputs["dsci_startup_overhead_summary"] = summary_md
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-events", default=str(DEFAULT_TRAINING_EVENTS))
    parser.add_argument("--output-dir", default=str(EXP3_RESULT_DIR))
    args = parser.parse_args(argv)

    outputs = generate_startup_overhead_artifacts(
        training_events=args.training_events,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
