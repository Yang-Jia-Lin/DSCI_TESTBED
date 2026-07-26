"""Plot the matched cold-to-warm PPO policy-update sequence from SolutionCache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Scripts.EvaluationCommon.config import DEFAULT_TRAINING_EVENTS
from Scripts.EvaluationCommon.paper_figure_style import (
    apply_large_single_panel_style,
    bold_tick_labels,
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "result_figure"
)
DEFAULT_SEQUENCE_PREFIX = "dsci-fixedp-"
MODE_ORDER = ("cold", "medium", "near")
MODE_LABELS = {
    "cold": "Cold training",
    "medium": "Medium warm-start",
    "near": "Near warm-start",
    "reuse": "Exact cache reuse",
}
MODE_AXIS_LABELS = {
    "cold": "Cold\ntraining",
    "medium": "Medium\nwarm-start",
    "near": "Near\nwarm-start",
    "reuse": "Exact cache\nreuse",
}
MODE_COLORS = {
    "cold": "#B95F5F",
    "medium": "#F2A65A",
    "near": "#59A14F",
    "reuse": "#4E79A7",
}


def load_training_events(path: str | Path) -> pd.DataFrame:
    """Load completed PPO training events from the SolutionCache JSONL file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training-event log does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if record.get("event") != "training_complete":
                continue
            rows.append(
                {
                    "round_id": str(record.get("round_id") or ""),
                    "training_mode": str(record.get("training_mode") or ""),
                    "duration_s": float(record["duration_s"]),
                    "started_at": float(record.get("started_at") or 0.0),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No training_complete events found in {path}")
    return frame


def select_matched_sequence(
    events: pd.DataFrame,
    *,
    sequence_prefix: str = DEFAULT_SEQUENCE_PREFIX,
) -> pd.DataFrame:
    """Select one cold/medium/near sequence sharing the requested round prefix."""

    matched = events[events["round_id"].str.startswith(sequence_prefix)].copy()
    selected: list[pd.Series] = []
    missing: list[str] = []
    for mode in MODE_ORDER:
        candidates = matched[matched["training_mode"] == mode].sort_values(
            ["started_at", "round_id"]
        )
        if candidates.empty:
            missing.append(mode)
        else:
            selected.append(candidates.iloc[0])

    if missing:
        available = sorted(matched["training_mode"].unique().tolist())
        raise ValueError(
            f"Sequence prefix {sequence_prefix!r} is missing modes {missing}; "
            f"available modes: {available}"
        )

    sequence = pd.DataFrame(selected).reset_index(drop=True)
    cold_duration = float(
        sequence.loc[sequence["training_mode"] == "cold", "duration_s"].iloc[0]
    )
    sequence["speedup_vs_cold"] = cold_duration / sequence["duration_s"]

    reuse = pd.DataFrame(
        [
            {
                "round_id": "cache-reuse",
                "training_mode": "reuse",
                "duration_s": 0.0,
                "started_at": np.nan,
                "speedup_vs_cold": np.nan,
            }
        ]
    )
    return pd.concat([sequence, reuse], ignore_index=True)


def plot_policy_update_overhead(
    sequence: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Create one directly labelled figure of background PPO update time."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_large_single_panel_style()

    frame = sequence.set_index("training_mode").loc[
        ["cold", "medium", "near", "reuse"]
    ].reset_index()
    positions = np.arange(len(frame))
    durations = frame["duration_s"].to_numpy(dtype=float)
    colors = [MODE_COLORS[mode] for mode in frame["training_mode"]]

    # Match the final physical canvas of the ablation figure.
    fig, ax = plt.subplots(figsize=(3.3393, 2.6318))
    bars = ax.bar(
        positions,
        durations,
        color=colors,
        edgecolor="black",
        linewidth=0.9,
        width=0.62,
    )
    ax.scatter(
        [positions[-1]],
        [0.0],
        marker="D",
        s=26,
        color=MODE_COLORS["reuse"],
        zorder=3,
        clip_on=False,
    )
    ax.set_xticks(
        positions,
        [MODE_AXIS_LABELS[mode] for mode in frame["training_mode"]],
    )
    bold_tick_labels(ax)
    ax.set_ylabel("Policy-update time (s)")
    ax.grid(axis="x", visible=False)

    cold_duration = float(durations[0])
    ax.set_ylim(0.0, cold_duration * 1.18)
    for bar, row in zip(bars, frame.itertuples(index=False)):
        duration = float(row.duration_s)
        y = duration + cold_duration * 0.025
        label = "0 retraining" if row.training_mode == "reuse" else f"{duration:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=8.0,
        )
        if row.training_mode in {"medium", "near"}:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + cold_duration * 0.04,
                f"{float(row.speedup_vs_cold):.1f}x",
                ha="center",
                va="bottom",
                color="#D62728",
                fontweight="bold",
                fontsize=13.0,
            )

    target = output_dir / "scheduling_overhead"
    fig.subplots_adjust(left=0.20, right=0.98, top=0.94, bottom=0.24)
    fig.savefig(
        target.with_suffix(".pdf"),
        format="pdf",
        bbox_inches=fig.bbox_inches,
        pad_inches=0,
    )
    fig.savefig(
        target.with_suffix(".png"),
        format="png",
        bbox_inches=fig.bbox_inches,
        pad_inches=0,
        dpi=300,
    )
    plt.close(fig)
    return target


def generate_figure(
    *,
    training_events: str | Path = DEFAULT_TRAINING_EVENTS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sequence_prefix: str = DEFAULT_SEQUENCE_PREFIX,
) -> dict[str, Path]:
    events = load_training_events(training_events)
    sequence = select_matched_sequence(
        events,
        sequence_prefix=sequence_prefix,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cache_policy_update_overhead.csv"
    sequence.to_csv(csv_path, index=False, encoding="utf-8-sig")
    figure_path = plot_policy_update_overhead(sequence, output_dir)
    return {
        "data": csv_path,
        "figure_pdf": figure_path.with_suffix(".pdf"),
        "figure_png": figure_path.with_suffix(".png"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-events", default=str(DEFAULT_TRAINING_EVENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sequence-prefix", default=DEFAULT_SEQUENCE_PREFIX)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = generate_figure(
        training_events=args.training_events,
        output_dir=args.output_dir,
        sequence_prefix=args.sequence_prefix,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
