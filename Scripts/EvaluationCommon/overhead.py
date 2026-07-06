"""Overhead summaries for Evaluation Section V.D."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from Src.Shared.Utils.plot_utils import save_fig_for_ieee, set_ieee_style

def load_metrics_jsonl(metrics_path: str | Path) -> pd.DataFrame:
    metrics_path = Path(metrics_path)
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_training_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    if not path.is_file():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def write_metrics_jsonl(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def normalize_ppo_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["algorithm", "step", "best_obj", "current_obj"])
    return pd.DataFrame(
        {
            "algorithm": "PPO",
            "step": metrics["epoch"],
            "best_obj": metrics["outer_obj"],
            "current_obj": metrics.get("inner_best_obj", metrics["outer_obj"]),
            "evaluations": metrics.get("steps_collected", metrics["epoch"]),
            "elapsed_s": metrics.get("elapsed_s", 0.0),
        }
    )


def summarize_phase1_overhead(path: str | Path) -> pd.DataFrame:
    events = load_training_events(path)
    if events.empty:
        return pd.DataFrame(
            columns=["step", "bundle_id", "runs", "duration_mean_s", "duration_min_s", "duration_max_s"]
        )
    return (
        events.groupby(["step", "bundle_id"], dropna=False)
        .agg(
            runs=("duration_s", "count"),
            duration_mean_s=("duration_s", "mean"),
            duration_min_s=("duration_s", "min"),
            duration_max_s=("duration_s", "max"),
        )
        .reset_index()
    )


def summarize_training_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "training_mode",
                "runs",
                "duration_mean_s",
                "duration_min_s",
                "duration_max_s",
                "objective_mean",
            ]
        )
    return (
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


def load_measurements(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return pd.DataFrame()
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return pd.DataFrame(rows)


def summarize_latency_breakdown(measurements: pd.DataFrame) -> pd.DataFrame:
    if measurements.empty:
        return pd.DataFrame(columns=["component", "mean_ms", "share_of_total"])
    total = float(measurements["T_total"].mean()) if "T_total" in measurements else 0.0
    rows = []
    for column in sorted(c for c in measurements.columns if c.startswith("T_")):
        mean_s = float(measurements[column].dropna().mean())
        rows.append(
            {
                "component": column,
                "mean_ms": mean_s * 1000.0,
                "share_of_total": mean_s / total if total > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_exit_distribution(measurements: pd.DataFrame) -> pd.DataFrame:
    if measurements.empty or "exit_location" not in measurements:
        return pd.DataFrame(columns=["exit_location", "exit_id", "samples", "rate"])
    frame = measurements.copy()
    if "exit_id" not in frame:
        frame["exit_id"] = ""
    grouped = (
        frame.groupby(["exit_location", "exit_id"], dropna=False)
        .size()
        .reset_index(name="samples")
    )
    grouped["rate"] = grouped["samples"] / float(len(frame))
    return grouped


def plot_optimizer_comparison(metrics: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    for algorithm, group in metrics.groupby("algorithm", sort=False):
        group = group.sort_values("step")
        ax.plot(group["step"], group["best_obj"], label=str(algorithm), linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Best Utility")
    ax.legend(frameon=False)
    plt.tight_layout(pad=0.2)
    target = output_dir / "optimizer_comparison"
    save_fig_for_ieee(target)
    return target


def plot_latency_breakdown(breakdown: pd.DataFrame, output_dir: str | Path) -> Path | None:
    if breakdown.empty:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    labels = breakdown["component"].astype(str).tolist()
    values = breakdown["mean_ms"].astype(float).tolist()
    ax.bar(labels, values)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_xlabel("Component")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout(pad=0.2)
    target = output_dir / "phase3_latency_breakdown"
    save_fig_for_ieee(target)
    return target


def plot_exit_distribution(distribution: pd.DataFrame, output_dir: str | Path) -> Path | None:
    if distribution.empty:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    labels = [
        f"{row.exit_location}\n{row.exit_id}" if str(row.exit_id) else str(row.exit_location)
        for row in distribution.itertuples(index=False)
    ]
    ax.bar(labels, distribution["rate"].astype(float).tolist())
    ax.set_ylabel("Exit rate")
    ax.set_xlabel("Exit point")
    plt.tight_layout(pad=0.2)
    target = output_dir / "phase3_exit_distribution"
    save_fig_for_ieee(target)
    return target
