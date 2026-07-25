"""Overhead summaries for Evaluation Section V.D."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Src.Shared.Utils.plot_utils import save_fig_for_ieee, set_ieee_style


PHASE3_COMPONENT_LABELS = {
    "device_compute": "Device compute",
    "d2e_transmission": "D2E transmission",
    "edge_compute": "Edge compute",
    "e2c_transmission": "E2C transmission",
    "cloud_compute": "Cloud compute",
}


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
        return pd.DataFrame(
            columns=[
                "algorithm",
                "step",
                "episode",
                "best_obj",
                "current_obj",
                "utility",
                "curve_type",
            ]
        )
    if "num_episodes" in metrics:
        episode = metrics["num_episodes"].fillna(0).astype(int).cumsum()
    else:
        episode = metrics["epoch"]
    # One completed PPO episode produces one joint solution for all users.
    # ``steps_collected`` counts per-user decisions, so using it would inflate
    # the candidate-evaluation axis by ``n`` in multi-device scenarios.
    evaluations = episode
    return pd.DataFrame(
        {
            "algorithm": "PPO",
            "step": episode,
            "episode": episode,
            "best_obj": metrics["outer_obj"],
            "current_obj": metrics.get("inner_best_obj", metrics["outer_obj"]),
            "utility": metrics["outer_obj"],
            "batch_mean_obj": metrics.get("inner_mean_obj", metrics["outer_obj"]),
            "batch_std_obj": metrics.get("inner_std_obj", np.nan),
            "batch_size": metrics.get("num_episodes", 1),
            "evaluations": evaluations,
            "elapsed_s": metrics.get("elapsed_s", 0.0),
            "curve_type": "ppo",
        }
    )


def summarize_phase1_overhead(
    path: str | Path, *, include_steps: tuple[str, ...] | None = None
) -> pd.DataFrame:
    events = load_training_events(path)
    if events.empty:
        return pd.DataFrame(
            columns=["step", "bundle_id", "runs", "duration_mean_s", "duration_min_s", "duration_max_s"]
        )
    if include_steps is not None and "step" in events:
        events = events[events["step"].isin(include_steps)]
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


def load_measurement_tree(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    frames = []
    for path in sorted(root.rglob("*_measurements.jsonl")):
        frame = load_measurements(path)
        if frame.empty:
            continue
        frame["measurement_file"] = str(path)
        frame["round_id"] = path.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def summarize_expected_latency_breakdown(
    components: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not components:
        empty = pd.DataFrame(columns=["component", "label", "mean_ms", "share_of_total"])
        return empty, pd.DataFrame()
    ordered = [name for name in PHASE3_COMPONENT_LABELS if name in components]
    matrix = np.vstack([np.asarray(components[name], dtype=np.float64) for name in ordered])
    totals = matrix.sum(axis=0)
    mean_total = float(np.mean(totals))
    rows = []
    for index, name in enumerate(ordered):
        mean_s = float(np.mean(matrix[index]))
        rows.append(
            {
                "component": name,
                "label": PHASE3_COMPONENT_LABELS[name],
                "mean_ms": mean_s * 1000.0,
                "share_of_total": mean_s / mean_total if mean_total > 0 else 0.0,
            }
        )
    per_user = pd.DataFrame(
        {
            "user_id": np.arange(totals.shape[0], dtype=int),
            **{name: np.asarray(components[name], dtype=np.float64) for name in ordered},
            "total_s": totals,
        }
    )
    return pd.DataFrame(rows), per_user


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


def summarize_expected_exit_distribution(
    exit_ids: list[str],
    exit_probabilities: np.ndarray,
) -> pd.DataFrame:
    probabilities = np.asarray(exit_probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("exit_probabilities must be shaped as users x exits")
    if probabilities.shape[1] != len(exit_ids):
        raise ValueError("exit_ids length does not match exit_probabilities")

    rows = []
    for index, exit_id in enumerate(exit_ids):
        values = probabilities[:, index]
        row = {
            "kind": "exit_point",
            "exit_id": str(exit_id),
            "mean_rate": float(np.mean(values)),
            "min_rate": float(np.min(values)),
            "max_rate": float(np.max(values)),
        }
        for user_index, value in enumerate(values):
            row[f"user_{user_index}_rate"] = float(value)
        rows.append(row)

    final_index = exit_ids.index("final") if "final" in exit_ids else len(exit_ids) - 1
    early_values = 1.0 - probabilities[:, final_index]
    early_row = {
        "kind": "aggregate",
        "exit_id": "early_exit_total",
        "mean_rate": float(np.mean(early_values)),
        "min_rate": float(np.min(early_values)),
        "max_rate": float(np.max(early_values)),
    }
    for user_index, value in enumerate(early_values):
        early_row[f"user_{user_index}_rate"] = float(value)
    rows.append(early_row)
    return pd.DataFrame(rows)


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


def plot_ppo_vs_baselines(metrics: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()

    ppo = metrics[metrics["curve_type"] == "ppo"].sort_values("episode")
    if not ppo.empty:
        ax.plot(
            ppo["episode"],
            ppo["utility"],
            label="PPO",
            linewidth=1.8,
        )
    baselines = metrics[metrics["curve_type"] == "final_baseline"]
    for algorithm, group in baselines.groupby("algorithm", sort=False):
        group = group.sort_values("episode")
        ax.plot(
            group["episode"],
            group["utility"],
            linestyle="--",
            linewidth=1.4,
            label=str(algorithm),
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Utility")
    ax.legend(frameon=False)
    plt.tight_layout(pad=0.2)
    target = output_dir / "ppo_vs_baselines"
    save_fig_for_ieee(target)
    return target


def plot_latency_breakdown(breakdown: pd.DataFrame, output_dir: str | Path) -> Path | None:
    if breakdown.empty:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    if "label" in breakdown:
        labels = breakdown["label"].astype(str).tolist()
    else:
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
    if "mean_rate" in distribution:
        labels = distribution["exit_id"].astype(str).tolist()
        values = distribution["mean_rate"].astype(float).tolist()
    else:
        labels = [
            f"{row.exit_location}\n{row.exit_id}" if str(row.exit_id) else str(row.exit_location)
            for row in distribution.itertuples(index=False)
        ]
        values = distribution["rate"].astype(float).tolist()
    ax.bar(labels, values)
    ax.set_ylabel("Exit rate")
    ax.set_xlabel("Exit point")
    plt.tight_layout(pad=0.2)
    target = output_dir / "phase3_exit_distribution"
    save_fig_for_ieee(target)
    return target
