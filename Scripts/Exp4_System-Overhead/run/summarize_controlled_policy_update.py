"""Summarize preserved controlled-policy-update artifacts without running PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODE_ORDER = ["cold", "medium", "near", "reuse"]
CONVERGENCE_FIELDS = (
    "outer_objective",
    "paper_utility_sum",
    "expected_accuracy_sum",
    "expected_accuracy_mean",
    "expected_latency_sum_s",
    "expected_latency_mean_s",
    "split_entropy_HX",
    "exit_entropy_HY",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def load_mode_records(experiment_dir: Path) -> pd.DataFrame:
    paths = sorted(experiment_dir.glob("seed_*/modes/*/record.json"))
    if not paths:
        raise FileNotFoundError(f"No mode records found under {experiment_dir}")
    rows = [_load_json(path) for path in paths]
    frame = pd.DataFrame(rows)
    frame["actual_training_mode"] = pd.Categorical(
        frame["actual_training_mode"],
        categories=MODE_ORDER,
        ordered=True,
    )
    return frame.sort_values(["seed", "actual_training_mode"]).reset_index(drop=True)


def _convergence_path(
    experiment_dir: Path,
    record: dict[str, Any],
) -> Path | None:
    value = record.get("convergence_path")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    path = Path(str(value))
    resolved = path if path.is_absolute() else experiment_dir / path
    if resolved.exists():
        return resolved
    relocated = sorted(
        experiment_dir.glob(
            f"seed_{int(record['seed'])}_*/convergence/{path.name}"
        )
    )
    if len(relocated) == 1:
        return relocated[0]
    if not relocated:
        return (
            experiment_dir
            / f"seed_{int(record['seed'])}"
            / "convergence"
            / path.name
        )
    raise ValueError(
        f"Multiple relocated convergence logs match {path.name}: "
        + ", ".join(str(item) for item in relocated)
    )


def load_convergence_runs(
    experiment_dir: Path,
    mode_records: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in mode_records.to_dict("records"):
        path = _convergence_path(experiment_dir, record)
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing convergence log: {path}")
        epochs = [
            item for item in _load_jsonl(path) if item.get("event") == "ppo_epoch"
        ]
        running_best = -np.inf
        best_epoch = None
        converted: list[dict[str, Any]] = []
        for item in epochs:
            outer = float(item["outer_obj"])
            accuracy_sum = float(item["acc"])
            latency_sum = float(item["latency"])
            paper_utility = accuracy_sum - latency_sum
            is_current_best = bool(outer > running_best)
            if is_current_best:
                running_best = outer
                best_epoch = int(item["epoch"])
            converted.append(
                {
                    "experiment_id": record["experiment_id"],
                    "run_id": record["run_id"],
                    "seed": int(record["seed"]),
                    "state_id": record["state_id"],
                    "training_mode": str(record["actual_training_mode"]),
                    "epoch": int(item["epoch"]),
                    "wall_clock_s": float(item["elapsed_s"]),
                    "outer_objective": outer,
                    "paper_utility_sum": paper_utility,
                    "expected_accuracy_sum": accuracy_sum,
                    "expected_accuracy_mean": accuracy_sum
                    / float(record["n_users"]),
                    "expected_latency_sum_s": latency_sum,
                    "expected_latency_mean_s": latency_sum
                    / float(record["n_users"]),
                    "split_entropy_HX": float(item["entropy_X"]),
                    "exit_entropy_HY": float(item["entropy_Y"]),
                    "relative_objective_change": item.get(
                        "convergence_rel_change"
                    ),
                    "objective_cv": item.get("convergence_cv"),
                    "best_epoch": best_epoch,
                    "is_current_best": is_current_best,
                    "converged": bool(item.get("converged", False)),
                    "stopping_reason": "",
                    "utility_consistent": bool(
                        np.isclose(
                            outer,
                            paper_utility,
                            rtol=1e-6,
                            atol=1e-8,
                        )
                    ),
                    "source_path": str(path),
                }
            )
        if converted:
            converted[-1]["stopping_reason"] = (
                "stopping_criterion"
                if converted[-1]["converged"]
                else "max_epochs"
            )
        rows.extend(converted)
    if not rows:
        raise ValueError("No ppo_epoch records were found")
    frame = pd.DataFrame(rows)
    if not bool(frame["utility_consistent"].all()):
        inconsistent = frame.loc[
            ~frame["utility_consistent"],
            ["run_id", "epoch", "outer_objective", "paper_utility_sum"],
        ]
        raise ValueError(
            "alpha=beta=1 objective mismatch:\n"
            + inconsistent.to_string(index=False)
        )
    return frame.sort_values(
        ["seed", "state_id", "epoch"]
    ).reset_index(drop=True)


def convergence_summary(convergence: pd.DataFrame) -> pd.DataFrame:
    cold = convergence.loc[convergence["training_mode"] == "cold"].copy()
    if cold.empty:
        raise ValueError("No cold convergence rows found")
    epochs = range(int(cold["epoch"].min()), int(cold["epoch"].max()) + 1)
    carried_runs: list[pd.DataFrame] = []
    for seed, run in cold.groupby("seed", sort=True):
        carried = (
            run.sort_values("epoch")
            .set_index("epoch")[list(CONVERGENCE_FIELDS)]
            .reindex(epochs)
            .ffill()
        )
        if carried.isna().any().any():
            raise ValueError(
                f"Cold convergence run for seed {seed} does not start at "
                f"epoch {min(epochs)}"
            )
        carried["seed"] = seed
        carried["epoch"] = carried.index
        carried_runs.append(carried.reset_index(drop=True))
    carried_cold = pd.concat(carried_runs, ignore_index=True)
    observed_seed_count = cold.groupby("epoch")["seed"].nunique()
    aggregations: dict[str, tuple[str, str]] = {
        "seed_count": ("seed", "nunique"),
    }
    for field in CONVERGENCE_FIELDS:
        aggregations[f"{field}_mean"] = (field, "mean")
        aggregations[f"{field}_std"] = (field, "std")
    summary = carried_cold.groupby("epoch", as_index=False).agg(**aggregations)
    summary.insert(
        2,
        "observed_seed_count",
        summary["epoch"].map(observed_seed_count).fillna(0).astype(int),
    )
    std_columns = [column for column in summary if column.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return summary


def policy_update_summary(mode_records: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "foreground_service_ms",
        "background_update_s",
        "utility_retention",
        "objective_gap",
        "accuracy_gap_pp",
        "latency_gap_ms",
    )
    aggregations: dict[str, tuple[str, str]] = {
        "repetitions": ("seed", "nunique"),
    }
    for metric in metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    summary = (
        mode_records.groupby(
            "actual_training_mode",
            observed=True,
            sort=False,
        )
        .agg(**aggregations)
        .reset_index()
    )
    summary["actual_training_mode"] = summary["actual_training_mode"].astype(str)
    std_columns = [column for column in summary if column.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    summary["utility_retention_percent_2dp"] = (
        np.floor(summary["utility_retention_mean"] * 10000.0 + 1e-9) / 100.0
    )
    summary["utility_retention_std_percent_2dp"] = (
        np.floor(summary["utility_retention_std"] * 10000.0 + 1e-9) / 100.0
    )
    order = {mode: index for index, mode in enumerate(MODE_ORDER)}
    summary["_order"] = summary["actual_training_mode"].map(order)
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _mean_std(mean: Any, std: Any, digits: int) -> str:
    if pd.isna(mean):
        return "--"
    std = 0.0 if pd.isna(std) else float(std)
    return f"{float(mean):.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    labels = {
        "cold": "Cold",
        "medium": "Medium",
        "near": "Near",
        "reuse": "Reuse",
    }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Mode} & \textbf{Response (ms)} & "
        r"\textbf{Update (s)} & \textbf{Utility Retention (\%)} \\",
        r"\hline",
    ]
    for row in summary.to_dict("records"):
        mode = str(row["actual_training_mode"])
        response = _mean_std(
            row["foreground_service_ms_mean"],
            row["foreground_service_ms_std"],
            2,
        )
        update = _mean_std(
            row["background_update_s_mean"],
            row["background_update_s_std"],
            2,
        )
        retention = _mean_std(
            row["utility_retention_percent_2dp"],
            row["utility_retention_std_percent_2dp"],
            2,
        )
        lines.append(
            f"{labels.get(mode, mode)} & {response} & {update} & {retention} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(experiment_dir: Path) -> Path:
    experiment_dir = experiment_dir.resolve()
    manifest_paths = sorted(experiment_dir.glob("manifest_*.json"))
    legacy_manifest = experiment_dir / "manifest.json"
    if legacy_manifest.is_file():
        manifest_path = legacy_manifest
    elif len(manifest_paths) == 1:
        manifest_path = manifest_paths[0]
    elif not manifest_paths:
        raise FileNotFoundError(f"No manifest found under {experiment_dir}")
    else:
        raise ValueError(
            "Multiple manifests found; summarize one timestamped run at a time"
        )
    manifest = _load_json(manifest_path)
    if not (
        np.isclose(float(manifest["objective_alpha"]), 1.0)
        and np.isclose(float(manifest["objective_beta"]), 1.0)
    ):
        raise ValueError(
            "paper_utility_sum validation requires objective_alpha=objective_beta=1"
        )
    output_dir = experiment_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = load_mode_records(experiment_dir)
    convergence = load_convergence_runs(experiment_dir, modes)
    convergence_agg = convergence_summary(convergence)
    policy_agg = policy_update_summary(modes)

    modes.to_csv(
        output_dir / "controlled_policy_update.csv",
        index=False,
        encoding="utf-8-sig",
    )
    convergence.to_csv(
        output_dir / "convergence_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    convergence_agg.to_csv(
        output_dir / "convergence_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    policy_agg.to_csv(
        output_dir / "policy_update_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_latex_table(
        policy_agg,
        output_dir / "policy_update_summary.tex",
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Directory containing manifest_*.json and timestamped seed_* outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = summarize(args.experiment_dir)
    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
