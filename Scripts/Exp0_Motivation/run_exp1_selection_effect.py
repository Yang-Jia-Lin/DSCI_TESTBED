"""Prepare Figure 1 data for the threshold-induced selection effect."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.Exp0_Motivation.config import (  # noqa: E402
    DEFAULT_CONFIG,
    canonical_curve_path,
    data_dir,
    prepare_run_dir,
    required_curve_columns,
    save_config,
    update_paper_numbers,
)
from Src.Shared.Config.model_config import get_bundle  # noqa: E402


def _first_crossing(frame: pd.DataFrame, accuracy_col: str, rate_col: str, baseline: float) -> float | None:
    selected = frame[(frame[rate_col] > 0.0) & (frame[accuracy_col] >= baseline)]
    if selected.empty:
        return None
    return float(selected.iloc[0]["threshold"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    cfg = DEFAULT_CONFIG
    run_dir = prepare_run_dir(args.run_id, prefer_latest=True)
    save_config(run_dir, cfg)
    curve_path = canonical_curve_path(run_dir)
    if not curve_path.is_file():
        raise FileNotFoundError(f"Canonical curve not found: {curve_path}")
    curves = pd.read_csv(curve_path)
    missing = sorted(required_curve_columns(cfg).difference(curves.columns))
    if missing:
        raise ValueError(f"Canonical curve misses columns: {missing}")

    bundle = get_bundle(cfg.bundle_id)
    final_accuracy = float(curves["final_accuracy"].iloc[0])
    distribution_cols = [f"{item.exit_id}_sequential_rate" for item in bundle.exits] + ["final_rate"]
    distribution_sum = curves[distribution_cols].sum(axis=1)
    if not ((distribution_sum - 100.0).abs() <= 1e-6).all():
        raise AssertionError("Sequential exit distribution does not sum to 100%")

    rows = []
    for _, row in curves.iterrows():
        record = {
            "threshold": float(row["threshold"]),
            "main_exit_accuracy_pct": final_accuracy,
            "overall_accuracy_pct": float(row["overall_accuracy"]),
            "final_rate_pct": float(row["final_rate"]),
        }
        for item in bundle.exits:
            record[f"{item.exit_id}_conditional_accuracy_pct"] = float(
                row[f"{item.exit_id}_sequential_accuracy"]
            )
            record[f"{item.exit_id}_sequential_rate_pct"] = float(
                row[f"{item.exit_id}_sequential_rate"]
            )
        rows.append(record)
    out = pd.DataFrame(rows)
    out_path = data_dir(run_dir) / "exp1_selection_effect.csv"
    out.to_csv(out_path, index=False)

    numbers = {
        "main_exit_accuracy_pct": final_accuracy,
        "overall_accuracy_max_pct": float(curves["overall_accuracy"].max()),
        "output_csv": str(out_path),
    }
    for item in bundle.exits:
        acc_col = f"{item.exit_id}_sequential_accuracy"
        rate_col = f"{item.exit_id}_sequential_rate"
        valid = curves[curves[rate_col] > 0.0]
        numbers[f"{item.exit_id}_max_conditional_accuracy_pct"] = (
            float(valid[acc_col].max()) if not valid.empty else None
        )
        numbers[f"{item.exit_id}_first_threshold_ge_main"] = _first_crossing(
            curves, acc_col, rate_col, final_accuracy
        )
    update_paper_numbers(run_dir, "figure1", numbers)
    print(f"Figure 1 data: {out_path}")


if __name__ == "__main__":
    main()
