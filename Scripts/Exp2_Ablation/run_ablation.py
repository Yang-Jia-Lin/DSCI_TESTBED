"""Section V.C ablation runner.

The ablation is evaluated from a cached DSCI solution and the current bundle's
manifest/profile artifacts.  It is safe to run before all future bundles and
true-device profiles are available; use Exp1's readiness report to see pending
artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Scripts.EvaluationCommon.ablation_cases import build_ablation_cases
from Scripts.EvaluationCommon.config import (
    DEFAULT_SOLUTION_META,
    DEFAULT_SOLUTION_NPZ,
    EXP2_RESULT_DIR,
)
from Scripts.EvaluationCommon.solutions import (
    evaluate_matrices,
    load_solution_bundle,
    read_solution_index,
    write_rows,
)
from Scripts.Exp2_Ablation.plot_ablation import plot_bubble_chart, plot_utility_bar


def run_ablation(
    *,
    solution_npz: str | Path = DEFAULT_SOLUTION_NPZ,
    solution_meta: str | Path = DEFAULT_SOLUTION_META,
    output_dir: str | Path = EXP2_RESULT_DIR,
    plot: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_solution_bundle(solution_npz, solution_meta)
    rows = []
    for case in build_ablation_cases(bundle.paras, bundle.X, bundle.Y):
        rows.append(
            evaluate_matrices(
                name=case["name"],
                group=case["group"],
                X=case["X"],
                Y=case["Y"],
                F_e=bundle.F_e,
                F_c=bundle.F_c,
                paras=bundle.paras,
            )
        )
    result_path = write_rows(rows, output_dir / "ablation_results.csv")
    if plot:
        df = pd.DataFrame(rows).rename(
            columns={
                "latency_mean_ms": "latency_ms",
                "accuracy_mean": "accuracy",
                "utility": "objective",
            }
        )
        plot_bubble_chart(df, output_dir)
        plot_utility_bar(df, output_dir)
    return result_path


def run_ablation_for_index(
    *,
    solution_index: str | Path,
    output_dir: str | Path = EXP2_RESULT_DIR,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index_row in read_solution_index(solution_index):
        bundle = load_solution_bundle(index_row["solution_npz"], index_row["solution_meta"])
        for case in build_ablation_cases(bundle.paras, bundle.X, bundle.Y):
            row = evaluate_matrices(
                name=case["name"],
                group=case["group"],
                X=case["X"],
                Y=case["Y"],
                F_e=bundle.F_e,
                F_c=bundle.F_c,
                paras=bundle.paras,
            )
            for key, value in index_row.items():
                if key not in {"solution_npz", "solution_meta"}:
                    row[key] = value
            row["solution_npz"] = str(index_row["solution_npz"])
            row["solution_meta"] = str(index_row["solution_meta"])
            rows.append(row)
    return write_rows(rows, output_dir / "ablation_results_index.csv")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-npz", default=str(DEFAULT_SOLUTION_NPZ))
    parser.add_argument("--solution-meta", default=str(DEFAULT_SOLUTION_META))
    parser.add_argument("--solution-index")
    parser.add_argument("--output-dir", default=str(EXP2_RESULT_DIR))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.solution_index:
        result_path = run_ablation_for_index(
            solution_index=args.solution_index,
            output_dir=args.output_dir,
        )
    else:
        result_path = run_ablation(
            solution_npz=args.solution_npz,
            solution_meta=args.solution_meta,
            output_dir=args.output_dir,
            plot=not args.no_plot,
        )
    print(f"ablation: {result_path}")


if __name__ == "__main__":
    main()
