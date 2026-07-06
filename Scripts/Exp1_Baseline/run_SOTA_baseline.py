"""Section V.B overall performance output for this work.

This script does not reproduce external-paper baselines.  It exports the
current work's bundle-aware DSCI result, artifact readiness table, and
bandwidth solution-index template.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from Scripts.EvaluationCommon.artifacts import write_readiness_report
from Scripts.EvaluationCommon.config import (
    DEFAULT_SOLUTION_META,
    DEFAULT_SOLUTION_NPZ,
    EXP1_RESULT_DIR,
    iter_bandwidth_cases,
)
from Scripts.EvaluationCommon.solutions import (
    evaluate_solution_row,
    evaluate_matrices,
    load_solution_bundle,
    read_solution_index,
    write_rows,
)


def write_solution_index_template(output_dir: Path) -> Path:
    return write_rows(
        iter_bandwidth_cases(),
        output_dir / "solution_index_template.csv",
    )


def run_overall_performance(
    *,
    solution_npz: str | Path = DEFAULT_SOLUTION_NPZ,
    solution_meta: str | Path = DEFAULT_SOLUTION_META,
    solution_index: str | Path | None = None,
    output_dir: str | Path = EXP1_RESULT_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    readiness_path = write_readiness_report(output_dir)
    template_path = write_solution_index_template(output_dir)
    if solution_index:
        rows = [evaluate_solution_row(row) for row in read_solution_index(solution_index)]
    else:
        bundle = load_solution_bundle(solution_npz, solution_meta)
        rows = [
            evaluate_matrices(
                name="Ours",
                X=bundle.X,
                Y=bundle.Y,
                F_e=bundle.F_e,
                F_c=bundle.F_c,
                paras=bundle.paras,
                group="ours",
            )
        ]
    result_path = write_rows(rows, output_dir / "overall_performance_ours.csv")
    return {
        "readiness": readiness_path,
        "solution_index_template": template_path,
        "overall": result_path,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-npz", default=str(DEFAULT_SOLUTION_NPZ))
    parser.add_argument("--solution-meta", default=str(DEFAULT_SOLUTION_META))
    parser.add_argument("--solution-index")
    parser.add_argument("--output-dir", default=str(EXP1_RESULT_DIR))
    args = parser.parse_args(argv)
    outputs = run_overall_performance(
        solution_npz=args.solution_npz,
        solution_meta=args.solution_meta,
        solution_index=args.solution_index,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
