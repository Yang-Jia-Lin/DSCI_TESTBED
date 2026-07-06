"""Section V.D convergence and system-overhead summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Scripts.EvaluationCommon.config import (
    DEFAULT_SOLUTION_META,
    DEFAULT_SOLUTION_NPZ,
)
from Scripts.EvaluationCommon.config import DEFAULT_TRAINING_EVENTS, EXP3_RESULT_DIR
from Scripts.EvaluationCommon.overhead import (
    load_measurements,
    load_metrics_jsonl,
    load_training_events,
    normalize_ppo_metrics,
    plot_optimizer_comparison,
    plot_exit_distribution,
    plot_latency_breakdown,
    summarize_phase1_overhead,
    summarize_exit_distribution,
    summarize_latency_breakdown,
    summarize_training_events,
    write_metrics_jsonl,
)
from Scripts.EvaluationCommon.solutions import load_solution_bundle
from Scripts.Exp3_Convergency_and_Overhead.plot_convergency import (
    plot_convergence,
    plot_entropy,
    plot_lan_and_acc,
)
from Src.Phase2_Scheduler.Optimizer.BF.alg_BF import optimize_BF
from Src.Phase2_Scheduler.Optimizer.Greedy.alg_greedy import optimize_greedy
from Src.Phase2_Scheduler.Optimizer.Random.alg_random import optimize_random
from Src.Shared.Utils.phase_timing import PHASE1_OVERHEAD_LOG


def run_convergence_analysis(
    data_dir: Path,
    output_dir: Path = EXP3_RESULT_DIR,
) -> Path:
    metrics_path = Path(data_dir) / "metrics.jsonl"
    df = load_metrics_jsonl(metrics_path)
    fig_save_dir = Path(output_dir) / Path(data_dir).name
    fig_save_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(df["outer_obj"], fig_save_dir)
    plot_entropy(df["entropy_X"], df["entropy_Y"], fig_save_dir)
    plot_lan_and_acc(df["latency"], df["acc"], fig_save_dir)
    csv_path = fig_save_dir / "ppo_convergence.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def run_overhead_summary(
    *,
    output_dir: str | Path = EXP3_RESULT_DIR,
    training_events: str | Path = DEFAULT_TRAINING_EVENTS,
    phase1_events: str | Path = PHASE1_OVERHEAD_LOG,
    measurements: str | Path | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    events = load_training_events(training_events)
    event_summary = summarize_training_events(events)
    event_path = output_dir / "phase2_training_overhead.csv"
    event_summary.to_csv(event_path, index=False, encoding="utf-8-sig")
    outputs["phase2_training_overhead"] = event_path

    phase1 = summarize_phase1_overhead(phase1_events)
    phase1_path = output_dir / "phase1_overhead.csv"
    phase1.to_csv(phase1_path, index=False, encoding="utf-8-sig")
    outputs["phase1_overhead"] = phase1_path

    if measurements:
        measurement_df = load_measurements(measurements)
        breakdown = summarize_latency_breakdown(measurement_df)
        distribution = summarize_exit_distribution(measurement_df)
        breakdown_path = output_dir / "phase3_latency_breakdown.csv"
        distribution_path = output_dir / "phase3_exit_distribution.csv"
        breakdown.to_csv(breakdown_path, index=False, encoding="utf-8-sig")
        distribution.to_csv(distribution_path, index=False, encoding="utf-8-sig")
        outputs["phase3_latency_breakdown"] = breakdown_path
        outputs["phase3_exit_distribution"] = distribution_path
        latency_fig = plot_latency_breakdown(breakdown, output_dir)
        exit_fig = plot_exit_distribution(distribution, output_dir)
        if latency_fig:
            outputs["phase3_latency_breakdown_fig"] = latency_fig
        if exit_fig:
            outputs["phase3_exit_distribution_fig"] = exit_fig

    return outputs


def _metrics_from_history(algorithm: str, history: list[float]) -> list[dict]:
    rows = []
    for step, value in enumerate(history):
        rows.append(
            {
                "algorithm": algorithm,
                "step": step,
                "current_obj": float(value),
                "best_obj": float(value),
                "evaluations": step + 1,
                "elapsed_s": 0.0,
            }
        )
    return rows


def run_optimizer_baselines(
    *,
    solution_npz: str | Path = DEFAULT_SOLUTION_NPZ,
    solution_meta: str | Path = DEFAULT_SOLUTION_META,
    output_dir: str | Path = EXP3_RESULT_DIR,
    random_iterations: int = 200,
    greedy_passes: int = 2,
    bf_max_iter: int = 3,
) -> dict[str, Path]:
    output_dir = Path(output_dir) / "optimizer_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    paras = load_solution_bundle(solution_npz, solution_meta).paras
    outputs: dict[str, Path] = {}

    _, _, _, random_metrics = optimize_random(paras, iterations=random_iterations)
    outputs["random_metrics"] = write_metrics_jsonl(
        random_metrics, output_dir / "random_metrics.jsonl"
    )

    _, _, _, greedy_metrics = optimize_greedy(paras, passes=greedy_passes)
    outputs["greedy_metrics"] = write_metrics_jsonl(
        greedy_metrics, output_dir / "greedy_metrics.jsonl"
    )

    _, _, bf_history = optimize_BF(paras, max_iter=bf_max_iter, threshold_step=0.25)
    bf_metrics = _metrics_from_history("BF coordinate search", [float(x) for x in bf_history])
    outputs["bf_metrics"] = write_metrics_jsonl(
        bf_metrics, output_dir / "bf_coordinate_metrics.jsonl"
    )

    combined = pd.concat(
        [
            pd.DataFrame(random_metrics),
            pd.DataFrame(greedy_metrics),
            pd.DataFrame(bf_metrics),
        ],
        ignore_index=True,
    )
    combined_path = output_dir / "optimizer_baselines.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    outputs["optimizer_baselines"] = combined_path
    outputs["optimizer_comparison_fig"] = plot_optimizer_comparison(combined, output_dir)
    return outputs


def run_optimizer_comparison_with_ppo(
    *,
    ppo_dir: str | Path,
    solution_npz: str | Path = DEFAULT_SOLUTION_NPZ,
    solution_meta: str | Path = DEFAULT_SOLUTION_META,
    output_dir: str | Path = EXP3_RESULT_DIR,
    random_iterations: int = 200,
    greedy_passes: int = 2,
    bf_max_iter: int = 3,
) -> dict[str, Path]:
    outputs = run_optimizer_baselines(
        solution_npz=solution_npz,
        solution_meta=solution_meta,
        output_dir=output_dir,
        random_iterations=random_iterations,
        greedy_passes=greedy_passes,
        bf_max_iter=bf_max_iter,
    )
    ppo = normalize_ppo_metrics(load_metrics_jsonl(Path(ppo_dir) / "metrics.jsonl"))
    baselines = pd.read_csv(outputs["optimizer_baselines"])
    combined = pd.concat([ppo, baselines], ignore_index=True)
    target_dir = Path(output_dir) / "optimizer_baselines"
    combined_path = target_dir / "optimizer_comparison_with_ppo.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    outputs["optimizer_comparison_with_ppo"] = combined_path
    outputs["optimizer_comparison_with_ppo_fig"] = plot_optimizer_comparison(
        combined, target_dir
    )
    return outputs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-dir", help="Directory containing metrics.jsonl.")
    parser.add_argument("--solution-npz", default=str(DEFAULT_SOLUTION_NPZ))
    parser.add_argument("--solution-meta", default=str(DEFAULT_SOLUTION_META))
    parser.add_argument("--run-optimizer-baselines", action="store_true")
    parser.add_argument("--random-iterations", type=int, default=200)
    parser.add_argument("--greedy-passes", type=int, default=2)
    parser.add_argument("--bf-max-iter", type=int, default=3)
    parser.add_argument("--training-events", default=str(DEFAULT_TRAINING_EVENTS))
    parser.add_argument("--phase1-events", default=str(PHASE1_OVERHEAD_LOG))
    parser.add_argument("--measurements", help="Optional JSON/JSONL/CSV runtime measurements.")
    parser.add_argument("--output-dir", default=str(EXP3_RESULT_DIR))
    args = parser.parse_args(argv)

    if args.ppo_dir:
        path = run_convergence_analysis(Path(args.ppo_dir), Path(args.output_dir))
        print(f"convergence: {path}")
        if args.run_optimizer_baselines:
            comparison_outputs = run_optimizer_comparison_with_ppo(
                ppo_dir=args.ppo_dir,
                solution_npz=args.solution_npz,
                solution_meta=args.solution_meta,
                output_dir=args.output_dir,
                random_iterations=args.random_iterations,
                greedy_passes=args.greedy_passes,
                bf_max_iter=args.bf_max_iter,
            )
            for name, path in comparison_outputs.items():
                print(f"{name}: {path}")
    elif args.run_optimizer_baselines:
        baseline_outputs = run_optimizer_baselines(
            solution_npz=args.solution_npz,
            solution_meta=args.solution_meta,
            output_dir=args.output_dir,
            random_iterations=args.random_iterations,
            greedy_passes=args.greedy_passes,
            bf_max_iter=args.bf_max_iter,
        )
        for name, path in baseline_outputs.items():
            print(f"{name}: {path}")
    outputs = run_overhead_summary(
        output_dir=args.output_dir,
        training_events=args.training_events,
        phase1_events=args.phase1_events,
        measurements=args.measurements,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
