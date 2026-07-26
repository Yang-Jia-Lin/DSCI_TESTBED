"""Section V.D convergence and system-overhead summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLOT_DIR = Path(__file__).resolve().parents[1] / "plot"
for path in (REPO_ROOT, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from Scripts.EvaluationCommon.config import (
    DEFAULT_SOLUTION_META,
    DEFAULT_SOLUTION_NPZ,
)
from Scripts.EvaluationCommon.config import (
    DEFAULT_TRAINING_EVENTS,
    EXP4_RESULT_DATA_DIR,
)
from Scripts.EvaluationCommon.overhead import (
    load_measurement_tree,
    load_measurements,
    load_metrics_jsonl,
    load_training_events,
    normalize_ppo_metrics,
    plot_ppo_vs_baselines,
    plot_optimizer_comparison,
    plot_exit_distribution,
    plot_latency_breakdown,
    summarize_expected_exit_distribution,
    summarize_expected_latency_breakdown,
    summarize_phase1_overhead,
    summarize_exit_distribution,
    summarize_latency_breakdown,
    summarize_training_events,
    write_metrics_jsonl,
)
from Scripts.EvaluationCommon.solutions import load_solution_bundle
from plot_convergency import (
    plot_convergence,
    plot_entropy,
    plot_lan_and_acc,
)
from plot_dsci_startup_overhead import (
    generate_startup_overhead_artifacts,
)
from Src.Phase2_Scheduler.Optimizer.BF.alg_BF import optimize_BF
from Src.Phase2_Scheduler.Optimizer.Greedy.alg_greedy import optimize_greedy
from Src.Phase2_Scheduler.Optimizer.Random.alg_random import optimize_random
from Src.Phase2_Scheduler.Objective.compute_P import compute_layer_exit_probs
from Src.Phase2_Scheduler.Objective.compute_latency import compute_5_latency
from Src.Shared.Utils.phase_timing import PHASE1_OVERHEAD_LOG


DEFAULT_PPO_DIR = (
    EXP4_RESULT_DATA_DIR / "legacy" / "DSCI_20260202_040737"
)
DEFAULT_TRAINING_CONVERGENCE_DIR = Path(
    "Data/Runtime/SolutionCache/TrainingConvergence"
)
DEFAULT_DEVICE_RESULTS = Path("Data/Runtime/DeviceResults")
PHASE1_PROFILE_STEPS = ("profile_segments",)
PPO_CONVERGENCE_COLUMNS = (
    "epoch",
    "outer_obj",
    "entropy_X",
    "entropy_Y",
    "latency",
    "acc",
)
PHASE3_COMPONENTS = (
    "device_compute",
    "d2e_transmission",
    "edge_compute",
    "e2c_transmission",
    "cloud_compute",
)


def _read_ppo_metrics_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = load_metrics_jsonl(path)
    if "event" in frame:
        frame = frame[frame["event"] == "ppo_epoch"].copy()
        frame = frame.dropna(axis="columns", how="all")
    return frame


def load_ppo_convergence_metrics(
    source: str | Path,
) -> tuple[pd.DataFrame, Path]:
    """Load one old metrics.jsonl or one Scheduler convergence JSONL run."""
    source = Path(source)
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        legacy_path = source / "metrics.jsonl"
        if legacy_path.is_file():
            candidates = [legacy_path]
        else:
            candidates = sorted(
                source.glob("*.jsonl"),
                key=lambda path: (path.stat().st_mtime_ns, path.name),
                reverse=True,
            )
    else:
        raise FileNotFoundError(f"PPO convergence source does not exist: {source}")

    if not candidates:
        raise FileNotFoundError(
            f"No metrics.jsonl or convergence *.jsonl file was found in: {source}"
        )

    rejected: list[str] = []
    for path in candidates:
        frame = _read_ppo_metrics_file(path)
        if frame.empty:
            rejected.append(f"{path.name}: no ppo_epoch rows")
            continue
        missing = [column for column in PPO_CONVERGENCE_COLUMNS if column not in frame]
        if missing:
            rejected.append(f"{path.name}: missing {', '.join(missing)}")
            continue
        frame = frame.copy()
        for column in PPO_CONVERGENCE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.equal(frame["epoch"], np.floor(frame["epoch"])).all():
            raise ValueError(f"Epoch values must be integers in: {path}")
        frame["epoch"] = frame["epoch"].astype(int)
        frame = frame.sort_values("epoch", kind="stable").reset_index(drop=True)
        return frame, path

    details = "; ".join(rejected)
    raise ValueError(f"No plottable PPO epoch log was found in {source}. {details}")


def default_ppo_convergence_source() -> Path:
    """Prefer a fresh Scheduler convergence run, then fall back to legacy data."""
    if DEFAULT_TRAINING_CONVERGENCE_DIR.is_dir():
        try:
            load_ppo_convergence_metrics(DEFAULT_TRAINING_CONVERGENCE_DIR)
        except (FileNotFoundError, ValueError):
            pass
        else:
            return DEFAULT_TRAINING_CONVERGENCE_DIR
    return DEFAULT_PPO_DIR


def run_convergence_analysis(
    data_source: str | Path,
    output_dir: Path = EXP4_RESULT_DATA_DIR,
) -> Path:
    df, metrics_path = load_ppo_convergence_metrics(data_source)
    source_path = Path(data_source)
    run_name = source_path.name if source_path.is_file() else metrics_path.parent.name
    if metrics_path.name != "metrics.jsonl":
        run_name = metrics_path.stem
    fig_save_dir = Path(output_dir) / run_name
    fig_save_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(df["outer_obj"], fig_save_dir, show=False)
    plot_entropy(df["entropy_X"], df["entropy_Y"], fig_save_dir, show=False)
    plot_lan_and_acc(df["latency"], df["acc"], fig_save_dir, show=False)
    csv_path = fig_save_dir / "ppo_convergence.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"convergence source: {metrics_path}")
    return csv_path


def run_overhead_summary(
    *,
    output_dir: str | Path = EXP4_RESULT_DATA_DIR,
    training_events: str | Path = DEFAULT_TRAINING_EVENTS,
    phase1_events: str | Path = PHASE1_OVERHEAD_LOG,
    solution_npz: str | Path = DEFAULT_SOLUTION_NPZ,
    solution_meta: str | Path = DEFAULT_SOLUTION_META,
    device_results: str | Path | None = DEFAULT_DEVICE_RESULTS,
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

    phase1 = summarize_phase1_overhead(
        phase1_events, include_steps=PHASE1_PROFILE_STEPS
    )
    phase1_path = output_dir / "phase1_overhead.csv"
    phase1.to_csv(phase1_path, index=False, encoding="utf-8-sig")
    outputs["phase1_overhead"] = phase1_path

    phase3_outputs = run_phase3_expected_summary(
        solution_npz=solution_npz,
        solution_meta=solution_meta,
        output_dir=output_dir,
    )
    outputs.update(phase3_outputs)

    if device_results:
        observed_outputs = run_observed_device_summary(
            device_results=device_results,
            output_dir=output_dir,
        )
        outputs.update(observed_outputs)

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

    summary_path = write_exp3_summary(output_dir=output_dir, outputs=outputs)
    outputs["exp3_overhead_summary"] = summary_path
    return outputs


def _metrics_from_history(algorithm: str, history: list[float]) -> list[dict]:
    rows = []
    for step, value in enumerate(history):
        rows.append({
            "algorithm": algorithm,
            "step": step,
            "current_obj": float(value),
            "best_obj": float(value),
            "evaluations": step + 1,
            "elapsed_s": 0.0,
        })
    return rows


def run_optimizer_baselines(
    *,
    solution_npz: str | Path = DEFAULT_SOLUTION_NPZ,
    solution_meta: str | Path = DEFAULT_SOLUTION_META,
    output_dir: str | Path = EXP4_RESULT_DATA_DIR,
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
    bf_metrics = _metrics_from_history("Exhaustive/BF", [float(x) for x in bf_history])
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
    outputs["optimizer_comparison_fig"] = plot_optimizer_comparison(
        combined, output_dir
    )
    return outputs


def run_ppo_vs_baselines(
    *,
    ppo_source: str | Path,
    solution_npz: str | Path = DEFAULT_SOLUTION_NPZ,
    solution_meta: str | Path = DEFAULT_SOLUTION_META,
    output_dir: str | Path = EXP4_RESULT_DATA_DIR,
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
    ppo_metrics, _ = load_ppo_convergence_metrics(ppo_source)
    ppo = normalize_ppo_metrics(ppo_metrics)
    baselines = pd.read_csv(outputs["optimizer_baselines"])
    max_episode = int(ppo["episode"].max()) if not ppo.empty else 0
    baseline_rows = []
    for algorithm, group in baselines.groupby("algorithm", sort=False):
        final_utility = float(group.sort_values("step")["best_obj"].iloc[-1])
        for episode in (0, max_episode):
            baseline_rows.append({
                "algorithm": algorithm,
                "step": episode,
                "episode": episode,
                "best_obj": final_utility,
                "current_obj": final_utility,
                "utility": final_utility,
                "evaluations": float(group["evaluations"].max()),
                "elapsed_s": float(group["elapsed_s"].max()),
                "curve_type": "final_baseline",
            })

    ppo_rows = ppo[
        [
            "algorithm",
            "step",
            "episode",
            "best_obj",
            "current_obj",
            "utility",
            "evaluations",
            "elapsed_s",
            "curve_type",
        ]
    ]
    combined = pd.concat([ppo_rows, pd.DataFrame(baseline_rows)], ignore_index=True)
    output_dir = Path(output_dir)
    combined_path = output_dir / "ppo_vs_baselines.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    outputs["ppo_vs_baselines"] = combined_path
    outputs["ppo_vs_baselines_fig"] = plot_ppo_vs_baselines(combined, output_dir)

    target_dir = output_dir / "optimizer_baselines"
    ppo_old_style = ppo.copy()
    ppo_old_style["step"] = ppo_old_style["episode"]
    old_style = pd.concat(
        [ppo_old_style, baselines],
        ignore_index=True,
    )
    outputs["optimizer_comparison_fig"] = plot_optimizer_comparison(
        old_style, target_dir
    )
    return outputs


def run_optimizer_comparison_with_ppo(**kwargs) -> dict[str, Path]:
    return run_ppo_vs_baselines(**kwargs)


def run_phase3_expected_summary(
    *,
    solution_npz: str | Path = DEFAULT_SOLUTION_NPZ,
    solution_meta: str | Path = DEFAULT_SOLUTION_META,
    output_dir: str | Path = EXP4_RESULT_DATA_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_solution_bundle(solution_npz, solution_meta)
    paras = bundle.paras
    P = compute_layer_exit_probs(bundle.Y, paras)
    component_arrays = dict(
        zip(
            PHASE3_COMPONENTS,
            compute_5_latency(bundle.X, P, bundle.F_e, bundle.F_c, paras),
        )
    )
    breakdown, per_user = summarize_expected_latency_breakdown(component_arrays)
    breakdown_path = output_dir / "phase3_latency_breakdown.csv"
    per_user_path = output_dir / "phase3_latency_breakdown_per_user.csv"
    breakdown.to_csv(breakdown_path, index=False, encoding="utf-8-sig")
    per_user.to_csv(per_user_path, index=False, encoding="utf-8-sig")

    exit_boundaries = [int(value) for value in paras.E] + [int(paras.m - 1)]
    exit_ids = [str(value) for value in paras.exit_ids] + ["final"]
    exit_probs = P[:, exit_boundaries]
    distribution = summarize_expected_exit_distribution(exit_ids, exit_probs)
    distribution_path = output_dir / "phase3_exit_distribution.csv"
    distribution.to_csv(distribution_path, index=False, encoding="utf-8-sig")

    outputs = {
        "phase3_latency_breakdown": breakdown_path,
        "phase3_latency_breakdown_per_user": per_user_path,
        "phase3_exit_distribution": distribution_path,
    }
    latency_fig = plot_latency_breakdown(breakdown, output_dir)
    exit_fig = plot_exit_distribution(distribution, output_dir)
    if latency_fig:
        outputs["phase3_latency_breakdown_fig"] = latency_fig
    if exit_fig:
        outputs["phase3_exit_distribution_fig"] = exit_fig
    return outputs


def run_observed_device_summary(
    *,
    device_results: str | Path = DEFAULT_DEVICE_RESULTS,
    output_dir: str | Path = EXP4_RESULT_DATA_DIR,
) -> dict[str, Path]:
    frame = load_measurement_tree(device_results)
    if frame.empty:
        return {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = (
        frame
        .groupby(["round_id", "exit_location"], dropna=False)
        .agg(
            samples=("request_id", "count"),
            accuracy=("is_correct", "mean"),
            mean_total_ms=("T_total", lambda values: float(values.mean()) * 1000.0),
        )
        .reset_index()
    )
    total_by_round = grouped.groupby("round_id")["samples"].transform("sum")
    grouped["rate"] = grouped["samples"] / total_by_round
    path = output_dir / "phase3_observed_device_results.csv"
    grouped.to_csv(path, index=False, encoding="utf-8-sig")
    return {"phase3_observed_device_results": path}


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _first_value(frame: pd.DataFrame, column: str, default: str = "n/a") -> str:
    if frame.empty or column not in frame:
        return default
    return str(frame[column].iloc[0])


def write_exp3_summary(*, output_dir: str | Path, outputs: dict[str, Path]) -> Path:
    output_dir = Path(output_dir)
    phase1 = pd.read_csv(outputs["phase1_overhead"])
    phase2 = pd.read_csv(outputs["phase2_training_overhead"])
    latency = pd.read_csv(outputs["phase3_latency_breakdown"])
    exits = pd.read_csv(outputs["phase3_exit_distribution"])
    observed_path = outputs.get("phase3_observed_device_results")
    observed = pd.read_csv(observed_path) if observed_path else pd.DataFrame()

    lines = [
        "# Exp3-D System Overhead Summary",
        "",
        "Scope: resnet50-cifar10-ee-v1; cached PPO metrics, SolutionCache, segment profiles, and runtime logs are reused.",
        "",
        "## Phase 1: profiling overhead",
    ]
    if phase1.empty:
        lines.append("- No phase1 profile_segments timing event was found.")
    else:
        for row in phase1.itertuples(index=False):
            lines.append(
                f"- {row.step} / {row.bundle_id}: mean {_fmt(row.duration_mean_s)} s "
                f"(min {_fmt(row.duration_min_s)} s, max {_fmt(row.duration_max_s)} s, runs {int(row.runs)})."
            )

    lines.extend(["", "## Phase 2: optimization overhead"])
    if phase2.empty:
        lines.append("- No training event was found.")
    else:
        for row in phase2.itertuples(index=False):
            lines.append(
                f"- {row.training_mode}: mean {_fmt(row.duration_mean_s)} s "
                f"(min {_fmt(row.duration_min_s)} s, max {_fmt(row.duration_max_s)} s, runs {int(row.runs)}, "
                f"mean objective {_fmt(row.objective_mean, 4)})."
            )

    lines.extend(["", "## Phase 3: expected inference latency breakdown"])
    for row in latency.itertuples(index=False):
        lines.append(
            f"- {row.label}: {_fmt(row.mean_ms)} ms, share {_fmt(100.0 * row.share_of_total, 2)}%."
        )
    lines.append(
        f"- Component shares sum to {_fmt(latency['share_of_total'].sum(), 6)}."
    )

    lines.extend(["", "## Phase 3: expected early-exit distribution"])
    for row in exits.itertuples(index=False):
        lines.append(f"- {row.exit_id}: {_fmt(100.0 * row.mean_rate, 2)}%.")
    exit_rows = exits[exits["kind"] == "exit_point"]
    early_row = exits[exits["exit_id"] == "early_exit_total"]
    if not early_row.empty:
        lines.append(
            f"- Early-exit total equals {_fmt(100.0 * float(early_row['mean_rate'].iloc[0]), 2)}%; "
            f"exit-point rates sum to {_fmt(float(exit_rows['mean_rate'].sum()), 6)}."
        )

    lines.extend(["", "## Runtime-log sanity check"])
    if observed.empty:
        lines.append("- No DeviceResults measurement file was found.")
    else:
        total_samples = int(observed["samples"].sum())
        locations = ", ".join(
            sorted(str(value) for value in observed["exit_location"].dropna().unique())
        )
        lines.append(
            f"- Observed DeviceResults cover {total_samples} samples across {observed['round_id'].nunique()} rounds."
        )
        lines.append(f"- Observed exit locations: {locations}.")
        if set(observed["exit_location"].astype(str)) == {"device"}:
            lines.append(
                "- Existing logs all terminate at device, so they validate the realized device-side fast path but do not contain per-head exit_id."
            )

    lines.extend([
        "",
        "## Artifacts",
        f"- PPO vs baselines: {_first_value(pd.DataFrame({'path': [outputs.get('ppo_vs_baselines', '')]}), 'path')}",
        f"- Phase 3 latency breakdown: {outputs['phase3_latency_breakdown']}",
        f"- Phase 3 exit distribution: {outputs['phase3_exit_distribution']}",
    ])
    target = output_dir / "exp3_overhead_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppo-source",
        "--ppo-dir",
        dest="ppo_source",
        help=(
            "Old experiment directory/metrics.jsonl, or a Scheduler "
            "TrainingConvergence directory/JSONL file. Defaults to the latest "
            "fresh Scheduler run, falling back to the legacy experiment."
        ),
    )
    parser.add_argument(
        "--convergence-only",
        action="store_true",
        help="Only export the PPO convergence CSV and figures.",
    )
    parser.add_argument("--solution-npz", default=str(DEFAULT_SOLUTION_NPZ))
    parser.add_argument("--solution-meta", default=str(DEFAULT_SOLUTION_META))
    parser.add_argument("--run-optimizer-baselines", action="store_true")
    parser.add_argument("--skip-optimizer-baselines", action="store_true")
    parser.add_argument("--random-iterations", type=int, default=200)
    parser.add_argument("--greedy-passes", type=int, default=2)
    parser.add_argument("--bf-max-iter", type=int, default=3)
    parser.add_argument("--training-events", default=str(DEFAULT_TRAINING_EVENTS))
    parser.add_argument("--phase1-events", default=str(PHASE1_OVERHEAD_LOG))
    parser.add_argument("--device-results", default=str(DEFAULT_DEVICE_RESULTS))
    parser.add_argument(
        "--measurements", help="Optional JSON/JSONL/CSV runtime measurements."
    )
    parser.add_argument("--output-dir", default=str(EXP4_RESULT_DATA_DIR))
    args = parser.parse_args(argv)

    generated_outputs: dict[str, Path] = {}
    ppo_source = (
        Path(args.ppo_source) if args.ppo_source else default_ppo_convergence_source()
    )
    if ppo_source:
        path = run_convergence_analysis(ppo_source, Path(args.output_dir))
        print(f"convergence: {path}")
        if not args.convergence_only and (
            args.run_optimizer_baselines or not args.skip_optimizer_baselines
        ):
            comparison_outputs = run_ppo_vs_baselines(
                ppo_source=ppo_source,
                solution_npz=args.solution_npz,
                solution_meta=args.solution_meta,
                output_dir=args.output_dir,
                random_iterations=args.random_iterations,
                greedy_passes=args.greedy_passes,
                bf_max_iter=args.bf_max_iter,
            )
            for name, path in comparison_outputs.items():
                print(f"{name}: {path}")
            generated_outputs.update(comparison_outputs)
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
        generated_outputs.update(baseline_outputs)
    if args.convergence_only:
        return
    outputs = run_overhead_summary(
        output_dir=args.output_dir,
        training_events=args.training_events,
        phase1_events=args.phase1_events,
        solution_npz=args.solution_npz,
        solution_meta=args.solution_meta,
        device_results=args.device_results,
        measurements=args.measurements,
    )
    generated_outputs.update(outputs)
    startup_outputs = generate_startup_overhead_artifacts(
        training_events=args.training_events,
        output_dir=args.output_dir,
    )
    generated_outputs.update(startup_outputs)
    summary_path = write_exp3_summary(
        output_dir=args.output_dir, outputs=generated_outputs
    )
    generated_outputs["exp3_overhead_summary"] = summary_path
    for name, path in outputs.items():
        print(f"{name}: {path}")
    for name, path in startup_outputs.items():
        print(f"{name}: {path}")
    print(f"exp3_overhead_summary: {summary_path}")


if __name__ == "__main__":
    main()
