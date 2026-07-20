"""Run Exp4 algorithm contrast for PPO, random search, greedy search, and BF.

Cached PPO metrics are used only when their experiment configuration matches
the same Paras used by the baselines. Otherwise the script tries to train PPO
on the current Paras and finally falls back to the cached PPO solution point.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Scripts.EvaluationCommon.config import DEFAULT_SOLUTION_META, DEFAULT_SOLUTION_NPZ
from Scripts.EvaluationCommon.overhead import (
    load_metrics_jsonl,
    normalize_ppo_metrics,
    write_metrics_jsonl,
)
from Scripts.EvaluationCommon.solutions import SolutionBundle, load_solution_bundle
from Src.Phase2_Scheduler.Optimizer.BF.alg_BF import optimize_BF
from Src.Phase2_Scheduler.Optimizer.GA.alg_GA import optimize_GA
from Src.Phase2_Scheduler.Optimizer.Greedy.alg_greedy import optimize_greedy
from Src.Phase2_Scheduler.Optimizer.Random.alg_random import optimize_random
from Src.Phase2_Scheduler.Optimizer.baseline_common import evaluate_candidate
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Config.paths import RESULT_DIR
from Src.Shared.Partitioning.split_actions import enumerate_deployment_pairs
from Src.Shared.Utils.plot_utils import save_fig_for_ieee, set_ieee_style


DEFAULT_PPO_DIR = Path("Scripts/Results/Optimize/DSCI_20260202_040737")
EXP4_RESULT_DIR = RESULT_DIR / "Exp4_Algorithm_Contrast"
DEFAULT_PPO_PARAMS: dict[str, Any] = {
    "gamma": 0.95,
    "lam": 0.95,
    "lr": 1e-4,
    "eps_clip": 0.15,
    "max_epochs": 80,
    "target_steps": 600,
    "k_epochs": 6,
    "entropy_coef": 0.01,
    "entropy_decay": 0.995,
    "grad_clip": 0.5,
    "obj_scale": 1000.0,
    "outer_ema": 0.02,
    "min_epochs": 20,
    "patience": 10,
    "rel_tolerance": 1e-4,
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _make_output_dir(base_dir: str | Path) -> Path:
    output_dir = Path(base_dir) / _timestamp()
    output_dir.mkdir(parents=True, exist_ok=False)
    latest = Path(base_dir) / "latest.txt"
    latest.write_text(str(output_dir), encoding="utf-8")
    return output_dir


def _parse_algorithms(value: str) -> tuple[str, ...]:
    aliases = {
        "random": "random",
        "rand": "random",
        "greedy": "greedy",
        "ga": "ga",
        "genetic": "ga",
        "bf": "bf",
        "exhaustive": "bf",
    }
    algorithms = []
    for raw in str(value).split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown algorithm '{raw}'. Valid: random, greedy, ga, bf")
        canonical = aliases[key]
        if canonical not in algorithms:
            algorithms.append(canonical)
    if not algorithms:
        raise ValueError("At least one baseline algorithm is required")
    return tuple(algorithms)


def _metrics_from_history(
    algorithm: str,
    history: list[float],
    *,
    elapsed_s: float,
    evals_per_step: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_index = max(len(history) - 1, 1)
    for step, value in enumerate(history):
        rows.append(
            {
                "algorithm": algorithm,
                "step": int(step),
                "current_obj": float(value),
                "best_obj": float(value),
                "evaluations": int(1 + step * evals_per_step),
                "elapsed_s": float(elapsed_s * step / final_index),
            }
        )
    return rows


def _bf_evals_per_step(paras: Paras, threshold_step: float) -> int:
    split_count = len(enumerate_deployment_pairs(paras.partition_boundary_ids))
    grid = np.arange(0.0, 1.0 + threshold_step / 2.0, threshold_step)
    threshold_count = len(list(product(grid, repeat=len(paras.E))))
    return int(paras.n * split_count * threshold_count)


def run_baselines(
    paras: Paras,
    *,
    algorithms: tuple[str, ...],
    random_iterations: int,
    greedy_passes: int,
    bf_max_iter: int,
    threshold_step: float,
    seed: int,
    ga_population_size: int,
    ga_generations: int,
    ga_mutation_rate: float,
    verbose_optimizers: bool,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    solutions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    frames: list[pd.DataFrame] = []

    if "random" in algorithms:
        random_best, random_sol, _, random_metrics = optimize_random(
            paras,
            iterations=random_iterations,
            seed=seed,
            threshold_step=threshold_step,
        )
        solutions["Random"] = random_sol
        if random_metrics:
            random_metrics[-1]["final_obj"] = float(random_best)
        frames.append(pd.DataFrame(random_metrics))

    if "greedy" in algorithms:
        greedy_best, greedy_sol, _, greedy_metrics = optimize_greedy(
            paras,
            passes=greedy_passes,
            threshold_step=threshold_step,
        )
        solutions["Greedy"] = greedy_sol
        if greedy_metrics:
            greedy_metrics[-1]["final_obj"] = float(greedy_best)
        frames.append(pd.DataFrame(greedy_metrics))

    if "ga" in algorithms:
        started = time.perf_counter()
        stdout = None if verbose_optimizers else io.StringIO()
        with contextlib.redirect_stdout(stdout):
            ga_best, ga_sol, ga_history, ga_metrics = optimize_GA(
                paras,
                population_size=ga_population_size,
                generations=ga_generations,
                mutation_rate=ga_mutation_rate,
                return_metrics=True,
            )
        ga_elapsed = time.perf_counter() - started
        X, Y, _F_e, _F_c = ga_sol
        ga_value, ga_F_e, ga_F_c = evaluate_candidate(paras, X, Y)
        if ga_value >= float(ga_best):
            ga_best = ga_value
            ga_sol = (X, Y, ga_F_e, ga_F_c)
        solutions["GA"] = ga_sol
        if ga_metrics:
            ga_metrics[-1]["best_obj"] = float(max(ga_metrics[-1]["best_obj"], ga_best))
            ga_metrics[-1]["current_obj"] = float(ga_best)
            ga_metrics[-1]["final_obj"] = float(ga_best)
        frames.append(pd.DataFrame(ga_metrics))

    if "bf" in algorithms:
        started = time.perf_counter()
        bf_best, bf_sol, bf_history, bf_metrics = optimize_BF(
            paras,
            max_iter=bf_max_iter,
            threshold_step=threshold_step,
            return_metrics=True,
        )
        _bf_elapsed = time.perf_counter() - started
        solutions["BF"] = bf_sol
        if bf_metrics:
            bf_metrics[-1]["final_obj"] = float(bf_best)
        frames.append(pd.DataFrame(bf_metrics))

    if not frames:
        raise ValueError("No baseline algorithms selected")
    return pd.concat(frames, ignore_index=True), solutions


def _paras_signature(paras: Paras) -> dict[str, Any]:
    return {
        "bundle_id": str(paras.bundle_id),
        "n": int(paras.n),
        "m": int(paras.m),
        "E": [int(value) for value in paras.E],
        "resource_mode": str(paras.resource_mode),
    }


def _ppo_signature_from_config(ppo_dir: str | Path) -> dict[str, Any] | None:
    config_path = Path(ppo_dir) / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    system = config.get("System_Parameters") or config.get("system_parameters") or {}
    if not system:
        return None
    return {
        "bundle_id": str(system.get("bundle_id", "")),
        "n": int(system["n"]) if system.get("n") is not None else None,
        "m": int(system["m"]) if system.get("m") is not None else None,
        "E": [int(value) for value in system.get("E", [])],
        "resource_mode": str(system.get("resource_mode", "")),
    }


def _compatible_signatures(paras_sig: dict[str, Any], ppo_sig: dict[str, Any] | None) -> bool:
    if ppo_sig is None:
        return False
    for key in ("n", "m", "E"):
        if ppo_sig.get(key) != paras_sig.get(key):
            return False
    if ppo_sig.get("bundle_id") and ppo_sig.get("bundle_id") != paras_sig.get("bundle_id"):
        return False
    if ppo_sig.get("resource_mode") and ppo_sig.get("resource_mode") != paras_sig.get("resource_mode"):
        return False
    return True


def load_ppo_metrics_curve(ppo_dir: str | Path, paras: Paras) -> pd.DataFrame:
    paras_sig = _paras_signature(paras)
    ppo_sig = _ppo_signature_from_config(ppo_dir)
    if not _compatible_signatures(paras_sig, ppo_sig):
        raise ValueError(
            "PPO metrics are not compatible with the baseline Paras. "
            f"paras={paras_sig}, ppo_metrics={ppo_sig}"
        )
    metrics_path = Path(ppo_dir) / "metrics.jsonl"
    ppo = normalize_ppo_metrics(load_metrics_jsonl(metrics_path))
    if ppo.empty:
        raise ValueError(f"No PPO metrics found in {metrics_path}")
    return ppo


def ppo_solution_curve(bundle: SolutionBundle) -> pd.DataFrame:
    from Src.Phase2_Scheduler.Objective.objective import objective

    utility = float(objective(bundle.X, bundle.Y, bundle.F_e, bundle.F_c, bundle.paras))
    return pd.DataFrame(
        [
            {
                "algorithm": "DSCI",
                "step": 0,
                "episode": 0,
                "best_obj": utility,
                "current_obj": utility,
                "utility": utility,
                "evaluations": 1,
                "elapsed_s": 0.0,
                "curve_type": "ppo_solution",
            }
        ]
    )


def train_ppo_curve(bundle: SolutionBundle, args: argparse.Namespace) -> pd.DataFrame:
    from Src.Phase2_Scheduler.Optimizer.DSCI.agent import PPOAgent

    params = dict(DEFAULT_PPO_PARAMS)
    params.update(
        {
            "max_epochs": int(args.ppo_max_epochs),
            "target_steps": int(args.ppo_target_steps),
            "k_epochs": int(args.ppo_k_epochs),
        }
    )
    agent = PPOAgent(bundle.paras, params)
    started = time.perf_counter()
    best_val, best_sol, _history = agent.train(initial_solution=None)
    ppo = normalize_ppo_metrics(pd.DataFrame(agent.logs))
    if ppo.empty:
        raise RuntimeError("PPO training produced no metrics")
    ppo["algorithm"] = "DSCI"
    ppo["curve_type"] = "dsci_train"
    if float(best_val) > float(ppo["utility"].max()):
        last = ppo.sort_values("episode").iloc[-1].to_dict()
        last["step"] = int(last["step"]) + 1
        last["episode"] = int(last["episode"]) + int(max(1, ppo["episode"].diff().dropna().median() if len(ppo) > 1 else 1))
        last["best_obj"] = float(best_val)
        last["current_obj"] = float(best_val)
        last["utility"] = float(best_val)
        last["elapsed_s"] = float(time.perf_counter() - started)
        last["curve_type"] = "dsci_train"
        ppo = pd.concat([ppo, pd.DataFrame([last])], ignore_index=True)
    ppo["best_obj"] = ppo["utility"].astype(float).cummax()
    ppo["utility"] = ppo["best_obj"]
    if best_sol is not None:
        ppo.attrs["solution"] = best_sol
    return ppo


def load_or_build_ppo_curve(bundle: SolutionBundle, args: argparse.Namespace) -> pd.DataFrame:
    if args.ppo_source == "metrics":
        return load_ppo_metrics_curve(args.ppo_dir, bundle.paras)
    if args.ppo_source == "solution":
        return ppo_solution_curve(bundle)
    if args.ppo_source == "train":
        return train_ppo_curve(bundle, args)

    try:
        return load_ppo_metrics_curve(args.ppo_dir, bundle.paras)
    except Exception as exc:
        print(f"[Exp4] Skip cached PPO metrics: {exc}")
    try:
        return train_ppo_curve(bundle, args)
    except ModuleNotFoundError as exc:
        print(f"[Exp4] PPO training unavailable ({exc}); using cached PPO solution point.")
        return ppo_solution_curve(bundle)
    except Exception as exc:
        print(f"[Exp4] PPO training failed ({exc}); using cached PPO solution point.")
        return ppo_solution_curve(bundle)


def build_ppo_vs_baselines(ppo: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    baseline_rows: list[dict[str, Any]] = []
    for algorithm, group in baselines.groupby("algorithm", sort=False):
        group = group.sort_values("step")
        for row in group.itertuples(index=False):
            baseline_rows.append(
                {
                    "algorithm": algorithm,
                    "step": int(row.step),
                    "episode": float(row.evaluations),
                    "best_obj": float(row.best_obj),
                    "current_obj": float(row.current_obj),
                    "utility": float(row.best_obj),
                    "evaluations": float(row.evaluations),
                    "elapsed_s": float(row.elapsed_s),
                    "curve_type": "baseline_progress",
                }
            )

    ppo = ppo.copy()
    if len(ppo) == 1 and baseline_rows:
        max_episode = max(float(row["episode"]) for row in baseline_rows)
        start = ppo.iloc[0].to_dict()
        end = dict(start)
        end["episode"] = max_episode
        end["step"] = max_episode
        ppo = pd.DataFrame([start, end])

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
    return pd.concat([ppo_rows, pd.DataFrame(baseline_rows)], ignore_index=True)


def build_summary(ppo: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ppo_sorted = ppo.sort_values("episode")
    ppo_curve_type = str(ppo_sorted["curve_type"].iloc[0])
    rows.append(
        {
            "algorithm": "DSCI",
            "curve_role": "learning_curve" if ppo_curve_type == "dsci_train" else "cached_solution",
            "final_utility": float(ppo_sorted["utility"].iloc[-1]),
            "best_utility": float(ppo_sorted["utility"].max()),
            "final_step": int(ppo_sorted["step"].iloc[-1]),
            "final_episode": int(ppo_sorted["episode"].iloc[-1]),
            "evaluations": float(ppo_sorted["evaluations"].iloc[-1]),
            "elapsed_s": float(ppo_sorted["elapsed_s"].iloc[-1]),
        }
    )
    for algorithm, group in baselines.groupby("algorithm", sort=False):
        ordered = group.sort_values("step")
        rows.append(
            {
                "algorithm": str(algorithm),
                "curve_role": "baseline",
                "final_utility": float(ordered["best_obj"].iloc[-1]),
                "best_utility": float(ordered["best_obj"].max()),
                "final_step": int(ordered["step"].iloc[-1]),
                "final_episode": np.nan,
                "evaluations": float(ordered["evaluations"].max()),
                "elapsed_s": float(ordered["elapsed_s"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_ppo_vs_baselines(metrics: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()

    ppo = metrics[metrics["algorithm"] == "DSCI"].sort_values("episode")
    if not ppo.empty:
        label = "DSCI" if str(ppo["curve_type"].iloc[0]) == "dsci_train" else "DSCI cached solution"
        ax.plot(
            ppo["episode"],
            ppo["utility"],
            label=label,
            linewidth=1.8,
            drawstyle="steps-post",
        )

    baselines = metrics[metrics["curve_type"] == "baseline_progress"]
    for algorithm, group in baselines.groupby("algorithm", sort=False):
        group = group.sort_values("episode")
        ax.plot(
            group["episode"],
            group["utility"],
            linestyle="--",
            linewidth=1.4,
            label=str(algorithm),
            drawstyle="steps-post",
        )

    ax.set_xlabel("Episode / Candidate Evaluation")
    ax.set_ylabel("Utility")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.2)
    target = output_dir / "ppo_vs_baselines"
    save_fig_for_ieee(target, fig=fig)
    plt.close(fig)
    return target


def plot_optimizer_progress(metrics: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    for algorithm, group in metrics.groupby("algorithm", sort=False):
        group = group.sort_values("step")
        ax.plot(group["step"], group["best_obj"], label=str(algorithm), linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Best Utility")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.2)
    target = output_dir / "optimizer_progress"
    save_fig_for_ieee(target, fig=fig)
    plt.close(fig)
    return target


def plot_time_progress(metrics: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()

    max_elapsed = float(metrics["elapsed_s"].max()) if "elapsed_s" in metrics else 0.0
    ppo = metrics[metrics["algorithm"] == "DSCI"].sort_values("elapsed_s")
    if not ppo.empty:
        label = "DSCI" if str(ppo["curve_type"].iloc[0]) == "dsci_train" else "DSCI cached solution"
        if ppo["elapsed_s"].nunique() <= 1 and max_elapsed > 0.0:
            value = float(ppo["utility"].iloc[-1])
            ax.plot([0.0, max_elapsed], [value, value], label=label, linewidth=1.8)
        else:
            ax.plot(
                ppo["elapsed_s"],
                ppo["utility"],
                label=label,
                linewidth=1.8,
                drawstyle="steps-post",
            )

    baselines = metrics[metrics["curve_type"] == "baseline_progress"]
    for algorithm, group in baselines.groupby("algorithm", sort=False):
        group = group.sort_values("elapsed_s")
        ax.plot(
            group["elapsed_s"],
            group["utility"],
            linestyle="--",
            linewidth=1.4,
            label=str(algorithm),
            drawstyle="steps-post",
        )

    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Best Utility")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.2)
    target = output_dir / "time_progress"
    save_fig_for_ieee(target, fig=fig)
    plt.close(fig)
    return target


def plot_final_utility_bar(summary: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    set_ieee_style(mode="single")
    fig, ax = plt.subplots()
    ordered = summary.sort_values("final_utility", ascending=False)
    ax.bar(ordered["algorithm"].astype(str), ordered["final_utility"].astype(float))
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Final Utility")
    fig.tight_layout(pad=0.2)
    target = output_dir / "final_utility_bar"
    save_fig_for_ieee(target, fig=fig)
    plt.close(fig)
    return target


def write_config(args: argparse.Namespace, output_dir: Path, paras: Paras) -> Path:
    payload = {
        "bundle_id": paras.bundle_id,
        "manifest_id": paras.manifest_id,
        "num_users": int(paras.n),
        "num_layers": int(paras.m),
        "exit_boundaries": [int(value) for value in paras.E],
        "ppo_dir": str(args.ppo_dir),
        "ppo_source": str(args.ppo_source),
        "algorithms": list(_parse_algorithms(args.algorithms)),
        "solution_npz": str(args.solution_npz),
        "solution_meta": str(args.solution_meta),
        "random_iterations": int(args.random_iterations),
        "greedy_passes": int(args.greedy_passes),
        "bf_max_iter": int(args.bf_max_iter),
        "threshold_step": float(args.threshold_step),
        "seed": int(args.seed),
        "ga_population_size": int(args.ga_population_size),
        "ga_generations": int(args.ga_generations),
        "ga_mutation_rate": float(args.ga_mutation_rate),
        "bf_scope": "coordinate exhaustive baseline from optimize_BF",
    }
    target = output_dir / "config.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def write_solution_npz(
    output_dir: Path,
    name: str,
    solution: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Path:
    solution_dir = output_dir / "solutions"
    solution_dir.mkdir(parents=True, exist_ok=True)
    X, Y, F_e, F_c = solution
    target = solution_dir / f"{name}_solution.npz"
    np.savez_compressed(target, X=X, Y=Y, F_e=F_e, F_c=F_c)
    return target


def run_algorithm_contrast(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = _make_output_dir(args.output_dir)
    bundle = load_solution_bundle(args.solution_npz, args.solution_meta)
    paras = bundle.paras

    ppo = load_or_build_ppo_curve(bundle, args)
    baselines, baseline_solutions = run_baselines(
        paras,
        algorithms=_parse_algorithms(args.algorithms),
        random_iterations=args.random_iterations,
        greedy_passes=args.greedy_passes,
        bf_max_iter=args.bf_max_iter,
        threshold_step=args.threshold_step,
        seed=args.seed,
        ga_population_size=args.ga_population_size,
        ga_generations=args.ga_generations,
        ga_mutation_rate=args.ga_mutation_rate,
        verbose_optimizers=args.verbose_optimizers,
    )
    comparison = build_ppo_vs_baselines(ppo, baselines)
    summary = build_summary(ppo, baselines)

    paths: dict[str, Path] = {}
    paths["config"] = write_config(args, output_dir, paras)
    paths["ppo_metrics"] = output_dir / "ppo_metrics.csv"
    ppo.to_csv(paths["ppo_metrics"], index=False, encoding="utf-8-sig")
    paths["baseline_metrics"] = output_dir / "baseline_metrics.csv"
    baselines.to_csv(paths["baseline_metrics"], index=False, encoding="utf-8-sig")
    paths["baseline_metrics_jsonl"] = write_metrics_jsonl(
        baselines.to_dict("records"),
        output_dir / "baseline_metrics.jsonl",
    )
    paths["ppo_vs_baselines"] = output_dir / "ppo_vs_baselines.csv"
    comparison.to_csv(paths["ppo_vs_baselines"], index=False, encoding="utf-8-sig")
    paths["summary"] = output_dir / "final_summary.csv"
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    if ppo.attrs.get("solution") is not None:
        paths["dsci_solution_npz"] = write_solution_npz(
            output_dir,
            "DSCI",
            ppo.attrs["solution"],
        )
    for algorithm, solution in baseline_solutions.items():
        paths[f"{algorithm.lower()}_solution_npz"] = write_solution_npz(
            output_dir,
            algorithm,
            solution,
        )

    paths["ppo_vs_baselines_fig"] = plot_ppo_vs_baselines(comparison, output_dir)
    paths["optimizer_progress_fig"] = plot_optimizer_progress(baselines, output_dir)
    paths["time_progress_fig"] = plot_time_progress(comparison, output_dir)
    paths["final_utility_bar_fig"] = plot_final_utility_bar(summary, output_dir)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-dir", default=str(DEFAULT_PPO_DIR))
    parser.add_argument(
        "--ppo-source",
        choices=("auto", "metrics", "train", "solution"),
        default="train",
        help=(
            "Default trains DSCI on the loaded Paras. auto validates cached "
            "metrics, then tries training, then falls back to cached solution."
        ),
    )
    parser.add_argument("--solution-npz", default=str(DEFAULT_SOLUTION_NPZ))
    parser.add_argument("--solution-meta", default=str(DEFAULT_SOLUTION_META))
    parser.add_argument("--output-dir", default=str(EXP4_RESULT_DIR))
    parser.add_argument("--random-iterations", type=int, default=200)
    parser.add_argument("--greedy-passes", type=int, default=2)
    parser.add_argument("--bf-max-iter", type=int, default=3)
    parser.add_argument(
        "--algorithms",
        default="greedy,ga,bf",
        help="Comma-separated baselines. Default: greedy,ga,bf. Add random if needed.",
    )
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ga-population-size", type=int, default=50)
    parser.add_argument("--ga-generations", type=int, default=150)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.1)
    parser.add_argument("--verbose-optimizers", action="store_true")
    parser.add_argument("--ppo-max-epochs", type=int, default=80)
    parser.add_argument("--ppo-target-steps", type=int, default=600)
    parser.add_argument("--ppo-k-epochs", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = run_algorithm_contrast(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
