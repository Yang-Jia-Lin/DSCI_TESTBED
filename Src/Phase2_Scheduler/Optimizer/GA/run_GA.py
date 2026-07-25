"""Runnable wrapper for the Phase 2 genetic-algorithm optimizer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from Src.Phase2_Scheduler.Optimizer.GA.alg_GA import optimize_GA
from Src.Phase2_Scheduler.Utils.log_function import save_experiment_results
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Config.paths import RESULT_GA_PATH

_DEFAULT_GA_PARAMS: dict[str, Any] = {
    "population_size": 50,
    "generations": 150,
    "mutation_rate": 0.1,
    "seed": 42,
    "crossover_rate": 0.9,
    "elite_size": 2,
    "tournament_size": 3,
    "threshold_mutation_std": 0.1,
    "patience": None,
    "rel_tolerance": 1e-6,
}


def _build_ga_params(custom_ga_hyperparams: dict | None) -> dict[str, Any]:
    params = dict(_DEFAULT_GA_PARAMS)
    if custom_ga_hyperparams:
        unknown = sorted(set(custom_ga_hyperparams) - set(params))
        if unknown:
            raise ValueError(f"Unknown GA hyperparameters: {unknown}")
        params.update(custom_ga_hyperparams)
    return params


def run_ga_experiment(
    paras: Paras | None = None,
    *,
    state: dict | None = None,
    custom_ga_hyperparams: dict | None = None,
    initial_solution=None,
    save_log: bool = True,
):
    """Run GA on an existing ``Paras`` or a deploy-facing scheduler state.

    This mirrors the reusable DSCI experiment shape and returns
    ``best_value, best_solution, history, paras``. Passing ``state`` uses the
    same ``Paras.from_state`` adapter as the online scheduler.
    """

    if paras is None:
        if state is None:
            raise ValueError("Provide either paras or state")
        paras = Paras.from_state(state)
    elif state is not None:
        raise ValueError("Provide paras or state, not both")

    ga_params = _build_ga_params(custom_ga_hyperparams)
    best_val, best_sol, history, metrics = optimize_GA(
        paras,
        **ga_params,
        initial_solution=initial_solution,
        return_metrics=True,
    )
    if save_log:
        save_experiment_results(
            save_dir=Path(RESULT_GA_PATH),
            algo_name="GA",
            paras=paras,
            best_val=best_val,
            best_sol=best_sol,
            history=history,
            hyper_params=ga_params,
            extra_logs=metrics,
        )
    return best_val, best_sol, history, paras


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution-npz",
        default="Data/Runtime/SolutionCache/latest_solution.npz",
        help="Existing solution used to reconstruct the current scheduler Paras.",
    )
    parser.add_argument(
        "--solution-meta",
        default="Data/Runtime/SolutionCache/latest_solution_meta.json",
    )
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--threshold-mutation-std", type=float, default=0.1)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--rel-tolerance", type=float, default=1e-6)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--no-save-log", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    from Scripts.EvaluationCommon.solutions import load_solution_bundle

    bundle = load_solution_bundle(args.solution_npz, args.solution_meta)
    params = {
        "population_size": args.population_size,
        "generations": args.generations,
        "mutation_rate": args.mutation_rate,
        "seed": args.seed,
        "crossover_rate": args.crossover_rate,
        "elite_size": args.elite_size,
        "tournament_size": args.tournament_size,
        "threshold_mutation_std": args.threshold_mutation_std,
        "patience": args.patience,
        "rel_tolerance": args.rel_tolerance,
    }
    best_val, _best_sol, history, _paras = run_ga_experiment(
        bundle.paras,
        custom_ga_hyperparams=params,
        initial_solution=(
            (bundle.X, bundle.Y, bundle.F_e, bundle.F_c)
            if args.warm_start
            else None
        ),
        save_log=not args.no_save_log,
    )
    print(
        f"GA complete: objective={best_val:.8f}, "
        f"generations={max(0, len(history) - 1)}"
    )


if __name__ == "__main__":
    main()
