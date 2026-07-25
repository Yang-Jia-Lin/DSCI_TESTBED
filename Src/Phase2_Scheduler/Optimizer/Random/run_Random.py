"""Runnable wrapper for the Phase 2 random-search optimizer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from Src.Phase2_Scheduler.Optimizer.Random.alg_random import optimize_random
from Src.Phase2_Scheduler.Utils.log_function import save_experiment_results
from Src.Phase2_Scheduler.paras import Paras
from Src.Shared.Config.paths import RESULT_RANDOM_PATH

_DEFAULT_RANDOM_PARAMS: dict[str, Any] = {
    "iterations": 200,
    "seed": 42,
    "threshold_step": 0.05,
    "allocate_resources_enabled": True,
    "verbose": False,
}


def _build_random_params(
    custom_random_hyperparams: dict | None,
) -> dict[str, Any]:
    params = dict(_DEFAULT_RANDOM_PARAMS)
    if custom_random_hyperparams:
        unknown = sorted(set(custom_random_hyperparams) - set(params))
        if unknown:
            raise ValueError(f"Unknown Random hyperparameters: {unknown}")
        params.update(custom_random_hyperparams)
    return params


def run_random_experiment(
    paras: Paras | None = None,
    *,
    state: dict | None = None,
    custom_random_hyperparams: dict | None = None,
    initial_solution=None,
    save_log: bool = True,
):
    """Run random search on an existing ``Paras`` or scheduler state.

    Returns ``best_value, best_solution, history, paras``, matching the reusable
    experiment interface exposed by DSCI and GA.
    """

    if paras is None:
        if state is None:
            raise ValueError("Provide either paras or state")
        paras = Paras.from_state(state)
    elif state is not None:
        raise ValueError("Provide paras or state, not both")

    random_params = _build_random_params(custom_random_hyperparams)
    best_val, best_sol, history, metrics = optimize_random(
        paras,
        **random_params,
        initial_solution=initial_solution,
    )
    if save_log:
        save_experiment_results(
            save_dir=Path(RESULT_RANDOM_PATH),
            algo_name="Random",
            paras=paras,
            best_val=best_val,
            best_sol=best_sol,
            history=history,
            hyper_params=random_params,
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
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Threshold grid step; use 0 for continuous random thresholds.",
    )
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--no-save-log", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    from Scripts.EvaluationCommon.solutions import load_solution_bundle

    bundle = load_solution_bundle(args.solution_npz, args.solution_meta)
    threshold_step = None if args.threshold_step == 0.0 else args.threshold_step
    params = {
        "iterations": args.iterations,
        "seed": args.seed,
        "threshold_step": threshold_step,
        "allocate_resources_enabled": True,
        "verbose": args.verbose,
    }
    best_val, _best_sol, history, _paras = run_random_experiment(
        bundle.paras,
        custom_random_hyperparams=params,
        initial_solution=(
            (bundle.X, bundle.Y, bundle.F_e, bundle.F_c)
            if args.warm_start
            else None
        ),
        save_log=not args.no_save_log,
    )
    print(
        f"Random search complete: objective={best_val:.8f}, "
        f"evaluations={len(history)}"
    )


if __name__ == "__main__":
    main()
