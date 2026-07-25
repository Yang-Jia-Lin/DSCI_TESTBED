"""Genetic-algorithm optimizer for Phase 2 scheduling."""

from Src.Phase2_Scheduler.Optimizer.GA.alg_GA import optimize_GA


def run_ga_experiment(*args, **kwargs):
    """Lazily import the runnable wrapper to keep ``python -m ...run_GA`` clean."""
    from Src.Phase2_Scheduler.Optimizer.GA.run_GA import (
        run_ga_experiment as _run_ga_experiment,
    )

    return _run_ga_experiment(*args, **kwargs)


__all__ = ["optimize_GA", "run_ga_experiment"]
