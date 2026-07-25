"""Random-search optimizer for Phase 2 scheduling."""

from Src.Phase2_Scheduler.Optimizer.Random.alg_random import optimize_random


def run_random_experiment(*args, **kwargs):
    """Lazily import the runnable wrapper for clean ``python -m`` execution."""
    from Src.Phase2_Scheduler.Optimizer.Random.run_Random import (
        run_random_experiment as _run_random_experiment,
    )

    return _run_random_experiment(*args, **kwargs)


__all__ = ["optimize_random", "run_random_experiment"]
