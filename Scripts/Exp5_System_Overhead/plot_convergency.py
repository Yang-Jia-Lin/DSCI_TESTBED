"""Experiment-facing wrappers for scheduler convergence plots."""

from pathlib import Path

from Src.Phase2_Scheduler.Reporting.plot_convergence import (
    plot_convergence,
    plot_entropy,
    plot_lan_and_acc,
)
from Src.Phase2_Scheduler.Utils.log_function import load_and_analyze_results
from Src.Shared.Config.paths import RESULT_DIR, RESULT_TEST_PATH

__all__ = ["plot_convergence", "plot_entropy", "plot_lan_and_acc"]


if __name__ == "__main__":
    target = RESULT_DIR / "Optimize" / "PPO" / "PPO_20260129_202802"
    _, _, _, _, history, _ = load_and_analyze_results(target, analysis=False)
    plot_convergence(history, save_dir=Path(RESULT_TEST_PATH))
