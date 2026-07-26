"""Experiment-facing wrappers for scheduler convergence plots."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Src.Phase2_Scheduler.Reporting.plot_convergence import (
    plot_convergence,
    plot_entropy,
    plot_lan_and_acc,
)
from Src.Phase2_Scheduler.Utils.log_function import load_and_analyze_results
from Scripts.EvaluationCommon.config import (
    EXP4_RESULT_DATA_DIR,
    EXP4_RESULT_FIGURE_DIR,
)

__all__ = ["plot_convergence", "plot_entropy", "plot_lan_and_acc"]


if __name__ == "__main__":
    target = EXP4_RESULT_DATA_DIR / "legacy" / "PPO_20260129_202802"
    _, _, _, _, history, _ = load_and_analyze_results(target, analysis=False)
    plot_convergence(history, save_dir=Path(EXP4_RESULT_FIGURE_DIR))
