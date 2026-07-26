"""Create the explicitly incomplete convergence/overhead paper figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Scripts.EvaluationCommon.paper_figure_style import (  # noqa: E402
    apply_compact_ieee_style,
    pending_panel,
    save_pdf,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RESULT_FIGURE_DIR = EXPERIMENT_ROOT / "result_figure"

def plot(output: Path) -> Path:
    apply_compact_ieee_style()
    fig, (ax_convergence, ax_overhead) = plt.subplots(
        1,
        2,
        figsize=(3.55, 1.82),
        gridspec_kw={"wspace": 0.14},
    )
    pending_panel(
        ax_convergence,
        detail="PPO / Random Search / GA",
    )
    pending_panel(
        ax_overhead,
        detail="Cold / hot timings for N=1-4",
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.96, bottom=0.06)
    return save_pdf(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_FIGURE_DIR / "algorithm_analysis_overhead.pdf",
    )
    args = parser.parse_args()
    print(plot(args.output))


if __name__ == "__main__":
    main()
