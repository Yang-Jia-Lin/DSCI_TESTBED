"""Rebuild the four compact Evaluation PDFs and their data manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "Scripts" / "Results"
MANIFEST_DIR = RESULTS_ROOT / "EvaluationFigures"

FIGURES = [
    {
        "name": "generalization",
        "script": REPO_ROOT / "Scripts" / "Exp1_SEAM" / "plot_generalization.py",
        "output": RESULTS_ROOT / "Exp1_SEAM" / "generalization.pdf",
        "status": "complete_from_draft",
        "source": "User-provided Evaluation handoff draft table",
        "selection": "Pi 5; ResNet-50/ViT-Base; CIFAR-10/ImageNet-100/NEU-CLS-64",
        "aggregation": "Values copied verbatim; no repository re-aggregation requested",
        "notes": [
            "Uses the generalization-table I-SplitEE R-C10 values: 94.53%, 321.28 ms.",
            "The conflicting baseline/scalability values remain unresolved.",
        ],
    },
    {
        "name": "scalability",
        "script": REPO_ROOT / "Scripts" / "Exp3_Scalable" / "plot_scalability.py",
        "output": RESULTS_ROOT / "Exp3_Scalable" / "scalability.pdf",
        "status": "complete_from_draft_with_flagged_value",
        "source": "User-provided Evaluation handoff draft table",
        "selection": "ResNet-50/CIFAR-10; N=1..4",
        "aggregation": "Draft means/P95/worst-device values; error bars are reported SD",
        "notes": [
            "I-SplitEE N=2 accuracy is plotted as the draft value 97.55%.",
            "Its underlying invalid 101.08% run was not re-aggregated in this draft-only pass.",
            "I-SplitEE N=4 is missing and is not interpolated or connected.",
        ],
    },
    {
        "name": "algorithm_analysis_overhead",
        "script": REPO_ROOT
        / "Scripts"
        / "Exp5_System_Overhead"
        / "plot_algorithm_analysis_overhead_paper.py",
        "output": RESULTS_ROOT
        / "Exp5_System_Overhead"
        / "algorithm_analysis_overhead.pdf",
        "status": "incomplete",
        "source": "No numeric experiment data in the paper draft",
        "selection": "Planned PPO/Random Search/GA and N=1..4 cold/hot overhead",
        "aggregation": "None",
        "notes": ["PDF contains explicit Data pending panels; no values were fabricated."],
    },
    {
        "name": "ablation",
        "script": REPO_ROOT / "Scripts" / "Exp4_Ablation" / "plot_ablation.py",
        "output": RESULTS_ROOT / "Exp4_Ablation" / "ablation.pdf",
        "status": "incomplete",
        "source": "User-provided Evaluation handoff draft table",
        "selection": "ResNet-50/ImageNet-100; single Pi 5",
        "aggregation": "Values copied verbatim; no repository re-aggregation requested",
        "notes": [
            "Only SEAM is available (240.82 ms, 93.22%).",
            "End only, Split only, and EE only are explicitly marked pending.",
        ],
    },
]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _check_dependencies() -> None:
    missing = [
        name for name in ("matplotlib", "numpy") if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(f"Missing plotting dependencies: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return status 2 after writing outputs if any figure is incomplete.",
    )
    args = parser.parse_args()
    _check_dependencies()

    for figure in FIGURES:
        subprocess.run(
            [
                sys.executable,
                str(figure["script"]),
                "--output",
                str(figure["output"]),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / "figure_data_manifest.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "data_policy": (
            "Draft-only pass requested by user; repository experiment records were not "
            "used to overwrite the supplied tables."
        ),
        "figures": [
            {
                **figure,
                "script": str(figure["script"].relative_to(REPO_ROOT)),
                "output": str(figure["output"].relative_to(REPO_ROOT)),
            }
            for figure in FIGURES
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)

    incomplete = [item["name"] for item in FIGURES if item["status"] == "incomplete"]
    if incomplete:
        print("Incomplete figures: " + ", ".join(incomplete), file=sys.stderr)
        if args.strict:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
