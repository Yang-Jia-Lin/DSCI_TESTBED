"""Rebuild the compact Evaluation PDFs and their data manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "Scripts"
MANIFEST_DIR = Path(__file__).resolve().parent / "result_data"
EXP2_ROOT = SCRIPTS_ROOT / "Exp2_Cross-Arch-Dataset"
EXP3_ROOT = SCRIPTS_ROOT / "Exp3_Multi-Device"
EXP4_ROOT = SCRIPTS_ROOT / "Exp4_System-Overhead"
EXP5_ROOT = SCRIPTS_ROOT / "Exp5_Ablation"

FIGURES = [
    {
        "name": "cross_arch_dataset_latency",
        "script": EXP2_ROOT / "plot_generalization.py",
        "data": EXP2_ROOT / "result_data" / "cross_arch_dataset_plot_data.csv",
        "argument_output": EXP2_ROOT / "result_figure" / "cross_arch_dataset.pdf",
        "output": EXP2_ROOT
        / "result_figure"
        / "2_a_cross_arch_dataset_latency.pdf",
        "status": "complete_from_draft",
        "source": "Exp2_Cross-Arch-Dataset/result_data/cross_arch_dataset_plot_data.csv",
        "selection": "Pi 5; ResNet-50/ViT-Base; CIFAR-10/ImageNet-100/NEU-CLS-64",
        "aggregation": "Values copied verbatim; no repository re-aggregation requested",
        "notes": [
            "Latency view; uses the latest six-bundle generalization table.",
        ],
    },
    {
        "name": "cross_arch_dataset_accuracy",
        "script": EXP2_ROOT / "plot_generalization.py",
        "data": EXP2_ROOT / "result_data" / "cross_arch_dataset_plot_data.csv",
        "argument_output": EXP2_ROOT / "result_figure" / "cross_arch_dataset.pdf",
        "output": EXP2_ROOT
        / "result_figure"
        / "2_b_cross_arch_dataset_accuracy.pdf",
        "status": "complete_from_draft",
        "source": "Exp2_Cross-Arch-Dataset/result_data/cross_arch_dataset_plot_data.csv",
        "selection": "Pi 5; ResNet-50/ViT-Base; CIFAR-10/ImageNet-100/NEU-CLS-64",
        "aggregation": "Values copied verbatim; no repository re-aggregation requested",
        "notes": [
            "Accuracy view; uses the latest six-bundle generalization table.",
        ],
    },
    {
        "name": "multi_device",
        "script": EXP3_ROOT / "plot_multi_device.py",
        "data": EXP3_ROOT / "result_data" / "multi_device_plot_data.csv",
        "argument_output": EXP3_ROOT / "result_figure" / "3_multi_device.pdf",
        "output": EXP3_ROOT / "result_figure" / "3_multi_device.pdf",
        "status": "complete_from_current_plot_data",
        "source": "Exp3_Multi-Device/result_data/multi_device_plot_data.csv",
        "selection": "ResNet-50/CIFAR-10; N=1..4",
        "aggregation": "Paper-facing mean/P95/worst-device latency and accuracy values",
        "notes": [
            "The figure contains the four configured device-count points.",
        ],
    },
    {
        "name": "algorithm_analysis_overhead",
        "script": EXP4_ROOT
        / "plot"
        / "plot_algorithm_analysis_overhead_paper.py",
        "argument_output": EXP4_ROOT
        / "result_figure"
        / "algorithm_analysis_overhead.pdf",
        "output": EXP4_ROOT / "result_figure" / "algorithm_analysis_overhead.pdf",
        "status": "incomplete",
        "source": "No numeric experiment data in the paper draft",
        "selection": "Planned PPO/Random Search/GA and N=1..4 cold/hot overhead",
        "aggregation": "None",
        "notes": ["PDF contains explicit Data pending panels; no values were fabricated."],
    },
    {
        "name": "ablation",
        "script": EXP5_ROOT / "plot_ablation.py",
        "data": EXP5_ROOT / "result_data" / "ablation_plot_data.csv",
        "argument_output": EXP5_ROOT / "result_figure" / "5_ablation.pdf",
        "output": EXP5_ROOT / "result_figure" / "5_ablation.pdf",
        "status": "complete_from_measured_results",
        "source": "Exp5_Ablation/result_data/ablation_plot_data.csv",
        "selection": "ResNet-50/CIFAR-10; single Pi 5",
        "aggregation": "Paper-facing values from the retained real-device ablation summary",
        "notes": [
            "The figure contains Cloud only, End only, Split only, EE only, and Ours.",
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


def _manifest_figure(figure: dict) -> dict:
    payload = {
        **figure,
        "script": str(figure["script"].relative_to(REPO_ROOT)),
        "argument_output": str(
            figure["argument_output"].relative_to(REPO_ROOT)
        ),
        "output": str(figure["output"].relative_to(REPO_ROOT)),
    }
    if "data" in figure:
        payload["data"] = str(figure["data"].relative_to(REPO_ROOT))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return status 2 after writing outputs if any figure is incomplete.",
    )
    args = parser.parse_args()
    _check_dependencies()

    generated: set[tuple[Path, Path, Path | None]] = set()
    for figure in FIGURES:
        command = (
            figure["script"],
            figure["argument_output"],
            figure.get("data"),
        )
        if command in generated:
            continue
        arguments = [
            sys.executable,
            str(figure["script"]),
            "--output",
            str(figure["argument_output"]),
        ]
        if "data" in figure:
            arguments.extend(["--data", str(figure["data"])])
        subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            check=True,
        )
        generated.add(command)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / "figure_data_manifest.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "data_policy": (
            "Each figure records its own paper-facing source and aggregation status; "
            "missing Exp4 values remain explicitly incomplete."
        ),
        "figures": [_manifest_figure(figure) for figure in FIGURES],
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
