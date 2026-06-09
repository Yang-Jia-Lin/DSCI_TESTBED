from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Src.Models.model_config import RESNET50 as MODEL_CFG


NAMING_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+_.+\.(csv|json)$")
REQUIRED_ARTIFACTS = ("rates", "accs", "layer_stats", "difficulty_labeled")
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit canonical offline tables under Data/OfflineTables."
    )
    parser.add_argument(
        "--offline-dir",
        default=str(MODEL_CFG.profile_dir),
        help="Canonical offline table directory.",
    )
    return parser.parse_args()


def audit_naming(offline_dir: Path) -> list[str]:
    errors = []
    for path in sorted(offline_dir.iterdir()):
        if not path.is_file():
            continue
        if not NAMING_RE.match(path.name):
            errors.append(
                f"Bad filename {path.name!r}; expected model_dataset_artifact.csv|json."
            )
    return errors


def audit_required_files(offline_dir: Path) -> list[str]:
    errors = []
    for artifact in REQUIRED_ARTIFACTS:
        suffix = "csv"
        path = offline_dir / f"{MODEL_CFG.artifact_prefix}_{artifact}.{suffix}"
        if not path.exists():
            errors.append(f"Missing active-model offline table: {path}")
    return errors


def audit_difficulty_table(offline_dir: Path) -> list[str]:
    errors = []
    counts = {label: 0 for label in sorted(VALID_DIFFICULTIES)}
    path = offline_dir / f"{MODEL_CFG.artifact_prefix}_difficulty_labeled.csv"
    if not path.exists():
        return [f"Missing difficulty table: {path}"]

    required = {"image_id", "true_label", "confidence", "difficulty"}
    seen: set[int] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            return [f"{path} is missing required columns: {sorted(missing)}"]
        for row in reader:
            image_id = int(row["image_id"])
            if image_id in seen:
                errors.append(f"Duplicate image_id in {path}: {image_id}")
            seen.add(image_id)
            if image_id < 0 or image_id >= 10000:
                errors.append(f"image_id outside CIFAR-10 test range: {image_id}")
            difficulty = row["difficulty"]
            if difficulty not in VALID_DIFFICULTIES:
                errors.append(f"Invalid difficulty label {difficulty!r} at image_id={image_id}")
            else:
                counts[difficulty] += 1

    missing_labels = [label for label, count in counts.items() if count == 0]
    if missing_labels:
        errors.append(f"Difficulty table has no samples for: {missing_labels}")

    print("Difficulty counts:", counts)
    return errors


def main():
    args = parse_args()
    offline_dir = Path(args.offline_dir)
    if not offline_dir.exists():
        raise FileNotFoundError(f"Offline table directory does not exist: {offline_dir}")

    errors = []
    errors.extend(audit_naming(offline_dir))
    errors.extend(audit_required_files(offline_dir))
    errors.extend(audit_difficulty_table(offline_dir))

    if errors:
        print("Offline table audit failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(f"Offline table audit passed: {offline_dir}")


if __name__ == "__main__":
    main()
