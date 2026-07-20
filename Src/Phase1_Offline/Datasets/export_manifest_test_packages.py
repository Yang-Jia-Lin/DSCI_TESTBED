"""Export class-balanced terminal test packages from prepared split manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import DATASET_DIR

MANIFEST_COLUMNS = (
    "sample_id", "source_index", "label", "difficulty", "confidence",
    "entropy", "correct", "relative_path",
)
SPECS = {
    "cifar10": ("resnet50-cifar10", "CIFAR10", 10),
    "imagenet100": ("resnet50-imagenet100", "ImageNet100", 100),
    "imagenet1000": (None, "ImageNet1000", 1000),
    "neucls64": ("resnet50-neucls64", "NEU-CLS-64", 6),
}


def package_name(dataset_id: str, samples_per_class: int, seed: int) -> str:
    return f"{dataset_id}__test__balanced__{samples_per_class}pc__seed{seed}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"relative_path", "label", "source_index", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid test split manifest: {path}")
    if any(row["split"] != "test" for row in rows):
        raise ValueError(f"Non-test samples found in {path}")
    return rows


def select_balanced(rows: list[dict[str, str]], num_classes: int, count: int, seed: int):
    grouped = defaultdict(list)
    for position, row in enumerate(rows):
        grouped[int(row["label"])].append((position, row))
    missing = {label: len(grouped[label]) for label in range(num_classes) if len(grouped[label]) < count}
    if missing:
        raise ValueError(f"Not enough test samples per class: {missing}")
    rng = random.Random(seed)
    selected = []
    for label in range(num_classes):
        candidates = list(grouped[label])
        rng.shuffle(candidates)
        selected.extend(sorted(candidates[:count], key=lambda item: item[0]))
    return [row for _, row in selected]


def source_image(dataset_id: str, data_root: Path, row: dict[str, str], cifar=None):
    if dataset_id == "cifar10":
        image, label = cifar[int(row["source_index"])]
        if int(label) != int(row["label"]):
            raise ValueError(f"CIFAR label mismatch at {row['source_index']}")
        return image, None
    base = data_root / "source" if dataset_id in ("imagenet100", "imagenet1000") else data_root
    path = base / row["relative_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return None, path


def export_one(dataset_id: str, *, samples_per_class: int, seed: int, overwrite: bool, copy_workers: int = 16) -> Path:
    bundle_id, directory, num_classes = SPECS[dataset_id]
    data_root = DATASET_DIR / directory
    split_manifest = data_root / "metadata" / "test_manifest.csv"
    selected = select_balanced(read_rows(split_manifest), num_classes, samples_per_class, seed)
    output = data_root / "TestSets" / package_name(dataset_id, samples_per_class, seed)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Test package already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    cifar = None
    if dataset_id == "cifar10":
        from torchvision.datasets import CIFAR10
        cifar = CIFAR10(str(data_root), train=False, download=False)
    package_rows = []
    copy_jobs = []
    class_counts = defaultdict(int)
    for row in selected:
        label = int(row["label"])
        rank = class_counts[label]
        source_index = int(row["source_index"])
        sample_id = f"balanced_{label:03d}_{rank:04d}_{source_index:08d}"
        image, path = source_image(dataset_id, data_root, row, cifar)
        class_dir = output / "images" / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        if path is None:
            destination = class_dir / f"{sample_id}.png"
            image.save(destination)
        else:
            suffix = path.suffix.lower() or ".jpg"
            destination = class_dir / f"{sample_id}{suffix}"
            copy_jobs.append((path, destination))
        package_rows.append({
            "sample_id": sample_id,
            "source_index": source_index,
            "label": label,
            "difficulty": "unknown",
            "confidence": "",
            "entropy": "",
            "correct": "",
            "relative_path": destination.relative_to(output).as_posix(),
        })
        class_counts[label] += 1
    if copy_jobs:
        with ThreadPoolExecutor(max_workers=copy_workers) as executor:
            list(executor.map(lambda pair: shutil.copy2(*pair), copy_jobs))
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(package_rows)
    metadata = {
        "bundle_id": None,
        "validation_bundle_id": bundle_id,
        "testset_id": dataset_id,
        "dataset_id": dataset_id,
        "split": "test",
        "mode": "balanced",
        "num_classes": num_classes,
        "samples_per_class": samples_per_class,
        "total_samples": len(package_rows),
        "seed": seed,
        "difficulty": "unknown; package selection is model-independent",
        "source_manifest": str(split_manifest.relative_to(DATASET_DIR.parent.parent)),
        "source_manifest_sha256": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
        "class_counts": {str(label): class_counts[label] for label in range(num_classes)},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def validate_package(path: Path, bundle_id: str | None, expected: int):
    if bundle_id is not None:
        from Src.Shared.Data.registry import build_test_package_dataset
        dataset = build_test_package_dataset(get_bundle(bundle_id), path)
        if len(dataset) != expected:
            raise ValueError(f"Package {path} has {len(dataset)} samples, expected {expected}")
        dataset[0]
        dataset[len(dataset) - 1]
        return
    rows = list(csv.DictReader((path / "manifest.csv").open("r", encoding="utf-8", newline="")))
    if len(rows) != expected:
        raise ValueError(f"Package {path} has {len(rows)} samples, expected {expected}")
    from PIL import Image
    for row in (rows[0], rows[-1]):
        image_path = path / row["relative_path"]
        with Image.open(image_path) as image:
            image.verify()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(SPECS), default=tuple(SPECS))
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
    if args.copy_workers <= 0:
        raise ValueError("--copy-workers must be positive")
    outputs = []
    for dataset_id in args.datasets:
        output = export_one(dataset_id, samples_per_class=args.samples_per_class, seed=args.seed, overwrite=args.overwrite, copy_workers=args.copy_workers)
        bundle_id, _, classes = SPECS[dataset_id]
        validate_package(output, bundle_id, classes * args.samples_per_class)
        outputs.append({"dataset_id": dataset_id, "samples": classes * args.samples_per_class, "path": str(output)})
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
