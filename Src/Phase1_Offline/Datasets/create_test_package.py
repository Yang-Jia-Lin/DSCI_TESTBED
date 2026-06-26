"""Create small, class-balanced test packages for device-side evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Src.Shared.Config.model_config import ModelBundleSpec, get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_transform
from Src.Shared.Models.ModelNet.MultiExitResNet import build_model


MANIFEST_COLUMNS = (
    "sample_id",
    "source_index",
    "label",
    "difficulty",
    "confidence",
    "entropy",
    "correct",
    "relative_path",
)
EASY_MIN_CONFIDENCE = 0.9
HARD_MAX_CONFIDENCE = 0.6


def _split_dir_name(root: Path, split: str) -> str:
    if split == "train":
        return "train"
    if split == "test" and (root / "test").is_dir():
        return "test"
    return "val"


def _load_source_dataset(
    bundle: ModelBundleSpec,
    split: str,
    *,
    data_root: str | Path | None,
    download: bool,
    transform=None,
):
    from torchvision import datasets

    root = Path(data_root or bundle_paths(bundle.bundle_id).dataset_root)
    train = split == "train"
    if bundle.dataset_id == "cifar10":
        return datasets.CIFAR10(
            root=str(root),
            train=train,
            transform=transform,
            download=download,
        )
    if bundle.dataset_id == "imagenet100":
        if download:
            raise ValueError("ImageNet100 download is not supported")
        directory = root / _split_dir_name(root, split)
        dataset = datasets.ImageFolder(str(directory), transform=transform)
        if len(dataset.classes) != bundle.num_classes:
            raise ValueError(
                f"ImageNet100 requires exactly {bundle.num_classes} classes, "
                f"found {len(dataset.classes)} under {directory}"
            )
        return dataset
    raise ValueError(f"Unsupported dataset_id: {bundle.dataset_id}")


def _targets(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(value) for value in dataset.targets]
    return [int(label) for _, label in dataset.samples]


def _source_path(dataset, index: int) -> Path | None:
    if hasattr(dataset, "samples"):
        return Path(dataset.samples[index][0])
    return None


def _clean_row(row: dict) -> dict:
    return {
        str(key).strip(): str(value).strip()
        for key, value in row.items()
        if key is not None
    }


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _classify_difficulty(correct: bool, confidence: float) -> str:
    if correct and confidence >= EASY_MIN_CONFIDENCE:
        return "easy"
    if (not correct) or confidence <= HARD_MAX_CONFIDENCE:
        return "hard"
    return "medium"


def _find_existing_difficulty_csv(bundle: ModelBundleSpec, split: str) -> Path | None:
    analysis_root = bundle_paths(bundle.bundle_id).analysis_root
    candidates = sorted(analysis_root.glob("*difficulty_labeled.csv"))
    if not candidates:
        return None
    split_matches = [path for path in candidates if split in path.stem]
    return split_matches[0] if split_matches else candidates[0]


def _read_difficulty_csv(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row = _clean_row(raw_row)
            index_value = row.get("source_index", row.get("image_id"))
            if index_value is None or index_value == "":
                continue
            confidence = float(row.get("confidence", 0.0))
            correct = _truthy(row.get("correct", "false"))
            difficulty = row.get("difficulty") or _classify_difficulty(correct, confidence)
            rows[int(index_value)] = {
                "source_index": int(index_value),
                "label": int(row.get("true_label", row.get("label", -1))),
                "difficulty": difficulty,
                "confidence": confidence,
                "entropy": float(row.get("entropy", 0.0)),
                "correct": correct,
            }
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * fraction)), len(ordered) - 1)
    return float(ordered[index])


def _write_difficulty_analysis(
    bundle: ModelBundleSpec,
    split: str,
    rows: list[dict],
) -> Path:
    paths = bundle_paths(bundle.bundle_id)
    paths.analysis_root.mkdir(parents=True, exist_ok=True)
    raw_path = paths.analysis_root / f"{bundle.bundle_id}_{split}_difficulty_raw.csv"
    labeled_path = paths.analysis_root / f"{bundle.bundle_id}_{split}_difficulty_labeled.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_index", "true_label", "pred_label", "correct", "confidence", "entropy"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_index": row["source_index"],
                    "true_label": row["label"],
                    "pred_label": row["pred_label"],
                    "correct": row["correct"],
                    "confidence": row["confidence"],
                    "entropy": row["entropy"],
                }
            )
    with labeled_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_index",
                "true_label",
                "pred_label",
                "correct",
                "confidence",
                "entropy",
                "difficulty",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_index": row["source_index"],
                    "true_label": row["label"],
                    "pred_label": row["pred_label"],
                    "correct": row["correct"],
                    "confidence": row["confidence"],
                    "entropy": row["entropy"],
                    "difficulty": row["difficulty"],
                }
            )
    confidences = [float(row["confidence"]) for row in rows]
    counts = defaultdict(int)
    correct_count = 0
    for row in rows:
        counts[row["difficulty"]] += 1
        correct_count += int(bool(row["correct"]))
    stats_path = paths.analysis_root / f"{bundle.bundle_id}_{split}_confidence_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "total_samples": len(rows),
                "overall_accuracy": correct_count / max(len(rows), 1),
                "confidence_percentiles": {
                    "p10": _percentile(confidences, 0.10),
                    "p25": _percentile(confidences, 0.25),
                    "p50": _percentile(confidences, 0.50),
                    "p75": _percentile(confidences, 0.75),
                    "p90": _percentile(confidences, 0.90),
                },
                "suggested_thresholds": {
                    "easy_min": EASY_MIN_CONFIDENCE,
                    "hard_max": HARD_MAX_CONFIDENCE,
                },
                "suggested_counts": dict(counts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return labeled_path


def _generate_difficulty_csv(
    bundle: ModelBundleSpec,
    split: str,
    *,
    data_root: str | Path | None,
    download: bool,
    batch_size: int,
) -> Path:
    paths = bundle_paths(bundle.bundle_id)
    if not paths.weight_path.is_file():
        raise FileNotFoundError(f"Bundle weights not found: {paths.weight_path}")
    dataset = _load_source_dataset(
        bundle,
        split,
        data_root=data_root,
        download=download,
        transform=build_transform(bundle, train=False),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(bundle).to(device)
    model.load_state_dict(torch.load(paths.weight_path, map_location=device, weights_only=True))
    model.eval()
    rows = []
    source_index = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            probabilities = torch.softmax(model(images), dim=1)
            confidences, predictions = probabilities.max(dim=1)
            entropies = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            for item_index in range(images.shape[0]):
                confidence = float(confidences[item_index].item())
                correct = bool(predictions[item_index].item() == labels[item_index].item())
                rows.append(
                    {
                        "source_index": source_index,
                        "label": int(labels[item_index].item()),
                        "pred_label": int(predictions[item_index].item()),
                        "correct": correct,
                        "confidence": confidence,
                        "entropy": float(entropies[item_index].item()),
                        "difficulty": _classify_difficulty(correct, confidence),
                    }
                )
                source_index += 1
    return _write_difficulty_analysis(bundle, split, rows)


def _difficulty_rows(
    bundle: ModelBundleSpec,
    split: str,
    *,
    data_root: str | Path | None,
    download: bool,
    batch_size: int,
    refresh: bool,
) -> dict[int, dict]:
    existing = None if refresh else _find_existing_difficulty_csv(bundle, split)
    path = existing or _generate_difficulty_csv(
        bundle,
        split,
        data_root=data_root,
        download=download,
        batch_size=batch_size,
    )
    print(f"Using difficulty labels: {path}")
    return _read_difficulty_csv(path)


def _package_name(bundle_id: str, split: str, mode: str, samples_per_class: int, seed: int) -> str:
    return f"{bundle_id}__{split}__{mode}__{samples_per_class}pc__seed{seed}"


def _mode_candidates(targets: list[int], difficulty: dict[int, dict], mode: str):
    candidates: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        row = difficulty.get(index)
        sample_difficulty = row["difficulty"] if row else "unknown"
        if mode == "balanced" or sample_difficulty == mode:
            candidates[int(label)].append(index)
    return candidates


def _select_balanced_indices(
    targets: list[int],
    difficulty: dict[int, dict],
    *,
    mode: str,
    num_classes: int,
    samples_per_class: int,
    seed: int,
    allow_fewer_per_class: bool,
) -> tuple[list[int], int]:
    candidates = _mode_candidates(targets, difficulty, mode)
    available = {label: len(candidates.get(label, [])) for label in range(num_classes)}
    actual_samples_per_class = samples_per_class
    if allow_fewer_per_class:
        actual_samples_per_class = min(samples_per_class, min(available.values()))
        if actual_samples_per_class <= 0:
            raise ValueError(f"No class-balanced {mode} package can be created; availability: {available}")
    else:
        insufficient = {
            label: count
            for label, count in available.items()
            if count < samples_per_class
        }
        if insufficient:
            details = ", ".join(
                f"class {label}: found {count}" for label, count in insufficient.items()
            )
            raise ValueError(
                f"Not enough {mode} samples for {samples_per_class} per class; {details}. "
                "Use a smaller --samples-per-class or pass --allow-fewer-per-class."
            )
    selected = []
    rng = random.Random(seed)
    for label in range(num_classes):
        class_candidates = list(candidates.get(label, []))
        rng.shuffle(class_candidates)
        selected.extend(sorted(class_candidates[:actual_samples_per_class]))
    return selected, actual_samples_per_class


def _default_difficulty(index: int, label: int) -> dict:
    return {
        "source_index": index,
        "label": label,
        "difficulty": "unknown",
        "confidence": "",
        "entropy": "",
        "correct": "",
    }


def _export_image(dataset, source_index: int, label: int, sample_id: str, output_dir: Path) -> str:
    class_dir = output_dir / "images" / str(label)
    class_dir.mkdir(parents=True, exist_ok=True)
    source_path = _source_path(dataset, source_index)
    if source_path is not None and source_path.is_file():
        suffix = source_path.suffix or ".jpg"
        destination = class_dir / f"{sample_id}{suffix.lower()}"
        shutil.copy2(source_path, destination)
    else:
        image, _ = dataset[source_index]
        destination = class_dir / f"{sample_id}.png"
        image.save(destination)
    return destination.relative_to(output_dir).as_posix()


def _write_package(
    bundle: ModelBundleSpec,
    split: str,
    mode: str,
    samples_per_class: int,
    seed: int,
    output_root: Path,
    dataset,
    targets: list[int],
    selected_indices: list[int],
    difficulty: dict[int, dict],
    *,
    overwrite: bool,
) -> Path:
    package_root = output_root / _package_name(bundle.bundle_id, split, mode, samples_per_class, seed)
    if package_root.exists():
        if not overwrite:
            raise FileExistsError(f"Test package already exists: {package_root}")
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True)
    manifest_rows = []
    per_class_counts = defaultdict(int)
    for source_index in selected_indices:
        label = int(targets[source_index])
        rank = per_class_counts[label]
        sample_id = f"{mode}_{label:03d}_{rank:04d}_{source_index:08d}"
        relative_path = _export_image(dataset, source_index, label, sample_id, package_root)
        row = difficulty.get(source_index, _default_difficulty(source_index, label))
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source_index": source_index,
                "label": label,
                "difficulty": row["difficulty"],
                "confidence": row["confidence"],
                "entropy": row["entropy"],
                "correct": row["correct"],
                "relative_path": relative_path,
            }
        )
        per_class_counts[label] += 1
    manifest_path = package_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "bundle_id": bundle.bundle_id,
        "dataset_id": bundle.dataset_id,
        "split": split,
        "mode": mode,
        "num_classes": bundle.num_classes,
        "samples_per_class": samples_per_class,
        "total_samples": len(manifest_rows),
        "seed": seed,
        "difficulty_thresholds": {
            "easy_min_confidence": EASY_MIN_CONFIDENCE,
            "hard_max_confidence": HARD_MAX_CONFIDENCE,
        },
        "class_counts": {str(label): per_class_counts[label] for label in range(bundle.num_classes)},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (package_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return package_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("balanced", "easy", "hard"),
        default=("balanced", "easy", "hard"),
    )
    parser.add_argument("--samples-per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--refresh-difficulty", action="store_true")
    parser.add_argument(
        "--allow-fewer-per-class",
        action="store_true",
        help=(
            "Keep packages class-balanced by reducing a mode to the largest "
            "per-class count available when the requested count is impossible."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
    bundle = get_bundle(args.bundle_id)
    difficulty = _difficulty_rows(
        bundle,
        args.split,
        data_root=args.data_root,
        download=args.download,
        batch_size=args.batch_size,
        refresh=args.refresh_difficulty,
    )
    dataset = _load_source_dataset(
        bundle,
        args.split,
        data_root=args.data_root,
        download=args.download,
        transform=None,
    )
    targets = _targets(dataset)
    output_root = Path(args.output_root) if args.output_root else bundle_paths(bundle.bundle_id).test_package_root
    output_root.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        selected, actual_samples_per_class = _select_balanced_indices(
            targets,
            difficulty,
            mode=mode,
            num_classes=bundle.num_classes,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            allow_fewer_per_class=args.allow_fewer_per_class,
        )
        if actual_samples_per_class < args.samples_per_class:
            print(
                f"Using {actual_samples_per_class} samples per class for {mode}; "
                f"requested {args.samples_per_class}."
            )
        package_root = _write_package(
            bundle,
            args.split,
            mode,
            actual_samples_per_class,
            args.seed,
            output_root,
            dataset,
            targets,
            selected,
            difficulty,
            overwrite=args.overwrite,
        )
        print(f"Created test package: {package_root}")


if __name__ == "__main__":
    main()
