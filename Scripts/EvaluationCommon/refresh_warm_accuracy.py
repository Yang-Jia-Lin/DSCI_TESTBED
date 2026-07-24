"""Refresh accuracy metadata for warm DeviceResults without touching raw latency.

The script evaluates the early-exit policy stored in each warm
``user_*_summary.json`` against a deterministic, stratified subset of the
dataset-level ``*__test__balanced__full`` package.  It adds an
``accuracy_refresh`` object to the summary and leaves the original accuracy,
measurements JSONL, inference CSV, latency summary, and utility fields intact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import DEVICE_RESULTS_DIR, PROJECT_ROOT, bundle_paths
from Src.Shared.Data.registry import (
    build_test_package_dataset,
    stratified_sample_indices,
)
from Src.Shared.Models.factory import build_model
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Partitioning.pytorch_executor import PyTorchSegmentExecutor


def _stable_sample_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _sample_seeds(
    round_id: str,
    dataset_id: str,
    user_id: int,
) -> tuple[int, int]:
    base_seed = _stable_sample_seed("round", round_id)
    effective_seed = _stable_sample_seed(
        "device",
        base_seed,
        dataset_id,
        int(user_id),
    )
    return base_seed, effective_seed


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _full_test_package(bundle_id: str) -> Path:
    bundle = get_bundle(bundle_id)
    return (
        bundle_paths(bundle_id).test_package_root
        / f"{bundle.dataset_id}__test__balanced__full"
    )


class _ReadablePackageDataset(Dataset):
    """Read a test package, recovering empty CIFAR exports from source_index."""

    def __init__(self, bundle, package_root: Path):
        self.package = build_test_package_dataset(bundle, package_root)
        self.bundle = bundle
        self.package_root = package_root
        self.rows = self.package.rows
        self.invalid_package_indices = {
            index
            for index, row in enumerate(self.rows)
            if not (package_root / row["relative_path"]).is_file()
            or (package_root / row["relative_path"]).stat().st_size == 0
        }
        self.cifar_source = None
        if self.invalid_package_indices:
            if bundle.dataset_id != "cifar10":
                raise ValueError(
                    f"{package_root} contains "
                    f"{len(self.invalid_package_indices)} missing/empty images; "
                    "automatic source recovery is only supported for CIFAR-10"
                )
            from torchvision.datasets import CIFAR10

            self.cifar_source = CIFAR10(
                str(bundle_paths(bundle.bundle_id).dataset_root),
                train=False,
                download=False,
            )

    def __len__(self) -> int:
        return len(self.package)

    def __getitem__(self, index: int):
        if index not in self.invalid_package_indices:
            image, label, metadata = self.package[index]
            return image, label, {**metadata, "source_fallback": False}
        row = self.rows[index]
        source_index = int(row["source_index"])
        image, source_label = self.cifar_source[source_index]
        label = int(row["label"])
        if int(source_label) != label:
            raise ValueError(
                f"CIFAR source label mismatch at source_index={source_index}: "
                f"manifest={label}, source={int(source_label)}"
            )
        tensor = self.package.transform(image.convert("RGB"))
        metadata = {
            "sample_id": row.get("sample_id", ""),
            "source_index": row.get("source_index", ""),
            "difficulty": row.get("difficulty", ""),
            "source_fallback": True,
        }
        return tensor, label, metadata


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: dict) -> None:
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _discover_summaries(
    results_root: Path,
    *,
    overwrite_refresh: bool,
    limit: int | None,
) -> tuple[list[tuple[Path, dict]], int]:
    selected: list[tuple[Path, dict]] = []
    skipped_existing = 0
    for round_dir in sorted(results_root.rglob("warm-*")):
        if not round_dir.is_dir():
            continue
        relative_parts = round_dir.relative_to(results_root).parts
        if relative_parts and relative_parts[0] == "Archive":
            continue
        for summary_path in sorted(round_dir.glob("user_*_summary.json")):
            summary = _read_json(summary_path)
            if "accuracy_refresh" in summary and not overwrite_refresh:
                skipped_existing += 1
                continue
            selected.append((summary_path, summary))
            if limit is not None and len(selected) >= limit:
                return selected, skipped_existing
    return selected, skipped_existing


def _summary_policy(summary: dict) -> dict[str, float]:
    decision = summary.get("decision") or {}
    user = decision.get("user") or {}
    thresholds = user.get("exit_thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError(
            f"Missing decision.user.exit_thresholds in round "
            f"{summary.get('round_id')!r}, user {summary.get('user_id')!r}"
        )
    return {str(key): float(value) for key, value in thresholds.items()}


def _selection_for_summary(
    dataset,
    summary: dict,
    *,
    sample_count: int,
) -> dict:
    bundle = get_bundle(str(summary["bundle_id"]))
    round_id = str(summary["round_id"])
    user_id = int(summary["user_id"])
    base_seed, effective_seed = _sample_seeds(
        round_id,
        bundle.dataset_id,
        user_id,
    )
    indices = stratified_sample_indices(dataset, sample_count, effective_seed)
    return {
        "indices": indices,
        "base_seed": base_seed,
        "effective_seed": effective_seed,
    }


def _all_exit_outputs(
    model: torch.nn.Module,
    manifest,
    images: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    executor = PyTorchSegmentExecutor(model, manifest)
    tensors = {"main": images}
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for segment_id in manifest.segment_ids:
        tensors = executor.execute_segment(segment_id, tensors)
        item = manifest.exit_for_boundary(segment_id + 1)
        if item is None:
            continue
        logits = executor.exit_logits(segment_id + 1, tensors)
        if logits is None:
            raise RuntimeError(
                f"Exit {item.get('exit_id')!r} produced no logits at "
                f"boundary {segment_id + 1}"
            )
        confidence, prediction = torch.softmax(logits, dim=1).max(dim=1)
        outputs[str(item["exit_id"])] = (
            confidence.detach().cpu(),
            prediction.detach().cpu(),
        )
    return outputs


def _cache_union_outputs(
    bundle_id: str,
    dataset,
    indices: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, dict], torch.nn.Module, object]:
    bundle = get_bundle(bundle_id)
    paths = bundle_paths(bundle_id)
    manifest = load_partition_manifest(bundle_id)
    model = build_model(bundle).to(device)
    model.load_state_dict(
        torch.load(paths.weight_path, map_location=device, weights_only=True)
    )
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    cache: dict[int, dict] = {}
    cursor = 0
    with torch.inference_mode():
        for batch_number, (images, labels, metadata) in enumerate(loader, start=1):
            images = images.to(device)
            outputs = _all_exit_outputs(model, manifest, images)
            batch_indices = indices[cursor : cursor + len(labels)]
            for offset, dataset_index in enumerate(batch_indices):
                cache[int(dataset_index)] = {
                    "label": int(labels[offset].item()),
                    "source_fallback": bool(
                        metadata["source_fallback"][offset].item()
                    ),
                    "outputs": {
                        exit_id: {
                            "confidence": float(confidences[offset].item()),
                            "prediction": int(predictions[offset].item()),
                        }
                        for exit_id, (confidences, predictions) in outputs.items()
                    },
                }
            cursor += len(labels)
            if batch_number % 10 == 0 or cursor == len(indices):
                print(
                    f"[{bundle_id}] inferred {cursor}/{len(indices)} unique samples",
                    flush=True,
                )
    if cursor != len(indices):
        raise RuntimeError(
            f"Cached {cursor} samples for {bundle_id}, expected {len(indices)}"
        )
    return cache, model, manifest


def _apply_policy(
    cached: dict,
    manifest,
    thresholds: dict[str, float],
) -> tuple[int, str, int]:
    manifest.validate_exit_thresholds(thresholds)
    for item in manifest.early_exits:
        exit_id = str(item["exit_id"])
        output = cached["outputs"][exit_id]
        if item.get("final") or output["confidence"] >= thresholds[exit_id]:
            prediction = int(output["prediction"])
            return prediction, exit_id, int(item["boundary_id"])
    raise RuntimeError("Manifest did not provide a final exit")


def _verify_batched_policy(
    dataset,
    dataset_index: int,
    cached: dict,
    model: torch.nn.Module,
    manifest,
    thresholds: dict[str, float],
) -> None:
    image, label, _metadata = dataset[dataset_index]
    executor = PyTorchSegmentExecutor(model, manifest)
    sequential = executor.execute_range_with_exits(
        0,
        manifest.final_boundary_id,
        {"main": image.unsqueeze(0)},
        thresholds,
    )
    cached_prediction, cached_exit_id, cached_boundary = _apply_policy(
        cached,
        manifest,
        thresholds,
    )
    actual = (
        int(sequential["prediction"]),
        str(sequential["exit_id"]),
        int(sequential["exit_boundary_id"]),
    )
    expected = (cached_prediction, cached_exit_id, cached_boundary)
    if actual != expected or int(label) != int(cached["label"]):
        raise AssertionError(
            f"Batched early-exit verification failed at dataset index "
            f"{dataset_index}: sequential={actual}, cached={expected}, "
            f"labels=({int(label)}, {int(cached['label'])})"
        )


def _dataset_metadata(package_root: Path) -> dict:
    metadata_path = package_root / "metadata.json"
    metadata = _read_json(metadata_path)
    return {
        "dataset_id": metadata.get("dataset_id"),
        "package_root": _relative_path(package_root),
        "full_test_pool": bool(metadata.get("full_test_pool")),
        "total_samples": int(metadata.get("total_samples", 0)),
        "source_manifest_sha256": metadata.get("source_manifest_sha256"),
        "package_created_at": metadata.get("created_at"),
    }


def _refresh_payload(
    summary: dict,
    *,
    thresholds: dict[str, float],
    selection: dict,
    cache: dict[int, dict],
    manifest,
    package_metadata: dict,
    model_hash: str,
) -> dict:
    correct = 0
    exit_counts: Counter[str] = Counter()
    for dataset_index in selection["indices"]:
        prediction, exit_id, _boundary = _apply_policy(
            cache[int(dataset_index)],
            manifest,
            thresholds,
        )
        correct += int(prediction == int(cache[int(dataset_index)]["label"]))
        exit_counts[exit_id] += 1
    samples = len(selection["indices"])
    source_fallback_samples = sum(
        int(cache[int(dataset_index)].get("source_fallback", False))
        for dataset_index in selection["indices"]
    )
    return {
        "accuracy": correct / max(samples, 1),
        "backend": "pytorch",
        "bundle_id": str(summary["bundle_id"]),
        "correct": int(correct),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_mode": "offline_batched_early_exit",
        "exit_counts": dict(sorted(exit_counts.items())),
        "exit_thresholds": copy.deepcopy(thresholds),
        "model_hash": model_hash,
        "samples": int(samples),
        "source_fallback_samples": int(source_fallback_samples),
        "sampling": {
            "base_seed": int(selection["base_seed"]),
            "effective_seed": int(selection["effective_seed"]),
            "policy": "stratified",
            "seed_derivation": "round_id+dataset_id+user_id",
            "seed_source": "round_id",
        },
        "test_package": copy.deepcopy(package_metadata),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEVICE_RESULTS_DIR,
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite-refresh", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")
    device = torch.device(device_name)

    summaries, skipped_existing = _discover_summaries(
        args.results_root,
        overwrite_refresh=args.overwrite_refresh,
        limit=args.limit,
    )
    print(
        f"Selected {len(summaries)} warm summaries; "
        f"skipped_existing={skipped_existing}; device={device}",
        flush=True,
    )
    grouped: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for item in summaries:
        grouped[str(item[1]["bundle_id"])].append(item)

    updated = 0
    for bundle_id in sorted(grouped):
        bundle = get_bundle(bundle_id)
        package_root = _full_test_package(bundle_id)
        if not package_root.is_dir():
            raise FileNotFoundError(f"Full test package not found: {package_root}")
        dataset = _ReadablePackageDataset(bundle, package_root)
        package_metadata = _dataset_metadata(package_root)
        package_metadata["missing_or_empty_files"] = len(
            dataset.invalid_package_indices
        )
        if dataset.invalid_package_indices:
            package_metadata["source_fallback"] = (
                "torchvision CIFAR10 test split via manifest source_index"
            )
        selections: dict[Path, dict] = {}
        policies: dict[Path, dict[str, float]] = {}
        union_indices: set[int] = set()
        for summary_path, summary in grouped[bundle_id]:
            selection = _selection_for_summary(
                dataset,
                summary,
                sample_count=args.samples,
            )
            policy = _summary_policy(summary)
            selections[summary_path] = selection
            policies[summary_path] = policy
            union_indices.update(selection["indices"])

        sorted_indices = sorted(union_indices)
        print(
            f"[{bundle_id}] summaries={len(grouped[bundle_id])}, "
            f"selected={len(grouped[bundle_id]) * args.samples}, "
            f"unique={len(sorted_indices)}, pool={len(dataset)}",
            flush=True,
        )
        cache, model, manifest = _cache_union_outputs(
            bundle_id,
            dataset,
            sorted_indices,
            batch_size=args.batch_size,
            device=device,
        )
        first_path, _first_summary = grouped[bundle_id][0]
        first_index = selections[first_path]["indices"][0]
        _verify_batched_policy(
            dataset,
            first_index,
            cache[first_index],
            model,
            manifest,
            policies[first_path],
        )
        print(f"[{bundle_id}] batched early-exit verification passed", flush=True)
        weight_path = bundle_paths(bundle_id).weight_path
        model_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()

        for summary_path, summary in grouped[bundle_id]:
            refreshed = _refresh_payload(
                summary,
                thresholds=policies[summary_path],
                selection=selections[summary_path],
                cache=cache,
                manifest=manifest,
                package_metadata=package_metadata,
                model_hash=model_hash,
            )
            print(
                f"{summary_path}: "
                f"accuracy_refresh={refreshed['accuracy']:.4f} "
                f"({refreshed['correct']}/{refreshed['samples']})",
                flush=True,
            )
            if args.write:
                summary["accuracy_refresh"] = refreshed
                _atomic_write_json(summary_path, summary)
                updated += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    action = "updated" if args.write else "would_update"
    print(f"{action}={updated if args.write else len(summaries)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
