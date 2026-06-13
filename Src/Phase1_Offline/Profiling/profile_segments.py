"""Measure atomic partition segments on the target PyTorch device."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from Src.Shared.Config.paths import RESNET50_PATHS
from Src.Shared.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Partitioning.manifest import validate_model_file
from Src.Shared.Partitioning.pytorch_executor import PyTorchSegmentExecutor
from Src.Shared.Profiles.segment_profile import write_segment_profile

_EXECUTOR = None
_SAMPLE = None
_MANIFEST = None


def _init_profile_worker(manifest_id: str, device_name: str, threads_per_worker: int):
    global _EXECUTOR, _SAMPLE, _MANIFEST
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    torch.set_num_threads(threads_per_worker)
    device = torch.device(device_name)
    _MANIFEST = load_partition_manifest(manifest_id)
    validate_model_file(_MANIFEST, RESNET50_PATHS.resolve_weight_path())
    model = MultiEEResNet50(
        Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True
    ).to(device)
    model.load_state_dict(
        torch.load(
            RESNET50_PATHS.resolve_weight_path(),
            map_location=device,
            weights_only=True,
        )
    )
    _EXECUTOR = PyTorchSegmentExecutor(model, _MANIFEST)
    _SAMPLE = torch.randn(1, 3, 227, 227, device=device)


def _profile_once():
    bundle = {"main": _SAMPLE}
    elapsed = []
    exit_elapsed = {}
    for segment_id in _MANIFEST.segment_ids:
        started = time.perf_counter()
        bundle = _EXECUTOR.execute_segment(segment_id, bundle)
        elapsed.append(time.perf_counter() - started)
        boundary_id = segment_id + 1
        for item in _MANIFEST.early_exits:
            if (
                int(item["boundary_id"]) == boundary_id
                and boundary_id != _MANIFEST.final_boundary_id
            ):
                started_exit = time.perf_counter()
                _EXECUTOR.exit_logits(boundary_id, bundle)
                exit_elapsed[int(item["logical_layer"])] = (
                    time.perf_counter() - started_exit
                )
    return elapsed, float(sum(elapsed)), exit_elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_id")
    parser.add_argument("--manifest-id", default="resnet50-cifar10-partition-v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--profile-root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.worker_count <= 0 or args.threads_per_worker <= 0 or args.runs <= 0:
        raise ValueError("Worker settings and runs must be positive")
    if args.worker_count * args.threads_per_worker > (os.cpu_count() or 1):
        raise ValueError("worker_count * threads_per_worker exceeds logical CPU count")
    if args.device != "cpu" and args.worker_count != 1:
        raise ValueError("Non-CPU profiling currently requires worker_count=1")

    manifest = load_partition_manifest(args.manifest_id)
    samples = [[] for _ in manifest.segment_ids]
    exit_samples = {
        int(item["logical_layer"]): []
        for item in manifest.early_exits
        if int(item["boundary_id"]) != manifest.final_boundary_id
    }
    totals = []
    with ProcessPoolExecutor(
        max_workers=args.worker_count,
        initializer=_init_profile_worker,
        initargs=(args.manifest_id, args.device, args.threads_per_worker),
    ) as pool:
        for _ in range(args.warmup):
            list(pool.map(lambda_unused_profile_job, range(args.worker_count)))
        for _ in range(args.runs):
            batch = list(pool.map(lambda_unused_profile_job, range(args.worker_count)))
            for elapsed, total, exit_elapsed in batch:
                totals.append(total)
                for segment_id, value in enumerate(elapsed):
                    samples[segment_id].append(value)
                for logical_layer, value in exit_elapsed.items():
                    exit_samples[logical_layer].append(value)
    profile = write_segment_profile(
        profile_id=args.profile_id,
        manifest=manifest,
        backend="pytorch",
        worker_count=args.worker_count,
        threads_per_worker=args.threads_per_worker,
        samples_s=samples,
        total_model_latency_s=float(sum(totals) / len(totals)),
        exit_head_samples_s=exit_samples,
        profile_root=args.profile_root,
        overwrite=args.overwrite,
    )
    print(f"Saved segment profile: {profile.profile_dir}")


def lambda_unused_profile_job(_unused):
    """Picklable adapter for ProcessPoolExecutor.map."""
    return _profile_once()


if __name__ == "__main__":
    main()
