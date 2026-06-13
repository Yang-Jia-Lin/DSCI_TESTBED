"""Measure atomic bundle segments on a target PyTorch device."""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Models.ModelNet.MultiExitResNet import build_model
from Src.Shared.Partitioning.manifest import load_partition_manifest, validate_model_file
from Src.Shared.Partitioning.pytorch_executor import PyTorchSegmentExecutor
from Src.Shared.Profiles.segment_profile import write_segment_profile

_EXECUTOR = _SAMPLE = _MANIFEST = None


def _init_worker(bundle_id, device_name, threads):
    global _EXECUTOR, _SAMPLE, _MANIFEST
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    bundle = get_bundle(bundle_id)
    paths = bundle_paths(bundle_id)
    _MANIFEST = load_partition_manifest(bundle_id)
    validate_model_file(_MANIFEST, paths.weight_path)
    device = torch.device(device_name)
    model = build_model(bundle).to(device)
    model.load_state_dict(torch.load(paths.weight_path, map_location=device, weights_only=True))
    _EXECUTOR = PyTorchSegmentExecutor(model, _MANIFEST)
    _SAMPLE = torch.randn((1, *bundle.input_shape), device=device)


def _profile_once():
    bundle = {"main": _SAMPLE}
    elapsed, exit_elapsed = [], {}
    for segment_id in _MANIFEST.segment_ids:
        started = time.perf_counter()
        bundle = _EXECUTOR.execute_segment(segment_id, bundle)
        elapsed.append(time.perf_counter() - started)
        item = _MANIFEST.exit_for_boundary(segment_id + 1)
        if item is not None and not item.get("final"):
            started = time.perf_counter()
            _EXECUTOR.exit_logits(segment_id + 1, bundle)
            exit_elapsed[str(item["exit_id"])] = time.perf_counter() - started
    return elapsed, float(sum(elapsed)), exit_elapsed


def _job(_):
    return _profile_once()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_id")
    parser.add_argument("--bundle-id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--profile-root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    manifest = load_partition_manifest(bundle.bundle_id)
    samples = [[] for _ in manifest.segment_ids]
    exit_samples = {exit_id: [] for exit_id in manifest.exit_ids}
    totals = []
    with ProcessPoolExecutor(
        max_workers=args.worker_count,
        initializer=_init_worker,
        initargs=(bundle.bundle_id, args.device, args.threads_per_worker),
    ) as pool:
        for _ in range(args.warmup):
            list(pool.map(_job, range(args.worker_count)))
        for _ in range(args.runs):
            for elapsed, total, heads in pool.map(_job, range(args.worker_count)):
                totals.append(total)
                for index, value in enumerate(elapsed):
                    samples[index].append(value)
                for exit_id, value in heads.items():
                    exit_samples[exit_id].append(value)
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


if __name__ == "__main__":
    main()
