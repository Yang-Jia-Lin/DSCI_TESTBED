"""Generate a calibrated PyTorch or stage-distributed MNN compute profile."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Src.Shared.Profiles.compute_profile import write_compute_profile
from Src.Shared.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50
from Src.Shared.Config.model_config import RESNET50 as MODEL_CFG
from Src.Shared.Config.paths import RESNET50_PATHS as MODEL_PATHS

STAGE_ENDS = (3, 26, 56, 102, 127)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_id")
    parser.add_argument("--backend", choices=("pytorch", "mnn"), default="pytorch")
    parser.add_argument("--weights", default=str(MODEL_PATHS.resolve_weight_path()))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--profile-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mnn-stage-latencies-json",
        help="For MNN, JSON file containing five measured stage latencies in seconds.",
    )
    return parser.parse_args(argv)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_pytorch(args, stats):
    device = torch.device(args.device)
    torch.set_num_threads(max(1, args.threads))
    model = MultiEEResNet50(
        Bottleneck, [3, 4, 6, 3], num_classes=10, include_top=True
    ).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    sample = torch.randn(1, 3, 227, 227, device=device)

    layer_names = stats["layer"].astype(str).tolist()
    modules = dict(model.named_modules())
    missing = [name for name in layer_names if name not in modules]
    if missing:
        raise ValueError(f"Model is missing canonical profile modules: {missing}")

    elapsed = defaultdict(float)
    starts = {}
    handles = []

    def before(name):
        def hook(_module, _inputs):
            _sync(device)
            starts[name] = time.perf_counter()
        return hook

    def after(name):
        def hook(_module, _inputs, _output):
            _sync(device)
            elapsed[name] += time.perf_counter() - starts[name]
        return hook

    for name in layer_names:
        handles.append(modules[name].register_forward_pre_hook(before(name)))
        handles.append(modules[name].register_forward_hook(after(name)))

    with torch.no_grad():
        for _ in range(args.warmup):
            model(sample, stage=None)
        elapsed.clear()
        totals = []
        for _ in range(args.runs):
            _sync(device)
            started = time.perf_counter()
            model(sample, stage=None)
            _sync(device)
            totals.append(time.perf_counter() - started)
    for handle in handles:
        handle.remove()
    raw = np.array([elapsed[name] / args.runs for name in layer_names], dtype=np.float64)
    return raw, float(np.mean(totals)), "layer_measured"


def _profile_mnn(args, stats):
    if not args.mnn_stage_latencies_json:
        raise ValueError("--mnn-stage-latencies-json is required for backend=mnn")
    with Path(args.mnn_stage_latencies_json).open("r", encoding="utf-8") as handle:
        stage_latencies = np.asarray(json.load(handle), dtype=np.float64)
    if stage_latencies.shape != (5,) or np.any(stage_latencies <= 0):
        raise ValueError("MNN stage latency JSON must contain five positive seconds values")

    flops = stats["approx_flops"].to_numpy(dtype=np.float64)
    raw = np.zeros(len(flops), dtype=np.float64)
    start = 0
    for stage, end in enumerate(STAGE_ENDS):
        stage_flops = flops[start : end + 1]
        weights = stage_flops.copy()
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        raw[start : end + 1] = stage_latencies[stage] * weights / float(weights.sum())
        start = end + 1
    return raw, float(stage_latencies.sum()), "stage_distributed"


def main(argv=None):
    args = parse_args(argv)
    if args.runs <= 0 or args.warmup < 0:
        raise ValueError("--runs must be positive and --warmup must be non-negative")
    stats = pd.read_csv(MODEL_PATHS.resolve_layer_stats_csv(), skipinitialspace=True)
    stats.columns = [str(col).strip() for col in stats.columns]
    if args.backend == "pytorch":
        raw, total, granularity = _profile_pytorch(args, stats)
    else:
        raw, total, granularity = _profile_mnn(args, stats)

    profile = write_compute_profile(
        profile_id=args.profile_id,
        layer_names=stats["layer"].astype(str),
        theoretical_flops=stats["approx_flops"],
        raw_latencies_s=raw,
        total_latency_s=total,
        model_name=MODEL_CFG.name,
        backend=args.backend,
        profile_root=args.profile_root,
        overwrite=args.overwrite,
        metadata_extra={
            "device": args.device,
            "host": platform.node(),
            "threads": args.threads,
            "batch_size": 1,
            "warmup_runs": args.warmup,
            "measurement_runs": args.runs,
            "profile_granularity": granularity,
            "weights": str(args.weights),
        },
    )
    print(f"Saved compute profile: {profile.profile_dir}")
    print(f"theta_flops_per_s={profile.theta:.6f}")


if __name__ == "__main__":
    main()
