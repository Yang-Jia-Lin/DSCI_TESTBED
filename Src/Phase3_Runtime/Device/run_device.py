"""Bundle-aware Device runner for fixed-worker partition decisions."""

import argparse
import json
import threading
import uuid
from pathlib import Path

import requests

from Src.Phase3_Runtime.Device.runtime_v2 import run_partitioned_inference
from Src.Phase3_Runtime.Shared.bandwidth_iperf import measure_bandwidth_iperf
from Src.Phase3_Runtime.Shared.state_reporter import RoundClient
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_loader, build_test_package_loader
from Src.Shared.Profiles.segment_profile import segment_profile_state


def collect_device_state(bundle_id: str, backend: str):
    device = segment_profile_state("device", backend, bundle_id)
    return {
        **device,
        "BW_d2e": measure_bandwidth_iperf(
            TESTBED_CFG.edge_host, TESTBED_CFG.edge_iperf_port
        ),
    }


def registration_payload(user_id: int, device: dict) -> dict:
    return {
        "user_id": int(user_id),
        "bundle_id": device["bundle_id"],
        "resource_mode": "fixed_worker_pool",
        "device": device,
    }


def _heartbeat_loop(client: RoundClient, stop: threading.Event, interval_s: float):
    while not stop.wait(float(interval_s)):
        try:
            client.heartbeat()
        except Exception as exc:
            print(f"Heartbeat failed: {exc}")


def _metadata_value(metadata: dict | None, key: str):
    if not metadata or key not in metadata:
        return None
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    elif hasattr(value, "numel") and value.numel() == 1:
        value = value.item()
    if value == "":
        return None
    return value


def _measurement_record(result: dict, *, is_correct: bool, sample_metadata: dict | None = None) -> dict:
    record = {
        "request_id": str(result["request_id"]),
        "T_total": float(result["T_total"]),
        "is_correct": bool(is_correct),
    }
    for key in ("sample_id", "source_index", "difficulty"):
        value = _metadata_value(sample_metadata, key)
        if value is not None:
            if key == "source_index":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            record[key] = value
    for key, value in result.items():
        if key.startswith("T_") and key != "T_total" and value is not None:
            record[key] = float(value)
    return record


def _print_measurement_summary(measurements: list[dict], *, correct: int, total: int) -> None:
    if not measurements:
        print("samples=0 accuracy=0.0000")
        return
    print(f"samples={total} accuracy={correct / max(total, 1):.4f}")
    latency_keys = sorted(
        {
            key
            for record in measurements
            for key in record
            if key.startswith("T_")
        }
    )
    for key in latency_keys:
        values = [float(record[key]) for record in measurements if key in record]
        if not values:
            continue
        mean_ms = 1000.0 * sum(values) / len(values)
        print(f"{key}_avg_ms={mean_ms:.3f}")


def collect_state(bundle_id: str, backend: str):
    """Backward-compatible one-user v1 state builder."""
    device = collect_device_state(bundle_id, backend)
    edge = requests.get(
        f"http://{TESTBED_CFG.edge_host}:{TESTBED_CFG.edge_status_port}/status", timeout=10
    ).json()
    cloud = requests.get(
        f"http://{TESTBED_CFG.cloud_host}:{TESTBED_CFG.cloud_status_port}/status", timeout=10
    ).json()
    return {
        "bundle_id": bundle_id,
        "resource_mode": "fixed_worker_pool",
        "edge": edge,
        "cloud": {**cloud, "BW_e2c": edge["BW_e2c"]},
        "users": [device],
    }


def _package_name(bundle_id: str, split: str, mode: str, samples_per_class: int, seed: int) -> str:
    return f"{bundle_id}__{split}__{mode}__{samples_per_class}pc__seed{seed}"


def _resolve_test_package_root(bundle, args) -> Path | None:
    if args.test_package_root:
        return Path(args.test_package_root)
    if not args.test_package_mode:
        return None
    base = Path(args.test_package_base) if args.test_package_base else bundle_paths(bundle.bundle_id).test_package_root
    if args.test_package_samples_per_class is not None:
        return base / _package_name(
            bundle.bundle_id,
            args.test_package_split,
            args.test_package_mode,
            args.test_package_samples_per_class,
            args.test_package_seed,
        )
    pattern = f"{bundle.bundle_id}__{args.test_package_split}__{args.test_package_mode}__*pc__seed{args.test_package_seed}"
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No test package matched {base / pattern}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            "Multiple test packages matched; pass --test-package-samples-per-class "
            f"to choose one. Matches: {names}"
        )
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--backend", choices=("pytorch", "mnn"), default="pytorch")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--test-package-root", "--testset-root", dest="test_package_root")
    parser.add_argument("--test-package-mode", choices=("balanced", "easy", "hard"))
    parser.add_argument("--test-package-split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--test-package-samples-per-class", type=int)
    parser.add_argument("--test-package-seed", type=int, default=42)
    parser.add_argument("--test-package-base")
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--decision-timeout", type=float, default=90.0)
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    test_package_root = _resolve_test_package_root(bundle, args)
    device = collect_device_state(bundle.bundle_id, args.backend)
    client = RoundClient(TESTBED_CFG.algo_base_url, args.round_id, args.user_id)
    client.register(registration_payload(args.user_id, device))
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(client, heartbeat_stop, args.heartbeat_interval),
        daemon=True,
    )
    heartbeat.start()
    correct = total = 0
    measurements = []
    try:
        decision = client.wait_for_decision(timeout_s=args.decision_timeout)
        print(json.dumps(decision, indent=2))
        if test_package_root:
            loader = build_test_package_loader(bundle, test_package_root, batch_size=1)
            sample_limit = args.test_samples
            print(f"Loaded test package: {test_package_root}")
        else:
            loader = build_loader(bundle, "val", batch_size=1, data_root=args.data_root)
            sample_limit = args.test_samples if args.test_samples is not None else 100
        for batch in loader:
            if test_package_root:
                images, labels, sample_metadata = batch
            else:
                images, labels = batch
                sample_metadata = None
            request_id = uuid.uuid4().hex
            result = run_partitioned_inference(
                images,
                decision,
                user_id=args.user_id,
                request_id=request_id,
            )
            is_correct = result["prediction"] == int(labels.item())
            correct += int(is_correct)
            measurements.append(
                _measurement_record(result, is_correct=is_correct, sample_metadata=sample_metadata)
            )
            total += 1
            if sample_limit is not None and total >= sample_limit:
                break
        client.submit_measurements(
            {
                "decision_id": decision["decision_id"],
                "decision_version": decision["decision_version"],
                "measurements": measurements,
            }
        )
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=args.heartbeat_interval + 1.0)
    _print_measurement_summary(measurements, correct=correct, total=total)


if __name__ == "__main__":
    main()
