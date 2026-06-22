"""Bundle-aware Device runner for fixed-worker partition decisions."""

import argparse
import json
import threading
import uuid

import requests

from Src.Phase3_Runtime.Device.runtime_v2 import run_partitioned_inference
from Src.Phase3_Runtime.Shared.bandwidth_iperf import measure_bandwidth_iperf
from Src.Phase3_Runtime.Shared.state_reporter import RoundClient
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Data.registry import build_loader
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


def _measurement_record(result: dict, *, is_correct: bool) -> dict:
    record = {
        "request_id": str(result["request_id"]),
        "T_total": float(result["T_total"]),
        "is_correct": bool(is_correct),
    }
    for key, value in result.items():
        if key.startswith("T_") and key != "T_total" and value is not None:
            record[key] = float(value)
    return record


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--backend", choices=("pytorch", "mnn"), default="pytorch")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--decision-timeout", type=float, default=90.0)
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
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
        loader = build_loader(bundle, "val", batch_size=1, data_root=args.data_root)
        for images, labels in loader:
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
                _measurement_record(result, is_correct=is_correct)
            )
            total += 1
            if total >= args.test_samples:
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
    print(f"samples={total} accuracy={correct / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
