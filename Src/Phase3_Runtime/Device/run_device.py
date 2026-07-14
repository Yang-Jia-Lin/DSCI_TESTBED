"""Bundle-aware Device runner for fixed-worker partition decisions."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import threading
import time
import uuid
from pathlib import Path

import requests

from Src.Phase2_Scheduler.algo_config import DEFAULT as ALGO_CFG
from Src.Phase3_Runtime.Device.runtime_v2 import run_partitioned_inference
from Src.Phase3_Runtime.Shared.bandwidth_iperf import measure_bandwidth_iperf
from Src.Phase3_Runtime.Shared.state_reporter import RoundClient
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import DEVICE_RESULTS_DIR, bundle_paths
from Src.Shared.Data.registry import build_loader, build_test_package_loader
from Src.Shared.Profiles.segment_profile import segment_profile_state


def collect_device_state(
    bundle_id: str,
    backend: str,
    *,
    bw_d2e_override: float | None = None,
    iperf_duration: float | None = None,
    iperf_timeout: float | None = None,
    d2e_link_id: str | None = None,
    d2e_capacity_mbps: float | None = None,
):
    device = segment_profile_state("device", backend, bundle_id)
    if bw_d2e_override is not None:
        bw_d2e = float(bw_d2e_override)
    else:
        measured_bw = measure_bandwidth_iperf(
            TESTBED_CFG.edge_host,
            TESTBED_CFG.edge_iperf_port,
            duration=iperf_duration,
            timeout_s=iperf_timeout,
        )
        if measured_bw is None or float(measured_bw) <= 0:
            bw_d2e = float(TESTBED_CFG.default_bw_d2e)
            print(
                "Device->Edge iperf failed; "
                f"using fallback BW_d2e={bw_d2e:.4f} Mbps"
            )
        else:
            bw_d2e = float(measured_bw)
    result = {
        **device,
        "BW_d2e": bw_d2e,
    }
    if d2e_link_id:
        result["d2e_link_id"] = str(d2e_link_id)
    if d2e_capacity_mbps is not None:
        result["d2e_capacity_mbps"] = float(d2e_capacity_mbps)
    return result


def registration_payload(
    user_id: int,
    device: dict,
    *,
    decision_mode: str | None = None,
) -> dict:
    payload = {
        "user_id": int(user_id),
        "bundle_id": device["bundle_id"],
        "resource_mode": "fixed_worker_pool",
        "device": device,
    }
    if decision_mode:
        payload["decision_mode"] = str(decision_mode)
    return payload


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


def _measurement_record(
    result: dict,
    *,
    user_id: int,
    sample_index: int,
    label: int,
    is_correct: bool,
    sample_metadata: dict | None = None,
    synchronization: dict | None = None,
) -> dict:
    record = {
        "request_id": str(result["request_id"]),
        "user_id": int(user_id),
        "sample_index": int(sample_index),
        "label": int(label),
        "prediction": int(result["prediction"]),
        "exit_location": result.get("exit_location"),
        "T_total": float(result["T_total"]),
        "is_correct": bool(is_correct),
        "exit_id": result.get("exit_id"),
        "exit_boundary_id": result.get("exit_boundary_id"),
        "confidence": result.get("confidence"),
        "executed_segments_by_node": copy.deepcopy(
            (result.get("request_trace") or {}).get("executed_segments_by_node", {})
        ),
    }
    trace = copy.deepcopy(result.get("request_trace") or {})
    for key in (
        "device_compute",
        "d2e_transport",
        "edge_queue",
        "edge_segment_compute",
        "edge_exit_head_compute",
        "edge_exit_check",
        "e2c_transport",
        "cloud_queue",
        "cloud_segment_compute",
        "cloud_exit_head_compute",
        "cloud_exit_check",
        "unattributed_overhead",
        "total_latency",
    ):
        if key in trace:
            record[key] = float(trace[key])
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
    if synchronization:
        record.update(copy.deepcopy(synchronization))
    request_utility = (
        float(ALGO_CFG.alpha) * float(bool(is_correct))
        - float(ALGO_CFG.beta) * float(result["T_total"])
    )
    record["observed_accuracy"] = float(bool(is_correct))
    record["observed_latency"] = float(result["T_total"])
    record["observed_utility"] = request_utility
    trace.update(
        {
            "sample_id": record.get("sample_id"),
            "label": int(label),
            "correct": bool(is_correct),
            "observed_utility": request_utility,
        }
    )
    record["request_trace"] = trace
    _add_latency_aliases(record)
    return record


def _add_latency_aliases(record: dict) -> None:
    if "T_compute_device" in record:
        record["T_d_compute"] = float(record["T_compute_device"])
    if "T_compute_edge" in record:
        record["T_e_compute"] = float(record["T_compute_edge"])
    if "T_compute_cloud" in record:
        record["T_c_compute"] = float(record["T_compute_cloud"])
    if "T_device_edge_roundtrip" in record:
        value = float(record["T_device_edge_roundtrip"])
        if "T_node_edge" in record:
            value -= float(record["T_node_edge"])
        if "T_edge_cloud_roundtrip" in record:
            value -= float(record["T_edge_cloud_roundtrip"])
        record["T_d2e"] = max(0.0, value)
    if "T_edge_cloud_roundtrip" in record:
        value = float(record["T_edge_cloud_roundtrip"])
        if "T_node_cloud" in record:
            value -= float(record["T_node_cloud"])
        record["T_e2c"] = max(0.0, value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary_payload(
    measurements: list[dict],
    *,
    round_id: str,
    user_id: int,
    bundle_id: str,
    backend: str,
    decision: dict | None,
    device_state: dict,
    correct: int,
    total: int,
) -> dict:
    latency_summary = {}
    for key in (
        "T_d_compute",
        "T_e_compute",
        "T_c_compute",
        "T_d2e",
        "T_e2c",
        "T_total",
        "device_compute",
        "d2e_transport",
        "edge_queue",
        "edge_segment_compute",
        "edge_exit_head_compute",
        "edge_exit_check",
        "e2c_transport",
        "cloud_queue",
        "cloud_segment_compute",
        "cloud_exit_head_compute",
        "cloud_exit_check",
        "unattributed_overhead",
        "total_latency",
    ):
        values = [float(record[key]) for record in measurements if key in record]
        mean_value = _mean(values)
        latency_summary[f"{key}_avg_ms"] = (
            1000.0 * mean_value if mean_value is not None else None
        )
    return {
        "round_id": str(round_id),
        "user_id": int(user_id),
        "bundle_id": str(bundle_id),
        "backend": str(backend),
        "samples": int(total),
        "correct": int(correct),
        "accuracy": correct / max(total, 1),
        "utility_mean": _mean(
            [float(record["observed_utility"]) for record in measurements]
        ),
        "utility_sum": sum(
            float(record["observed_utility"]) for record in measurements
        ),
        "alpha": float(ALGO_CFG.alpha),
        "beta": float(ALGO_CFG.beta),
        "latency": latency_summary,
        "decision": decision,
        "device_state": device_state,
    }


def _write_device_results(
    measurements: list[dict],
    summary: dict,
    *,
    round_id: str,
    user_id: int,
) -> tuple[Path, Path, Path]:
    result_dir = DEVICE_RESULTS_DIR / str(round_id)
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = f"user_{int(user_id)}"
    jsonl_path = result_dir / f"{stem}_measurements.jsonl"
    summary_path = result_dir / f"{stem}_summary.json"
    csv_path = result_dir / f"{stem}_inference_results.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in measurements:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    fieldnames = sorted({key for record in measurements for key in record})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(measurements)

    return jsonl_path, summary_path, csv_path


def _print_measurement_summary(measurements: list[dict], *, correct: int, total: int) -> None:
    if not measurements:
        print("samples=0 accuracy=0.0000")
        return
    print(f"samples={total} accuracy={correct / max(total, 1):.4f}")
    for key in (
        "T_d_compute",
        "T_e_compute",
        "T_c_compute",
        "T_d2e",
        "T_e2c",
        "T_total",
    ):
        values = [float(record[key]) for record in measurements if key in record]
        if not values:
            continue
        mean_ms = 1000.0 * sum(values) / len(values)
        print(f"{key}_avg_ms={mean_ms:.3f}")


def collect_state(
    bundle_id: str,
    backend: str,
    *,
    bw_d2e_override: float | None = None,
    iperf_duration: float | None = None,
    iperf_timeout: float | None = None,
):
    """Backward-compatible one-user v1 state builder."""
    device = collect_device_state(
        bundle_id,
        backend,
        bw_d2e_override=bw_d2e_override,
        iperf_duration=iperf_duration,
        iperf_timeout=iperf_timeout,
    )
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


def _package_name(package_id: str, split: str, mode: str, samples_per_class: int, seed: int) -> str:
    return f"{package_id}__{split}__{mode}__{samples_per_class}pc__seed{seed}"


def _test_package_id(bundle, mode: str) -> str:
    # Balanced packages are dataset-level and shared by every compatible model.
    # Easy/hard packages remain model-level because difficulty is model-derived.
    return bundle.dataset_id if mode == "balanced" else bundle.bundle_id


def _resolve_test_package_root(bundle, args) -> Path | None:
    if args.test_package_root:
        return Path(args.test_package_root)
    if not args.test_package_mode:
        return None
    base = Path(args.test_package_base) if args.test_package_base else bundle_paths(bundle.bundle_id).test_package_root
    package_id = _test_package_id(bundle, args.test_package_mode)
    if args.test_package_samples_per_class is not None:
        return base / _package_name(
            package_id,
            args.test_package_split,
            args.test_package_mode,
            args.test_package_samples_per_class,
            args.test_package_seed,
        )
    pattern = f"{package_id}__{args.test_package_split}__{args.test_package_mode}__*pc__seed{args.test_package_seed}"
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
    parser.add_argument("--test-package-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--test-package-samples-per-class", type=int)
    parser.add_argument("--test-package-seed", type=int, default=42)
    parser.add_argument("--test-package-base")
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--decision-timeout", type=float, default=90.0)
    parser.add_argument(
        "--no-request-barrier",
        action="store_true",
        help="Run samples immediately instead of synchronizing every request across devices.",
    )
    parser.add_argument(
        "--override-bw-d2e",
        type=float,
        help="Use this Device->Edge bandwidth in Mbps instead of iperf.",
    )
    parser.add_argument(
        "--d2e-link-id",
        help="Shared-link identifier; use the same value for devices on one AP/link.",
    )
    parser.add_argument(
        "--d2e-capacity-mbps",
        type=float,
        help="Total capacity of the shared Device-to-Edge link in Mbps.",
    )
    parser.add_argument(
        "--iperf-duration",
        type=float,
        help="iperf3 measurement duration in seconds; default is 8 or DSCI_IPERF_DURATION_S.",
    )
    parser.add_argument(
        "--iperf-timeout",
        type=float,
        help="iperf3 subprocess timeout in seconds; default is duration + 15.",
    )
    parser.add_argument(
        "--decision-mode",
        choices=(
            "dsci",
            "device",
            "device_early_exit",
            "edge",
            "edge_early_exit",
            "cloud",
            "cloud_early_exit",
        ),
        help="Request a preset placement instead of DSCI for this round.",
    )
    args = parser.parse_args(argv)
    args.round_id = (
        str(args.round_id or "").strip()
        or str(os.environ.get("ROUND_ID") or "").strip()
        or str(os.environ.get("DSCI_ROUND_ID") or "").strip()
    )
    if not args.round_id:
        raise SystemExit(
            "--round-id is empty. Set it with: export ROUND_ID=$(date +%Y%m%d-%H%M) "
            'and pass --round-id "$ROUND_ID".'
        )
    bundle = get_bundle(args.bundle_id)
    test_package_root = _resolve_test_package_root(bundle, args)
    device = collect_device_state(
        bundle.bundle_id,
        args.backend,
        bw_d2e_override=args.override_bw_d2e,
        iperf_duration=args.iperf_duration,
        iperf_timeout=args.iperf_timeout,
        d2e_link_id=args.d2e_link_id,
        d2e_capacity_mbps=args.d2e_capacity_mbps,
    )
    print(f"Device state BW_d2e={float(device['BW_d2e']):.4f} Mbps")
    client = RoundClient(TESTBED_CFG.algo_base_url, args.round_id, args.user_id)
    client.register(
        registration_payload(
            args.user_id,
            device,
            decision_mode=args.decision_mode,
        )
    )
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
        user_decision = decision.get("user", {})
        print(
            "Decision summary: "
            f"source={decision.get('decision_source')}, "
            f"objective={decision.get('objective')}, "
            f"b1={user_decision.get('partition_boundary_1')}, "
            f"b2={user_decision.get('partition_boundary_2')}"
        )
        if test_package_root:
            loader = build_test_package_loader(bundle, test_package_root, batch_size=1)
            sample_limit = args.test_samples
            print(f"Loaded test package: {test_package_root}")
        else:
            loader = build_loader(bundle, "val", batch_size=1, data_root=args.data_root)
            sample_limit = args.test_samples if args.test_samples is not None else 100
        for sample_index, batch in enumerate(loader):
            if test_package_root:
                images, labels, sample_metadata = batch
            else:
                images, labels = batch
                sample_metadata = None
            request_id = uuid.uuid4().hex
            synchronization = None
            if not args.no_request_barrier:
                barrier = client.wait_for_request_release(
                    sample_index, timeout_s=args.decision_timeout
                )
                actual_start_at = time.time()
                release_at = float(barrier["release_at"])
                synchronization = {
                    "request_seq": int(sample_index),
                    "barrier_ready_at_utc": float(
                        barrier["ready_at"][str(args.user_id)]
                    ),
                    "barrier_release_at_utc": release_at,
                    "actual_start_at_utc": actual_start_at,
                    "start_skew_s": actual_start_at - release_at,
                }
            result = run_partitioned_inference(
                images,
                decision,
                user_id=args.user_id,
                request_id=request_id,
            )
            label = int(labels.item())
            is_correct = result["prediction"] == label
            correct += int(is_correct)
            measurements.append(
                _measurement_record(
                    result,
                    user_id=args.user_id,
                    sample_index=sample_index,
                    label=label,
                    is_correct=is_correct,
                    sample_metadata=sample_metadata,
                    synchronization=synchronization,
                )
            )
            total += 1
            if sample_limit is not None and total >= sample_limit:
                break
        summary = _summary_payload(
            measurements,
            round_id=args.round_id,
            user_id=args.user_id,
            bundle_id=bundle.bundle_id,
            backend=args.backend,
            decision=decision,
            device_state=device,
            correct=correct,
            total=total,
        )
        jsonl_path, summary_path, csv_path = _write_device_results(
            measurements,
            summary,
            round_id=args.round_id,
            user_id=args.user_id,
        )
        print(f"Saved measurements: {jsonl_path}")
        print(f"Saved summary: {summary_path}")
        print(f"Saved inference CSV: {csv_path}")
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
