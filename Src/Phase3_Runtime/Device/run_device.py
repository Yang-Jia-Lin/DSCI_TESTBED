"""Bundle-aware Device runner for fixed-worker partition decisions."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
from Src.Phase3_Runtime.Shared.dynamic_bandwidth import BandwidthEstimator
from Src.Phase3_Runtime.Shared.state_reporter import RoundClient
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import DEVICE_RESULTS_DIR, bundle_paths
from Src.Shared.Data.registry import build_loader, build_test_package_loader
from Src.Shared.Profiles.segment_profile import segment_profile_state

_BANDWIDTH_CACHE_LOCK = threading.Lock()


def _bandwidth_cache_path(device: dict) -> Path:
    profile_id = str(device["execution_profile_id"]).replace("/", "_").replace("\\", "_")
    return DEVICE_RESULTS_DIR / "bandwidth_cache" / f"{profile_id}__d2e.json"


def _load_cached_bandwidth(device: dict) -> float | None:
    path = _bandwidth_cache_path(device)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("edge_host") != TESTBED_CFG.edge_host:
            return None
        value = float(payload["bw_d2e_mbps"])
        return value if value > 0.0 else None
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_cached_bandwidth(device: dict, bw_mbps: float) -> None:
    path = _bandwidth_cache_path(device)
    payload = {
        "execution_profile_id": device["execution_profile_id"],
        "edge_host": TESTBED_CFG.edge_host,
        "bw_d2e_mbps": float(bw_mbps),
        "updated_at": float(time.time()),
    }
    with _BANDWIDTH_CACHE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)


def collect_device_state(
    bundle_id: str,
    backend: str,
    *,
    bw_d2e_override: float | None = None,
    iperf_duration: float | None = None,
    iperf_timeout: float | None = None,
    d2e_link_id: str | None = None,
    d2e_capacity_mbps: float | None = None,
    dynamic_bandwidth: bool = False,
):
    device = segment_profile_state("device", backend, bundle_id)
    if bw_d2e_override is not None:
        bw_d2e = float(bw_d2e_override)
    elif dynamic_bandwidth:
        cached_bw = _load_cached_bandwidth(device)
        bw_d2e = float(
            TESTBED_CFG.default_bw_d2e if cached_bw is None else cached_bw
        )
        print(
            "Dynamic bandwidth mode: registering with cached/fallback "
            f"BW_d2e={bw_d2e:.4f} Mbps before coordinated calibration"
        )
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
    dynamic_bandwidth: bool = False,
) -> dict:
    payload = {
        "user_id": int(user_id),
        "bundle_id": device["bundle_id"],
        "resource_mode": "fixed_worker_pool",
        "device": device,
        "dynamic_bandwidth": bool(dynamic_bandwidth),
    }
    if decision_mode:
        payload["decision_mode"] = str(decision_mode)
    return payload


def _decision_objective_weights(decision: dict | None) -> tuple[float, float]:
    decision = decision or {}
    alpha = decision.get("objective_alpha")
    beta = decision.get("objective_beta")
    return (
        float(ALGO_CFG.alpha if alpha is None else alpha),
        float(ALGO_CFG.beta if beta is None else beta),
    )


def _heartbeat_loop(client: RoundClient, stop: threading.Event, interval_s: float):
    while not stop.wait(float(interval_s)):
        try:
            client.heartbeat()
        except Exception as exc:
            print(f"Heartbeat failed: {exc}")


def _bandwidth_calibration_loop(
    client: RoundClient,
    estimator: BandwidthEstimator,
    stop: threading.Event,
    *,
    duration_s: float,
    device: dict,
):
    while not stop.wait(0.5):
        try:
            lease = client.acquire_bandwidth_lease()
            if lease.get("status") != "granted":
                continue
            token = str(lease["lease_token"])
            measured = measure_bandwidth_iperf(
                TESTBED_CFG.edge_host,
                TESTBED_CFG.edge_iperf_port,
                duration=duration_s,
                timeout_s=max(float(duration_s) + 5.0, 8.0),
                retries=1,
            )
            if measured is None or float(measured) <= 0.0:
                client.report_bandwidth(
                    {
                        "link": "d2e",
                        "source": "iperf",
                        "status": "failed",
                        "lease_token": token,
                    }
                )
                continue
            sample = estimator.observe(
                float(measured), source="iperf", decision_version=0
            )
            payload = sample.as_dict()
            payload["lease_token"] = token
            client.report_bandwidth(payload)
            _save_cached_bandwidth(device, sample.filtered_bw_mbps)
        except Exception as exc:
            print(f"Dynamic bandwidth calibration failed: {exc}")
            stop.wait(1.0)


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
    objective_alpha: float = float(ALGO_CFG.alpha),
    objective_beta: float = float(ALGO_CFG.beta),
) -> dict:
    record = {
        "request_id": str(result["request_id"]),
        "decision_id": str(result["decision_id"]),
        "decision_version": int(result["decision_version"]),
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
        float(objective_alpha) * float(bool(is_correct))
        - float(objective_beta) * float(result["T_total"])
    )
    record["observed_accuracy"] = float(bool(is_correct))
    record["observed_latency"] = float(result["T_total"])
    record["observed_utility"] = request_utility
    record["objective_alpha"] = float(objective_alpha)
    record["objective_beta"] = float(objective_beta)
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
    sample_selection: dict | None = None,
) -> dict:
    objective_alpha, objective_beta = _decision_objective_weights(decision)
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
        "alpha": objective_alpha,
        "beta": objective_beta,
        "objective_mode": (decision or {}).get("objective_mode", "weighted"),
        "target_accuracy": (decision or {}).get("target_accuracy"),
        "constraint_status": (decision or {}).get("constraint_status", "disabled"),
        "constraint_satisfied": (decision or {}).get("constraint_satisfied"),
        "latency": latency_summary,
        "decision": decision,
        "device_state": device_state,
        "sample_selection": copy.deepcopy(sample_selection),
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


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measurement_is_correct(record: dict) -> bool:
    value = record.get("is_correct", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _print_measurement_summary(
    measurements: list[dict],
    *,
    correct: int,
    total: int,
    decision: dict | None = None,
) -> None:
    width = 78
    print()
    print("=" * width)
    print("DEVICE INFERENCE SUMMARY / 端设备推理汇总")
    print("=" * width)
    if not measurements:
        print("样本数              : 0")
        print("准确率              : 0.00% (0/0)")
        print("=" * width)
        return

    decision = decision or {}
    decision_versions = sorted(
        {
            int(record["decision_version"])
            for record in measurements
            if record.get("decision_version") is not None
        }
    )
    user_decision = decision.get("user") or {}
    decision_id = decision.get("decision_id")

    print(f"样本数              : {total}")
    print(f"准确率              : {100.0 * correct / max(total, 1):.2f}% ({correct}/{total})")
    if decision_id:
        print(f"最终决策            : {decision_id}")
    if decision_versions:
        versions = ", ".join(f"v{version}" for version in decision_versions)
        print(f"本轮使用决策版本    : {versions}")

    b1 = user_decision.get("partition_boundary_1")
    b2 = user_decision.get("partition_boundary_2")
    if b1 is not None and b2 is not None:
        print(f"最终切分点          : b1={int(b1)}, b2={int(b2)}")

    thresholds = user_decision.get("exit_thresholds") or {}
    if thresholds:
        threshold_text = ", ".join(
            f"{exit_id}={float(threshold):.3f}"
            for exit_id, threshold in thresholds.items()
        )
        version_note = (
            f" (最终 v{int(decision['decision_version'])})"
            if decision.get("decision_version") is not None
            else ""
        )
        print(f"早退阈值{version_note:<10}: {threshold_text}")

    grouped: dict[tuple[str, int | None, str], list[dict]] = {}
    for record in measurements:
        exit_id = str(record.get("exit_id") or "unknown")
        boundary_value = record.get("exit_boundary_id")
        boundary = int(boundary_value) if boundary_value is not None else None
        location = str(record.get("exit_location") or "unknown")
        grouped.setdefault((exit_id, boundary, location), []).append(record)

    def _exit_sort_key(
        item: tuple[tuple[str, int | None, str], list[dict]],
    ) -> tuple[int, str, str]:
        (exit_id, boundary, location), _ = item
        return (
            boundary if boundary is not None else 10**9,
            exit_id,
            location,
        )

    print()
    print("-" * width)
    print("退出分布")
    print("-" * width)
    print(
        f"{'Exit point':<24} {'Node':<9} {'Count':>7} "
        f"{'Rate':>9} {'Acc.':>9} {'Avg ms':>10}"
    )
    print("-" * width)
    for (exit_id, boundary, location), records in sorted(
        grouped.items(), key=_exit_sort_key
    ):
        point = f"{exit_id} (b{boundary})" if boundary is not None else exit_id
        count = len(records)
        accuracy = sum(_measurement_is_correct(record) for record in records) / count
        latencies_ms = [
            1000.0 * float(record["T_total"])
            for record in records
            if record.get("T_total") is not None
        ]
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        print(
            f"{point:<24} {location:<9} {count:>7d} "
            f"{100.0 * count / max(total, 1):>8.2f}% "
            f"{100.0 * accuracy:>8.2f}% {avg_latency_ms:>10.3f}"
        )

    early_count = sum(
        1
        for record in measurements
        if str(record.get("exit_id") or "") not in {"", "final"}
    )
    final_count = sum(
        1 for record in measurements if str(record.get("exit_id") or "") == "final"
    )
    print("-" * width)
    print(
        f"早退合计            : {early_count}/{total} "
        f"({100.0 * early_count / max(total, 1):.2f}%)"
    )
    print(
        f"最终出口            : {final_count}/{total} "
        f"({100.0 * final_count / max(total, 1):.2f}%)"
    )

    total_latencies_ms = [
        1000.0 * float(record["T_total"])
        for record in measurements
        if record.get("T_total") is not None
    ]
    if total_latencies_ms:
        mean_ms = sum(total_latencies_ms) / len(total_latencies_ms)
        variance = sum(
            (latency_ms - mean_ms) ** 2 for latency_ms in total_latencies_ms
        ) / len(total_latencies_ms)
        print()
        print("-" * width)
        print("总时延统计 (ms)")
        print("-" * width)
        print(f"累计 / 平均         : {sum(total_latencies_ms) / 1000.0:.3f} s / {mean_ms:.3f} ms")
        print(
            f"P50 / P95 / P99     : {_percentile(total_latencies_ms, 0.50):.3f} / "
            f"{_percentile(total_latencies_ms, 0.95):.3f} / "
            f"{_percentile(total_latencies_ms, 0.99):.3f}"
        )
        print(
            f"最小 / 最大         : {min(total_latencies_ms):.3f} / "
            f"{max(total_latencies_ms):.3f}"
        )
        print(f"标准差              : {variance**0.5:.3f}")

    component_keys = (
        ("Device compute", "T_d_compute"),
        ("Device -> Edge", "T_d2e"),
        ("Edge queue", "edge_queue"),
        ("Edge compute", "T_e_compute"),
        ("Edge -> Cloud", "T_e2c"),
        ("Cloud queue", "cloud_queue"),
        ("Cloud compute", "T_c_compute"),
    )
    component_rows = []
    for label, key in component_keys:
        values = [float(record[key]) for record in measurements if key in record]
        if not values or not any(value != 0.0 for value in values):
            continue
        mean_ms = 1000.0 * sum(values) / len(values)
        component_rows.append((label, mean_ms))
    if component_rows:
        print()
        print("-" * width)
        print("平均阶段时延 (ms)")
        print("-" * width)
        for label, mean_ms in component_rows:
            print(f"{label:<22}: {mean_ms:>10.3f}")

    print("=" * width)


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


def _full_package_name(package_id: str, split: str, mode: str) -> str:
    return f"{package_id}__{split}__{mode}__full"


def _stable_sample_seed(*parts) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _sample_seeds(round_id: str, dataset_id: str, user_id: int, explicit_seed: int | None):
    base_seed = (
        int(explicit_seed)
        if explicit_seed is not None
        else _stable_sample_seed("round", round_id)
    )
    effective_seed = _stable_sample_seed("device", base_seed, dataset_id, int(user_id))
    return base_seed, effective_seed


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
    if args.test_package_full:
        if args.test_package_mode != "balanced":
            raise ValueError("--test-package-full requires --test-package-mode balanced")
        if args.test_package_samples_per_class is not None:
            raise ValueError(
                "--test-package-full cannot be combined with "
                "--test-package-samples-per-class"
            )
        return base / _full_package_name(
            package_id,
            args.test_package_split,
            args.test_package_mode,
        )
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
    parser.add_argument(
        "--test-package-full",
        action="store_true",
        help="Use the dataset-level full test pool exported for per-round sampling.",
    )
    parser.add_argument("--test-package-base")
    parser.add_argument("--test-samples", type=int)
    parser.add_argument(
        "--test-sample-seed",
        type=int,
        help=(
            "Base seed for per-device stratified sampling. If omitted, a stable "
            "seed is derived from --round-id, so a new round selects new samples."
        ),
    )
    parser.add_argument(
        "--test-sampling",
        choices=("stratified", "sequential"),
        default="stratified",
        help="Sample selection policy; stratified is the experiment default.",
    )
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--decision-timeout", type=float, default=90.0)
    parser.add_argument("--dynamic-bandwidth", action="store_true")
    parser.add_argument("--bandwidth-ewma-alpha", type=float, default=0.3)
    parser.add_argument("--bandwidth-change-threshold", type=float, default=0.20)
    parser.add_argument(
        "--bandwidth-min-reschedule-interval", type=float, default=30.0
    )
    parser.add_argument("--bandwidth-stale-after", type=float, default=300.0)
    parser.add_argument("--iperf-calibration-duration", type=float, default=3.0)
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
    if args.test_samples is not None and args.test_samples <= 0:
        parser.error("--test-samples must be positive")
    if args.test_package_samples_per_class is not None and args.test_package_samples_per_class <= 0:
        parser.error("--test-package-samples-per-class must be positive")
    if args.test_package_full and args.test_package_mode != "balanced":
        parser.error("--test-package-full requires --test-package-mode balanced")
    if args.test_package_full and args.test_package_samples_per_class is not None:
        parser.error(
            "--test-package-full cannot be combined with "
            "--test-package-samples-per-class"
        )
    if args.dynamic_bandwidth and args.no_request_barrier:
        parser.error("--dynamic-bandwidth cannot be used with --no-request-barrier")
    if not 0.0 < args.bandwidth_ewma_alpha <= 1.0:
        parser.error("--bandwidth-ewma-alpha must be in (0, 1]")
    if args.bandwidth_change_threshold <= 0.0:
        parser.error("--bandwidth-change-threshold must be positive")
    if args.bandwidth_min_reschedule_interval < 0.0:
        parser.error("--bandwidth-min-reschedule-interval must be non-negative")
    if args.bandwidth_stale_after <= 0.0:
        parser.error("--bandwidth-stale-after must be positive")
    if args.iperf_calibration_duration <= 0.0:
        parser.error("--iperf-calibration-duration must be positive")
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
    if test_package_root is not None and not test_package_root.is_dir():
        raise FileNotFoundError(f"Test package directory not found: {test_package_root}")
    base_sample_seed, effective_sample_seed = _sample_seeds(
        args.round_id,
        bundle.dataset_id,
        args.user_id,
        args.test_sample_seed,
    )
    requested_samples = args.test_samples if args.test_samples is not None else 100
    print(
        "Test sampling: "
        f"policy={args.test_sampling}, requested={requested_samples}, "
        f"base_seed={base_sample_seed}, effective_seed={effective_sample_seed}"
    )
    device = collect_device_state(
        bundle.bundle_id,
        args.backend,
        bw_d2e_override=args.override_bw_d2e,
        iperf_duration=args.iperf_duration,
        iperf_timeout=args.iperf_timeout,
        d2e_link_id=args.d2e_link_id,
        d2e_capacity_mbps=args.d2e_capacity_mbps,
        dynamic_bandwidth=args.dynamic_bandwidth,
    )
    print(f"Device state BW_d2e={float(device['BW_d2e']):.4f} Mbps")
    client = RoundClient(TESTBED_CFG.algo_base_url, args.round_id, args.user_id)
    client.register(
        registration_payload(
            args.user_id,
            device,
            decision_mode=args.decision_mode,
            dynamic_bandwidth=args.dynamic_bandwidth,
        )
    )
    heartbeat_stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(client, heartbeat_stop, args.heartbeat_interval),
        daemon=True,
    )
    heartbeat.start()
    bandwidth_estimator = None
    bandwidth_stop = threading.Event()
    bandwidth_thread = None
    if args.dynamic_bandwidth:
        bandwidth_estimator = BandwidthEstimator(
            link="d2e",
            initial_mbps=float(device["BW_d2e"]),
            alpha=args.bandwidth_ewma_alpha,
            stale_after_s=args.bandwidth_stale_after,
        )
        bandwidth_thread = threading.Thread(
            target=_bandwidth_calibration_loop,
            args=(client, bandwidth_estimator, bandwidth_stop),
            kwargs={
                "duration_s": args.iperf_calibration_duration,
                "device": device,
            },
            daemon=True,
        )
        bandwidth_thread.start()
    correct = total = 0
    measurements = []
    effective_decision_timeout = (
        max(float(args.decision_timeout), 180.0)
        if args.dynamic_bandwidth
        else float(args.decision_timeout)
    )
    try:
        decision = client.wait_for_decision(timeout_s=effective_decision_timeout)
        objective_alpha, objective_beta = _decision_objective_weights(decision)
        user_decision = decision.get("user", {})
        print(
            "Decision summary: "
            f"source={decision.get('decision_source')}, "
            f"objective={decision.get('objective')}, "
            f"constraint={decision.get('constraint_status')}, "
            f"b1={user_decision.get('partition_boundary_1')}, "
            f"b2={user_decision.get('partition_boundary_2')}"
        )
        if test_package_root:
            loader = build_test_package_loader(
                bundle,
                test_package_root,
                batch_size=1,
                sample_count=requested_samples,
                sample_seed=effective_sample_seed,
                stratified_sample=args.test_sampling == "stratified",
            )
            print(f"Loaded test package: {test_package_root}")
        else:
            loader = build_loader(
                bundle,
                "val",
                batch_size=1,
                data_root=args.data_root,
                sample_count=requested_samples,
                sample_seed=effective_sample_seed,
                stratified_sample=args.test_sampling == "stratified",
            )
        sample_limit = len(loader.dataset)
        sample_selection = {
            "policy": args.test_sampling,
            "requested_samples": int(requested_samples),
            "selected_samples": int(sample_limit),
            "base_seed": int(base_sample_seed),
            "effective_seed": int(effective_sample_seed),
            "seed_source": "explicit" if args.test_sample_seed is not None else "round_id",
            "dataset_id": bundle.dataset_id,
            "user_id": int(args.user_id),
            "test_package_root": str(test_package_root) if test_package_root else None,
        }
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
                    sample_index, timeout_s=effective_decision_timeout
                )
                if barrier.get("decision") is not None:
                    decision = barrier["decision"]
                    objective_alpha, objective_beta = _decision_objective_weights(
                        decision
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
                    "T_bandwidth_calibration": float(
                        barrier.get("T_bandwidth_calibration") or 0.0
                    ),
                }
            result = run_partitioned_inference(
                images,
                decision,
                user_id=args.user_id,
                request_id=request_id,
                measure_bandwidth=args.dynamic_bandwidth,
            )
            bandwidth_metrics = result.pop("_bandwidth_sample_d2e", None)
            if bandwidth_metrics is not None and bandwidth_estimator is not None:
                sample = bandwidth_estimator.observe(
                    bandwidth_metrics["bw_mbps"],
                    source="passive",
                    payload_bytes=bandwidth_metrics["payload_bytes"],
                    elapsed_s=bandwidth_metrics["elapsed_s"],
                    decision_version=int(decision["decision_version"]),
                    observed_at=bandwidth_metrics["observed_at"],
                )
                if sample is not None:
                    device["BW_d2e"] = sample.filtered_bw_mbps
                    _save_cached_bandwidth(device, sample.filtered_bw_mbps)
                    try:
                        client.report_bandwidth(sample.as_dict())
                    except Exception as exc:
                        print(f"Dynamic bandwidth report failed: {exc}")
            label = int(labels.item())
            is_correct = result["prediction"] == label
            print(
                f"[sample {sample_index + 1}/{sample_limit or '?'}] "
                f"prediction={result['prediction']} "
                f"label={label} "
                f"correct={bool(is_correct)} "
                f"exit={result.get('exit_location')} "
                f"latency_ms={float(result['T_total']) * 1000:.3f}",
                flush=True,
            )
            correct += int(is_correct)
            measurement = _measurement_record(
                result,
                user_id=args.user_id,
                sample_index=sample_index,
                label=label,
                is_correct=is_correct,
                sample_metadata=sample_metadata,
                synchronization=synchronization,
                objective_alpha=objective_alpha,
                objective_beta=objective_beta,
            )
            measurement["test_sample_seed"] = int(effective_sample_seed)
            measurements.append(measurement)
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
            sample_selection=sample_selection,
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
        bandwidth_stop.set()
        if bandwidth_thread is not None:
            bandwidth_thread.join(timeout=args.iperf_calibration_duration + 2.0)
        heartbeat_stop.set()
        heartbeat.join(timeout=args.heartbeat_interval + 1.0)
    _print_measurement_summary(
        measurements,
        correct=correct,
        total=total,
        decision=decision,
    )


if __name__ == "__main__":
    main()
