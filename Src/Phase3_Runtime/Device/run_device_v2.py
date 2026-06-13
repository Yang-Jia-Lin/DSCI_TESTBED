"""Device batch runner for partition-manifest and fixed-worker decisions."""

from __future__ import annotations

import csv
import json

import requests

from Src.Phase3_Runtime.Device.run_device import (
    CSV_OUTPUT,
    cifar10_test_loader,
    convert_to_jsonable,
    parse_args,
    print_summary_statistics,
    resolve_difficulty_table,
    unpack_loader_sample,
)
from Src.Phase3_Runtime.Device.runtime_v2 import run_partitioned_inference
from Src.Phase3_Runtime.Shared.bandwidth_iperf import measure_bandwidth_iperf
from Src.Phase3_Runtime.Shared.state_reporter import report_status
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Profiles.segment_profile import segment_profile_state


def collect_state(backend: str = "pytorch") -> tuple[dict, dict, dict]:
    device = segment_profile_state("device", backend)
    bw_d2e = measure_bandwidth_iperf(
        TESTBED_CFG.edge_host, TESTBED_CFG.edge_iperf_port
    )
    edge_response = requests.get(
        f"http://{TESTBED_CFG.edge_host}:{TESTBED_CFG.edge_status_port}/status",
        timeout=10,
    )
    edge_response.raise_for_status()
    edge = edge_response.json()
    cloud_response = requests.get(
        f"http://{TESTBED_CFG.cloud_host}:{TESTBED_CFG.cloud_status_port}/status",
        timeout=10,
    )
    cloud_response.raise_for_status()
    cloud = cloud_response.json()
    state = {
        "model_name": "Resnet50",
        "resource_mode": "fixed_worker_pool",
        "edge": edge,
        "cloud": {**cloud, "BW_e2c": edge["BW_e2c"]},
        "users": [{**device, "BW_d2e": bw_d2e}],
    }
    return state, edge, cloud


def main(argv=None):
    args = parse_args(argv)
    difficulty_table = resolve_difficulty_table(args)
    state, edge, _cloud = collect_state()
    decision = report_status(TESTBED_CFG.algo_decision_url, state)
    if not decision:
        raise RuntimeError("Scheduler returned no decision")
    if decision.get("resource_mode") != "fixed_worker_pool":
        raise ValueError("run_device_v2 only accepts fixed_worker_pool decisions")
    print("Received decision:", json.dumps(decision, indent=2, ensure_ascii=False))

    loader = cifar10_test_loader(
        args.data_root,
        difficulty_table_path=difficulty_table,
        difficulty=args.difficulty,
        include_difficulty_metadata=(difficulty_table is not None),
        include_image_id=True,
    )
    results = []
    for _ in range(args.test_samples):
        try:
            image, label, metadata = unpack_loader_sample(next(loader))
        except StopIteration:
            break
        result = convert_to_jsonable(run_partitioned_inference(image, decision))
        result.update(metadata)
        result.update(
            {
                "ground_truth": label,
                "is_correct": int(result.get("prediction", -1)) == label,
                "partition_s1": decision["users"][0]["partition_boundary_1"],
                "partition_s2": decision["users"][0]["partition_boundary_2"],
                "T_device": float(result.get("T_compute_device", 0.0)) * 1000,
                "T_edge": float(result.get("T_node_edge", 0.0)) * 1000,
                "T_cloud": float(result.get("T_node_cloud", 0.0)) * 1000,
                "T_total": float(result.get("T_total", 0.0)) * 1000,
                "BW_d2e": state["users"][0]["BW_d2e"],
                "BW_e2c": edge["BW_e2c"],
            }
        )
        results.append(result)

    if results:
        keys = list(dict.fromkeys(key for row in results for key in row))
        CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    print_summary_statistics(results)


if __name__ == "__main__":
    main()
