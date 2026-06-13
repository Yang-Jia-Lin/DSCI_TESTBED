"""Bundle-aware Device runner for fixed-worker partition decisions."""

import argparse
import json

import requests

from Src.Phase3_Runtime.Device.runtime_v2 import run_partitioned_inference
from Src.Phase3_Runtime.Shared.bandwidth_iperf import measure_bandwidth_iperf
from Src.Phase3_Runtime.Shared.state_reporter import report_status
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Data.registry import build_loader
from Src.Shared.Profiles.segment_profile import segment_profile_state


def collect_state(bundle_id: str, backend: str):
    device = segment_profile_state("device", backend, bundle_id)
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
        "users": [{
            **device,
            "BW_d2e": measure_bandwidth_iperf(TESTBED_CFG.edge_host, TESTBED_CFG.edge_iperf_port),
        }],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--backend", choices=("pytorch", "mnn"), default="pytorch")
    parser.add_argument("--data-root")
    parser.add_argument("--test-samples", type=int, default=100)
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    state = collect_state(bundle.bundle_id, args.backend)
    decision = report_status(TESTBED_CFG.algo_decision_url, state)
    if not decision:
        raise RuntimeError("Scheduler returned no decision")
    print(json.dumps(decision, indent=2))
    loader = build_loader(bundle, "val", batch_size=1, data_root=args.data_root)
    correct = total = 0
    for images, labels in loader:
        result = run_partitioned_inference(images, decision)
        correct += int(result["prediction"] == int(labels.item()))
        total += 1
        if total >= args.test_samples:
            break
    print(f"samples={total} accuracy={correct / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
