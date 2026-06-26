"""Fixed-worker Edge runtime for partition-manifest decisions."""

from __future__ import annotations

import argparse
import threading
import time

from flask import Flask, jsonify

from Src.Phase3_Runtime.Device.comm import send_tensor
from Src.Phase3_Runtime.Shared.fixed_worker_pool import FixedWorkerPool, WorkerPoolConfig
from Src.Phase3_Runtime.Shared.mnn_segment_worker import execute_mnn_range, init_mnn_worker
from Src.Phase3_Runtime.Shared.pytorch_segment_worker import (
    execute_pytorch_range,
    init_pytorch_worker,
)
from Src.Phase3_Runtime.Shared.request_identity import request_identity
from Src.Phase3_Runtime.Shared.socket_server import serve_requests
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Profiles.segment_profile import segment_profile_state


def _strip_tensors_if_terminal(result: dict) -> dict:
    if result.get("prediction") is not None:
        result = dict(result)
        result.pop("tensors", None)
    return result


def create_runtime(bundle_id: str, backend: str):
    state = segment_profile_state("edge", backend, bundle_id)
    manifest = load_partition_manifest(state["bundle_id"])
    state["final_boundary_id"] = manifest.final_boundary_id
    config = WorkerPoolConfig(
        worker_count=int(state["worker_count"]),
        threads_per_worker=int(state["threads_per_worker"]),
        max_queue_size=64,
    )
    fn = execute_pytorch_range if backend == "pytorch" else execute_mnn_range
    initializer = init_pytorch_worker if backend == "pytorch" else init_mnn_worker
    pool = FixedWorkerPool(
        config, fn, initializer=initializer, initargs=(state["bundle_id"],)
    )
    return state, pool, manifest


def _validate_edge_request(payload: dict, state: dict, manifest) -> tuple[dict, int, int]:
    if payload.get("bundle_id") != state["bundle_id"]:
        raise ValueError("Request bundle_id does not match Edge runtime")
    if payload.get("manifest_id") != state["manifest_id"]:
        raise ValueError("Request manifest_id does not match Edge runtime")
    if payload.get("model_hash") != state["model_hash"]:
        raise ValueError("Request model_hash does not match Edge runtime")
    meta = payload["meta"]
    b1 = int(payload["boundary_id"])
    b2 = int(meta["partition_boundary_2"])
    manifest.validate_boundary_pair(b1, b2)
    return meta, b1, b2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--backend", choices=("pytorch", "mnn"), default="pytorch")
    parser.add_argument(
        "--override-bw-e2c",
        type=float,
        help="Report this Edge->Cloud bandwidth in Mbps instead of the default.",
    )
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    state, pool, manifest = create_runtime(bundle.bundle_id, args.backend)
    bw_e2c = (
        float(args.override_bw_e2c)
        if args.override_bw_e2c is not None
        else TESTBED_CFG.default_bw_e2c
    )
    status_app = Flask(__name__)

    @status_app.route("/status")
    def status():
        current = {**state, **pool.state()}
        current["BW_e2c"] = bw_e2c
        return jsonify(current)

    def handle(payload):
        identity = request_identity(payload)
        meta, b1, b2 = _validate_edge_request(payload, state, manifest)
        node_started = time.perf_counter()
        result = pool.submit(
            b1, b2, payload["tensors"], meta.get("exit_thresholds", {})
        ).result()
        t_node_edge = time.perf_counter() - node_started
        print(
            f"[edge] local range {b1}->{b2} done, "
            f"prediction={result.get('prediction')}"
        )
        if result.get("prediction") is not None:
            result = _strip_tensors_if_terminal(result)
            return {
                **result,
                **identity,
                "exit_location": "edge",
                "T_compute_edge": float(result.get("T_compute_s", 0.0)),
                "T_node_edge": t_node_edge,
            }
        if b2 == int(state.get("final_boundary_id", -1)):
            result = _strip_tensors_if_terminal(result)
            return {
                **result,
                **identity,
                "exit_location": "edge",
                "T_compute_edge": float(result.get("T_compute_s", 0.0)),
                "T_node_edge": t_node_edge,
            }
        cloud_payload = {
            **identity,
            "bundle_id": state["bundle_id"],
            "manifest_id": state["manifest_id"],
            "model_hash": state["model_hash"],
            "boundary_id": b2,
            "tensors": result["tensors"],
            "meta": meta,
        }
        tx_started = time.perf_counter()
        print(
            f"[edge] forwarding to cloud "
            f"{TESTBED_CFG.cloud_host}:{TESTBED_CFG.cloud_feature_port}, "
            f"boundary={b2}"
        )
        cloud = send_tensor(
            cloud_payload, TESTBED_CFG.cloud_host, TESTBED_CFG.cloud_feature_port
        )
        print("[edge] cloud response received")
        cloud = _strip_tensors_if_terminal(cloud)
        cloud["T_edge_cloud_roundtrip"] = time.perf_counter() - tx_started
        return {
            **cloud,
            **identity,
            "T_compute_edge": float(result.get("T_compute_s", 0.0)),
            "T_node_edge": t_node_edge,
        }

    threading.Thread(
        target=lambda: status_app.run(
            host=TESTBED_CFG.listen_host,
            port=TESTBED_CFG.edge_status_port,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    ).start()
    serve_requests(TESTBED_CFG.listen_host, TESTBED_CFG.edge_feature_port, handle)


if __name__ == "__main__":
    main()
