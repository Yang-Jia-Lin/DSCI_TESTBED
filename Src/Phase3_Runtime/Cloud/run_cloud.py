"""Fixed-worker Cloud runtime for partition-manifest decisions."""

from __future__ import annotations

import argparse
import threading
import time

from flask import Flask, jsonify

from Src.Phase3_Runtime.Shared.fixed_worker_pool import FixedWorkerPool, WorkerPoolConfig
from Src.Phase3_Runtime.Shared.segment_worker import (
    execute_mnn_range,
    execute_pytorch_range,
    init_mnn_worker,
    init_pytorch_worker,
)
from Src.Phase3_Runtime.Shared.socket_server import serve_requests
from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Partitioning.manifest import load_partition_manifest
from Src.Shared.Profiles.segment_profile import segment_profile_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--backend", choices=("pytorch", "mnn"), default="pytorch")
    args = parser.parse_args(argv)
    bundle = get_bundle(args.bundle_id)
    state = segment_profile_state("cloud", args.backend, bundle.bundle_id)
    manifest = load_partition_manifest(state["bundle_id"])
    fn = execute_pytorch_range if args.backend == "pytorch" else execute_mnn_range
    initializer = init_pytorch_worker if args.backend == "pytorch" else init_mnn_worker
    pool = FixedWorkerPool(
        WorkerPoolConfig(
            int(state["worker_count"]), int(state["threads_per_worker"]), 64
        ),
        fn,
        initializer=initializer,
        initargs=(state["bundle_id"],),
    )
    status_app = Flask(__name__)

    @status_app.route("/status")
    def status():
        return jsonify({**state, **pool.state()})

    def handle(payload):
        if payload.get("bundle_id") != state["bundle_id"]:
            raise ValueError("Request bundle_id does not match Cloud runtime")
        if payload.get("manifest_id") != state["manifest_id"]:
            raise ValueError("Request manifest_id does not match Cloud runtime")
        if payload.get("model_hash") != state["model_hash"]:
            raise ValueError("Request model_hash does not match Cloud runtime")
        node_started = time.perf_counter()
        result = pool.submit(
            int(payload["boundary_id"]),
            manifest.final_boundary_id,
            payload["tensors"],
            payload.get("meta", {}).get("exit_thresholds", {}),
        ).result()
        return {
            **result,
            "exit_location": "cloud",
            "T_compute_cloud": float(result.get("T_compute_s", 0.0)),
            "T_node_cloud": time.perf_counter() - node_started,
        }

    threading.Thread(
        target=lambda: status_app.run(
            host=TESTBED_CFG.listen_host,
            port=TESTBED_CFG.cloud_status_port,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    ).start()
    serve_requests(TESTBED_CFG.listen_host, TESTBED_CFG.cloud_feature_port, handle)


if __name__ == "__main__":
    main()
