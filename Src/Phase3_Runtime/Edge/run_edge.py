import threading
import time

import torch
from flask import Flask, jsonify

from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Phase3_Runtime.Edge.comm import receive_tensor, send_response
from Src.Phase3_Runtime.Edge.resource_ctrl import get_max_cpu
from Src.Phase3_Runtime.Shared.bandwidth_iperf import measure_bandwidth_iperf
from Src.Phase3_Runtime.Shared.model_loader import load_full_model, threshold_for_stage
from Src.Shared.Profiles.compute_profile import compute_profile_state

# ── 配置 ──────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

status_app = Flask(__name__)


@status_app.route("/status")
def status():
    state = compute_profile_state("edge", "pytorch")
    state["BW_e2c"] = measure_bandwidth_iperf(
        TESTBED_CFG.cloud_host, TESTBED_CFG.cloud_iperf_port
    )
    return jsonify(state)


def run_feature_server():
    model = load_full_model()
    while True:
        payload, conn = receive_tensor(
            TESTBED_CFG.listen_host, TESTBED_CFG.edge_feature_port
        )
        if payload is None:
            if conn:
                conn.close()
            continue

        tensor = payload["tensor"]
        meta = payload["meta"]
        edge_start = int(meta.get("device_end", 2)) + 1
        edge_end = int(meta.get("edge_end", 3))
        exit_layer, threshold = threshold_for_stage(meta["exit_thresholds"], edge_end)

        torch.set_num_threads(
            max(1, int(meta.get("edge_compute_quota", 1.0) * get_max_cpu()))
        )

        t0 = time.perf_counter()
        if edge_start <= edge_end:
            with torch.no_grad():
                features, logits, conf, pred = model.forward_partial(
                    tensor, edge_start, edge_end
                )
        else:
            features, _, conf, pred = tensor, None, None, None
        T_edge = (time.perf_counter() - t0) * 1000

        if conf is not None and (
            edge_end == 4 or (threshold is not None and conf >= threshold)
        ):
            final_resp = {
                "T_edge": T_edge,
                "T_cloud": 0.0,
                "T_trans_e2c": 0.0,
                "exit_layer": exit_layer,
                "exit_location": "edge",
                "exit_confidence": conf,
                "prediction": pred,
            }
            send_response(conn, final_resp)
            conn.close()
            continue

        from Src.Phase3_Runtime.Device.comm import send_tensor as send_to_cloud

        cloud_payload = {"tensor": features, "meta": meta}
        t_fwd = time.perf_counter()
        try:
            cloud_resp = send_to_cloud(
                cloud_payload,
                TESTBED_CFG.cloud_host,
                TESTBED_CFG.cloud_feature_port,
            )
        except Exception as e:
            send_response(conn, {"status": "error", "message": str(e)})
            conn.close()
            continue
        T_trans_e2c = (time.perf_counter() - t_fwd) * 1000

        final_resp = {
            "T_edge": T_edge,
            "T_cloud": cloud_resp.get("T_cloud", 0),
            "T_trans_e2c": T_trans_e2c,
            "exit_layer": cloud_resp.get("exit_layer"),
            "exit_location": cloud_resp.get("exit_location", "cloud"),
            "exit_confidence": cloud_resp.get("exit_confidence"),
            "prediction": cloud_resp.get("prediction", cloud_resp.get("predicted")),
        }
        send_response(conn, final_resp)
        conn.close()


if __name__ == "__main__":
    threading.Thread(
        target=lambda: status_app.run(
            host=TESTBED_CFG.listen_host,
            port=TESTBED_CFG.edge_status_port,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    ).start()
    run_feature_server()
