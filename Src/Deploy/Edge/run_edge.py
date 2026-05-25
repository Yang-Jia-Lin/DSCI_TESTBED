import threading
import time

import torch
from flask import Flask, jsonify

from Src.Deploy.edge.comm import receive_tensor, send_response
from Src.Deploy.edge.resource_ctrl import get_max_cpu
from Src.Deploy.monitor.bandwidth import measure_bandwidth_iperf
from Src.Deploy.shared.model_loader import load_full_model, threshold_for_stage

status_app = Flask(__name__)


@status_app.route("/status")
def status():
    return jsonify(
        {"f_e_max": get_max_cpu(), "BW_e2c": measure_bandwidth_iperf("127.0.0.1", 5002)}
    )


def run_feature_server():
    model = load_full_model()
    while True:
        payload, conn = receive_tensor("0.0.0.0", 9001)
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

        from Src.Deploy.device.comm import send_tensor as send_to_cloud

        cloud_payload = {"tensor": features, "meta": meta}
        t_fwd = time.perf_counter()
        try:
            cloud_resp = send_to_cloud(cloud_payload, "127.0.0.1", 9004)
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
            host="0.0.0.0", port=9002, debug=False, use_reloader=False
        ),
        daemon=True,
    ).start()
    run_feature_server()
