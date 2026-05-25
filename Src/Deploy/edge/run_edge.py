import threading
import time
from pathlib import Path

import torch
from flask import Flask, jsonify

from Src.Deploy.edge.comm import receive_tensor, send_response
from Src.Deploy.edge.resource_ctrl import get_max_cpu
from Src.Deploy.monitor.bandwidth import measure_bandwidth_iperf

status_app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = BASE_DIR / "Data" / "Weights"


@status_app.route("/status")
def status():
    return jsonify(
        {"f_e_max": get_max_cpu(), "BW_e2c": measure_bandwidth_iperf("127.0.0.1", 5002)}
    )


def run_feature_server():
    me = torch.load(
        WEIGHTS_DIR / "me.pth", map_location="cpu", weights_only=False
    ).eval()
    while True:
        payload, conn = receive_tensor("0.0.0.0", 9001)
        if payload is None:
            if conn:
                conn.close()
            continue
        tensor = payload["tensor"]  # x2
        meta = payload["meta"]

        torch.set_num_threads(
            max(1, int(meta.get("edge_compute_quota", 1.0) * get_max_cpu()))
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            x3 = me(tensor)  # layer3 输出
        T_edge = (time.perf_counter() - t0) * 1000

        # 转发给云
        from Src.Deploy.device.comm import send_tensor as send_to_cloud

        cloud_payload = {"tensor": x3, "meta": meta}
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
            "prediction": cloud_resp.get("predicted"),
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
