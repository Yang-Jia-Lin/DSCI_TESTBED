import pickle
import threading
import time

import torch
from flask import Flask, jsonify

from Src.Deploy.Cloud.comm import receive_tensor
from Src.Deploy.Cloud.resource_ctrl import get_max_cpu
from Src.Deploy.shared.model_loader import load_full_model

status_app = Flask(__name__)


@status_app.route("/status")
def status():
    return jsonify({"f_c_max": get_max_cpu()})


def run_feature_server():
    model = load_full_model()

    while True:
        payload, conn = receive_tensor("0.0.0.0", 32266)
        if payload is None:
            if conn:
                conn.close()
            continue

        tensor = payload["tensor"]
        meta = payload["meta"]
        cloud_start = int(meta.get("edge_end", 3)) + 1

        torch.set_num_threads(
            max(1, int(meta.get("cloud_compute_quota", 1.0) * get_max_cpu()))
        )

        t_start = time.perf_counter()
        with torch.no_grad():
            features, logits, conf, pred = model.forward_partial(tensor, cloud_start, 4)
        T_cloud = (time.perf_counter() - t_start) * 1000

        result = {
            "exit_layer": 128,
            "exit_location": "cloud",
            "exit_confidence": conf,
            "prediction": pred,
            "predicted": pred,
            "T_cloud": T_cloud,
        }
        send_data = pickle.dumps(result)
        try:
            conn.sendall(send_data)
        finally:
            conn.close()


if __name__ == "__main__":
    threading.Thread(
        target=lambda: status_app.run(
            host="0.0.0.0", port=32265, debug=False, use_reloader=False
        ),
        daemon=True,
    ).start()
    run_feature_server()
