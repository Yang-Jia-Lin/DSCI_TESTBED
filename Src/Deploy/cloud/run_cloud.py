import threading
import torch
import torch.nn as nn
import time
import pickle
from flask import Flask, jsonify
from Src.deploy.cloud.comm import receive_tensor
from Src.deploy.cloud.resource_ctrl import get_max_cpu

status_app = Flask(__name__)


@status_app.route('/status')
def status():
    return jsonify({"f_c_max": get_max_cpu()})


def run_feature_server():
    mc = torch.load("Models/Weights/mc.pth",
                    map_location="cpu", weights_only=False).eval()
    ec_cloud = nn.Linear(1024, 10)  # 256*4 = 1024
    ec_cloud.load_state_dict(torch.load(
        "Models/Weights/exit2_fc.pth", map_location="cpu"))
    ec_cloud.eval()

    while True:
        payload, conn = receive_tensor("0.0.0.0", 9004)
        if payload is None:
            if conn:
                conn.close()
            continue
        x3 = payload["tensor"]
        meta = payload["meta"]
        threshold_103 = meta["exit_thresholds"]["103"]

        torch.set_num_threads(
            max(1, int(meta.get("cloud_compute_quota", 1.0) * get_max_cpu())))

        t_start = time.perf_counter()
        # 先尝试早退（在 x3 上应用 fc3）
        with torch.no_grad():
            pooled = nn.AdaptiveAvgPool2d((1, 1))(x3)
            flat = torch.flatten(pooled, 1)
            logits_early = ec_cloud(flat)
            probs_early = torch.softmax(logits_early, dim=1)
            conf_early, pred_early = torch.max(probs_early, dim=1)

        if conf_early.item() >= threshold_103:
            T_cloud = (time.perf_counter() - t_start) * 1000
            result = {
                "exit_layer": 103,
                "exit_location": "cloud",
                "exit_confidence": conf_early.item(),
                "predicted": pred_early.item(),
                "T_cloud": T_cloud
            }
        else:
            # 执行剩余层 (layer4 + avgpool + fc)
            with torch.no_grad():
                x_final = mc(x3)
                probs_final = torch.softmax(x_final, dim=1)
                conf_final, pred_final = torch.max(probs_final, dim=1)
            T_cloud = (time.perf_counter() - t_start) * 1000
            result = {
                "exit_layer": 128,   # 最后一层
                "exit_location": "cloud",
                "exit_confidence": conf_final.item(),
                "predicted": pred_final.item(),
                "T_cloud": T_cloud
            }
        send_data = pickle.dumps(result)
        try:
            conn.sendall(send_data)
        finally:
            conn.close()


if __name__ == "__main__":
    threading.Thread(target=lambda: status_app.run(
        host="0.0.0.0", port=9003, debug=False, use_reloader=False), daemon=True).start()
    run_feature_server()
