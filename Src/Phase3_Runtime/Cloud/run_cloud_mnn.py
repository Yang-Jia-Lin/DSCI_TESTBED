import pickle
import threading
import time

import torch
from flask import Flask, jsonify

from Src.Shared.Config.deploy_config import DEFAULT as TESTBED_CFG
from Src.Phase3_Runtime.Cloud.comm import receive_tensor
from Src.Phase3_Runtime.Cloud.resource_ctrl import get_max_cpu
# 修改：导入云端 MNN 模型加载器
from Src.Phase3_Runtime.Shared.model_loader_mnn_cloud import load_cloud_model
from Src.Shared.Profiles.compute_profile import compute_profile_state

status_app = Flask(__name__)


@status_app.route("/status")
def status():
    return jsonify(compute_profile_state("cloud", "mnn"))


def run_feature_server():
    cloud_model = load_cloud_model()      # 加载 MNN 云端模型

    while True:
        payload, conn = receive_tensor(
            TESTBED_CFG.listen_host, TESTBED_CFG.cloud_feature_port
        )
        if payload is None:
            if conn:
                conn.close()
            continue

        tensor = payload["tensor"]        # 边缘传来的特征
        meta = payload["meta"]
        # 云端始终从 stage4 开始，但我们的云端模型直接接收 stage3 特征，不需要 start 参数
        # 原来的 torch 代码中 cloud_start = int(meta.get("edge_end", 3)) + 1，即 4
        # 但 MNN 模型直接输出最终结果，不需要部分执行

        torch.set_num_threads(
            max(1, int(meta.get("cloud_compute_quota", 1.0) * get_max_cpu()))
        )

        t_start = time.perf_counter()
        # 云端推理
        logits, conf, pred = cloud_model.forward(tensor)
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
            if conn is not None:
                conn.sendall(send_data)
        finally:
            if conn is not None:
                conn.close()


if __name__ == "__main__":
    threading.Thread(
        target=lambda: status_app.run(
            host=TESTBED_CFG.listen_host,
            port=TESTBED_CFG.cloud_status_port,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    ).start()
    run_feature_server()
