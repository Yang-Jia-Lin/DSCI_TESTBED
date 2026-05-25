import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from Src.Deploy.Device.comm import send_tensor
from Src.Deploy.monitor.bandwidth import measure_bandwidth_iperf
from Src.Deploy.monitor.cpu_monitor import get_cpu_available_cores
from Src.Deploy.monitor.state_reporter import report_status
from Src.Deploy.shared.model_loader import (
    load_full_model,
    stage_end_from_partition_boundary,
    threshold_for_stage,
)

# 閰嶇疆甯搁噺
EDGE_IP = "127.0.0.1"
CLOUD_IP = "127.0.0.1"
EDGE_PORT_FEATURE = 9001
IPERF_PORT_EDGE = 5001  # iperf3 杈圭紭娴嬮€熺鍙?
IPERF_PORT_CLOUD = 5002  # iperf3 浜戠娴嬮€熺鍙?
EDGE_STATUS_PORT = 9002  # 杈圭紭鐘舵€佹帴鍙ｇ鍙?
CLOUD_STATUS_PORT = 9003  # 浜戠鐘舵€佹帴鍙ｇ鍙?
ALGO_URL = "http://127.0.0.1:8000/api/v1/decision"
TEST_SAMPLES = 100  # Test image count
RESULTS_DIR = Path(__file__).resolve().parent / "Results"
CSV_OUTPUT = RESULTS_DIR / f"test_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

# ==================== 杈呭姪鍑芥暟 ====================


def convert_to_jsonable(obj):
    """灏?torch 寮犻噺銆乶umpy 鏍囬噺绛夎浆鎹负鍙?JSON 搴忓垪鍖栫殑鍘熺敓 Python 绫诲瀷"""
    if isinstance(obj, dict):
        return {k: convert_to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_jsonable(i) for i in obj]
    elif torch.is_tensor(obj) and obj.numel() == 1:
        return obj.item()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def print_summary_statistics(results):
    if not results:
        print("\n=== Test Summary ===")
        print("No test results to summarize.")
        return

    total = len(results)
    valid = [
        r
        for r in results
        if r.get("exit_location") != "error" and r.get("T_total") is not None
    ]
    correct = sum(1 for r in results if bool(r.get("is_correct", False)))
    accuracy = correct / total if total else 0.0

    print("\n=== Test Summary ===")
    print(f"Samples: {total}")
    print(f"Valid inferences: {len(valid)}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

    if not valid:
        print("No valid latency results.")
        return

    def _print_latency_stats(name, key):
        values = np.array([float(r.get(key, 0.0)) for r in valid], dtype=np.float64)
        print(
            f"{name}: mean={values.mean():.2f} ms, "
            f"var={values.var():.2f}, std={values.std():.2f}, "
            f"min={values.min():.2f}, max={values.max():.2f}"
        )

    _print_latency_stats("T_device", "T_device")
    _print_latency_stats("T_trans_d2e", "T_trans_d2e")
    _print_latency_stats("T_edge", "T_edge")
    _print_latency_stats("T_trans_e2c", "T_trans_e2c")
    _print_latency_stats("T_cloud", "T_cloud")
    _print_latency_stats("T_total", "T_total")

    split_points = {}
    layers = {}
    for r in results:
        split_key = (r.get("partition_s1", "unknown"), r.get("partition_s2", "unknown"))
        layer = r.get("exit_layer", "unknown")
        split_points[split_key] = split_points.get(split_key, 0) + 1
        layers[layer] = layers.get(layer, 0) + 1
    print(f"Split decision distribution (p1, p2): {split_points}")
    print(f"Exit layer distribution: {layers}")


# ==================== 鍗曟鎺ㄧ悊鏍稿績 ====================


def run_single_inference(input_tensor, label, decision, bw_d2e, bw_e2c, cpu_avail):
    """
    瀵瑰崟寮犲浘鐗囨墽琛屾帹鐞嗭紝杩斿洖缁撴灉瀛楀吀銆?
    杈撳叆:
        input_tensor: [1, 3, 224, 224] 寮犻噺
        label: 鐪熷疄鏍囩 (int)
        decision: 绠楁硶杩斿洖鐨勫喅绛?JSON
        bw_d2e, bw_e2c: 甯﹀
        cpu_avail: 璁惧鍙敤 CPU
    杩斿洖:
        result: 鍖呭惈鍚勭鎸囨爣鐨勫瓧鍏?
    """
    user = decision["users"][0]
    model = load_full_model()
    exit_thresholds = user["exit_thresholds"]

    # Decision JSON fields from api_server.py/decision_codec.py:
    # partition_s1 is the Edge start layer, partition_s2 is the Cloud start layer.
    device_end = stage_end_from_partition_boundary(user.get("partition_s1"), 2)
    edge_end = stage_end_from_partition_boundary(user.get("partition_s2"), 3)
    exit_layer, threshold = threshold_for_stage(exit_thresholds, device_end)
    decision_info = {
        "partition_s1": user.get("partition_s1"),
        "partition_s2": user.get("partition_s2"),
        "threshold_57": exit_thresholds.get("57"),
        "threshold_103": exit_thresholds.get("103"),
        "decision_source": decision.get("decision_source", "unknown"),
    }

    t_total_start = time.perf_counter()

    # 璁惧娈垫寜 stage 鎵ц銆?
    with torch.no_grad():
        features, logits, conf, pred = model.forward_partial(
            input_tensor, 0, device_end
        )
    T_device = (time.perf_counter() - t_total_start) * 1000  # ms

    # 璁惧鏃╅€€鍒ゆ柇
    if conf is not None and (
        device_end == 4 or (threshold is not None and conf >= threshold)
    ):
        result = {
            "decision_id": decision["decision_id"],
            "user_id": user["user_id"],
            "T_device": T_device,
            "T_edge": 0.0,
            "T_cloud": 0.0,
            "T_trans_d2e": 0.0,
            "T_trans_e2c": 0.0,
            "T_total": T_device,
            "exit_layer": exit_layer,
            "exit_location": "device",
            "exit_confidence": conf,
            "prediction": pred,
            "ground_truth": label,
            "is_correct": pred == label,
            "BW_d2e": bw_d2e,
            "BW_e2c": bw_e2c,
            "cpu_util_device": 0.05,
            "cpu_util_edge": 0.0,
            "cpu_util_cloud": 0.0,
            **decision_info,
        }
        return convert_to_jsonable(result)

    # 鏈棭閫€锛屽彂閫佺壒寰佺粰杈圭紭
    t_send = time.perf_counter()
    meta = {
        "decision_id": decision["decision_id"],
        "user_id": user["user_id"],
        "exit_thresholds": exit_thresholds,
        "Y_row": user.get("Y_row", []),
        "device_end": device_end,
        "edge_end": edge_end,
        "edge_compute_quota": user.get("edge_compute_quota", 1.0),
        "cloud_compute_quota": user.get("cloud_compute_quota", 1.0),
    }
    payload = {"tensor": features, "meta": meta}
    try:
        response = send_tensor(payload, EDGE_IP, EDGE_PORT_FEATURE)
    except Exception as e:
        print(f"鍙戦€佺壒寰佸埌杈圭紭澶辫触: {e}")
        # 杩斿洖閿欒缁撴灉
        return {
            "decision_id": decision["decision_id"],
            "user_id": user["user_id"],
            "T_device": T_device,
            "T_edge": 0,
            "T_cloud": 0,
            "T_trans_d2e": 0,
            "T_trans_e2c": 0,
            "T_total": T_device,
            "exit_layer": -1,
            "exit_location": "error",
            "exit_confidence": 0.0,
            "prediction": -1,
            "ground_truth": label,
            "is_correct": False,
            "BW_d2e": bw_d2e,
            "BW_e2c": bw_e2c,
            "cpu_util_device": 0.05,
            "cpu_util_edge": 0.0,
            "cpu_util_cloud": 0.0,
            **decision_info,
        }

    t_recv = time.perf_counter()
    T_trans_total = (t_recv - t_send) * 1000
    T_edge = response.get("T_edge", 0.0)
    T_cloud = response.get("T_cloud", 0.0)
    T_trans_e2c = response.get("T_trans_e2c", 0.0)
    T_trans_d2e = max(T_trans_total - T_edge - T_cloud - T_trans_e2c, 0.0)

    result = {
        "decision_id": decision["decision_id"],
        "user_id": user["user_id"],
        "T_device": T_device,
        "T_edge": T_edge,
        "T_cloud": T_cloud,
        "T_trans_d2e": T_trans_d2e,
        "T_trans_e2c": T_trans_e2c,
        "T_total": T_device + T_edge + T_cloud + T_trans_d2e + T_trans_e2c,
        "exit_layer": response.get("exit_layer", 128),
        "exit_location": response.get("exit_location", "cloud"),
        "exit_confidence": response.get("exit_confidence", 0.0),
        "prediction": response.get("prediction", -1),
        "ground_truth": label,
        "is_correct": response.get("prediction", -1) == label,
        "BW_d2e": bw_d2e,
        "BW_e2c": bw_e2c,
        "cpu_util_device": 0.05,
        "cpu_util_edge": 0.0,
        "cpu_util_cloud": 0.0,
        **decision_info,
    }
    return convert_to_jsonable(result)


# ==================== 涓诲嚱鏁?====================


def main():
    print("=== Batch test started ===")

    # 1. Collect current state once for this batch.
    print("Collecting current state...")
    cpu_avail = get_cpu_available_cores()
    bw_d2e = measure_bandwidth_iperf(EDGE_IP, IPERF_PORT_EDGE)
    try:
        edge_status = requests.get(f"http://{EDGE_IP}:{EDGE_STATUS_PORT}/status").json()
        cloud_status = requests.get(
            f"http://{CLOUD_IP}:{CLOUD_STATUS_PORT}/status"
        ).json()
    except (requests.RequestException, ValueError, KeyError):
        edge_status = {"f_e_max": 4.0, "BW_e2c": 500}
        cloud_status = {"f_c_max": 8.0}

    algo_state = {
        "model_name": "Resnet50",
        "edge": {"f_e_max": edge_status["f_e_max"]},
        "cloud": {"f_c_max": cloud_status["f_c_max"], "BW_e2c": edge_status["BW_e2c"]},
        "users": [{"f_u": cpu_avail, "BW_d2e": bw_d2e}],
    }
    print("Collected state:", algo_state)

    # 2. Fetch one decision for this batch.
    decision = report_status(ALGO_URL, algo_state)
    if not decision:
        print("No decision received; exiting.")
        return
    print("Received decision:", json.dumps(decision, indent=2, ensure_ascii=False))

    # 3. Prepare dataset.
    print("Loading CIFAR-10 test set...")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # 璋冩暣鍥惧儚澶у皬锛屽鏋滃師鏈浘鍍忎负224x224鍙敞閲婃帀
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = CIFAR10(
        root="Data/CIFAR10", train=False, download=False, transform=transform
    )
    num_samples = min(TEST_SAMPLES, len(testset))
    print(f"Testing {num_samples} images.")

    # 4. Batch inference.
    results = []
    for i in range(num_samples):
        image, label = testset[i]
        input_tensor = image.unsqueeze(0)  # [1, 3, 224, 224]
        print(f"\n[{i + 1}/{num_samples}] image={i}, label={label}")

        try:
            res = run_single_inference(
                input_tensor, label, decision, bw_d2e, edge_status["BW_e2c"], cpu_avail
            )
            res = convert_to_jsonable(res)
            if not isinstance(res, dict):
                raise TypeError(
                    f"Unexpected inference result type: {type(res).__name__}"
                )
            results.append(res)
            print(
                f"  decision: p1={res.get('partition_s1', 'N/A')}, p2={res.get('partition_s2', 'N/A')}, "
                f"thr57={res.get('threshold_57', 'N/A')}, thr103={res.get('threshold_103', 'N/A')}, "
                f"exit_layer={res.get('exit_layer', 'N/A')}, location={res.get('exit_location', 'N/A')}, "
                f"confidence={res.get('exit_confidence', 0.0):.4f}, prediction={res.get('prediction', -1)}, "
                f"correct={res.get('is_correct', False)}, total={res.get('T_total', 0.0):.2f} ms"
            )
        except Exception as e:
            print(f"  inference failed: {e}")
            # Record a failed inference row.
            fail_result = {
                "decision_id": decision.get("decision_id", "unknown"),
                "user_id": 0,
                "T_device": 0,
                "T_edge": 0,
                "T_cloud": 0,
                "T_trans_d2e": 0,
                "T_trans_e2c": 0,
                "T_total": 0,
                "exit_layer": -1,
                "exit_location": "error",
                "exit_confidence": 0.0,
                "prediction": -1,
                "ground_truth": label,
                "is_correct": False,
                "BW_d2e": bw_d2e,
                "BW_e2c": edge_status["BW_e2c"],
                "cpu_util_device": 0.05,
                "cpu_util_edge": 0.0,
                "cpu_util_cloud": 0.0,
                "partition_s1": decision.get("users", [{}])[0].get("partition_s1"),
                "partition_s2": decision.get("users", [{}])[0].get("partition_s2"),
                "threshold_57": decision.get("users", [{}])[0]
                .get("exit_thresholds", {})
                .get("57"),
                "threshold_103": decision.get("users", [{}])[0]
                .get("exit_thresholds", {})
                .get("103"),
                "decision_source": decision.get("decision_source", "unknown"),
            }
            results.append(fail_result)

    # 5. Save results.
    print(f"\nAll images processed; saving results to {CSV_OUTPUT} ...")
    if results:
        keys = results[0].keys()
        CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print("Results saved.")
    print_summary_statistics(results)


if __name__ == "__main__":
    main()
