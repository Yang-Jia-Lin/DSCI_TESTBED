import csv
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F

# from torchvision import transforms
# from torchvision.datasets import CIFAR10
from Src.Deploy.Device.comm import send_tensor
from Src.Deploy.deploy_config import DEFAULT as TESTBED_CFG
from Src.Deploy.Shared.bandwidth_iperf import measure_bandwidth_iperf
from Src.compute_profile import compute_profile_state
from Src.Deploy.Shared.state_reporter import report_status
from Src.Deploy.Shared.model_loader_mnn import (
    load_full_model,
    stage_end_from_partition_boundary,
    threshold_for_stage,
)

TEST_SAMPLES = 100
RESULTS_DIR = Path(__file__).resolve().parent / "Results"
CSV_OUTPUT = RESULTS_DIR / \
    f"test_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

# ==================== 辅助函数 ====================


def convert_to_jsonable(obj):
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


def normalize(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        1, -1, 1, 1
    )
    std = torch.tensor(std, dtype=tensor.dtype,
                       device=tensor.device).view(1, -1, 1, 1)
    return (tensor - mean) / std


def cifar10_test_loader(data_dir="Data/CIFAR10/cifar-10-batches-py"):
    """逐张返回测试图片和标签，图片形状 (1, 3, 224, 224)，已归一化"""

    def load_cifar10_batch(file_path):
        with open(file_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        images = batch[b"data"].reshape(-1, 3,
                                        32, 32).astype(np.float32) / 255.0
        labels = np.array(batch[b"labels"])
        return images, labels

    images, labels = load_cifar10_batch(f"{data_dir}/test_batch")
    for img, label in zip(images, labels):
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, 32, 32)
        tensor = F.interpolate(
            tensor, size=(224, 224), mode="bilinear", align_corners=False
        )
        tensor = normalize(tensor, [0.4914, 0.4822, 0.4465], [
                           0.2023, 0.1994, 0.2010])
        yield tensor, label


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
        values = np.array([float(r.get(key, 0.0))
                          for r in valid], dtype=np.float64)
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
        split_key = (r.get("partition_s1", "unknown"),
                     r.get("partition_s2", "unknown"))
        layer = r.get("exit_layer", "unknown")
        split_points[split_key] = split_points.get(split_key, 0) + 1
        layers[layer] = layers.get(layer, 0) + 1
    print(f"Split decision distribution (p1, p2): {split_points}")
    print(f"Exit layer distribution: {layers}")


# ==================== 单次推理核心 ====================


def run_single_inference(input_tensor, label, decision, bw_d2e, bw_e2c, cpu_avail):
    user = decision["users"][0]
    model = load_full_model()
    exit_thresholds = user["exit_thresholds"]

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

    with torch.no_grad():
        features, logits, conf, pred = model.forward_partial(
            input_tensor, 0, device_end
        )
    T_device = (time.perf_counter() - t_total_start) * 1000

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
        response = send_tensor(
            payload, TESTBED_CFG.edge_host, TESTBED_CFG.edge_feature_port
        )
    except Exception as e:
        print(f"发送特征到边缘失败: {e}")
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


# ==================== 主函数 ====================


def main():
    print("=== Batch test started ===")

    print("Collecting current state...")
    device_status = compute_profile_state("device", "mnn")
    bw_d2e = measure_bandwidth_iperf(
        TESTBED_CFG.edge_host, TESTBED_CFG.edge_iperf_port
    )
    edge_response = requests.get(
        f"http://{TESTBED_CFG.edge_host}:{TESTBED_CFG.edge_status_port}/status"
    )
    edge_response.raise_for_status()
    edge_status = edge_response.json()
    cloud_response = requests.get(
        f"http://{TESTBED_CFG.cloud_host}:{TESTBED_CFG.cloud_status_port}/status"
    )
    cloud_response.raise_for_status()
    cloud_status = cloud_response.json()

    algo_state = {
        "model_name": "Resnet50",
        "edge": {
            "f_e_max": edge_status["f_e_max"],
            "compute_profile_id": edge_status["compute_profile_id"],
        },
        "cloud": {
            "f_c_max": cloud_status["f_c_max"],
            "compute_profile_id": cloud_status["compute_profile_id"],
            "BW_e2c": edge_status["BW_e2c"],
        },
        "users": [{**device_status, "BW_d2e": bw_d2e}],
    }
    print("Collected state:", algo_state)

    decision = report_status(TESTBED_CFG.algo_decision_url, algo_state)
    if not decision:
        print("No decision received; exiting.")
        return
    print("Received decision:", json.dumps(
        decision, indent=2, ensure_ascii=False))

    print("Loading CIFAR-10 test set...")
    test_loader = cifar10_test_loader("Data/CIFAR10/cifar-10-batches-py")
    print(f"Testing {TEST_SAMPLES} images.")

    results = []
    for i in range(TEST_SAMPLES):
        try:
            image, label = next(test_loader)
        except StopIteration:
            print(f"数据集已用完，实际测试了 {i} 张")
            break

        input_tensor = image  # 已经是 (1, 3, 224, 224)
        print(f"\n[{i + 1}/{TEST_SAMPLES}] image={i}, label={label}")

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
