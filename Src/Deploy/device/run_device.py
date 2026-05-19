import torch
import torch.nn as nn
import time
import json
import pickle
import requests
import numpy as np
import csv
from torchvision.datasets import CIFAR10
from torchvision import transforms
from Src.deploy.monitor.cpu_monitor import get_cpu_available_cores
from Src.deploy.monitor.bandwidth import measure_bandwidth_iperf
from Src.deploy.monitor.state_reporter import report_status
from Src.deploy.device.comm import send_tensor

# 配置常量
EDGE_IP = "127.0.0.1"
CLOUD_IP = "127.0.0.1"
EDGE_PORT_FEATURE = 9001
IPERF_PORT_EDGE = 5001       # iperf3 边缘测速端口
IPERF_PORT_CLOUD = 5002      # iperf3 云端测速端口
EDGE_STATUS_PORT = 9002      # 边缘状态接口端口
CLOUD_STATUS_PORT = 9003     # 云端状态接口端口
ALGO_URL = "http://127.0.0.1:8000/api/v1/decision"
WEIGHTS_DIR = "Models/Weights"
MU_PATH = f"{WEIGHTS_DIR}/mu.pth"
EXIT1_FC_PATH = f"{WEIGHTS_DIR}/exit1_fc.pth"
EXIT2_FC_PATH = f"{WEIGHTS_DIR}/exit2_fc.pth"
NUM_CLASSES = 10
TEST_SAMPLES = 100               # 测试图片数量，可调
CSV_OUTPUT = "test_results.csv"

# ==================== 辅助函数 ====================


def convert_to_jsonable(obj):
    """将 torch 张量、numpy 标量等转换为可 JSON 序列化的原生 Python 类型"""
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


def load_early_exit_fc(path, in_features, num_classes=NUM_CLASSES):
    fc = nn.Linear(in_features, num_classes)
    fc.load_state_dict(torch.load(path, map_location="cpu"))
    fc.eval()
    return fc

# ==================== 单次推理核心 ====================


def run_single_inference(input_tensor, label, decision, bw_d2e, bw_e2c, cpu_avail):
    """
    对单张图片执行推理，返回结果字典。
    输入:
        input_tensor: [1, 3, 224, 224] 张量
        label: 真实标签 (int)
        decision: 算法返回的决策 JSON
        bw_d2e, bw_e2c: 带宽
        cpu_avail: 设备可用 CPU
    返回:
        result: 包含各种指标的字典
    """
    user = decision["users"][0]
    threshold_57 = user["exit_thresholds"]["57"]
    threshold_103 = user["exit_thresholds"]["103"]

    # 加载模型
    mu = torch.load(MU_PATH, map_location="cpu", weights_only=False).eval()
    ec_device = load_early_exit_fc(
        EXIT1_FC_PATH, 512, NUM_CLASSES)  # 128*4=512
    ec_cloud = load_early_exit_fc(
        EXIT2_FC_PATH, 1024, NUM_CLASSES)  # 256*4=1024

    t_total_start = time.perf_counter()

    # 设备段推理 (0~93)
    with torch.no_grad():
        x2 = mu(input_tensor)      # shape [1, 512, 14, 14]
    T_device = (time.perf_counter() - t_total_start) * 1000  # ms

    # 设备早退判断
    pooled = nn.AdaptiveAvgPool2d((1, 1))(x2)
    flat = torch.flatten(pooled, 1)
    logits_dev = ec_device(flat)
    probs_dev = torch.softmax(logits_dev, dim=1)
    conf_dev, pred_dev = torch.max(probs_dev, dim=1)

    if conf_dev.item() >= threshold_57:
        result = {
            "decision_id": decision["decision_id"],
            "user_id": user["user_id"],
            "T_device": T_device,
            "T_edge": 0.0, "T_cloud": 0.0,
            "T_trans_d2e": 0.0, "T_trans_e2c": 0.0,
            "T_total": T_device,
            "exit_layer": 57,
            "exit_location": "device",
            "exit_confidence": conf_dev.item(),
            "prediction": pred_dev.item(),
            "ground_truth": label,
            "is_correct": pred_dev.item() == label,
            "BW_d2e": bw_d2e,
            "BW_e2c": bw_e2c,
            "cpu_util_device": 0.05,
            "cpu_util_edge": 0.0,
            "cpu_util_cloud": 0.0
        }
        return convert_to_jsonable(result)

    # 未早退，发送特征给边缘
    t_send = time.perf_counter()
    meta = {
        "decision_id": decision["decision_id"],
        "user_id": user["user_id"],
        "exit_thresholds": user["exit_thresholds"],
        "Y_row": user.get("Y_row", []),
        "edge_compute_quota": user.get("edge_compute_quota", 1.0),
        "cloud_compute_quota": user.get("cloud_compute_quota", 1.0)
    }
    payload = {"tensor": x2, "meta": meta}
    try:
        response = send_tensor(payload, EDGE_IP, EDGE_PORT_FEATURE)
    except Exception as e:
        print(f"发送特征到边缘失败: {e}")
        # 返回错误结果
        return {
            "decision_id": decision["decision_id"],
            "user_id": user["user_id"],
            "T_device": T_device,
            "T_edge": 0, "T_cloud": 0,
            "T_trans_d2e": 0, "T_trans_e2c": 0,
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
            "cpu_util_cloud": 0.0
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
        "cpu_util_cloud": 0.0
    }
    return convert_to_jsonable(result)

# ==================== 主函数 ====================


def main():
    print("=== 批量测试启动 ===")

    # 1. 采集一次状态（本地环境稳定，可复用）
    print("采集当前状态...")
    cpu_avail = get_cpu_available_cores()
    bw_d2e = measure_bandwidth_iperf(EDGE_IP, IPERF_PORT_EDGE)
    try:
        edge_status = requests.get(
            f"http://{EDGE_IP}:{EDGE_STATUS_PORT}/status").json()
        cloud_status = requests.get(
            f"http://{CLOUD_IP}:{CLOUD_STATUS_PORT}/status").json()
    except:
        edge_status = {"f_e_max": 4.0, "BW_e2c": 500}
        cloud_status = {"f_c_max": 8.0}

    algo_state = {
        "model_name": "Resnet50",
        "edge": {"f_e_max": edge_status["f_e_max"]},
        "cloud": {"f_c_max": cloud_status["f_c_max"], "BW_e2c": edge_status["BW_e2c"]},
        "users": [{"f_u": cpu_avail, "BW_d2e": bw_d2e}]
    }
    print("采集状态:", algo_state)

    # 2. 获取决策（固定一次，若想每张图动态决策，移入循环内）
    decision = report_status(ALGO_URL, algo_state)
    if not decision:
        print("未获得决策，退出")
        return
    print("收到决策:", json.dumps(decision, indent=2, ensure_ascii=False))

    # 3. 准备数据集
    print(f"加载 CIFAR-10 测试集...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),          # 调整图像大小，如果原本图像为224x224可注释掉
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    testset = CIFAR10(root='Data/CIFAR10', train=False,
                      download=False, transform=transform)
    num_samples = min(TEST_SAMPLES, len(testset))
    print(f"将测试 {num_samples} 张图片")

    # 4. 批量测试
    results = []
    for i in range(num_samples):
        image, label = testset[i]
        input_tensor = image.unsqueeze(0)  # [1, 3, 224, 224]
        print(f"\n[{i+1}/{num_samples}] 处理图片 {i}, 真实标签 {label}")

        try:
            res = run_single_inference(input_tensor, label, decision,
                                       bw_d2e, edge_status["BW_e2c"], cpu_avail)
            results.append(res)
            print(f"  退出层: {res['exit_layer']}, 位置: {res['exit_location']}, "
                  f"置信度: {res['exit_confidence']:.4f}, 预测: {res['prediction']}, "
                  f"正确: {res['is_correct']}, 总时延: {res['T_total']:.2f} ms")
        except Exception as e:
            print(f"  推理失败: {e}")
            # 记录失败条目
            fail_result = {
                "decision_id": decision.get("decision_id", "unknown"),
                "user_id": 0,
                "T_device": 0, "T_edge": 0, "T_cloud": 0,
                "T_trans_d2e": 0, "T_trans_e2c": 0,
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
                "cpu_util_cloud": 0.0
            }
            results.append(fail_result)

    # 5. 保存结果
    print(f"\n所有图片测试完成，正在保存结果到 {CSV_OUTPUT} ...")
    if results:
        keys = results[0].keys()
        with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print("结果已保存。")


if __name__ == "__main__":
    main()
