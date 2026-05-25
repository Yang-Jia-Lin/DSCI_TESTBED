"""
Src/Exp5_EE_Model/Resnet_Train_and_Evaluate/resnet50_thred_curve.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Src.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50
from Src.Algorithm.Utils.log_function import save_thr_data
from Src.Algorithm.Utils.utils_function import get_device, get_test_data_loaders
from Src.Configs.paras import DATA_ROOT, OFFLINE_TABLE_DIR, WEIGHTS_DIR


def evaluate_model(device, test_loader, model, threshold):
    # 1) Initialize
    ee1_counter = 0
    ee2_counter = 0
    # full_counter = 0

    n_samples = 0  # 总样本数
    correct = 0  # 总正确的样本数
    ee1_correct = 0  # 从ee1退出 & 正确的样本数
    ee2_correct = 0  # 从ee2退出 & 正确的样本数
    # full_correct = 0  # 从full退出 & 正确的样本数

    # 2) Testing with exit threshold
    with torch.no_grad():
        for images, true_labels in test_loader:
            # Preparation
            images = images.to(device, non_blocking=True)
            true_labels = true_labels.to(device, non_blocking=True)
            B = images.size(0)
            # Infer
            out_full, out_e1, out_e2 = model(images, stage=None)
            soft1 = F.softmax(out_e1, dim=1)
            conf1, p1 = soft1.max(dim=1)
            soft2 = F.softmax(out_e2, dim=1)
            conf2, p2 = soft2.max(dim=1)
            _, p3 = out_full.max(dim=1)

            # Counting
            for i in range(B):
                true = true_labels[i].item()
                n_samples += 1
                # with thr rate
                if conf1[i] >= threshold:
                    ee1_counter += 1
                    if p1[i].item() == true:
                        ee1_correct += 1
                if conf2[i] >= threshold:
                    ee2_counter += 1
                    if p2[i].item() == true:
                        ee2_correct += 1

                # 总体准确率逻辑
                if conf1[i] >= threshold:  # 注意这里建议和上面逻辑保持一致用 >=
                    if p1[i].item() == true:
                        correct += 1
                elif conf2[i] >= threshold:
                    if p2[i].item() == true:
                        correct += 1
                else:
                    if p3[i].item() == true:
                        correct += 1

    # 3) Results
    exit1_rate = 100.0 * ee1_counter / n_samples
    exit2_rate = 100.0 * ee2_counter / n_samples
    exit1_acc = (100.0 * ee1_correct / ee1_counter) if ee1_counter > 0 else 0
    exit2_acc = (100.0 * ee2_correct / ee2_counter) if ee2_counter > 0 else 0
    acc = 100.0 * correct / n_samples
    return (exit1_rate, exit2_rate), (exit1_acc, exit2_acc, acc)


def main():
    # 1) Parameters
    device = get_device()

    # 修复 pin_memory 警告
    if device.type == "cpu":
        warnings.filterwarnings("ignore", message=".*pin_memory.*")

    test_loader = get_test_data_loaders(root=str(DATA_ROOT), batch_size=128)
    blocks_num = [3, 4, 6, 3]  # ResNet‑50
    num_classes = 10  # CIFAR‑10
    include_top = True

    # 2) Load model and weights
    model_name = "ResNet50_multi_EE"
    model = MultiEEResNet50(
        block=Bottleneck,
        blocks_num=blocks_num,
        num_classes=num_classes,
        include_top=include_top,
    ).to(device)

    model_path = Path(WEIGHTS_DIR) / "ResNet50_multi_EE_model.pth"

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: 找不到权重文件 {model_path}")
        return

    model.eval()

    # 3) Initialization
    thresholds = np.linspace(0.0, 1.0, 100)
    exit1_rates = []
    exit2_rates = []
    exit1_accs = []
    exit2_accs = []
    full_accs = []

    # 4) Evaluation loop
    print(f"Evaluating {model_name} on {len(thresholds)} thresholds...")

    # 使用 tqdm 包装循环，否则在 CPU 上你可能以为它死机了
    for thr in tqdm(thresholds, desc="Threshold Progress"):
        (r1, r2), (a1, a2, a3) = evaluate_model(
            device=device, test_loader=test_loader, model=model, threshold=thr
        )
        exit1_rates.append(r1)
        exit2_rates.append(r2)
        exit1_accs.append(a1)
        exit2_accs.append(a2)
        full_accs.append(a3)

    # 5) Save Data
    rates = pd.DataFrame(
        {"threshold": thresholds, "exit1_rate": exit1_rates, "exit2_rate": exit2_rates}
    )
    accs = pd.DataFrame(
        {
            "threshold": thresholds,
            "exit1_accuracy": exit1_accs,
            "exit2_accuracy": exit2_accs,
            "accuracy": full_accs,
        }
    )

    rates_saved_path = save_thr_data(rates, "rates", Path(OFFLINE_TABLE_DIR))
    accs_saved_path = save_thr_data(accs, "accs", Path(OFFLINE_TABLE_DIR))
    print(f"\nSaved CSV: {rates_saved_path} and {accs_saved_path}\n")


if __name__ == "__main__":
    main()
