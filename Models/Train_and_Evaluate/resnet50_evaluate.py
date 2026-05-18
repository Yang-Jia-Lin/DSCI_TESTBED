"""
Src/Exp5_EE_Model/Resnet_Train_and_Evaluate/resnet50_evaluate.py
"""

import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50
from Src.Algorithm.Utils.utils_function import get_device, get_test_data_loaders

from Src.paras import DATA_ROOT, WEIGHTS_DIR


def main():
    # 1) device
    device = get_device()
    if device.type == "cpu":
        warnings.filterwarnings("ignore", message=".*pin_memory.*")

    # 2) data
    test_loader = get_test_data_loaders(root=str(DATA_ROOT), batch_size=128)
    blocks_num = [3, 4, 6, 3]  # ResNet‑50
    num_classes = 10  # CIFAR‑10
    include_top = True

    # 3) Model
    model_to_evaluate = MultiEEResNet50(
        block=Bottleneck,
        blocks_num=blocks_num,
        num_classes=num_classes,
        include_top=include_top,
    ).to(device)
    model_path = Path(WEIGHTS_DIR) / "ResNet50_multi_EE_model.pth"

    # 4) Load weights
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model_to_evaluate.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: 找不到模型文件 {model_path}")
        return

    model_to_evaluate.eval()

    # 5) Testing with exit threshold
    threshold = 0.8

    ee1_counter = 0
    ee2_counter = 0
    full_counter = 0

    n_samples = 0  # 总样本数
    correct = 0  # 总正确的样本数
    ee1_correct = 0  # 从ee1退出 & 正确的样本数
    ee2_correct = 0  # 从ee2退出 & 正确的样本数
    full_correct = 0  # 从full退出 & 正确的样本数

    ee1_correct_no_thr = 0  # 直接从ee1退出 正确的样本数
    ee2_correct_no_thr = 0  # 直接从ee2退出 正确的样本数
    full_correct_no_thr = 0  # 直接从full退出 正确的样本数

    print("Starting evaluation...")

    with torch.no_grad():
        for images, true_labels in test_loader:
            # Preparation
            images = images.to(device, non_blocking=True)
            true_labels = true_labels.to(device, non_blocking=True)
            B = images.size(0)

            # Infer
            out_full, out_e1, out_e2 = model_to_evaluate(images, stage=None)
            soft1 = F.softmax(out_e1, dim=1)
            conf1, p1 = soft1.max(dim=1)
            soft2 = F.softmax(out_e2, dim=1)
            conf2, p2 = soft2.max(dim=1)
            _, p3 = out_full.max(dim=1)

            # Counting
            for i in range(B):
                true = true_labels[i].item()
                n_samples += 1
                # with thr
                if conf1[i] >= threshold:
                    ee1_counter += 1
                    pred = p1[i].item()
                    if pred == true:
                        ee1_correct += 1
                        correct += 1
                elif conf2[i] >= threshold:
                    ee2_counter += 1
                    pred = p2[i].item()
                    if pred == true:
                        ee2_correct += 1
                        correct += 1
                else:
                    full_counter += 1
                    pred = p3[i].item()
                    if pred == true:
                        full_correct += 1
                        correct += 1

                # without threshold
                if p1[i].item() == true:
                    ee1_correct_no_thr += 1
                if p2[i].item() == true:
                    ee2_correct_no_thr += 1
                if p3[i].item() == true:
                    full_correct_no_thr += 1

    # 6) Test results
    if n_samples == 0:
        print("没有检测到样本。")
        return

    # Exit probability
    print(f"Full exit rate:\t{full_counter / n_samples * 100:.2f}%")
    print(f"Early exit1 rate:\t{ee1_counter / n_samples * 100:.2f}%")
    print(f"Early exit2 rate:\t{ee2_counter / n_samples * 100:.2f}%\n")

    # Correct
    print(f"Overall Test Accuracy:\t{correct / n_samples * 100:.2f}%")
    # 防止除以 0 的错误
    if full_counter > 0:
        print(f"Full exit Accuracy:\t{full_correct / full_counter * 100:.2f}%")
    if ee1_counter > 0:
        print(f"Early exit1 Accuracy:\t{ee1_correct / ee1_counter * 100:.2f}%")
    if ee2_counter > 0:
        print(f"Early exit2 Accuracy:\t{ee2_correct / ee2_counter * 100:.2f}%\n")

    # Direct Correct
    print(
        f"Full exit Direct Accuracy (no threshold):\t{full_correct_no_thr / n_samples * 100:.2f}%"
    )
    print(
        f"Early exit1 Direct Accuracy (no threshold):\t{ee1_correct_no_thr / n_samples * 100:.2f}%"
    )
    print(
        f"Early exit2 Direct Accuracy (no threshold):\t{ee2_correct_no_thr / n_samples * 100:.2f}%\n"
    )


if __name__ == "__main__":
    main()
