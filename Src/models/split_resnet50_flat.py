import os
from pathlib import Path

import torch
import torch.nn as nn

# 请确保可以从 Models.ModelNet.Resnet50 导入你的模型类
from Src.Models.ModelNet.Resnet50 import Bottleneck, MultiEEResNet50

BASE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = BASE_DIR / "Data" / "Weights"


def flatten_model(model):
    """递归遍历模型，将所有可执行层按顺序存入列表，并记录每层的类型和名称"""
    layers = []
    global_idx = 0
    name_mapping = {}

    def _add_module(module, prefix=""):
        nonlocal global_idx
        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.BatchNorm2d,
                nn.ReLU,
                nn.MaxPool2d,
                nn.AdaptiveAvgPool2d,
                nn.Linear,
                nn.Flatten,
            ),
        ):
            # 为层分配全局索引
            name_mapping[global_idx] = f"{prefix} ({type(module).__name__})"
            layers.append(module)
            global_idx += 1
        else:
            for child_name, child in module.named_children():
                _add_module(child, f"{prefix}.{child_name}" if prefix else child_name)

    _add_module(model)
    return nn.ModuleList(layers), name_mapping


def main():
    # 1. 创建模型结构并加载预训练权重
    model = MultiEEResNet50(
        block=Bottleneck,
        blocks_num=[3, 4, 6, 3],
        num_classes=10,  # 你的实际类别数
        include_top=True,
    )

    # 你的权重文件路径 (选一个)
    weight_path = WEIGHTS_DIR / "ResNet50_multi_EE_model.pth"  # 请替换为实际路径
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 展平整个模型
    flat_layers, name_mapping = flatten_model(model)
    total_layers = len(flat_layers)
    print(f"模型总层数: {total_layers}")

    # 3. 保存展平层列表（保存整个 ModuleList）
    save_dir = WEIGHTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    torch.save(flat_layers, save_dir / "all_layers.pth")

    # 4. 保存早退分类器权重 (fc2 和 fc3)
    # 注意：fc2 在 layer2 之后，fc3 在 layer3 之后，它们的权重直接提取
    torch.save(model.fc2.state_dict(), save_dir / "exit1_fc.pth")
    torch.save(model.fc3.state_dict(), save_dir / "exit2_fc.pth")

    # 5. 打印层索引与名称对照表（可写入文件）
    with open(save_dir / "layer_mapping.txt", "w") as f:
        for idx, name in sorted(name_mapping.items()):
            f.write(f"{idx}: {name}\n")
    print("展平完成！请查看 layer_mapping.txt 确认早退层对应的索引。")


if __name__ == "__main__":
    main()
