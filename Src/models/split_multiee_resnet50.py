import torch
import torch.nn as nn
import os
from collections import OrderedDict

# 请将你的模型类定义放在这里（或者从外部导入）
# 这里直接复制你提供的 MultiEEResNet50, Bottleneck 等类（略，因为太长，实际使用时从你的模型文件 import）
# 假设你将这些类定义在 models/multiee_resnet.py 中
from Models.ModelNet.Resnet50 import MultiEEResNet50, Bottleneck


def split_and_save(state_dict_path, save_dir="models/weights"):
    # 1. 实例化模型结构（用与训练时完全相同的参数）
    model = MultiEEResNet50(
        block=Bottleneck,
        blocks_num=[3, 4, 6, 3],
        num_classes=10,  # 你的类别数，如果是10就写10
        include_top=True
    )
    # 2. 加载预训练权重
    state_dict = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 切分模型
    # 设备段：到 layer2 结束（包含 fc2 的输入特征）
    mu = nn.Sequential(OrderedDict([
        ("conv1", model.conv1),
        ("bn1", model.bn1),
        ("relu", model.relu),
        ("maxpool", model.maxpool),
        ("layer1", model.layer1),
        ("layer2", model.layer2)
    ]))

    # 边缘段：layer3
    me = nn.Sequential(OrderedDict([
        ("layer3", model.layer3)
    ]))

    # 云端段：layer4 + avgpool + flatten + fc
    mc = nn.Sequential(OrderedDict([
        ("layer4", model.layer4),
        ("avgpool", model.avgpool),
        ("flatten", nn.Flatten(1)),
        ("fc", model.fc)
    ]))

    os.makedirs(save_dir, exist_ok=True)

    # 4. 保存各段模型（整个模型对象）
    torch.save(mu, os.path.join(save_dir, "mu.pth"))
    torch.save(me, os.path.join(save_dir, "me.pth"))
    torch.save(mc, os.path.join(save_dir, "mc.pth"))

    # 5. 保存早退分类器权重（state_dict）
    torch.save(model.fc2.state_dict(), os.path.join(
        save_dir, "exit1_fc.pth"))  # 设备端早退
    torch.save(model.fc3.state_dict(), os.path.join(
        save_dir, "exit2_fc.pth"))  # 边缘端早退

    print("MultiEEResNet50 切片完成！")


if __name__ == "__main__":
    # 替换为你的其中一个 .pth 文件路径
    split_and_save(
        "C:\\Users\\t1960\\Desktop\\Code\\DSCI_TESTBED-main\\Src\\models\\ResNet50_multi_EE_model.pth")
