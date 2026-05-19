# deploy/device/model_device.py
import torch
import torch.nn as nn


class DeviceModel(nn.Module):
    def __init__(self, weight_path="models/weights/mu.pth"):
        super().__init__()
        # 加载整个设备段模型
        self.model = torch.load(
            weight_path, map_location="cpu", weights_only=False)
        self.model.eval()

    def forward(self, x):
        return self.model(x)
