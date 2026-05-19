import torch
import torch.nn as nn


class EdgeModel(nn.Module):
    def __init__(self, weight_path="models/weights/me.pth"):
        super().__init__()
        self.model = torch.load(
            weight_path, map_location="cpu", weights_only=False)
        self.model.eval()

    def forward(self, x):
        return self.model(x)
