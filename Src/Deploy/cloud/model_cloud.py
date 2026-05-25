from pathlib import Path

import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_DIR = BASE_DIR / "Data" / "Weights"


class CloudModel(nn.Module):
    def __init__(self, weight_path: str | Path = DEFAULT_WEIGHTS_DIR / "mc.pth"):
        super().__init__()
        self.model = torch.load(weight_path, map_location="cpu", weights_only=False)
        self.model.eval()

    def forward(self, x):
        return self.model(x)
