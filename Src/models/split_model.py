# models/split_model.py
import torch
import torch.nn as nn
import os


class DemoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),   # 0
            nn.ReLU(),                        # 1
            nn.MaxPool2d(2),                  # 2
            nn.Conv2d(16, 32, 3, padding=1),  # 3
            nn.ReLU(),                        # 4
            nn.MaxPool2d(2),                  # 5
            nn.Conv2d(32, 64, 3, padding=1),  # 6
            nn.ReLU(),                        # 7
            nn.AdaptiveAvgPool2d((1, 1))      # 8
        )
        self.classifier = nn.Linear(64, 10)   # 9

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def split_and_save(save_dir="models/weights"):
    os.makedirs(save_dir, exist_ok=True)
    full = DemoCNN()

    # Device: 0~4 (Conv -> ReLU -> MaxPool -> Conv -> ReLU)
    mu = nn.Sequential(*list(full.features.children())[:5])
    # Edge: 5~7 (MaxPool -> Conv -> ReLU)
    me = nn.Sequential(*list(full.features.children())[5:8])
    # Cloud: 8~9 (AdaptiveAvgPool + Flatten + Linear)
    mc = nn.Sequential(
        list(full.features.children())[8],
        nn.Flatten(),
        full.classifier
    )

    torch.save(mu, f"{save_dir}/mu.pth")      # 直接保存整个模型
    torch.save(me, f"{save_dir}/me.pth")
    torch.save(mc, f"{save_dir}/mc.pth")
    print("模型切片（完整模型）已保存")


if __name__ == "__main__":
    split_and_save()
