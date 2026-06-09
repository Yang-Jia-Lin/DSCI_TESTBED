# deploy/device/early_exit.py
import torch
import torch.nn as nn


class EarlyExitClassifier(nn.Module):
    def __init__(self, in_channels=32, num_classes=10):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def check_early_exit(features, layer_idx, thresholds, exit_classifier):
    """如果当前层是早退层且置信度达到阈值，则返回 True 和结果"""
    if str(layer_idx) in thresholds:
        logits = exit_classifier(features)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        if confidence.item() >= thresholds[str(layer_idx)]:
            return True, {"predicted": predicted.item(), "confidence": confidence.item()}
    return False, None
