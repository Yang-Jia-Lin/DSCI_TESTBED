"""Generic multi-exit ResNet implementation.

The historical module path is retained to avoid breaking imports while the
model itself is selected exclusively through a model bundle.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from Src.Shared.Config.model_config import ModelBundleSpec


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class MultiExitResNet(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        blocks: Iterable[int],
        *,
        num_classes: int,
        exits: Iterable,
    ):
        super().__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        counts = tuple(blocks)
        self.layer1 = self._make_layer(block, 64, counts[0])
        self.layer2 = self._make_layer(block, 128, counts[1], stride=2)
        self.layer3 = self._make_layer(block, 256, counts[2], stride=2)
        self.layer4 = self._make_layer(block, 512, counts[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        channels = {
            "layer1": 64 * block.expansion,
            "layer2": 128 * block.expansion,
            "layer3": 256 * block.expansion,
            "layer4": 512 * block.expansion,
        }
        self.exit_heads = nn.ModuleDict(
            {
                item.exit_id: nn.Linear(channels[item.attach_point], num_classes)
                for item in exits
            }
        )
        self.exit_attach_points = {
            item.exit_id: item.attach_point for item in exits
        }
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, channels, count, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = [block(self.in_channel, channels, stride, downsample)]
        self.in_channel = channels * block.expansion
        layers.extend(block(self.in_channel, channels) for _ in range(1, count))
        return nn.Sequential(*layers)

    def classify_exit(self, exit_id: str, features: torch.Tensor) -> torch.Tensor:
        return self.exit_heads[exit_id](torch.flatten(self.avgpool(features), 1))

    def forward_features(self, x):
        features = {}
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(self, name)(x)
            features[name] = x
        return features

    def forward(self, x, exit_id: str | None = None):
        features = self.forward_features(x)
        if exit_id is not None:
            return self.classify_exit(exit_id, features[self.exit_attach_points[exit_id]])
        return self.fc(torch.flatten(self.avgpool(features["layer4"]), 1))


def build_model(bundle: ModelBundleSpec) -> MultiExitResNet:
    architectures = {
        "resnet18": (BasicBlock, (2, 2, 2, 2)),
        "resnet50": (Bottleneck, (3, 4, 6, 3)),
    }
    try:
        block, blocks = architectures[bundle.architecture]
    except KeyError as exc:
        raise ValueError(f"Unsupported architecture: {bundle.architecture}") from exc
    return MultiExitResNet(
        block, blocks, num_classes=bundle.num_classes, exits=bundle.exits
    )


def freeze_for_exit(model: MultiExitResNet, exit_id: str | None) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = model.fc if exit_id is None else model.exit_heads[exit_id]
    for parameter in head.parameters():
        parameter.requires_grad = True
