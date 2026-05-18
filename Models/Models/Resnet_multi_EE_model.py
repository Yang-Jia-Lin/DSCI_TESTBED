import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResnetMultiEEModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)  # 56x56
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 28x28
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 14x14
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 7x7 (exit1)
        self.layer5 = self._make_layer(512, 512, 2, stride=1)
        self.layer6 = self._make_layer(512, 512, 2, stride=1)  # exit2

        # Exit networks
        self.exit1_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes))

        self.exit2_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes))

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes))

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride):
        layers = [_Block(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(_Block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        exit1 = self.exit1_fc(x)

        x = self.layer5(x)
        x = self.layer6(x)
        exit2 = self.exit2_fc(x)

        full = self.fc(x)
        return exit1, exit2, full