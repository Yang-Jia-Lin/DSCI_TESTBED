import torch.nn as nn
import torch.nn.functional as F

class AlexnetMultiEEModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # main CNN model
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 0), nn.BatchNorm2d(96),
            nn.ReLU(),                 nn.MaxPool2d(3, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2), nn.BatchNorm2d(256),
            nn.ReLU(),                   nn.MaxPool2d(3, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1), nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1), nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1), nn.BatchNorm2d(256),
            nn.ReLU(),                    nn.MaxPool2d(3, 2))

        # exit_early_1
        self.exit1_fc1 = nn.Linear(384 * 13 * 13, 1024)
        self.exit1_fc2 = nn.Linear(1024, num_classes)

        # exit_early_2
        self.exit2_fc1 = nn.Linear(384 * 13 * 13, 1024)
        self.exit2_fc2 = nn.Linear(1024, num_classes)

        # exit_full
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        # layer3 exit_early_1
        x3 = self.layer3(x)
        feat1 = x3.view(x3.size(0), -1)
        out_e1 = self.exit1_fc2(F.relu(self.exit1_fc1(feat1)))

        # layer4 exit_early_1
        x4 = self.layer4(x3)
        feat2 = x4.view(x4.size(0), -1)
        out_e2 = self.exit2_fc2(F.relu(self.exit2_fc1(feat2)))

        # layer5 exit_full
        x5 = self.layer5(x4)
        flat = x5.view(x5.size(0), -1)
        y    = self.fc(flat)
        y    = self.fc1(y)
        out_full = self.fc2(y)

        return out_e1, out_e2, out_full
