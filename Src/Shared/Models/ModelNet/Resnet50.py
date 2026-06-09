import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # un squeeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [block(self.in_channel,
                        channel,
                        downsample=downsample,
                        stride=stride,
                        groups=self.groups,
                        width_per_group=self.width_per_group)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

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

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


class MultiEEResNet50(nn.Module):
    def __init__(self, block, blocks_num, num_classes=10, include_top=True):
        super(MultiEEResNet50, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 额外的全连接层，添加在第2和第3个Bottleneck后
        self.fc2 = nn.Linear(128 * block.expansion, num_classes)  # 对第2个Bottleneck进行分类
        self.fc3 = nn.Linear(256 * block.expansion, num_classes)  # 对第3个Bottleneck进行分类

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = [block(self.in_channel, channel, downsample=downsample, stride=stride)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x, stage='final'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 通过层1
        x1 = self.layer1(x)

        # 通过层2（提取第2个Bottleneck后的特征）
        x2 = self.layer2(x1)
        if self.include_top:
            x2_avg = self.avgpool(x2)  # 平均池化
            x2_flatten = torch.flatten(x2_avg, 1)  # 展平
            x2_fc = self.fc2(x2_flatten)  # 在第2个Bottleneck后通过分类器
        else:
            x2_fc = x2  # 如果不需要分类器输出，只保留特征

        # 通过层3（提取第3个Bottleneck后的特征）
        x3 = self.layer3(x2)
        if self.include_top:
            x3_avg = self.avgpool(x3)  # 平均池化
            x3_flatten = torch.flatten(x3_avg, 1)  # 展平
            x3_fc = self.fc3(x3_flatten)  # 在第3个Bottleneck后通过分类器
        else:
            x3_fc = x3  # 如果不需要分类器输出，只保留特征

        # 通过层4（输出最终的分类结果）
        x4 = self.layer4(x3)
        x4_avg = self.avgpool(x4)
        x4_flatten = torch.flatten(x4_avg, 1)  # 展平
        x_final = self.fc(x4_flatten)

        # 根据stage决定输出
        if stage == 'final':
            return x_final  # 只返回最终的输出，不返回x2_fc和x3_fc
        elif stage == 'x2_fc':
            return x2_fc  # 只返回第2个出口的分类器输出
        elif stage == 'x3_fc':
            return x3_fc  # 只返回第3个出口的分类器输出
        else:
            return x_final, x2_fc, x3_fc  # 默认返回所有输出

    def forward_partial(self, x: torch.Tensor, start: int, end: int):
        """
        Execute stages [start, end] inclusive (stage indices 0-4).
        Returns:
          features : torch.Tensor  (raw output of the last executed stage)
          logits   : torch.Tensor | None  (classification logits if end has an exit head)
          conf     : float | None
          pred     : int   | None
        Exit heads:
          end == 2 -> avgpool + fc2
          end == 3 -> avgpool + fc3
          end == 4 -> already included in stage 4, logits = features
        No exit head for end in {0, 1}.
        """
        if start < 0 or end > 4 or start > end:
            raise ValueError(f"require 0 <= start <= end <= 4, got ({start}, {end})")

        if start <= 0 and end >= 0:
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        if start <= 1 and end >= 1:
            x = self.layer1(x)
        if start <= 2 and end >= 2:
            x = self.layer2(x)
        if start <= 3 and end >= 3:
            x = self.layer3(x)
        if start <= 4 and end >= 4:
            x = self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))

        logits = None
        if end == 2:
            logits = self.fc2(torch.flatten(self.avgpool(x), 1))
        elif end == 3:
            logits = self.fc3(torch.flatten(self.avgpool(x), 1))
        elif end == 4:
            logits = x

        if logits is None:
            return x, None, None, None

        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return x, logits, float(conf.item()), int(pred.item())


# Helper function to control which layers to freeze/unfreeze
def freeze_layers(model, freeze_backbone=False, freeze_x2_fc=False, freeze_x3_fc=False):
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final output layer if needed
    for param in model.fc.parameters():
        param.requires_grad = True

    # Unfreeze the 2nd exit classifier if needed
    if not freeze_x2_fc:
        for param in model.fc2.parameters():
            param.requires_grad = True

    # Unfreeze the 3rd exit classifier if needed
    if not freeze_x3_fc:
        for param in model.fc3.parameters():
            param.requires_grad = True

    # Unfreeze the backbone if needed
    if not freeze_backbone:
        for param in model.layer1.parameters():
            param.requires_grad = True
        for param in model.layer2.parameters():
            param.requires_grad = True
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
