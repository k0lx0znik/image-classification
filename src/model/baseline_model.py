from torch import nn
import torch.nn as nn
import torchvision


class ReiAyanami(nn.Module):
    def __init__(self, num_classes=200, p=0.3):
        super().__init__()

        resnext = torchvision.models.resnext50_32x4d(weights=None)

        resnext.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        resnext.maxpool = nn.Identity()
        resnext.relu = nn.LeakyReLU(0.1)

        self.stem = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
        )

        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        return self.head(x)