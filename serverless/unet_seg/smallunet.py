# =============================================================================
# 小型 U-Net（セマンティックセグメンテーション）のアーキテクチャ定義
#
# **学習側のコードと同一の定義であること。** state_dict しか持たない
# チェックポイントを読むので、層の名前・並びが 1 つでもずれると
# load_state_dict が落ちる（あるいは黙って別のモデルになる）。
# 学習コードを変更したらこのファイルも合わせること。
# =============================================================================
import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, base_channels=24):
        super().__init__()
        c = int(base_channels)
        self.down1 = ConvBlock(in_channels, c)
        self.down2 = ConvBlock(c, c * 2)
        self.down3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.conv2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.conv1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        y = self.up2(x3)
        if y.shape[-2:] != x2.shape[-2:]:
            y = F.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.conv2(torch.cat([y, x2], dim=1))
        y = self.up1(y)
        if y.shape[-2:] != x1.shape[-2:]:
            y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.conv1(torch.cat([y, x1], dim=1))
        return self.head(y)


def create_model(in_channels=4, num_classes=2, base_channels=24):
    return SmallUNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(base_channels),
    )
