"""Class-specific CAM classifiers used for ROI-supervised experiments."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetCAM2(nn.Module):
    """ResNet-50 with a two-class 1x1 CAM head (benign/malignant)."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.cam_conv = nn.Conv2d(2048, n_classes, 1)

    def forward(self, x, return_cam: bool = False):
        features = self.features(x)
        cam = self.cam_conv(features)
        logits = F.adaptive_avg_pool2d(cam, 1).flatten(1)
        return (logits, cam) if return_cam else logits


class ConvNeXtCAM2(nn.Module):
    """ConvNeXt-Tiny with the corresponding two-class 1x1 CAM head."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.cam_conv = nn.Conv2d(768, n_classes, kernel_size=1)

    def forward(self, x, return_cam: bool = False):
        features = self.features(x)
        cam = self.cam_conv(features)
        logits = F.adaptive_avg_pool2d(cam, 1).flatten(1)
        return (logits, cam) if return_cam else logits
