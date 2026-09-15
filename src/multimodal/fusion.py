"""Image/metadata fusion architecture used in notebook 07."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageBranch(nn.Module):
    def __init__(self, roi_checkpoint: str | None = None):
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
        self.out_dim = 2048

        if roi_checkpoint:
            state = torch.load(roi_checkpoint, map_location="cpu")
            feature_state = {
                key[len("features."):]: value
                for key, value in state.items()
                if key.startswith("features.")
            }
            self.features.load_state_dict(feature_state)

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultimodalModel(nn.Module):
    def __init__(
        self,
        *,
        use_image: bool,
        use_meta: bool,
        meta_in: int,
        roi_checkpoint: str | None = None,
    ):
        super().__init__()
        if not (use_image or use_meta):
            raise ValueError("At least one branch must be enabled.")

        self.use_image = use_image
        self.use_meta = use_meta
        fused_dim = 0

        if use_image:
            self.image = ImageBranch(roi_checkpoint)
            fused_dim += self.image.out_dim
        if use_meta:
            self.meta = MetaMLP(meta_in)
            fused_dim += self.meta.out_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, image, meta):
        features = []
        if self.use_image:
            features.append(self.image(image))
        if self.use_meta:
            features.append(self.meta(meta))
        return self.head(torch.cat(features, dim=1))
