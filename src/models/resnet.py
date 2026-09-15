"""ResNet-50 classifiers used across the dissertation experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet50_baseline(num_classes: int = 2):
    """Notebook-02 baseline: ImageNet ResNet-50 with a two-class head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50_binary(dropout: float | None = None):
    """Binary ResNet-50 used by the later 512x512 experiments."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    if dropout is None:
        model.fc = nn.Linear(in_features, 1)
    else:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    return model


def freeze_all_backbone(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_layer4_and_head(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
