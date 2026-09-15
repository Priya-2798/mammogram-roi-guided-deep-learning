"""ResNet-50 baseline model used for CBIS-DDSM classification.

Refactored from the executed dissertation baseline notebook.
"""

import torch.nn as nn
from torchvision import models


def build_resnet50_baseline(num_classes=2):
    """Build the ImageNet-pretrained ResNet-50 baseline classifier."""
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V2
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model
