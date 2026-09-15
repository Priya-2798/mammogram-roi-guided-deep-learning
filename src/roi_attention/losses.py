"""ROI-guided true-class CAM supervision from notebooks 06 and 08."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice(prob, target, eps: float = 1.0):
    p = prob.flatten(1)
    t = target.flatten(1)
    intersection = (p * t).sum(1)
    return (1 - (2 * intersection + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def balanced_bce_logits(logits, target, eps: float = 1e-6):
    """Foreground/background-balanced BCE used for small lesion masks."""
    positive_fraction = target.mean(dim=(1, 2, 3), keepdim=True).clamp(eps, 1 - eps)
    weights = (
        (0.5 / positive_fraction) * target
        + (0.5 / (1 - positive_fraction)) * (1 - target)
    )
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (weights * raw).mean()


def attention_loss(cam, mask, label, has_mask, sup_size: int = 128):
    """Align the true-class CAM to the soft ROI target."""
    idx = torch.arange(label.size(0), device=label.device)
    true_class_cam = cam[idx, label].unsqueeze(1)
    selected = has_mask.view(-1) == 1

    if selected.sum() == 0:
        return cam.new_tensor(0.0)

    logits = F.interpolate(
        true_class_cam[selected],
        size=(sup_size, sup_size),
        mode="bilinear",
        align_corners=False,
    )
    target = F.interpolate(
        mask[selected],
        size=(sup_size, sup_size),
        mode="area",
    ).clamp(0.0, 1.0)

    return (
        0.5 * balanced_bce_logits(logits, target)
        + 0.5 * soft_dice(torch.sigmoid(logits), target)
    )
