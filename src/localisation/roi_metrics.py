"""ROI registration and localisation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_likeness(path) -> float:
    arr = np.asarray(Image.open(path).convert("L"))
    if arr.size == 0:
        return -1.0
    near_binary = np.mean((arr < 20) | (arr > 235))
    contrast = arr.std() / 255.0
    return float(near_binary + contrast)


def register_full_size_masks(
    mask_files,
    full_shape,
    *,
    output_size: int = 224,
    size_tolerance: float = 0.03,
    binary_min: float | None = None,
):
    """Union full-size ROI masks after size/binary-content validation."""
    full_h, full_w = full_shape
    union = np.zeros((output_size, output_size), dtype=np.uint8)
    found = False

    for path in mask_files:
        arr = np.asarray(Image.open(path).convert("L"))
        h, w = arr.shape
        if abs(h - full_h) / max(full_h, 1) > size_tolerance:
            continue
        if abs(w - full_w) / max(full_w, 1) > size_tolerance:
            continue
        if binary_min is not None:
            if float(np.mean((arr < 20) | (arr > 235))) < binary_min:
                continue

        if (h, w) != (full_h, full_w):
            arr = np.asarray(
                Image.fromarray(arr).resize((full_w, full_h), Image.NEAREST)
            )
        mask = arr > 127
        if mask.mean() > 0.5:
            mask = ~mask
        resized = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (output_size, output_size), Image.NEAREST
            )
        ) > 127
        union = np.maximum(union, resized.astype(np.uint8))
        found = True

    return union if found and union.sum() > 0 else None


def localisation_metrics(cam: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Pointing Game, pixel ROC-AUC/AP and Dice@0.5 from notebook 12."""
    mask = mask.astype(bool)
    if mask.sum() == 0:
        raise ValueError("ROI mask is empty.")

    peak = np.unravel_index(np.argmax(cam), cam.shape)
    pointing_game = int(mask[peak])

    if 0 < mask.sum() < mask.size:
        pixel_auc = float(roc_auc_score(mask.ravel(), cam.ravel()))
        pixel_ap = float(average_precision_score(mask.ravel(), cam.ravel()))
    else:
        pixel_auc = np.nan
        pixel_ap = np.nan

    pred_mask = cam >= 0.5
    intersection = np.logical_and(pred_mask, mask).sum()
    denom = pred_mask.sum() + mask.sum()
    dice = float(2 * intersection / denom) if denom else 0.0

    return {
        "pointing_game": pointing_game,
        "pixel_auc": pixel_auc,
        "pixel_ap": pixel_ap,
        "dice@0.5": dice,
    }
