"""Breast crop/pad/resize preprocessing used by the 512x512 experiments."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def load_gray(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(gray.size, 1)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_total = mu[-1]
    score = (mu_total * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    score[(omega <= 0) | (omega >= 1)] = 0
    return int(np.argmax(score))


def breast_bbox(gray: np.ndarray, pad_frac: float = 0.02) -> tuple[int, int, int, int]:
    h, w = gray.shape
    mask = gray > max(otsu_threshold(gray), 5)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)))
    labels, n = ndimage.label(mask)
    if n == 0:
        return (0, 0, w, h)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    ys, xs = np.where(labels == counts.argmax())
    if len(xs) == 0:
        return (0, 0, w, h)

    pad = int(round(pad_frac * max(h, w)))
    return (
        max(int(xs.min()) - pad, 0),
        max(int(ys.min()) - pad, 0),
        min(int(xs.max()) + 1 + pad, w),
        min(int(ys.max()) + 1 + pad, h),
    )


def _crop_pad_resize(arr: np.ndarray, bbox, size: int, resample) -> np.ndarray:
    left, top, right, bottom = bbox
    crop = Image.fromarray(arr).crop((left, top, right, bottom))
    crop_w, crop_h = crop.size
    side = max(crop_w, crop_h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(crop, ((side - crop_w) // 2, (side - crop_h) // 2))
    return np.asarray(canvas.resize((size, size), resample))


def crop_pad_resize_image(gray: np.ndarray, bbox, size: int = 512) -> np.ndarray:
    return _crop_pad_resize(gray, bbox, size, Image.BILINEAR).astype(np.uint8)


def crop_pad_resize_mask(mask: np.ndarray, bbox, size: int = 512) -> np.ndarray:
    return (_crop_pad_resize(mask, bbox, size, Image.NEAREST) > 127).astype(np.uint8)


def preprocess_mammogram(path, size: int = 512, pad_frac: float = 0.02):
    original = load_gray(path)
    bbox = breast_bbox(original, pad_frac=pad_frac)
    processed = crop_pad_resize_image(original, bbox, size=size)
    return original, processed, bbox
