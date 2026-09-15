"""Visualisation utilities for CAM/ROI figures."""

from __future__ import annotations

import matplotlib
import numpy as np


def overlay_cam(gray, cam, alpha: float = 0.45, cmap: str = "jet"):
    gray_norm = gray.astype(float)
    if gray_norm.max() > 1:
        gray_norm = gray_norm / 255.0
    rgb = np.dstack([gray_norm, gray_norm, gray_norm])
    heat = matplotlib.colormaps[cmap](cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)
