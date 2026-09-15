"""Grad-CAM utilities used in the baseline spatial analysis."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def tensor_from_array(arr: np.ndarray, device) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack([arr] * 3, axis=0)).float() / 255.0
    return ((tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)


def compute_resnet_gradcam(
    model,
    arr: np.ndarray,
    *,
    device,
    target_class: int = 1,
    output_size: int = 224,
) -> np.ndarray:
    """Compute Grad-CAM at ResNet-50 layer4[-1], matching notebook 03."""
    activations = []
    gradients = []
    target_layer = model.layer4[-1]

    h1 = target_layer.register_forward_hook(lambda _m, _i, o: activations.append(o))
    h2 = target_layer.register_full_backward_hook(
        lambda _m, _gi, go: gradients.append(go[0])
    )
    try:
        model.zero_grad(set_to_none=True)
        x = tensor_from_array(arr, device)
        logits = model(x)
        logits[:, target_class].sum().backward()

        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
    finally:
        h1.remove()
        h2.remove()
