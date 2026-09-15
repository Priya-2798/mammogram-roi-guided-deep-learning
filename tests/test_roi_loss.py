import torch

from src.roi_attention.losses import attention_loss


def test_attention_loss_no_masks_is_zero():
    cam = torch.randn(2, 2, 16, 16)
    mask = torch.zeros(2, 1, 128, 128)
    label = torch.tensor([0, 1], dtype=torch.long)
    has_mask = torch.zeros(2, dtype=torch.long)
    loss = attention_loss(cam, mask, label, has_mask)
    assert float(loss) == 0.0
