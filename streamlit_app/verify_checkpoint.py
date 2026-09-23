from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

MODEL = Path(__file__).with_name("resnet50_cam2_lambda0.5.pth")

class ResNetCAM2(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet50(weights=None)
        self.features = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2, bb.layer3, bb.layer4)
        self.cam_conv = nn.Conv2d(2048, 2, 1)
    def forward(self, x, return_cam=False):
        f = self.features(x)
        cam = self.cam_conv(f)
        logits = F.adaptive_avg_pool2d(cam, 1).flatten(1)
        return (logits, cam) if return_cam else logits

if not MODEL.exists():
    raise SystemExit(f"Missing {MODEL.name}. Put it next to this script.")

try:
    state = torch.load(MODEL, map_location="cpu", weights_only=True)
except TypeError:
    state = torch.load(MODEL, map_location="cpu")

m = ResNetCAM2()
m.load_state_dict(state, strict=True)
m.eval()
with torch.no_grad():
    logits, cam = m(torch.zeros(1, 3, 512, 512), return_cam=True)
print("Checkpoint loaded successfully")
print("logits shape:", tuple(logits.shape))
print("CAM shape:", tuple(cam.shape))
