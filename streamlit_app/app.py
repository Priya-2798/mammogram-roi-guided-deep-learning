from pathlib import Path
import io
import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Mammogram AI Research Demo",
    page_icon="🩻",
    layout="wide",
)

MODEL_FILENAME = "resnet50_cam2_lambda0.5.pth"
MODEL_PATH = Path(os.environ.get("MAMMOGRAM_MODEL_PATH", Path(__file__).with_name(MODEL_FILENAME)))
IMG_SIZE = 512
CROP_PAD_FRAC = 0.02
MALIGNANT_THRESHOLD = 0.29  # validation-selected Youden-J threshold from the dissertation run
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLASS_NAMES = {0: "Benign class", 1: "Malignant class"}

# -----------------------------
# Model definition from Notebook 06
# -----------------------------
class ResNetCAM2(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # weights=None avoids any internet download; the full trained state_dict is loaded below.
        bb = models.resnet50(weights=None)
        self.features = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.relu,
            bb.maxpool,
            bb.layer1,
            bb.layer2,
            bb.layer3,
            bb.layer4,
        )
        self.cam_conv = nn.Conv2d(2048, n_classes, 1)  # ch0 benign, ch1 malignant

    def forward(self, x, return_cam=False):
        f = self.features(x)
        cam = self.cam_conv(f)  # B, 2, 16, 16 for 512x512 input
        logits = F.adaptive_avg_pool2d(cam, 1).flatten(1)
        return (logits, cam) if return_cam else logits


@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    model = ResNetCAM2()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")

    # The submitted checkpoint is a plain state_dict.
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# -----------------------------
# Dissertation-faithful preprocessing
# -----------------------------
def otsu_threshold(gray_u8: np.ndarray) -> int:
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(float)
    p = hist / max(gray_u8.size, 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    score = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    score[(omega <= 0) | (omega >= 1)] = 0
    return int(np.argmax(score))


def breast_bbox(gray_u8: np.ndarray, pad_frac: float = CROP_PAD_FRAC):
    h, w = gray_u8.shape
    mask = gray_u8 > max(otsu_threshold(gray_u8), 5)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)))
    labelled, n = ndimage.label(mask)

    if n == 0:
        return (0, 0, w, h)

    counts = np.bincount(labelled.ravel())
    counts[0] = 0
    ys, xs = np.where(labelled == counts.argmax())
    if len(xs) == 0:
        return (0, 0, w, h)

    q = int(round(pad_frac * max(h, w)))
    return (
        max(int(xs.min()) - q, 0),
        max(int(ys.min()) - q, 0),
        min(int(xs.max()) + 1 + q, w),
        min(int(ys.max()) + 1 + q, h),
    )


def crop_pad_resize(gray_u8: np.ndarray, bbox, size: int = IMG_SIZE) -> np.ndarray:
    l, t, r, b = bbox
    crop = Image.fromarray(gray_u8).crop((l, t, r, b))
    cw, ch = crop.size
    side = max(cw, ch)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(crop, ((side - cw) // 2, (side - ch) // 2))
    return np.array(canvas.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def preprocess_image(pil_img: Image.Image):
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)
    bbox = breast_bbox(gray)
    processed = crop_pad_resize(gray, bbox, IMG_SIZE)

    x = np.stack([processed.astype(np.float32) / 255.0] * 3, axis=0)
    tensor = torch.from_numpy(x)
    tensor = (tensor - MEAN) / STD
    return processed, tensor.float(), bbox


# -----------------------------
# Inference / visualisation
# -----------------------------
def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    mx = arr.max()
    if mx > 0:
        arr = arr / mx
    return arr


@torch.no_grad()
def infer(model: nn.Module, tensor: torch.Tensor):
    logits, cams = model(tensor.unsqueeze(0), return_cam=True)
    probs = torch.softmax(logits, dim=1)[0]
    malignant_score = float(probs[1].cpu())
    predicted_class = 1 if malignant_score >= MALIGNANT_THRESHOLD else 0

    # Dissertation model directly produces class-specific CAMs.
    cam_up = F.interpolate(
        cams,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    )[0].cpu().numpy()

    benign_cam = normalize_map(cam_up[0])
    malignant_cam = normalize_map(cam_up[1])
    decision_cam = malignant_cam if predicted_class == 1 else benign_cam

    return {
        "malignant_score": malignant_score,
        "benign_score": float(probs[0].cpu()),
        "predicted_class": predicted_class,
        "benign_cam": benign_cam,
        "malignant_cam": malignant_cam,
        "decision_cam": decision_cam,
    }


def make_overlay(gray_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0
    rgb = np.dstack([gray, gray, gray])
    heat = plt.get_cmap("magma")(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
    <style>
    .main-title {font-size: 2.1rem; font-weight: 700; margin-bottom: 0.2rem;}
    .subtle {color: #6b7280; font-size: 0.95rem;}
    .research-box {padding: 0.8rem 1rem; border-radius: 0.6rem; background: rgba(128,128,128,0.08);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Mammogram AI Research Demo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">MSc Artificial Intelligence dissertation prototype — ROI-guided ResNet-50 (λ = 0.5)</div>',
    unsafe_allow_html=True,
)

st.warning(
    "Research demonstration only. This system is not intended for clinical diagnosis, screening, "
    "treatment, triage, or medical decision-making. Do not use its output to make decisions about a real patient."
)

analysis_tab, results_tab, about_tab = st.tabs(["Mammogram Analysis", "Dissertation Results", "About"])

with analysis_tab:
    if not MODEL_PATH.exists():
        st.error(
            f"Model checkpoint not found. Put `{MODEL_FILENAME}` in the same folder as `app.py`, "
            "or set the environment variable `MAMMOGRAM_MODEL_PATH` to its full path."
        )
        st.stop()

    model = load_model(str(MODEL_PATH))

    uploaded = st.file_uploader(
        "Upload a mammogram image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="For the most faithful demonstration, use a de-identified image in the same style as the dissertation's CBIS-DDSM JPEG inputs.",
    )

    st.caption(
        "The app applies the dissertation's breast-focused crop, square padding, 512×512 resize, "
        "and ImageNet normalisation before inference."
    )

    if uploaded is not None:
        try:
            original = Image.open(io.BytesIO(uploaded.getvalue()))
            processed, x, bbox = preprocess_image(original)
            out = infer(model, x)
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")
            st.stop()

        pred_name = CLASS_NAMES[out["predicted_class"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Model prediction", pred_name)
        c2.metric("Malignant model score", f"{100*out['malignant_score']:.1f}%")
        c3.metric("Decision threshold", f"{100*MALIGNANT_THRESHOLD:.0f}%")

        st.info(
            "The displayed score is the model's softmax output and should not be interpreted as a calibrated clinical probability of cancer."
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.subheader("Uploaded image")
            st.image(original, use_container_width=True)
        with col_b:
            st.subheader("Processed input")
            st.image(processed, clamp=True, use_container_width=True)
        with col_c:
            st.subheader("Decision CAM overlay")
            st.image(make_overlay(processed, out["decision_cam"]), clamp=True, use_container_width=True)

        with st.expander("Show class-specific activation maps", expanded=False):
            e1, e2 = st.columns(2)
            with e1:
                st.markdown("**Benign-class CAM**")
                st.image(make_overlay(processed, out["benign_cam"]), clamp=True, use_container_width=True)
            with e2:
                st.markdown("**Malignant-class CAM**")
                st.image(make_overlay(processed, out["malignant_cam"]), clamp=True, use_container_width=True)

            st.caption(
                "These are class-specific activation maps produced directly by the model's 1×1 CAM head. "
                "They are visual evidence maps, not lesion segmentations and not proof of causal reasoning."
            )

        with st.expander("Preprocessing details", expanded=False):
            st.write(
                f"Detected breast-region bounding box in the uploaded image: left={bbox[0]}, top={bbox[1]}, "
                f"right={bbox[2]}, bottom={bbox[3]}."
            )
            st.write(
                "Pipeline: grayscale → Otsu-based foreground estimate → largest connected component → "
                "2% padding → square zero-padding → 512×512 bilinear resize → three-channel replication → ImageNet normalisation."
            )

with results_tab:
    st.subheader("Selected ROI-guided ResNet-50 results")
    st.markdown(
        "The λ = 0.5 model was selected using validation localisation subject to the predefined AUC constraint. "
        "The locked-test classification change relative to the matched λ = 0 control was small and not statistically supported."
    )

    st.dataframe(
        {
            "Model": ["Matched control", "ROI-guided selected"],
            "λ": [0.0, 0.5],
            "ROC-AUC": [0.7771, 0.7808],
            "PR-AUC": [0.7297, 0.7271],
            "Sensitivity": [0.8000, 0.8414],
            "Specificity": [0.6111, 0.5000],
            "F1": [0.6725, 0.6507],
        },
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Locked-test localisation (all masked cases, n = 348)**")
    st.dataframe(
        {
            "Model": ["Matched control", "ROI-guided selected"],
            "Pointing Game": [0.0661, 0.1063],
            "Pixel ROC-AUC": [0.5787, 0.9053],
            "Pixel AP": [0.0647, 0.1217],
            "Dice@0.5": [0.0528, 0.0094],
        },
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Paired ΔROC-AUC (selected − control): +0.0037, 95% CI [−0.0222, 0.0298]. "
        "Therefore, the dissertation did not establish a statistically detectable classification AUC improvement."
    )

with about_tab:
    st.subheader("Research prototype")
    st.markdown(
        """
        This prototype reproduces the inference path of the dissertation's selected **ROI-guided ResNet-50 (λ = 0.5)**.

        **Architecture**
        - ImageNet-style ResNet-50 feature extractor
        - 1×1 convolution producing two class-specific activation maps
        - Channel 0: benign evidence
        - Channel 1: malignant evidence
        - Global average pooling of the CAMs to obtain the two classification logits

        **Training concept**
        - Image-level cross-entropy classification loss
        - Auxiliary ROI-alignment loss applied to the true-class CAM during training
        - Radiologist ROI masks are **not needed at inference**

        **Important limitation**
        - The model was developed as an MSc research classifier on public research data.
        - The app is a post-dissertation demonstration layer and was not itself part of the submitted model evaluation.
        - It has not undergone external clinical validation, prospective evaluation, calibration for clinical probability, or regulatory assessment.
        """
    )

    st.markdown(
        '<div class="research-box"><b>Privacy:</b> For demonstrations, use only de-identified research images. Do not upload identifiable patient data.</div>',
        unsafe_allow_html=True,
    )
