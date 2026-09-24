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

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Mammogram AI Research Studio",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILENAME = "resnet50_cam2_lambda0.5.pth"
MODEL_PATH = Path(os.environ.get("MAMMOGRAM_MODEL_PATH", Path(__file__).with_name(MODEL_FILENAME)))
IMG_SIZE = 512
CROP_PAD_FRAC = 0.02
MALIGNANT_THRESHOLD = 0.29
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLASS_NAMES = {0: "Benign class", 1: "Malignant class"}

# --------------------------------------------------
# Premium research-dashboard styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ink: #0f172a;
        --muted: #64748b;
        --panel: #ffffff;
        --line: #e2e8f0;
        --soft: #f8fafc;
        --navy: #0f3d56;
        --teal: #0f766e;
        --teal2: #14b8a6;
        --rose: #be123c;
        --rose-soft: #fff1f2;
        --green-soft: #ecfdf5;
        --amber-soft: #fffbeb;
    }

    html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}

    /* Keep the app readable even when the viewer's OS/browser/Streamlit is in dark mode.
       This dashboard intentionally uses a light main canvas with a dark sidebar. */
    html, body, .stApp, [data-testid="stAppViewContainer"] {color-scheme: light !important;}
    .stApp, [data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #f8fbfc 0%, #f8fafc 38%, #ffffff 100%) !important; color: var(--ink) !important;}
    .block-container {max-width: 1320px; padding-top: 1.5rem; padding-bottom: 3rem;}

    /* Native Streamlit text: force dark text on the light main canvas. */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6 {color: #0f172a !important;}

    [data-testid="stAppViewContainer"] .stMarkdown p,
    [data-testid="stAppViewContainer"] .stMarkdown li,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] *,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
    [data-testid="stAppViewContainer"] [data-testid="stProgress"] p {color: #475569 !important;}

    /* Metrics stay readable in dark system mode. */
    [data-testid="stMetric"] {background:#ffffff !important;}
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {color:#64748b !important;}
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {color:#0f172a !important;}
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {color:#334155 !important;}

    /* Tabs: Streamlit otherwise inherits white text from dark mode. */
    button[data-baseweb="tab"],
    button[data-baseweb="tab"] * {color:#334155 !important;}
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] * {color:#0f172a !important;}

    /* Buttons and file uploader. */
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFileUploader"] button {
        color:#0f172a !important;
        background:#ffffff !important;
        border-color:#cbd5e1 !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stFileUploader"] button:hover {
        color:#0f3d56 !important;
        border-color:#0f766e !important;
        background:#f8fafc !important;
    }

    div[data-testid="stFileUploader"],
    div[data-testid="stFileUploader"] section {background:#ffffff !important;}
    div[data-testid="stFileUploader"] *,
    div[data-testid="stFileUploader"] section * {color:#334155 !important;}

    /* Expanders and regular input surfaces. */
    [data-testid="stExpander"] {background:#ffffff !important;}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] div {color:#334155;}

    input, textarea, [role="combobox"] {color:#0f172a !important; background:#ffffff !important;}

    /* Sidebar */
    [data-testid="stSidebar"] {background: #0b2230; border-right: 1px solid rgba(255,255,255,.08);}
    [data-testid="stSidebar"] * {color: #eaf4f7;}
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p {color: #b9d0d8 !important;}
    [data-testid="stSidebar"] hr {border-color: rgba(255,255,255,.12);}

    /* Hero */
    .hero {
        background: linear-gradient(120deg, #0b2f42 0%, #0f5361 52%, #0f766e 100%);
        border-radius: 24px;
        padding: 30px 34px;
        color: white;
        box-shadow: 0 18px 50px rgba(15, 61, 86, .18);
        margin-bottom: 18px;
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 280px; height: 280px;
        right: -85px; top: -115px;
        border-radius: 50%;
        border: 36px solid rgba(255,255,255,.08);
    }
    .hero-kicker {font-size: .79rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; opacity: .82;}
    .hero-title {font-size: 2.25rem; line-height: 1.1; font-weight: 800; margin: 7px 0 9px 0; max-width: 840px;}
    .hero-sub {font-size: 1rem; line-height: 1.6; max-width: 850px; color: #dceef1;}
    .hero-pills {display:flex; flex-wrap:wrap; gap:8px; margin-top:18px;}
    .pill {display:inline-block; padding:6px 10px; border-radius:999px; font-size:.78rem; font-weight:700; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.16);}

    /* Generic card */
    .card {
        background: rgba(255,255,255,.96);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 8px 26px rgba(15,23,42,.055);
    }
    .card-title {font-size: .84rem; color: var(--muted); font-weight: 750; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px;}
    .card-value {font-size: 1.45rem; font-weight: 800; color: var(--ink);}
    .card-note {font-size: .82rem; color: var(--muted); margin-top: 4px;}

    .feature-card {height: 100%; min-height: 154px;}
    .feature-icon {font-size:1.4rem; margin-bottom:8px;}
    .feature-title {font-weight:800; font-size:1.02rem; margin-bottom:5px;}
    .feature-text {color:var(--muted); font-size:.9rem; line-height:1.5;}

    /* Banner */
    .research-banner {
        background: var(--amber-soft);
        border: 1px solid #fde68a;
        border-left: 5px solid #d97706;
        color: #854d0e;
        padding: 13px 16px;
        border-radius: 14px;
        margin: 8px 0 18px 0;
        font-size: .9rem;
        line-height: 1.5;
    }

    /* Prediction banners */
    .prediction {
        border-radius: 20px;
        padding: 20px 22px;
        border: 1px solid var(--line);
        box-shadow: 0 8px 28px rgba(15,23,42,.06);
        margin: 8px 0 16px 0;
    }
    .prediction-benign {background: linear-gradient(110deg, #f0fdfa 0%, #ecfdf5 100%); border-color:#99f6e4;}
    .prediction-malignant {background: linear-gradient(110deg, #fff1f2 0%, #fff7ed 100%); border-color:#fecdd3;}
    .prediction-label {font-size:.77rem; text-transform:uppercase; letter-spacing:.08em; font-weight:800; color:var(--muted);}
    .prediction-value {font-size:1.8rem; font-weight:850; margin-top:3px;}

    .image-label {font-size:.85rem; font-weight:800; color:var(--ink); margin-bottom:7px;}
    .section-title {font-size:1.28rem; font-weight:850; color:var(--ink); margin:8px 0 5px 0;}
    .section-sub {font-size:.91rem; color:var(--muted); margin-bottom:14px;}

    /* Streamlit components */
    div[data-testid="stFileUploader"] {background:#ffffff; border:1px dashed #94a3b8; border-radius:18px; padding:9px; box-shadow:0 8px 24px rgba(15,23,42,.04);}
    div[data-testid="stFileUploader"] section {background:#f8fafc !important; border-radius:14px;}
    div[data-baseweb="tab-list"] {gap: 7px; background:#eef4f6; padding:6px; border-radius:14px;}
    button[data-baseweb="tab"] {border-radius:10px; padding:10px 14px; font-weight:700;}
    button[data-baseweb="tab"][aria-selected="true"] {background:white; box-shadow:0 2px 10px rgba(15,23,42,.08);}

    [data-testid="stMetric"] {background:white; border:1px solid var(--line); padding:14px 16px; border-radius:16px; box-shadow:0 6px 22px rgba(15,23,42,.045);}
    [data-testid="stMetricLabel"] {color:var(--muted);}
    [data-testid="stMetricValue"] {font-weight:850;}

    .pipeline {display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 18px 0;}
    .pipe {background:#eef6f7; color:#164e63; border:1px solid #cfe6ea; padding:7px 10px; border-radius:999px; font-size:.8rem; font-weight:700;}
    .arrow {color:#94a3b8; align-self:center; font-weight:700;}

    .mini-note {font-size:.82rem; color:var(--muted); line-height:1.45;}
    .interpretation-box {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(15,23,42,.045);
        margin: 14px 0 18px 0;
    }
    .interpretation-heading {font-size:1.06rem; font-weight:850; color:var(--ink); margin-bottom:8px;}
    .interpretation-text {font-size:.92rem; color:#334155; line-height:1.62;}
    .interpretation-text b {color:var(--ink);}
    .footer-note {margin-top:24px; color:var(--muted); font-size:.78rem; text-align:center;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Model definition
# --------------------------------------------------
class ResNetCAM2(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        bb = models.resnet50(weights=None)
        self.features = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2, bb.layer3, bb.layer4,
        )
        self.cam_conv = nn.Conv2d(2048, n_classes, 1)

    def forward(self, x, return_cam=False):
        f = self.features(x)
        cam = self.cam_conv(f)
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
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
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

# --------------------------------------------------
# Inference
# --------------------------------------------------
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
    cam_up = F.interpolate(cams, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0].cpu().numpy()
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


def make_overlay(gray_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    gray = gray_u8.astype(np.float32) / 255.0
    rgb = np.dstack([gray, gray, gray])
    heat = plt.get_cmap("magma")(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## Mammogram AI")
    st.caption("MSc Artificial Intelligence research prototype")
    st.divider()
    st.markdown("**Model**")
    st.caption("ROI-guided ResNet-50")
    st.markdown("**Selected λ**")
    st.caption("0.5")
    st.markdown("**Input**")
    st.caption("512 × 512 mammogram")
    st.markdown("**Task**")
    st.caption("Benign vs malignant classification")
    st.divider()
    st.markdown("**Research status**")
    st.caption("Post-dissertation demonstration")
    st.markdown("**Inference ROI required?**")
    st.caption("No")
    st.divider()
    st.caption("Use only de-identified research images. Not for clinical use.")

# --------------------------------------------------
# Hero + disclaimer
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Explainable medical imaging research</div>
      <div class="hero-title">Mammogram AI Research Studio</div>
      <div class="hero-sub">Interactive demonstration of ROI-guided mammogram classification, class-specific attention maps, and dissertation results.</div>
      <div class="hero-pills">
        <span class="pill">ResNet-50</span>
        <span class="pill">ROI-guided attention</span>
        <span class="pill">512×512 inference</span>
        <span class="pill">CAM explainability</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="research-banner"><b>Research demonstration only.</b> This system is not intended for clinical diagnosis, screening, treatment, triage, or medical decision-making. The model score is a research output, not a calibrated probability of cancer.</div>
    """,
    unsafe_allow_html=True,
)

analysis_tab, explain_tab, results_tab, method_tab = st.tabs([
    "Analyze Mammogram", "Explainability", "Research Results", "Methodology"
])

# --------------------------------------------------
# Analysis tab
# --------------------------------------------------
with analysis_tab:
    if not MODEL_PATH.exists():
        st.error(f"Model checkpoint not found: `{MODEL_FILENAME}`")
        st.stop()

    model = load_model(str(MODEL_PATH))

    st.markdown('<div class="section-title">Upload a mammogram</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Use a de-identified research mammogram. The same dissertation preprocessing pipeline is applied automatically.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Mammogram image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        label_visibility="collapsed",
        help="For the most faithful demo, use a CBIS-DDSM-style image.",
    )

    st.markdown(
        """
        <div class="pipeline">
          <span class="pipe">1. Grayscale</span><span class="arrow">→</span>
          <span class="pipe">2. Breast crop</span><span class="arrow">→</span>
          <span class="pipe">3. Square padding</span><span class="arrow">→</span>
          <span class="pipe">4. 512×512 resize</span><span class="arrow">→</span>
          <span class="pipe">5. ImageNet normalisation</span><span class="arrow">→</span>
          <span class="pipe">6. ROI-guided ResNet-50</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded is None:
        c1, c2, c3 = st.columns(3)
        cards = [
            ("🧠", "Real trained model", "Inference uses the selected λ = 0.5 checkpoint from the dissertation experiment."),
            ("🎯", "Spatial attention", "Class-specific CAMs reveal where the network places benign or malignant evidence."),
            ("📊", "Research context", "Classification and localisation are shown separately, matching the dissertation framing."),
        ]
        for col, (icon, title, text) in zip([c1, c2, c3], cards):
            with col:
                st.markdown(f'<div class="card feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-text">{text}</div></div>', unsafe_allow_html=True)
    else:
        try:
            original = Image.open(io.BytesIO(uploaded.getvalue()))
            processed, x, bbox = preprocess_image(original)
            out = infer(model, x)
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")
            st.stop()

        pred_name = CLASS_NAMES[out["predicted_class"]]
        pred_css = "prediction-malignant" if out["predicted_class"] == 1 else "prediction-benign"
        pred_caption = "Model output crossed the dissertation-selected malignant decision threshold." if out["predicted_class"] == 1 else "Model output remained below the dissertation-selected malignant decision threshold."
        st.markdown(
            f'<div class="prediction {pred_css}"><div class="prediction-label">Research classification</div><div class="prediction-value">{pred_name}</div><div class="mini-note">{pred_caption}</div></div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Malignant model score", f"{100*out['malignant_score']:.1f}%")
        m2.metric("Benign model score", f"{100*out['benign_score']:.1f}%")
        m3.metric("Decision threshold", f"{100*MALIGNANT_THRESHOLD:.0f}%")
        m4.metric("Input resolution", "512×512")

        st.progress(min(max(out["malignant_score"], 0.0), 1.0), text="Malignant model score")
        st.caption("Softmax score shown for research demonstration only; it is not a calibrated clinical probability.")

        # Dynamic research interpretation
        mal_score = 100 * out["malignant_score"]
        ben_score = 100 * out["benign_score"]
        threshold_pct = 100 * MALIGNANT_THRESHOLD

        if out["predicted_class"] == 1:
            interpretation_html = f"""
            <div class="interpretation-box">
              <div class="interpretation-heading">AI Research Interpretation</div>
              <div class="interpretation-text">
                <b>Classification:</b> The ROI-guided ResNet-50 assigned this mammogram to the <b>malignant class</b>.<br><br>
                <b>Why:</b> The malignant model score was <b>{mal_score:.1f}%</b>, compared with <b>{ben_score:.1f}%</b> for the benign class.
                Because the malignant score is above the dissertation-selected decision threshold of <b>{threshold_pct:.0f}%</b>, the model assigns the image to the malignant class.<br><br>
                <b>What this means:</b> The model detected image patterns that its learned representation associates more strongly with the malignant class than with the benign class. This describes the behaviour of the research model only; it does not establish that a person has breast cancer.<br><br>
                <b>How to read the CAM:</b> The Decision CAM below highlights regions contributing to the selected class output. Brighter or more strongly activated areas indicate stronger class-related activation. The CAM is not a tumour segmentation and should not be interpreted as an exact lesion boundary.<br><br>
                <b>Clinical limitation:</b> The displayed score is a neural-network softmax score, not a clinically calibrated probability of cancer. This MSc research model has not been externally validated for clinical diagnosis.
              </div>
            </div>
            """
        else:
            interpretation_html = f"""
            <div class="interpretation-box">
              <div class="interpretation-heading">AI Research Interpretation</div>
              <div class="interpretation-text">
                <b>Classification:</b> The ROI-guided ResNet-50 assigned this mammogram to the <b>benign class</b>.<br><br>
                <b>Why:</b> The malignant model score was <b>{mal_score:.1f}%</b>, compared with <b>{ben_score:.1f}%</b> for the benign class.
                Because the malignant score remains below the dissertation-selected decision threshold of <b>{threshold_pct:.0f}%</b>, the model assigns the image to the benign class.<br><br>
                <b>What this means:</b> The uploaded image was more consistent with image patterns learned for the benign class under the decision rule used in this research. A benign model prediction does not rule out cancer and should not be interpreted as a clinical diagnosis.<br><br>
                <b>How to read the CAM:</b> The Decision CAM below highlights regions contributing to the selected class output. Stronger activation shows where the network is responding, but it does not represent an exact lesion boundary or clinical explanation.<br><br>
                <b>Clinical limitation:</b> The displayed score is a neural-network softmax score, not a clinically calibrated probability. This MSc research model has not been externally validated for clinical diagnosis.
              </div>
            </div>
            """

        st.markdown(interpretation_html, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Visual analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Compare the uploaded image, the processed model input, and the class-specific decision attention.</div>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown('<div class="image-label">Original mammogram</div>', unsafe_allow_html=True)
            st.image(original, use_container_width=True)
        with col_b:
            st.markdown('<div class="image-label">Processed model input</div>', unsafe_allow_html=True)
            st.image(processed, clamp=True, use_container_width=True)
        with col_c:
            st.markdown('<div class="image-label">Decision CAM overlay</div>', unsafe_allow_html=True)
            st.image(make_overlay(processed, out["decision_cam"]), clamp=True, use_container_width=True)

        with st.expander("View preprocessing details"):
            st.write(
                f"Detected breast-region bounding box: left={bbox[0]}, top={bbox[1]}, right={bbox[2]}, bottom={bbox[3]}."
            )
            st.write(
                "Pipeline: grayscale → Otsu foreground estimate → largest connected component → 2% padding → square zero-padding → 512×512 resize → three-channel replication → ImageNet normalisation."
            )

# --------------------------------------------------
# Explainability tab
# --------------------------------------------------
with explain_tab:
    st.markdown('<div class="section-title">Class-specific attention maps</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">The ROI-guided model directly produces separate benign and malignant class activation maps through its 1×1 CAM head.</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.info("Upload a mammogram in the **Analyze Mammogram** tab to view class-specific attention maps here.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><div class="card-title">Benign CAM</div><div class="card-value">Channel 0</div><div class="card-note">Spatial evidence associated with the benign class.</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><div class="card-title">Malignant CAM</div><div class="card-value">Channel 1</div><div class="card-note">Spatial evidence associated with the malignant class.</div></div>', unsafe_allow_html=True)
    else:
        e1, e2 = st.columns(2)
        with e1:
            st.markdown('<div class="image-label">Benign-class CAM</div>', unsafe_allow_html=True)
            st.image(make_overlay(processed, out["benign_cam"]), clamp=True, use_container_width=True)
        with e2:
            st.markdown('<div class="image-label">Malignant-class CAM</div>', unsafe_allow_html=True)
            st.image(make_overlay(processed, out["malignant_cam"]), clamp=True, use_container_width=True)

        decision_class_name = "malignant" if out["predicted_class"] == 1 else "benign"
        alternative_class_name = "benign" if out["predicted_class"] == 1 else "malignant"

        st.markdown(
            f"""
            <div class="interpretation-box">
              <div class="interpretation-heading">Understanding the attention maps</div>
              <div class="interpretation-text">
                The ROI-guided ResNet-50 produces two separate class-specific activation maps through its 1×1 CAM head.<br><br>
                <b>Benign-class CAM:</b> highlights image regions associated with evidence learned for the benign class.<br><br>
                <b>Malignant-class CAM:</b> highlights image regions associated with evidence learned for the malignant class.<br><br>
                <b>Decision map for this case:</b> the model predicted the <b>{decision_class_name} class</b>, so the <b>{decision_class_name}-class CAM</b> is the decision-class activation map. The {alternative_class_name}-class CAM shows the spatial evidence associated with the alternative class.<br><br>
                <b>How to read the colours:</b> brighter or more strongly coloured regions indicate locations with stronger class-related activation. These maps show where the network is responding, not an exact tumour boundary.<br><br>
                <b>Important:</b> the colour intensity of the benign and malignant maps should not be compared as if they were probabilities. The class scores in the <b>Analyze Mammogram</b> tab determine the classification result; these maps provide a spatial visualisation of model evidence.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="research-banner"><b>Interpretation note:</b> activation maps are visual evidence maps, not lesion segmentations. Spatial overlap with a lesion does not prove causal or clinical reasoning.</div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------
# Results tab
# --------------------------------------------------
with results_tab:
    st.markdown('<div class="section-title">Dissertation results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Locked-test performance of the matched control and the selected ROI-guided ResNet-50.</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("ROI model ROC-AUC", "0.7808")
    r2.metric("Control ROC-AUC", "0.7771")
    r3.metric("ROI Pointing Game", "0.1063")
    r4.metric("ROI Pixel AP", "0.1217")

    st.markdown("#### Classification")
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

    st.markdown("#### Localisation")
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
    st.caption("Paired ΔROC-AUC = +0.0037, 95% CI [−0.0222, 0.0298]. The dissertation did not establish a statistically detectable classification AUC improvement.")

# --------------------------------------------------
# Methodology tab
# --------------------------------------------------
with method_tab:
    st.markdown('<div class="section-title">How the research model works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A concise view of the model and how ROI supervision was used.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card feature-card"><div class="feature-icon">1</div><div class="feature-title">Image encoder</div><div class="feature-text">A ResNet-50 extracts spatial feature maps from the 512×512 processed mammogram.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card feature-card"><div class="feature-icon">2</div><div class="feature-title">Class-specific CAM head</div><div class="feature-text">A 1×1 convolution produces benign and malignant activation maps before global average pooling.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card feature-card"><div class="feature-icon">3</div><div class="feature-title">ROI supervision</div><div class="feature-text">During training, the true-class CAM receives an auxiliary ROI-alignment loss. ROI masks are not used at inference.</div></div>', unsafe_allow_html=True)

    st.markdown("#### Training objective")
    st.latex(r"L_{total} = L_{CE} + \lambda\left(0.5L_{BCE} + 0.5L_{Dice}\right)")
    st.markdown(
        """
        **Prototype scope**
        - Uses the dissertation's selected **λ = 0.5** ROI-guided checkpoint.
        - Reproduces the image-only inference path and class-specific CAM visualisation.
        - The Streamlit interface itself is a post-dissertation demonstration layer.
        - No external clinical validation, prospective evaluation, calibration for clinical probability, or regulatory assessment has been performed.
        """
    )

st.markdown('<div class="footer-note">MSc Artificial Intelligence research prototype · De-identified research images only · Not for clinical use</div>', unsafe_allow_html=True)
