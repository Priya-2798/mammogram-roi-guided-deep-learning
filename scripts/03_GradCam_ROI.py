"""Code-cell export from 03_GradCam_ROI.ipynb.

The executed notebook is the experiment record/source of truth.
This export is provided for code review and searchability.
"""

# %% [notebook cell 1]
from google.colab import drive
drive.mount("/content/drive")

# %% [notebook cell 3]
import os

PROJECT_DIR = "/content/drive/MyDrive/BreastCancer_dissertation"
DATA_DIR    = os.path.join(PROJECT_DIR, "Data", "CBIS-DDSM")
JPEG_DIR    = os.path.join(DATA_DIR, "jpeg")
CACHE_DIR   = os.path.join(PROJECT_DIR, "cache")
MODEL_DIR   = os.path.join(PROJECT_DIR, "Models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")

MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_mass_baseline_patient_split.pth")
FIG_PATH   = os.path.join(RESULTS_DIR, "resnet50_gradcam_roi_examples.png")

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# %% [notebook cell 5]
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

# %% [notebook cell 7]
def uid_from(path):
    return str(path).split("/")[-2]


def jpegs_from_metadata_path(metadata_path):
    folder = os.path.join(JPEG_DIR, uid_from(metadata_path))
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, fname)
        for fname in sorted(os.listdir(folder))
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def first_jpeg_from_metadata_path(metadata_path):
    files = jpegs_from_metadata_path(metadata_path)
    return files[0] if files else None


def join_unique_paths(paths):
    out = []
    for p in paths:
        if isinstance(p, str) and p not in out:
            out.append(p)
    return "||".join(out)


def build_img_df_with_roi(case_csv):
    df = pd.read_csv(os.path.join(DATA_DIR, case_csv))
    df["label"] = (df["pathology"].str.upper() == "MALIGNANT").astype(int)
    df["full_jpeg"] = df["image file path"].apply(first_jpeg_from_metadata_path)

    out = (
        df.dropna(subset=["full_jpeg"])
          .groupby(["patient_id", "left or right breast", "image view"], as_index=False)
          .agg(
              full_jpeg=("full_jpeg", "first"),
              label=("label", "max"),
              pathology=("pathology", lambda x: "; ".join(sorted(set(map(str, x))))),
              assessment=("assessment", lambda x: "; ".join(sorted(set(map(str, x))))),
              breast_density=("breast_density", "first"),
              mass_shape=("mass shape", lambda x: "; ".join(sorted(set(map(str, x))))),
              mass_margins=("mass margins", lambda x: "; ".join(sorted(set(map(str, x))))),
              roi_mask_paths=("ROI mask file path", join_unique_paths),
              cropped_paths=("cropped image file path", join_unique_paths),
          )
    )
    return out


img_df_test = build_img_df_with_roi("mass_case_description_test_set.csv")
print("Locked test images:", len(img_df_test))
display(img_df_test.head())

# %% [notebook cell 9]
def row_keys(df):
    return (
        df["patient_id"].astype(str) + "|" +
        df["left or right breast"].astype(str) + "|" +
        df["image view"].astype(str)
    ).to_numpy()


def preprocess(df, size=IMG_SIZE):
    X = np.zeros((len(df), size, size), dtype=np.uint8)
    y = df["label"].to_numpy(dtype=np.int64)
    for i, path in enumerate(tqdm(df["full_jpeg"].tolist(), desc="preprocessing test images")):
        X[i] = np.array(Image.open(path).convert("L").resize((size, size)), dtype=np.uint8)
    return X, y


def load_or_build_test_cache(df):
    cache = os.path.join(CACHE_DIR, "mass_test_224.npz")
    expected_keys = row_keys(df)
    expected_y = df["label"].to_numpy(dtype=np.int64)

    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if "row_key" in d.files and np.array_equal(d["row_key"].astype(str), expected_keys.astype(str)) and np.array_equal(d["y"], expected_y):
            print("Loaded validated cache: mass_test_224.npz")
            return d["X"], d["y"]
        print("Cache mismatch detected; rebuilding test cache")

    X, y = preprocess(df)
    np.savez_compressed(cache, X=X, y=y, row_key=expected_keys)
    return X, y


X_test, y_test = load_or_build_test_cache(img_df_test)
print("Test array:", X_test.shape)

# %% [notebook cell 11]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ArrayDS(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        t = torch.from_numpy(np.stack([self.X[i]] * 3, axis=0)).float() / 255.0
        return (t - MEAN) / STD, int(self.y[i])


test_loader = DataLoader(
    ArrayDS(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Loaded model:", MODEL_PATH)

# %% [notebook cell 13]
@torch.no_grad()
def eval_probs(loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
        ps.extend(p.tolist())
        ys.extend(y.tolist())
    return np.array(ys), np.array(ps)


yt, pt = eval_probs(test_loader)
pred = (pt >= 0.5).astype(int)

print("Locked test AUC:", round(roc_auc_score(yt, pt), 4))
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(yt, pred))

pred_df = img_df_test.copy()
pred_df["y_true"] = yt
pred_df["p_malignant"] = pt
pred_df["y_pred"] = pred
pred_df["outcome"] = np.select(
    [
        (pred_df["y_true"] == 1) & (pred_df["y_pred"] == 1),
        (pred_df["y_true"] == 1) & (pred_df["y_pred"] == 0),
        (pred_df["y_true"] == 0) & (pred_df["y_pred"] == 1),
        (pred_df["y_true"] == 0) & (pred_df["y_pred"] == 0),
    ],
    ["true_malignant", "false_negative", "false_positive", "true_not_malignant"],
    default="other",
)

display(pred_df[["patient_id", "left or right breast", "image view", "y_true", "p_malignant", "y_pred", "outcome"]].head())

# %% [notebook cell 15]
def pick_case(df, outcome, ascending):
    sub = df[df["outcome"] == outcome].copy()
    if len(sub) == 0:
        return None
    return sub.sort_values("p_malignant", ascending=ascending).index[0]


case_indices = {
    "True malignant correctly classified": pick_case(pred_df, "true_malignant", ascending=False),
    "False negative": pick_case(pred_df, "false_negative", ascending=True),
    "False positive": pick_case(pred_df, "false_positive", ascending=False),
    "True not-malignant correctly classified": pick_case(pred_df, "true_not_malignant", ascending=True),
}

case_indices = {k: v for k, v in case_indices.items() if v is not None}

for label, idx in case_indices.items():
    r = pred_df.loc[idx]
    print(f"{label}: index={idx}, patient={r['patient_id']}, true={r['y_true']}, pred={r['y_pred']}, p_malignant={r['p_malignant']:.3f}")

# %% [notebook cell 17]
def tensor_from_array(arr):
    t = torch.from_numpy(np.stack([arr] * 3, axis=0)).float() / 255.0
    return ((t - MEAN) / STD).unsqueeze(0).to(device)


def compute_gradcam(model, arr, target_class=1):
    activations = []
    gradients = []

    target_layer = model.layer4[-1]

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    x = tensor_from_array(arr)
    logits = model(x)
    score = logits[:, target_class].sum()
    score.backward()

    act = activations[0]
    grad = gradients[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam_map = (weights * act).sum(dim=1, keepdim=True)
    cam_map = F.relu(cam_map)
    cam_map = F.interpolate(cam_map, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam_map = cam_map.squeeze().detach().cpu().numpy()

    cam_map = cam_map - cam_map.min()
    if cam_map.max() > 0:
        cam_map = cam_map / cam_map.max()

    h1.remove()
    h2.remove()
    return cam_map


def overlay_cam(gray, cam_map, alpha=0.45):
    gray_norm = gray.astype(float) / 255.0
    rgb = np.dstack([gray_norm, gray_norm, gray_norm])
    heat = matplotlib.colormaps["jet"](cam_map)[:, :, :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


def binary_likeness(path):
    arr = np.array(Image.open(path).convert("L"))
    if arr.size == 0:
        return -1
    near_binary = np.mean((arr < 20) | (arr > 235))
    contrast = arr.std() / 255.0
    return near_binary + contrast


def best_roi_mask_path(row):
    paths = []
    for metadata_path in str(row["roi_mask_paths"]).split("||"):
        paths.extend(jpegs_from_metadata_path(metadata_path))
    paths = list(dict.fromkeys(paths))
    if not paths:
        return None
    return sorted(paths, key=binary_likeness, reverse=True)[0]


def load_roi_mask_for_display(row):
    path = best_roi_mask_path(row)
    if path is None:
        return None, None
    arr = np.array(Image.open(path).convert("L"))
    return arr, path

# %% [notebook cell 19]
n_cases = len(case_indices)
fig, axes = plt.subplots(n_cases, 3, figsize=(13, 4 * n_cases))
if n_cases == 1:
    axes = np.array([axes])

for row_i, (case_label, idx) in enumerate(case_indices.items()):
    row = pred_df.loc[idx]
    gray = X_test[idx]
    cam_map = compute_gradcam(model, gray, target_class=1)
    overlay = overlay_cam(gray, cam_map)
    roi_arr, roi_path = load_roi_mask_for_display(row)

    title = (
        f"{case_label}\n"
        f"true={int(row['y_true'])}, pred={int(row['y_pred'])}, p(malignant)={row['p_malignant']:.3f}\n"
        f"patient={row['patient_id']}, {row['left or right breast']} {row['image view']}"
    )

    axes[row_i, 0].imshow(gray, cmap="gray")
    axes[row_i, 0].set_title("Full mammogram")
    axes[row_i, 0].axis("off")

    axes[row_i, 1].imshow(overlay)
    axes[row_i, 1].set_title("Grad-CAM for malignant class")
    axes[row_i, 1].axis("off")

    if roi_arr is not None:
        axes[row_i, 2].imshow(roi_arr, cmap="gray")
        axes[row_i, 2].set_title("CBIS-DDSM ROI mask")
    else:
        axes[row_i, 2].text(0.5, 0.5, "ROI mask not found", ha="center", va="center")
        axes[row_i, 2].set_title("CBIS-DDSM ROI mask")
    axes[row_i, 2].axis("off")

    axes[row_i, 0].text(
        0.0, -0.12, title,
        transform=axes[row_i, 0].transAxes,
        fontsize=10,
        va="top",
    )

plt.tight_layout()
plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
plt.show()
print("Saved Grad-CAM figure to:", FIG_PATH)

# %% [notebook cell 21]
def first_existing_jpeg_from_joined_paths(joined_paths):
    for metadata_path in str(joined_paths).split("||"):
        files = jpegs_from_metadata_path(metadata_path)
        if files:
            return files[0]
    return None


def load_cropped_lesion_for_display(row):
    path = first_existing_jpeg_from_joined_paths(row["cropped_paths"])
    if path is None:
        return None, None
    return np.array(Image.open(path).convert("L")), path


def select_alignment_cases(pred_df, max_cases=8):
    selected = []

    # Keep the four outcome examples already used for Grad-CAM interpretation.
    for idx in case_indices.values():
        if idx not in selected:
            selected.append(idx)

    # Add extra malignant and non-malignant examples for a broader visual check.
    candidates = (
        pred_df.assign(confidence=np.abs(pred_df["p_malignant"] - 0.5))
               .sort_values("confidence", ascending=False)
               .index
               .tolist()
    )
    for idx in candidates:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= max_cases:
            break

    return selected[:max_cases]


alignment_indices = select_alignment_cases(pred_df, max_cases=8)
print("Selected cases for ROI alignment check:", alignment_indices)

# %% [notebook cell 22]
n_align = len(alignment_indices)
fig, axes = plt.subplots(n_align, 4, figsize=(16, 4 * n_align))
if n_align == 1:
    axes = np.array([axes])

for row_i, idx in enumerate(alignment_indices):
    row = pred_df.loc[idx]
    gray = X_test[idx]
    cam_map = compute_gradcam(model, gray, target_class=1)
    overlay = overlay_cam(gray, cam_map)
    crop_arr, crop_path = load_cropped_lesion_for_display(row)
    roi_arr, roi_path = load_roi_mask_for_display(row)

    axes[row_i, 0].imshow(gray, cmap="gray")
    axes[row_i, 0].set_title("Full mammogram")
    axes[row_i, 0].axis("off")

    axes[row_i, 1].imshow(overlay)
    axes[row_i, 1].set_title("Grad-CAM")
    axes[row_i, 1].axis("off")

    if crop_arr is not None:
        axes[row_i, 2].imshow(crop_arr, cmap="gray")
        axes[row_i, 2].set_title("Cropped lesion")
    else:
        axes[row_i, 2].text(0.5, 0.5, "Crop not found", ha="center", va="center")
        axes[row_i, 2].set_title("Cropped lesion")
    axes[row_i, 2].axis("off")

    if roi_arr is not None:
        axes[row_i, 3].imshow(roi_arr, cmap="gray")
        axes[row_i, 3].set_title("ROI mask")
    else:
        axes[row_i, 3].text(0.5, 0.5, "ROI not found", ha="center", va="center")
        axes[row_i, 3].set_title("ROI mask")
    axes[row_i, 3].axis("off")

    caption = (
        f"{row['outcome']} | true={int(row['y_true'])}, pred={int(row['y_pred'])}, "
        f"p(malignant)={row['p_malignant']:.3f} | patient={row['patient_id']}, "
        f"{row['left or right breast']} {row['image view']}"
    )
    axes[row_i, 0].text(0.0, -0.12, caption, transform=axes[row_i, 0].transAxes, fontsize=10, va="top")

plt.tight_layout()
plt.show()

