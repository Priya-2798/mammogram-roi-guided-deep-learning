"""Code-cell export from 04_ROI_Registration_and_Localisation_Metrics.ipynb.

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
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "resnet50_mass_baseline_patient_split.pth")
METRICS_PATH = os.path.join(RESULTS_DIR, "localisation_metrics.csv")
FIG_PATH     = os.path.join(RESULTS_DIR, "gradcam_localisation_examples.png")

IMG_SIZE       = 224
SIZE_TOL       = 0.03    # mask accepted as "full-size" if within 3% of full-image dims
CAM_THRESHOLD  = 0.5     # normalised-CAM threshold for IoU/Dice (state this in the write-up)
SEED           = 42

# %% [notebook cell 5]
import random
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from PIL import Image
import matplotlib, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# %% [notebook cell 7]
def uid_from(p): return str(p).split("/")[-2]

def jpegs_from_metadata_path(mp):
    folder = os.path.join(JPEG_DIR, uid_from(mp))
    if not os.path.isdir(folder): return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

def first_jpeg(mp):
    fs = jpegs_from_metadata_path(mp); return fs[0] if fs else None

def join_unique(paths):
    out = []
    for p in paths:
        if isinstance(p, str) and p not in out: out.append(p)
    return "||".join(out)

def build_test_df(case_csv):
    df = pd.read_csv(os.path.join(DATA_DIR, case_csv))
    df["label"] = (df["pathology"].str.upper() == "MALIGNANT").astype(int)
    df["full_jpeg"] = df["image file path"].apply(first_jpeg)
    return (df.dropna(subset=["full_jpeg"])
              .groupby(["patient_id","left or right breast","image view"], as_index=False)
              .agg(full_jpeg=("full_jpeg","first"), label=("label","max"),
                   roi_mask_paths=("ROI mask file path", join_unique)))

img_df_test = build_test_df("mass_case_description_test_set.csv")
print("Locked test images:", len(img_df_test))

# %% [notebook cell 9]
d = np.load(os.path.join(CACHE_DIR, "mass_test_224.npz"), allow_pickle=True)
X_test, y_test = d["X"], d["y"]
assert len(X_test) == len(img_df_test), "cache/metadata length mismatch"
assert np.array_equal(d["y"], img_df_test["label"].to_numpy()), "label order mismatch"
print("Test array:", X_test.shape)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()
print("Loaded", MODEL_PATH)

# %% [notebook cell 11]
MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
STD  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def to_input(arr):
    t = torch.from_numpy(np.stack([arr]*3, 0)).float()/255.0
    return ((t-MEAN)/STD).unsqueeze(0).to(device)

def compute_gradcam(arr, target_class=1):
    acts, grads = [], []
    layer = model.layer4[-1]
    h1 = layer.register_forward_hook(lambda m,i,o: acts.append(o))
    h2 = layer.register_full_backward_hook(lambda m,gi,go: grads.append(go[0]))
    model.zero_grad(set_to_none=True)
    logits = model(to_input(arr))
    logits[:, target_class].sum().backward()
    w = grads[0].mean(dim=(2,3), keepdim=True)
    cam = F.relu((w*acts[0]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE,IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0: cam = cam/cam.max()
    h1.remove(); h2.remove()
    return cam

# %% [notebook cell 13]
def register_mask_224(row, full_size, size=IMG_SIZE, tol=SIZE_TOL):
    W, H = full_size
    union = np.zeros((size, size), dtype=np.uint8); found = False
    for mp in str(row["roi_mask_paths"]).split("||"):
        best, best_bin = None, -1.0
        for f in jpegs_from_metadata_path(mp):
            with Image.open(f) as im0: w, h = im0.size
            if abs(w-W)/max(W,1) > tol or abs(h-H)/max(H,1) > tol:
                continue                      # not full-size -> cannot register
            a = np.array(Image.open(f).convert("L"))
            binlike = np.mean((a < 20) | (a > 235))
            if binlike > best_bin:
                best_bin, best = binlike, Image.open(f).convert("L")
        if best is not None:
            m = np.array(best.resize((size, size), Image.NEAREST))
            union = np.maximum(union, (m > 127).astype(np.uint8)); found = True
    return union if found else None

# %% [notebook cell 15]
masks = np.zeros((len(img_df_test), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
valid = np.zeros(len(img_df_test), dtype=bool)
for i, row in tqdm(img_df_test.iterrows(), total=len(img_df_test), desc="registering masks"):
    with Image.open(row["full_jpeg"]) as im: full_size = im.size   # (W, H)
    m = register_mask_224(row, full_size)
    if m is not None and m.sum() > 0:
        masks[i] = m; valid[i] = True

cov = valid.mean()
areas = np.array([masks[i].mean() for i in range(len(masks)) if valid[i]])
print(f"Registrable masks: {valid.sum()}/{len(valid)}  ({cov:.1%} coverage)")
print(f"Mean lesion area fraction (valid): {areas.mean():.4f}  "
      f"[this is the chance Pointing-Game hit rate]")

# %% [notebook cell 17]
vidx = [i for i in range(len(masks)) if valid[i]]
show = random.sample(vidx, min(6, len(vidx)))
fig, ax = plt.subplots(1, len(show), figsize=(3*len(show), 3.2))
if len(show) == 1: ax = [ax]
for a, i in zip(ax, show):
    a.imshow(X_test[i], cmap="gray")
    a.contour(masks[i], levels=[0.5], colors="red", linewidths=1.2)
    a.set_title("malig" if y_test[i] else "benign", fontsize=9); a.axis("off")
plt.suptitle("Registered ROI mask (red contour) over 224x224 mammogram"); plt.tight_layout(); plt.show()

# %% [notebook cell 19]
cams = np.zeros((len(img_df_test), IMG_SIZE, IMG_SIZE), dtype=np.float32)
for i in tqdm(range(len(img_df_test)), desc="grad-cam"):
    cams[i] = compute_gradcam(X_test[i], target_class=1)

# %% [notebook cell 21]
def metrics_for(indices):
    pg, pauc, iou, dice = [], [], [], []
    for i in indices:
        cam, m = cams[i], masks[i]
        peak = np.unravel_index(cam.argmax(), cam.shape)
        pg.append(int(m[peak] == 1))
        if 0 < m.sum() < m.size:
            pauc.append(roc_auc_score(m.flatten(), cam.flatten()))
        cb = (cam >= CAM_THRESHOLD).astype(np.uint8)
        inter = np.logical_and(cb, m).sum(); union = np.logical_or(cb, m).sum()
        iou.append(inter/union if union else 0.0)
        dice.append(2*inter/(cb.sum()+m.sum()) if (cb.sum()+m.sum()) else 0.0)
    return dict(n=len(indices), pointing_game=np.mean(pg), pixel_auc=np.mean(pauc),
                iou=np.mean(iou), dice=np.mean(dice))

all_valid = [i for i in range(len(masks)) if valid[i]]
malig_valid = [i for i in all_valid if y_test[i] == 1]
chance_pg = float(np.mean([masks[i].mean() for i in all_valid]))

rows = []
for name, idx in [("all lesions", all_valid), ("malignant only", malig_valid)]:
    r = metrics_for(idx); r["subset"] = name; rows.append(r)
res = pd.DataFrame(rows)[["subset","n","pointing_game","pixel_auc","iou","dice"]]
res["pointing_game_chance"] = round(chance_pg, 4)
res["pixel_auc_chance"] = 0.5
res = res.round(4)
res.to_csv(METRICS_PATH, index=False)
print("Saved", METRICS_PATH)
display(res)

# %% [notebook cell 23]
def overlay(gray, cam, alpha=0.45):
    g = gray.astype(float)/255.0; rgb = np.dstack([g,g,g])
    heat = matplotlib.colormaps["jet"](cam)[:,:,:3]
    return np.clip((1-alpha)*rgb + alpha*heat, 0, 1)

show = [i for i in malig_valid][:4] or all_valid[:4]
fig, ax = plt.subplots(len(show), 3, figsize=(11, 3.6*len(show)))
if len(show) == 1: ax = ax[None, :]
for r, i in enumerate(show):
    peak = np.unravel_index(cams[i].argmax(), cams[i].shape)
    hit = "HIT" if masks[i][peak] == 1 else "MISS"
    ax[r,0].imshow(X_test[i], cmap="gray"); ax[r,0].set_title("mammogram"); ax[r,0].axis("off")
    ax[r,1].imshow(overlay(X_test[i], cams[i]))
    ax[r,1].plot(peak[1], peak[0], "w*", markersize=12)
    ax[r,1].set_title(f"Grad-CAM (peak {hit})"); ax[r,1].axis("off")
    ax[r,2].imshow(X_test[i], cmap="gray")
    ax[r,2].contour(masks[i], levels=[0.5], colors="red", linewidths=1.2)
    ax[r,2].set_title("registered ROI mask"); ax[r,2].axis("off")
plt.tight_layout(); plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight"); plt.show()
print("Saved", FIG_PATH)

