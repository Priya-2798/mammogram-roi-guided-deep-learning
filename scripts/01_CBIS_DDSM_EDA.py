"""Code-cell export from 01_CBIS_DDSM_EDA.ipynb.

The executed notebook is the experiment record/source of truth.
This export is provided for code review and searchability.
"""

# %% [notebook cell 0]
from google.colab import drive
drive.mount('/content/drive')

# %% [notebook cell 1]
PROJECT_DIR = "/content/drive/MyDrive/BreastCancer_Dissertation"
DATA_DIR = f"{PROJECT_DIR}/Data/CBIS-DDSM"

print(PROJECT_DIR)
print(DATA_DIR)

# %% [notebook cell 2]
import os

PROJECT_DIR = "/content/drive/MyDrive/BreastCancer_dissertation"
DATA_DIR = os.path.join(PROJECT_DIR, "Data", "CBIS-DDSM")

print("Project Directory:", PROJECT_DIR)
print("Data Directory:", DATA_DIR)

# %% [notebook cell 3]
import os

JPEG_DIR = os.path.join(DATA_DIR, "jpeg")

num_case_folders = len(os.listdir(JPEG_DIR))

print("=" * 50)
print("CBIS-DDSM JPEG Dataset Summary")
print("=" * 50)
print(f"Total DICOM study folders : {num_case_folders}")
print("Each folder contains JPEG images converted from one DICOM study.")

# %% [notebook cell 5]
import pandas as pd
import os

# Load the CBIS-DDSM mass training metadata
mass_train = pd.read_csv(
    os.path.join(DATA_DIR, "mass_case_description_train_set.csv")
)

print("Dataset columns:")
print(mass_train.columns.tolist())

display(mass_train.head())

# %% [notebook cell 7]
from PIL import Image
import matplotlib.pyplot as plt
import os

roi_folder = os.path.join(DATA_DIR, "jpeg", roi_uid)
files = os.listdir(roi_folder)

plt.figure(figsize=(8,4))

for i, file in enumerate(files):
    img_path = os.path.join(roi_folder, file)
    img = Image.open(img_path)

    plt.subplot(1, len(files), i+1)
    plt.imshow(img, cmap="gray")
    plt.title(file)
    plt.axis("off")

plt.show()

# %% [notebook cell 9]
import os

def uid_from(path):
    return str(path).split("/")[-2]

print("=" * 60)
print("Dataset Integrity Check")
print("=" * 60)

for col in [
    "image file path",
    "cropped image file path",
    "ROI mask file path"
]:
    uids = mass_train[col].apply(uid_from)
    found = uids.apply(
        lambda u: os.path.isdir(os.path.join(JPEG_DIR, u))
    )

    print(f"{col:<25} : {found.sum()}/{len(found)} ({found.mean():.1%})")

# %% [notebook cell 11]
mass_train["full_jpeg"] = mass_train["image file path"].apply(
    lambda p: next((os.path.join(JPEG_DIR, uid_from(p), f)
                    for f in os.listdir(os.path.join(JPEG_DIR, uid_from(p)))
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))), None)
    if os.path.isdir(os.path.join(JPEG_DIR, uid_from(p))) else None)

mass_train["label"] = (mass_train["pathology"].str.upper() == "MALIGNANT").astype(int)

# one row per physical image; malignant if ANY abnormality on it is malignant
img_df = (mass_train.dropna(subset=["full_jpeg"])
          .groupby(["patient_id", "left or right breast", "image view"], as_index=False)
          .agg(full_jpeg=("full_jpeg", "first"), label=("label", "max")))
print("Linked images:", len(img_df), "| malignant rate:", round(img_df["label"].mean(), 3))

# %% [notebook cell 14]
# Create binary label (0 = Benign, 1 = Malignant)
mass_train["label"] = (
    mass_train["pathology"].str.upper() == "MALIGNANT"
).astype(int)

# %% [notebook cell 15]
import matplotlib.pyplot as plt

# Create binary label if not already present
mass_train["label"] = (
    mass_train["pathology"].str.upper() == "MALIGNANT"
).astype(int)

# Class distribution
plt.figure(figsize=(6,4))
mass_train["pathology"].value_counts().plot(kind="bar")
plt.title("Pathology Distribution in Mass Training Set")
plt.xlabel("Pathology")
plt.ylabel("Number of Cases")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Malignancy rate by metadata
metadata_fields = [
    "assessment",
    "breast_density",
    "mass margins",
    "mass shape",
    "subtlety"
]

for col in metadata_fields:
    if col in mass_train.columns:

        g = (
            mass_train
            .groupby(mass_train[col].astype(str))["label"]
            .agg(["mean", "count"])
        )

        print("\n" + "="*60)
        print(f"Malignancy Rate by {col.title()}")
        print("="*60)

        display(
            g[g["count"] >= 10]
            .sort_values("mean")
            .round(3)
        )

# %% [notebook cell 17]
import os
def uid_from(p): return str(p).split("/")[-2]
def jpegs_in(p):
    folder = os.path.join(JPEG_DIR, uid_from(p))
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# %% [notebook cell 19]
from PIL import Image
import matplotlib.pyplot as plt

def show_grid(df, title, n=6):
    sub = df.sample(min(n, len(df)), random_state=1)
    fig, ax = plt.subplots(1, len(sub), figsize=(3*len(sub), 3.2))
    if len(sub) == 1: ax = [ax]
    for a, (_, r) in zip(ax, sub.iterrows()):
        a.imshow(Image.open(r["full_jpeg"]).convert("L"), cmap="gray"); a.axis("off")
    fig.suptitle(title); plt.tight_layout(); plt.show()

show_grid(img_df[img_df.label == 1], "Sample MALIGNANT mammograms")
show_grid(img_df[img_df.label == 0], "Sample NOT-MALIGNANT mammograms")

# %% [notebook cell 21]
r = mass_train.iloc[0]
full = jpegs_in(r["image file path"])[0]
roi_files = jpegs_in(r["ROI mask file path"])      # usually 2: cropped lesion + binary mask
panels = [("Full mammogram", full)] + [(f"ROI folder #{i+1}", f) for i, f in enumerate(roi_files)]
fig, ax = plt.subplots(1, len(panels), figsize=(4*len(panels), 4))
for a, (t, p) in zip(ax, panels):
    a.imshow(Image.open(p).convert("L"), cmap="gray"); a.set_title(t); a.axis("off")
plt.show()

# %% [notebook cell 24]
import numpy as np
samp = img_df["full_jpeg"].sample(min(150, len(img_df)), random_state=2)
sizes = np.array([Image.open(p).size for p in samp])     # (width, height)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(sizes[:,0], bins=30); ax[0].set_title("Width (px)")
ax[1].hist(sizes[:,1], bins=30); ax[1].set_title("Height (px)")
plt.tight_layout(); plt.show()
print("Width  min/median/max:", sizes[:,0].min(), int(np.median(sizes[:,0])), sizes[:,0].max())
print("Height min/median/max:", sizes[:,1].min(), int(np.median(sizes[:,1])), sizes[:,1].max())

# %% [notebook cell 26]
for col in ["mass shape", "mass margins"]:
    print(f"\n--- {col} ---")
    print(mass_train[col].value_counts(dropna=False))

# %% [notebook cell 28]
miss = mass_train.isnull().sum()
print(miss[miss > 0] if miss.sum() else "No missing values")

# %% [notebook cell 30]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
mass_train["image view"].value_counts().plot(kind="bar", ax=ax[0], title="View (CC vs MLO)")
mass_train["left or right breast"].value_counts().plot(kind="bar", ax=ax[1], title="Laterality")
plt.tight_layout(); plt.show()

# %% [notebook cell 31]


