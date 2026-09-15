# Datasets and Splits

This document summarises the datasets, label definitions, split strategy, and reproducibility manifests used in the dissertation experiments.

## 1. CBIS-DDSM

CBIS-DDSM is the primary dataset used for the benign-versus-malignant mammography experiments.

The main experiments use the **mass cases** and link the clinical case-description metadata to the corresponding full mammogram images.

### Labels

The binary classification target is:

- `0` — benign / non-malignant
- `1` — malignant

Where multiple lesion annotations refer to the same mammogram, records are consolidated at image level using:

- patient ID
- left/right breast
- image view

The resulting image-level samples retain relevant metadata including pathology, breast density, mass shape, mass margins, assessment, and ROI-mask information where available.

## 2. CBIS-DDSM Split Design

The official CBIS-DDSM training partition is treated as the **development cohort**.

The official CBIS-DDSM test partition is retained as the **locked test set**.

For the ResNet-50 V2 experiment, the development cohort is divided into training and validation subsets using `StratifiedGroupKFold`:

- grouping variable: patient ID
- stratification variable: binary pathology label
- training and validation patients are disjoint
- the locked test patients are disjoint from both training and validation

The locked test set is not used for model selection or threshold selection.

The same patient-disjoint principle is used by the later ROI-guided and localisation experiments.

## 3. CBIS-DDSM Image Preprocessing

Two main image-processing settings appear in the dissertation.

### Initial baseline

The initial ResNet-50 baseline uses a lower-resolution preprocessing pipeline with images resized to approximately:

`224 x 224`

### Breast-focused experiments

The later experiments use breast-focused preprocessing with:

1. background removal
2. breast-region cropping
3. square padding
4. resizing to `512 x 512`
5. ImageNet-compatible normalisation where required

The 512 x 512 pipeline is used by the main ResNet-50 V2, ROI-supervised ResNet-50, ConvNeXt-Tiny, localisation, multimodal and failure-analysis experiments.

## 4. ROI Information

CBIS-DDSM includes radiologist-provided lesion annotations.

These ROI masks are used for:

- Grad-CAM comparison
- ROI registration
- localisation metrics
- ROI-guided attention supervision
- failure-case analysis

ROI masks are used as an **auxiliary training/evaluation signal**.

The classification models do not require a ground-truth ROI mask at inference time.

## 5. Multimodal Metadata

The multimodal experiments use structured radiological metadata alongside mammogram features.

The retained metadata fields are:

- breast density
- mass shape
- mass margins

BI-RADS assessment is excluded from the final multimodal feature set to reduce label-leakage risk.

## 6. mini-MIAS

mini-MIAS is used for the separate normal-versus-abnormal anomaly-detection pathway.

The dataset contains:

- 322 mammograms
- 161 left/right patient pairs

### Label mapping

`NORM` is mapped to:

`0 = normal`

The following categories are mapped to:

`1 = abnormal`

- `CALC`
- `CIRC`
- `SPIC`
- `MISC`
- `ARCH`
- `ASYM`

Benign abnormalities remain part of the abnormal class because the task is **normal versus abnormal**, not benign versus malignant.

## 7. mini-MIAS Split Design

mini-MIAS uses a deterministic:

`70% / 15% / 15%`

patient-level split for:

- training
- validation
- locked test

Both left and right mammograms from the same patient remain in the same split.

Patient-level stratification is based on whether the patient contains at least one abnormal mammogram.

For the one-class anomaly experiment:

- only normal training images define the normal distribution
- validation data are used for model/threshold selection
- the locked test set is evaluated only after selection

The supervised mini-MIAS baseline uses the same patient-level split when the saved assignments are available.

## 8. Reproducibility Manifests Included in This Repository

The repository contains compact split manifests under `manifests/`.

### CBIS-DDSM ResNet-50 V2

`manifests/cbis_ddsm_resnet_v2_split.csv`

Copied from:

`results/05_ResNet_V2/split_assignments.csv`

### CBIS-DDSM ROI-attention experiments

`manifests/cbis_ddsm_roi_attention_split.csv`

Copied from:

`results/06_ROI_Attention/exact_split_assignments.csv`

### mini-MIAS

`manifests/mini_mias_patient_split.csv`

Copied from:

`results/10_Normal_vs_Abnormal_Anomaly_Detection/exact_patient_level_split_assignments.csv`

These files make the final experimental cohort assignments auditable without redistributing the raw medical-image datasets.

## 9. Data Excluded from the Repository

The raw datasets are intentionally not included.

Excluded data include:

- CBIS-DDSM JPEG/DICOM mammograms
- CBIS-DDSM raw ROI image data
- mini-MIAS PGM mammograms
- preprocessing caches
- temporary feature caches not required for the reported results
- large trained model checkpoints

The repository instead contains:

- notebooks
- reusable source code
- runnable experiment scripts
- split manifests
- experiment configurations
- predictions
- evaluation metrics
- selected figures
- final dissertation result tables

Users should obtain the original datasets from their official distribution sources before reproducing the experiments.
