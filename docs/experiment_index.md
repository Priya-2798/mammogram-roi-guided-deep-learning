@'
# Experiment Index

This document provides a compact index of the experimental pipeline used in the dissertation **Deep Learning in Tumour Detection and Mammogram Classification**.

For detailed links between dissertation sections, notebooks, scripts, source modules and outputs, see:

`docs/dissertation_mapping.md`

---

## Experiment Overview

| ID | Experiment | Dataset | Primary purpose | Main output |
|---|---|---|---|---|
| 01 | CBIS-DDSM exploratory data analysis | CBIS-DDSM | Dataset inspection, metadata analysis and image linkage | Dataset diagnostics |
| 02 | ResNet-50 V1 baseline | CBIS-DDSM | Initial benign-vs-malignant image classifier | Baseline classification metrics |
| 03 | Grad-CAM ROI analysis | CBIS-DDSM | Compare model attention with radiologist lesion regions | Grad-CAM visualisations |
| 04 | ROI registration and localisation metrics | CBIS-DDSM | Register ROI masks and quantify localisation | Pointing Game, pixel AUC/AP, Dice |
| 05 | ResNet-50 V2 breast-focused baseline | CBIS-DDSM | Improved 512 x 512 image-only classifier | Locked-test classification |
| 06 | ResNet-50 ROI-guided attention | CBIS-DDSM | Test auxiliary ROI supervision | Classification + localisation |
| 07 | Multimodal image-metadata fusion | CBIS-DDSM | Test contribution of structured radiological metadata | M1-M4 ablation |
| 08 | ConvNeXt-Tiny ROI replication | CBIS-DDSM | Cross-backbone validation of ROI supervision | Classification + localisation |
| 09 | One-class anomaly exploration | CBIS-DDSM | Exploratory deep-feature anomaly detection | One-class metrics |
| 10 | Normal-vs-abnormal anomaly detection | mini-MIAS | Final one-class normal/abnormal experiment | Mahalanobis, Isolation Forest, OCSVM |
| 11 | Supervised normal-vs-abnormal baseline | mini-MIAS | Compare one-class modelling with supervised learning | Supervised ResNet-50 metrics |
| 12 | Failure-case analysis | CBIS-DDSM | Analyse false positives, false negatives and attention patterns | Qualitative diagnostic panels |
| 13 | Final results audit | Combined outputs | Verify final dissertation values and generate Chapter 4 artefacts | Final tables and figures |

---

# 01 — CBIS-DDSM Exploratory Data Analysis

**Notebook**

`notebooks/01_eda/01_CBIS_DDSM_EDA.ipynb`

**Script**

`scripts/01_CBIS_DDSM_EDA.py`

**Purpose**

- inspect CBIS-DDSM metadata
- analyse pathology and clinical-variable distributions
- connect case descriptions to mammogram files
- check dataset structure before model development

---

# 02 — ResNet-50 V1 Baseline

**Notebook**

`notebooks/02_baseline_preprocessing/02_preprocessing_Resnet50_baseline.ipynb`

**Script**

`scripts/02_preprocessing_Resnet50_baseline.py`

**Results**

`results/02_ResNet_V1/`

**Purpose**

Establish the initial benign-versus-malignant classification baseline.

---

# 03 — Grad-CAM and ROI Analysis

**Notebook**

`notebooks/03_gradcam_roi/03_GradCam_ROI.ipynb`

**Script**

`scripts/03_GradCam_ROI.py`

**Purpose**

Visualise model attention and compare Grad-CAM activation with radiologist-provided lesion regions.

---

# 04 — ROI Registration and Localisation

**Notebook**

`notebooks/04_roi_registration_localisation/04_ROI_Registration_and_Localisation_Metrics.ipynb`

**Script**

`scripts/04_ROI_Registration_and_Localisation_Metrics.py`

**Purpose**

Prepare registered ROI masks and calculate localisation metrics used by later experiments.

**Metrics**

- Pointing Game
- pixel ROC-AUC
- pixel average precision
- Dice score

---

# 05 — ResNet-50 V2 Breast-Focused Baseline

**Notebook**

`notebooks/05_resnet50_v2/05_ResNet50_V2_BreastCrop_512_v5.ipynb`

**Script**

`scripts/05_ResNet50_V2_BreastCrop_512_v5.py`

**Results**

`results/05_ResNet_V2/`

**Manifest**

`manifests/cbis_ddsm_resnet_v2_split.csv`

**Purpose**

Establish the principal 512 x 512 breast-focused image-only baseline used for comparison with later models.

---

# 06 — ResNet-50 ROI-Guided Attention Supervision

**Notebook**

`notebooks/06_roi_attention_resnet50/06_TwoClass_ROI_Attention_Supervision.ipynb`

**Script**

`scripts/06_TwoClass_ROI_Attention_Supervision.py`

**Results**

`results/06_ROI_Attention/`

**Manifest**

`manifests/cbis_ddsm_roi_attention_split.csv`

**Purpose**

Test whether auxiliary radiologist-ROI supervision improves spatial alignment while maintaining image-level classification.

**Lambda sweep**

`0.0, 0.1, 0.5, 1.0`

The lambda-zero model is the matched architectural control.

---

# 07 — Multimodal Image + Metadata Fusion

**Notebook**

`notebooks/07_multimodal_fusion/07_Multimodal_Image_Metadata_Fusion_Corrected.ipynb`

**Script**

`scripts/07_Multimodal_Image_Metadata_Fusion_Corrected.py`

**Results**

`results/07_Multimodal/`

**Configurations**

- M1 — image only
- M2 — metadata only
- M3 — image + metadata
- M4 — ROI-initialised image representation + metadata

**Metadata**

- breast density
- mass shape
- mass margins

BI-RADS assessment is excluded from the final multimodal inputs.

---

# 08 — ConvNeXt-Tiny ROI-Attention Replication

**Notebook**

`notebooks/08_convnext_roi_attention/08_ConvNeXt_ROI_Attention_Supervision_Strong.ipynb`

**Script**

`scripts/08_ConvNeXt_ROI_Attention_Supervision_Strong.py`

**Results**

`results/08_ConvNeXt_ROI/`

**Purpose**

Evaluate whether ROI-supervision behaviour observed with ResNet-50 transfers to a different convolutional backbone.

---

# 09 — Exploratory One-Class Anomaly Detection

**Notebook**

`notebooks/09_anomaly_detection/09_Anomaly_Detection_OneClass.ipynb`

**Script**

`scripts/09_Anomaly_Detection_OneClass.py`

**Results**

`results/09_Anomaly/`

**Purpose**

Explore deep-feature one-class anomaly detection before the final same-source mini-MIAS experiment.

This experiment is retained for completeness but is separate from the final normal-versus-abnormal study.

---

# 10 — mini-MIAS Normal-vs-Abnormal Anomaly Detection

**Notebook**

`notebooks/10_mini_mias_normal_abnormal/10_Normal_vs_Abnormal_Anomaly_Detection_miniMIAS_FINAL_FIXED.ipynb`

**Script**

`scripts/10_Normal_vs_Abnormal_Anomaly_Detection_miniMIAS_FINAL_FIXED.py`

**Results**

`results/10_Normal_vs_Abnormal_Anomaly_Detection/`

**Manifest**

`manifests/mini_mias_patient_split.csv`

**Methods**

- Mahalanobis distance with covariance shrinkage
- Isolation Forest
- One-Class SVM

**Feature extractor**

Frozen ImageNet-pretrained ResNet-50.

**Training principle**

The one-class models are fitted using normal training mammograms only.

---

# 11 — Supervised mini-MIAS Comparator

**Notebook**

`notebooks/11_mini_mias_supervised_baseline/11_Supervised_Normal_vs_Abnormal_miniMIAS_Baseline.ipynb`

**Script**

`scripts/11_Supervised_Normal_vs_Abnormal_miniMIAS_Baseline.py`

**Results**

`results/11_Supervised_Normal_vs_Abnormal_miniMIAS/`

**Purpose**

Provide a conventional supervised ResNet-50 comparator for the same normal-versus-abnormal task.

---

# 12 — Failure-Case Analysis

**Notebook**

`notebooks/12_failure_case_analysis/12_Failure_Case_Analysis_ROI_Attention_PORTABLE.ipynb`

**Script**

`scripts/12_Failure_Case_Analysis_ROI_Attention_PORTABLE.py`

**Results**

`results/12_Failure_Case_Analysis/`

**Purpose**

Investigate:

- false positives
- false negatives
- model attention behaviour
- localisation quality in difficult cases

---

# 13 — Final Dissertation Results Audit

**Notebook**

`notebooks/13_results_audit/13_Dissertation_Results_Figures_and_Chapter_4_Audit.ipynb`

**Script**

`scripts/13_Dissertation_Results_Figures_and_Chapter_4_Audit.py`

**Results**

`results/13_Dissertation_Figures/`

**Purpose**

- audit saved experimental outputs
- verify final reported metrics
- reproduce Chapter 4 tables
- reproduce final comparison figures

---

# Principal Result Folders

```text
results/
├── 02_ResNet_V1/
├── 05_ResNet_V2/
├── 06_ROI_Attention/
├── 07_Multimodal/
├── 08_ConvNeXt_ROI/
├── 09_Anomaly/
├── 10_Normal_vs_Abnormal_Anomaly_Detection/
├── 11_Supervised_Normal_vs_Abnormal_miniMIAS/
├── 12_Failure_Case_Analysis/
└── 13_Dissertation_Figures/