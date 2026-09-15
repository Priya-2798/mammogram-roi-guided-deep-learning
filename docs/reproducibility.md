# Reproducibility Guide

## Overview

This repository contains the code, notebooks, result artifacts, figures, and supporting files for the dissertation project:

**Mammogram ROI-Guided Deep Learning**

The project investigates multiple deep learning pipelines for mammogram analysis, including:
- baseline ResNet50 classification,
- ROI registration and localisation,
- ROI attention supervision,
- multimodal image + metadata fusion,
- anomaly detection,
- supervised normal-vs-abnormal classification on mini-MIAS,
- failure-case analysis,
- final dissertation figure generation.

---

## Repository Structure

- `notebooks/` — main experiment notebooks organised by study
- `scripts/` — exported Python versions of the notebooks
- `results/` — saved experiment outputs, metrics, predictions, figures, and summaries
- `figures/` — selected dissertation figures and visual outputs
- `src/` — reusable utility modules for data handling, models, evaluation, localisation, preprocessing, multimodal fusion, and visualisation
- `tests/` — lightweight test files
- `configs/` — configuration files
- `docs/` — documentation files

---

## Main Experimental Notebook Order

The primary notebooks are organised in the following order:

1. `01_eda/01_CBIS_DDSM_EDA.ipynb`
2. `02_baseline_preprocessing/02_preprocessing_Resnet50_baseline.ipynb`
3. `03_gradcam_roi/03_GradCam_ROI.ipynb`
4. `04_roi_registration_localisation/04_ROI_Registration_and_Localisation_Metrics.ipynb`
5. `05_resnet50_v2/05_ResNet50_V2_BreastCrop_512_v5.ipynb`
6. `06_roi_attention_resnet50/06_TwoClass_ROI_Attention_Supervision.ipynb`
7. `07_multimodal_fusion/07_Multimodal_Image_Metadata_Fusion_Corrected.ipynb`
8. `08_convnext_roi_attention/08_ConvNeXt_ROI_Attention_Supervision_Strong.ipynb`
9. `09_anomaly_detection/09_Anomaly_Detection_OneClass.ipynb`
10. `10_mini_mias_normal_abnormal/10_Normal_vs_Abnormal_Anomaly_Detection_miniMIAS_FINAL_FIXED.ipynb`
11. `11_mini_mias_supervised_baseline/11_Supervised_Normal_vs_Abnormal_miniMIAS_Baseline.ipynb`
12. `12_failure_case_analysis/12_Failure_Case_Analysis_ROI_Attention_PORTABLE.ipynb`
13. `13_results_audit/13_Dissertation_Results_Figures_and_Chapter_4_Audit.ipynb`

Legacy or earlier notebook variants are stored in `notebooks/legacy/`.

---

## Data

The full medical imaging dataset is **not redistributed** in this repository.

The original project used:
- **CBIS-DDSM**
- **mini-MIAS**

Only lightweight derived outputs, summary files, result tables, and selected figure artifacts are included where relevant.

Large raw image data and full training datasets should be obtained separately from their original sources, subject to their usage terms.

---

## Environment

The project is based on Python and common scientific / deep learning libraries, including:

- Python
- NumPy
- Pandas
- Matplotlib
- PyTorch
- torchvision
- scikit-learn
- Pillow
- SciPy
- tqdm

Install dependencies using:

```bash
pip install -r requirements.txt