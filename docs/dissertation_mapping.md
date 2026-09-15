# Dissertation-to-Code Mapping

This document maps the dissertation experiments to the corresponding notebooks, Python scripts, reusable source modules, result folders, and figures in this repository.

The purpose is to make the submitted code auditable and to show how the experimental evidence in the dissertation was produced.

---

## 1. Dataset Exploration — CBIS-DDSM

### Dissertation role
Dataset understanding, class distributions, clinical metadata exploration, image linkage, and initial data-quality checks.

### Notebook
`notebooks/01_eda/01_CBIS_DDSM_EDA.ipynb`

### Script
`scripts/01_CBIS_DDSM_EDA.py`

### Main code
`src/data/cbis_ddsm.py`

---

## 2. Initial ResNet-50 Baseline

### Dissertation role
Initial benign-versus-malignant image-classification baseline using ResNet-50.

This experiment establishes the first reference point before the later breast-focused 512 x 512 pipeline.

### Notebook
`notebooks/02_baseline_preprocessing/02_preprocessing_Resnet50_baseline.ipynb`

### Script
`scripts/02_preprocessing_Resnet50_baseline.py`

### Main code
- `src/models/resnet50_baseline.py`
- `src/models/resnet.py`
- `src/preprocessing/mammogram.py`
- `src/evaluation/classification.py`

### Results
`results/02_ResNet_V1/`

---

## 3. Grad-CAM and Radiologist ROI Comparison

### Dissertation role
Qualitative investigation of whether model attention overlaps with radiologist-provided lesion regions.

This stage motivates the later quantitative localisation and ROI-supervision experiments.

### Notebook
`notebooks/03_gradcam_roi/03_GradCam_ROI.ipynb`

### Script
`scripts/03_GradCam_ROI.py`

### Main code
- `src/localisation/gradcam.py`
- `src/visualisation/overlays.py`

### Figures
Relevant visualisations are available under:

`figures/Gradcam/`

---

## 4. ROI Registration and Localisation Metrics

### Dissertation role
Alignment of radiologist ROI masks with model-resolution mammograms and implementation of localisation metrics.

The principal localisation metrics include:

- Pointing Game
- pixel ROC-AUC
- pixel average precision
- Dice score

### Notebook
`notebooks/04_roi_registration_localisation/04_ROI_Registration_and_Localisation_Metrics.ipynb`

### Script
`scripts/04_ROI_Registration_and_Localisation_Metrics.py`

### Main code
- `src/localisation/roi_metrics.py`
- `src/localisation/gradcam.py`
- `src/visualisation/overlays.py`

---

## 5. ResNet-50 V2 Breast-Focused Baseline

### Dissertation role
Improved image-only baseline using the breast-focused 512 x 512 preprocessing pipeline.

The development cohort is split at patient level, while the official CBIS-DDSM test cohort remains locked.

### Notebook
`notebooks/05_resnet50_v2/05_ResNet50_V2_BreastCrop_512_v5.ipynb`

### Script
`scripts/05_ResNet50_V2_BreastCrop_512_v5.py`

### Main code
- `src/data/cbis_ddsm.py`
- `src/preprocessing/mammogram.py`
- `src/models/resnet.py`
- `src/evaluation/classification.py`
- `src/evaluation/bootstrap.py`
- `src/localisation/gradcam.py`

### Results
`results/05_ResNet_V2/`

Important artefacts include:

- `test_metrics.json`
- `test_predictions.csv`
- `split_assignments.csv`
- `selected_threshold.json`
- `training_history.csv`
- `v1_v2_comparison.csv`
- Grad-CAM and error-analysis figures

### Split manifest
`manifests/cbis_ddsm_resnet_v2_split.csv`

---

## 6. ResNet-50 ROI-Guided Attention Supervision

### Dissertation role
Primary ROI-guided experiment.

A class-specific CAM is supervised against the radiologist ROI during training using an auxiliary spatial loss.

Classification remains image-level and no ground-truth ROI is required at inference.

A validation sweep is performed over:

`lambda = {0.0, 0.1, 0.5, 1.0}`

The lambda-zero model provides the identical-architecture control.

### Notebook
`notebooks/06_roi_attention_resnet50/06_TwoClass_ROI_Attention_Supervision.ipynb`

### Script
`scripts/06_TwoClass_ROI_Attention_Supervision.py`

### Main code
- `src/models/cam_models.py`
- `src/roi_attention/losses.py`
- `src/localisation/roi_metrics.py`
- `src/evaluation/classification.py`
- `src/evaluation/bootstrap.py`

### Results
`results/06_ROI_Attention/`

Important artefacts include:

- `validation_by_lambda.csv`
- `classification_test.csv`
- `localisation_test.csv`
- `paired_localisation_bootstrap.csv`
- `selected_lambda.json`
- `selected_thresholds.json`
- per-case localisation CSV files
- locked-test predictions
- qualitative attention panel

### Split manifest
`manifests/cbis_ddsm_roi_attention_split.csv`

---

## 7. Multimodal Image + Metadata Fusion

### Dissertation role
Evaluation of whether structured radiological metadata contributes information beyond mammogram-image features.

The experiment compares:

- M1 — image only
- M2 — metadata only
- M3 — image + metadata
- M4 — ROI-initialised image features + metadata

The retained metadata fields are:

- breast density
- mass shape
- mass margins

BI-RADS assessment is excluded from the final multimodal inputs to reduce leakage risk.

### Notebook
`notebooks/07_multimodal_fusion/07_Multimodal_Image_Metadata_Fusion_Corrected.ipynb`

### Script
`scripts/07_Multimodal_Image_Metadata_Fusion_Corrected.py`

### Main code
- `src/multimodal/encoding.py`
- `src/multimodal/fusion.py`
- `src/evaluation/classification.py`

### Results
`results/07_Multimodal/`

Important artefacts include:

- `ablation_table.csv`
- `metadata_categories.json`
- validation predictions for M1-M4
- locked-test predictions for M1-M4

---

## 8. ConvNeXt-Tiny ROI-Attention Replication

### Dissertation role
Cross-backbone replication of the ROI-guided attention experiment.

The purpose is to test whether the localisation behaviour observed with ResNet-50 generalises to another convolutional architecture.

The experiment again includes a lambda-zero matched control and validation-selected ROI-supervised models.

### Notebook
`notebooks/08_convnext_roi_attention/08_ConvNeXt_ROI_Attention_Supervision_Strong.ipynb`

### Script
`scripts/08_ConvNeXt_ROI_Attention_Supervision_Strong.py`

### Main code
- `src/models/cam_models.py`
- `src/roi_attention/losses.py`
- `src/localisation/roi_metrics.py`
- `src/evaluation/bootstrap.py`

### Results
`results/08_ConvNeXt_ROI/`

Important artefacts include:

- `validation_by_lambda.csv`
- `classification_test.csv`
- `localisation_test.csv`
- `paired_localisation_bootstrap.csv`
- `selected_lambda.json`
- `models_by_lambda.json`
- test predictions
- localisation case-level results
- qualitative attention panel

---

## 9. One-Class Anomaly Detection — CBIS-DDSM Exploration

### Dissertation role
Exploratory one-class anomaly-detection investigation using deep image features.

This experiment is separate from the final mini-MIAS normal-versus-abnormal pathway and is retained for experimental completeness.

### Notebook
`notebooks/09_anomaly_detection/09_Anomaly_Detection_OneClass.ipynb`

### Script
`scripts/09_Anomaly_Detection_OneClass.py`

### Main code
- `src/anomaly/scoring.py`
- `src/evaluation/classification.py`
- `src/evaluation/bootstrap.py`

### Results
`results/09_Anomaly/`

The folder contains:

- extracted feature cache
- validation metrics
- locked-test metrics
- predictions
- patient-level bootstrap confidence intervals
- method comparisons
- anomaly-detection figures

---

## 10. mini-MIAS Normal-vs-Abnormal One-Class Experiment

### Dissertation role
Secondary anomaly-detection pathway using a dataset that contains both genuinely normal and abnormal mammograms.

This avoids the major source-label confounding that would occur if normal and abnormal images came from different datasets.

The one-class detector is fitted using normal training mammograms only.

Methods include:

- Mahalanobis distance
- Isolation Forest
- One-Class SVM

Deep features are obtained from a frozen ImageNet-pretrained ResNet-50.

### Notebook
`notebooks/10_mini_mias_normal_abnormal/10_Normal_vs_Abnormal_Anomaly_Detection_miniMIAS_FINAL_FIXED.ipynb`

### Script
`scripts/10_Normal_vs_Abnormal_Anomaly_Detection_miniMIAS_FINAL_FIXED.py`

### Main code
- `src/data/mini_mias.py`
- `src/anomaly/scoring.py`
- `src/preprocessing/mammogram.py`
- `src/evaluation/classification.py`
- `src/evaluation/bootstrap.py`

### Results
`results/10_Normal_vs_Abnormal_Anomaly_Detection/`

Important artefacts include:

- `exact_patient_level_split_assignments.csv`
- `validation_metrics.csv`
- `validation_predictions.csv`
- `validation_selected_thresholds.json`
- `locked_test_metrics.csv`
- `locked_test_predictions.csv`
- `locked_test_patient_bootstrap_ci.csv`
- method comparisons
- dissertation summary
- test figures

### Split manifest
`manifests/mini_mias_patient_split.csv`

---

## 11. Supervised mini-MIAS Comparator

### Dissertation role
Supervised normal-versus-abnormal ResNet-50 comparator using the same mini-MIAS task.

Unlike the one-class experiment, this model uses both normal and abnormal training labels.

The purpose is to compare one-class anomaly modelling with conventional supervised discrimination.

### Notebook
`notebooks/11_mini_mias_supervised_baseline/11_Supervised_Normal_vs_Abnormal_miniMIAS_Baseline.ipynb`

### Script
`scripts/11_Supervised_Normal_vs_Abnormal_miniMIAS_Baseline.py`

### Main code
- `src/data/mini_mias.py`
- `src/models/resnet.py`
- `src/evaluation/classification.py`
- `src/evaluation/bootstrap.py`

### Results
`results/11_Supervised_Normal_vs_Abnormal_miniMIAS/`

Important artefacts include:

- `locked_test_metrics.csv`
- `locked_test_predictions.csv`
- `locked_test_patient_bootstrap_ci.csv`
- `paired_auc_vs_notebook10.csv`
- `training_history.csv`
- `validation_predictions.csv`
- supervised locked-test figures

---

## 12. Failure-Case Analysis

### Dissertation role
Qualitative analysis of important model errors and attention behaviour.

This experiment examines false positives, false negatives, and spatial attention patterns rather than reporting only aggregate metrics.

### Notebook
`notebooks/12_failure_case_analysis/12_Failure_Case_Analysis_ROI_Attention_PORTABLE.ipynb`

### Script
`scripts/12_Failure_Case_Analysis_ROI_Attention_PORTABLE.py`

### Main code
- `src/localisation/gradcam.py`
- `src/localisation/roi_metrics.py`
- `src/visualisation/overlays.py`

### Results
`results/12_Failure_Case_Analysis/`

Important artefacts include:

- `failure_case_summary.csv`
- `failure_case_localisation_metrics.csv`
- `failure_case_panel.png`
- `false_positive_attention_panel.png`
- `false_negative_attention_panel.png`

---

## 13. Final Results Audit and Dissertation Figures

### Dissertation role
Final consistency check across the principal experiments and generation of Chapter 4 tables and figures.

This notebook verifies the final reported values against saved experimental outputs wherever available.

### Notebook
`notebooks/13_results_audit/13_Dissertation_Results_Figures_and_Chapter_4_Audit.ipynb`

### Script
`scripts/13_Dissertation_Results_Figures_and_Chapter_4_Audit.py`

### Results
`results/13_Dissertation_Figures/`

The folder contains the final dissertation tables:

- `Table_4_1_resnet_validation_lambda_ablation.csv`
- `Table_4_2_resnet_locked_test_classification.csv`
- `Table_4_3_resnet_locked_test_localisation.csv`
- `Table_4_4_backbone_roi_attention_effect.csv`
- `Table_4_5_multimodal_ablation.csv`

and final figures including:

- `Figure_4_1_locked_test_roc_auc`
- `Figure_4_2_resnet_failure_cases`

---

# Repository Structure and Dissertation Evidence

The repository is organised so that each major experimental stage has four complementary forms of evidence:

1. **Notebook** — complete experimental workflow and analysis.
2. **Script** — Python representation of the experiment.
3. **Reusable source module** — common functionality separated into `src/`.
4. **Saved outputs** — metrics, predictions, figures, configs, and manifests under `results/`.

This structure separates reusable implementation code from experimental notebooks while preserving the artefacts used to support the dissertation conclusions.

---

# Excluded Artefacts

The following are intentionally not included in the repository:

- raw CBIS-DDSM mammograms
- raw mini-MIAS mammograms
- local preprocessing caches
- Python virtual environments
- notebook checkpoints
- temporary development files
- large trained `.pth` checkpoints

The trained checkpoints are retained separately in the original project archive because the complete model directory is approximately 1.6 GB and several individual checkpoints are unsuitable for standard Git repository storage.

The repository contains the code, experiment configurations, split assignments, predictions, metrics, and figures needed to document and audit the reported experiments.
