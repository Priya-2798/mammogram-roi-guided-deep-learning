## ROI-Guided Mammography Classification

🌐 **Live Project Website:** [View the interactive dissertation project page](https://priya-2798.github.io/mammogram-roi-guided-deep-learning/)

🧠 **Live AI Research Demo:** [Launch the Mammogram AI Demo](https://mammogram-roi-guided-deep-learning-nbsnno5p4tgzilh9caeiuk.streamlit.app/)

Research code for an MSc Artificial Intelligence dissertation investigating whether radiologist-defined regions of interest (ROIs) can guide model attention while maintaining useful benign-versus-malignant mammography classification performance.

The repository preserves the executed experiment notebooks, saved result artifacts, patient-level split manifests, dissertation figures, and reusable Python utilities.

## Main Findings

The primary experiments used CBIS-DDSM mass cases with patient-disjoint partitions.

A ResNet-50 baseline achieved a locked-test ROC-AUC of **0.7965**.

ROI-guided class-specific attention supervision improved several localisation measures, including Pointing Game and pixel-level average precision. However, the corresponding change in classification performance was small and was not statistically supported.

The ROI-attention experiment was also replicated using ConvNeXt-Tiny to assess whether the observed behaviour transferred across architectures.

A multimodal experiment combined mammogram image features with:

- breast density
- mass shape
- mass margins

The image-only configuration achieved ROC-AUC **0.7873**, while image plus metadata achieved **0.8598**.

BI-RADS assessment was deliberately excluded from the multimodal metadata features to reduce the risk of target leakage.

A separate mini-MIAS branch investigated both one-class anomaly detection and supervised normal-versus-abnormal classification.

These experiments are academic research analyses and are not intended for clinical diagnosis or deployment.

## Repository Structure

```text
mammogram-roi-guided-deep-learning/
|
|-- configs/
|-- docs/
|   |-- dataset_and_splits.md
|   |-- dissertation_mapping.md
|   |-- experiment_index.md
|   `-- reproducibility.md
|
|-- figures/
|-- manifests/
|
|-- notebooks/
|   |-- 01_eda/
|   |-- 02_baseline_preprocessing/
|   |-- 03_gradcam_roi/
|   |-- 04_roi_registration_localisation/
|   |-- 05_resnet50_v2/
|   |-- 06_roi_attention_resnet50/
|   |-- 07_multimodal_fusion/
|   |-- 08_convnext_roi_attention/
|   |-- 09_anomaly_detection/
|   |-- 10_mini_mias_normal_abnormal/
|   |-- 11_mini_mias_supervised_baseline/
|   |-- 12_failure_case_analysis/
|   |-- 13_results_audit/
|   `-- legacy/
|
|-- results/
|-- scripts/
|
|-- src/
|   |-- anomaly/
|   |-- data/
|   |-- evaluation/
|   |-- localisation/
|   |-- models/
|   |-- multimodal/
|   |-- preprocessing/
|   |-- roi_attention/
|   `-- visualisation/
|
|-- tests/
|
|-- .gitignore
|-- ENVIRONMENT_NOTES.md
|-- TEST_VERIFICATION.json
|-- environment.yml
|-- pyproject.toml
|-- README.md
|-- requirements-dev.txt
`-- requirements.txt
```

## Experiment Sequence

1. **CBIS-DDSM EDA**  
   Dataset checks, class inspection, and exploratory analysis.

2. **Preprocessing and ResNet-50 baseline**  
   Initial mammogram preprocessing and baseline classification.

3. **Grad-CAM ROI analysis**  
   Comparison between model activation maps and radiologist-defined ROIs.

4. **ROI registration and localisation metrics**  
   ROI registration and quantitative spatial alignment analysis.

5. **ResNet-50 V2**  
   Breast-focused crop, padding, 512 x 512 preprocessing, and improved classification pipeline.

6. **ROI-attention ResNet-50**  
   Class-specific CAM supervision with lambda ablation.

7. **Multimodal fusion**  
   Combination of mammogram image representations with structured radiological metadata.

8. **ConvNeXt ROI attention**  
   Cross-architecture replication of ROI-guided attention supervision.

9. **One-class anomaly detection**  
   Feature-space anomaly detection experiments.

10. **mini-MIAS normal vs abnormal anomaly detection**  
    Patient-disjoint one-class anomaly detection using mini-MIAS.

11. **mini-MIAS supervised baseline**  
    Supervised normal-versus-abnormal comparison.

12. **Failure-case analysis**  
    Classification and localisation failure analysis with qualitative examples.

13. **Results audit**  
    Final dissertation tables, figures, and consistency checks.

Earlier notebook variants are retained under `notebooks/legacy/` for provenance.

## Installation

### Virtual Environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the main dependencies:

```bash
python -m pip install -r requirements.txt
```

For testing and notebook development:

```bash
python -m pip install -r requirements-dev.txt
```

A Conda environment definition is also provided in:

```text
environment.yml
```

More information is available in `ENVIRONMENT_NOTES.md`.

## Testing

The repository includes lightweight unit tests for reusable implementation components.

Run:

```bash
python -m pytest -q
```

Latest verified result:

```text
4 passed in 4.66s
```

See `TEST_VERIFICATION.json` for the recorded verification result.

## Data

Raw medical-image datasets are intentionally not redistributed in this repository.

The project uses:

- CBIS-DDSM
- mini-MIAS

The CBIS-DDSM experiments use mass-case metadata and mammogram/ROI images.

The mini-MIAS experiments use the mammogram images and associated truth-data information.

Patient-level split information used by key experiments is preserved under `manifests/`.

Dataset paths must be configured locally before reproducing experiments that require the original images.

See `docs/dataset_and_splits.md` for further details.

## Results

Saved experimental artifacts are preserved under `results/`, including:

- classification metrics
- validation predictions
- locked-test predictions
- training histories
- experiment configuration files
- localisation metrics
- bootstrap results
- selected thresholds
- qualitative panels
- final dissertation figures and tables

These outputs allow the reported experimental analysis to be inspected without redistributing the complete medical imaging datasets.

## Model Checkpoints

Large trained model checkpoint files are intentionally excluded from this clean repository.

The complete original checkpoint collection is approximately **1.6 GB** and is retained separately in the project archive.

Checkpoint extensions such as `.pth`, `.pt`, and `.ckpt` are excluded through `.gitignore`.

## Reusable Source Package

Reusable implementation components are organised under `src/`.

Examples include:

```python
from src.preprocessing.mammogram import preprocess_mammogram
from src.models.cam_models import ResNetCAM2, ConvNeXtCAM2
from src.roi_attention.losses import attention_loss
from src.evaluation.classification import classification_metrics
```

The executed notebooks remain the primary record of exact experiment orchestration.

The `src/` package provides reusable implementations of important processing, modelling, evaluation, localisation, multimodal, and visualisation components.

## Reproducibility

The project uses patient-disjoint splitting for the main experiments and preserves relevant split assignments under `manifests/`.

Saved result artifacts and configuration files are retained to support inspection and reproducibility.

Some original notebooks were developed in Google Colab and therefore contain Google Drive or Colab-specific paths. These paths may need to be changed when reproducing experiments in another environment.

See:

- `docs/reproducibility.md`
- `docs/experiment_index.md`
- `docs/dissertation_mapping.md`
- `docs/dataset_and_splits.md`

for additional information.

## Clinical-Use Disclaimer

This repository is provided for academic research purposes only.

Improved spatial overlap between model attribution maps and radiologist-defined ROIs does not demonstrate causal reasoning, diagnostic validity, clinical safety, or readiness for deployment.

External clinical validation would be required before any clinical interpretation or use.

## License

This repository is released under the **MIT License**.

See [LICENSE](LICENSE).
