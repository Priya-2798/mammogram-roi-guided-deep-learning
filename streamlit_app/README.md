# Mammogram AI Research Demo

Streamlit prototype for the selected ROI-guided ResNet-50 (λ = 0.5) from the MSc dissertation.

## 1. Put the model next to the app

Copy this checkpoint into the same folder as `app.py`:

`resnet50_cam2_lambda0.5.pth`

Alternatively set:

```bash
export MAMMOGRAM_MODEL_PATH="/full/path/to/resnet50_cam2_lambda0.5.pth"
```

## 2. Create a virtual environment

Recommended: Python 3.10 or 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## What the app implements

- Dissertation-faithful breast-focused preprocessing
- 512×512 model input
- ImageNet normalisation
- Exact ResNetCAM2 architecture from Notebook 06
- The selected λ = 0.5 checkpoint
- Benign/malignant model score
- Validation-selected malignant decision threshold of 0.29
- Class-specific CAM visualisation and image overlay
- Dissertation results summary and research disclaimer

## Important

This is a research demonstration only. It is not a clinical diagnostic system and must not be used for real-patient medical decisions.
