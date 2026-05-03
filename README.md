# Plant Disease Detector API

FastAPI backend for a plant disease diagnosis prototype with:

- plant leaf image upload
- MobileNetV2-based prediction
- confidence estimation
- MC Dropout uncertainty estimation
- Grad-CAM generation
- risk-aware recommendation output

## Project overview

This API is part of a bachelor’s thesis workflow focused on robust plant disease diagnosis under real-world conditions.

The backend currently supports:

- real image validation and preprocessing
- real model inference using a trained MobileNetV2 checkpoint
- class mapping from the harmonized plant dataset
- Monte Carlo Dropout uncertainty estimation
- Grad-CAM explanation generation
- recommendation routing based on confidence, uncertainty, and decision logic

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open:

```text
http://localhost:8000/docs
```

## API endpoints

### `GET /health`

Returns a simple health check response.

Example response:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Accepts a plant image upload and returns the prediction result.

#### Request

- content type: `multipart/form-data`
- form field: `image`

#### Example response

```json
{
  "label": "Tomato leaf late blight",
  "confidence": 78,
  "uncertainty": 24,
  "recommendation": "MONITOR",
  "explanation_support": "moderate",
  "gradcam_base64": "..."
}
```

## Current inference pipeline

For each uploaded image, the API currently performs:

1. image validation and RGB conversion
2. resize to `224 x 224`
3. tensor conversion and ImageNet normalization
4. MobileNetV2 forward inference
5. softmax confidence extraction
6. MC Dropout uncertainty estimation
7. Grad-CAM heatmap generation
8. recommendation routing

## Model assets

The backend expects local model assets that are intentionally excluded from Git:

- `app/saved_model/mobilenet_v2_seed42_best.pt`
- `app/class_mapping.json`

These files must exist locally for inference to work.

## Environment and dependencies

Main dependencies:

- FastAPI
- Uvicorn
- PyTorch
- Torchvision
- Pillow
- NumPy

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- The backend is designed for local prototype development first.
- Model checkpoint files and environment files are excluded through `.gitignore`.
- The API currently returns Grad-CAM as a base64-encoded PNG string.
- Additional improvements such as stronger input validation and refined explanation-based decision logic can be added later.
