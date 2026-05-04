from io import BytesIO
from pathlib import Path
import base64
import json
import os
import numpy as np
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

from app.services.recommender import decide_recommendation


BASE_DIR = Path(__file__).resolve().parents[2]
SAVED_MODEL_DIR = BASE_DIR / "app" / "saved_model"
LOCAL_MODEL_PATH = SAVED_MODEL_DIR / "mobilenet_v2_seed42_best.pt"
CLASS_MAPPING_PATH = BASE_DIR / "app" / "class_mapping.json"
HF_MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "mobilenet_v2_seed42_best.pt")
HF_TOKEN = os.getenv("HF_TOKEN", "")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
MC_PASSES = 30
TEMPERATURE = 0.9974

ExplanationSupport = Literal["strong", "moderate", "weak"]



class ActivationGradientHook:
    """
    Capture activations and gradients from the final convolution block
    for Grad-CAM generation.
    """

    def __init__(self, module: nn.Module):
        self.activations = None
        self.gradients = None
        self._forward_handle = module.register_forward_hook(self._forward_hook)
        self._backward_handle = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def clear(self) -> None:
        self.activations = None
        self.gradients = None

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()



def load_class_names() -> list[str]:
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping["classes"]


CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])



def create_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(DEVICE)


def resolve_model_path() -> Path:
    """
    Return the local model path. If the checkpoint is missing locally,
    try downloading it from Hugging Face Hub.
    """
    if LOCAL_MODEL_PATH.exists():
        return LOCAL_MODEL_PATH

    if not HF_MODEL_REPO_ID:
        raise FileNotFoundError(
            f"Missing model checkpoint: {LOCAL_MODEL_PATH}. "
            "Set HF_MODEL_REPO_ID to download it from Hugging Face Hub."
        )

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=HF_MODEL_FILENAME,
        token=HF_TOKEN or None,
        local_dir=str(SAVED_MODEL_DIR),
        local_dir_use_symlinks=False,
    )

    return Path(downloaded_path)


def load_model() -> tuple[nn.Module, ActivationGradientHook]:
    model_path = resolve_model_path()

    model = create_model(NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    feature_hook = ActivationGradientHook(model.features[-1])
    return model, feature_hook


MODEL, FEATURE_HOOK = load_model()



def validate_image_bytes(contents: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image file.") from exc



def preprocess_image(image: Image.Image) -> torch.Tensor:
    tensor = EVAL_TRANSFORM(image).unsqueeze(0)
    return tensor.to(DEVICE)



def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference for MC Dropout.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()



def apply_temperature(logits: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """
    Apply temperature scaling. A value of 1.0 means no additional calibration.
    """
    if temperature <= 0:
        return logits
    return logits / temperature



def normalized_entropy_from_probs(mean_probs: torch.Tensor) -> float:
    """
    Compute entropy normalized to [0, 1] using the class count.
    """
    entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum(dim=1)
    max_entropy = torch.log(torch.tensor(mean_probs.shape[1], device=mean_probs.device, dtype=mean_probs.dtype))
    norm_entropy = entropy / (max_entropy + 1e-12)
    return float(norm_entropy.item())



def mc_dropout_predict(x: torch.Tensor, passes: int = MC_PASSES) -> tuple[torch.Tensor, float]:
    """
    Run MC Dropout and return mean probabilities and normalized entropy.
    """
    MODEL.eval()
    enable_dropout(MODEL)

    prob_list = []
    for _ in range(passes):
        logits = MODEL(x)
        logits = apply_temperature(logits)
        probs = F.softmax(logits, dim=1)
        prob_list.append(probs)

    stacked = torch.stack(prob_list, dim=0)
    mean_probs = stacked.mean(dim=0)
    norm_entropy = normalized_entropy_from_probs(mean_probs)

    MODEL.eval()
    return mean_probs, norm_entropy





def generate_gradcam_overlay(
    original_image: Image.Image,
    x: torch.Tensor,
    pred_idx: int,
) -> tuple[str | None, float]:
    """
    Generate a Grad-CAM overlay and return:
    - base64 encoded PNG
    - focus ratio used as explanation-support proxy
    """
    MODEL.eval()
    FEATURE_HOOK.clear()
    MODEL.zero_grad(set_to_none=True)

    logits = MODEL(x)
    calibrated_logits = apply_temperature(logits)
    score = calibrated_logits[:, pred_idx].sum()
    score.backward()

    activations = FEATURE_HOOK.activations
    gradients = FEATURE_HOOK.gradients

    if activations is None or gradients is None:
        return None, 0.0

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    if float(cam.max().item()) <= 0.0:
        return None, 0.0

    cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    cam = cam / (cam.max() + 1e-12)

    flat = cam.flatten()
    k = max(1, int(0.20 * flat.numel()))
    focus_ratio = float(torch.topk(flat, k=k).values.sum().item() / (cam.sum().item() + 1e-12))

    base_img = original_image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
    base_arr = np.array(base_img).astype(np.float32)

    heat = cam.detach().cpu().numpy()
    heat_uint8 = np.uint8(heat * 255.0)

    heat_rgb = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    heat_rgb[..., 0] = heat_uint8
    heat_rgb[..., 1] = np.uint8(heat_uint8 * 0.25)

    overlay = (0.6 * base_arr + 0.4 * heat_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    buffer = BytesIO()
    overlay_img.save(buffer, format="PNG")
    gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return gradcam_base64, focus_ratio



def focus_to_support(focus_ratio: float) -> ExplanationSupport:
    if focus_ratio >= 0.75:
        return "strong"
    if focus_ratio >= 0.60:
        return "moderate"
    return "weak"


def predict_image(contents: bytes) -> dict:
    image = validate_image_bytes(contents)
    x = preprocess_image(image)

    with torch.no_grad():
        logits = MODEL(x)
        calibrated_logits = apply_temperature(logits)
        probs = F.softmax(calibrated_logits, dim=1)

        confidence_value, pred_idx_tensor = probs.max(dim=1)
        pred_idx = int(pred_idx_tensor.item())
        confidence = int(round(float(confidence_value.item()) * 100))

    # Real uncertainty from MC Dropout
    mean_probs_mc, norm_entropy = mc_dropout_predict(x, passes=MC_PASSES)
    uncertainty = int(round(norm_entropy * 100))
    uncertainty = max(0, min(100, uncertainty))

    # Real Grad-CAM + explanation-support proxy
    gradcam_base64, focus_ratio = generate_gradcam_overlay(image, x, pred_idx)
    explanation_support = focus_to_support(focus_ratio)

    recommendation = decide_recommendation(
        confidence=confidence,
        uncertainty=uncertainty,
        explanation_support=explanation_support,
    )

    print("----- Prediction Debug -----")
    print("Label:", CLASS_NAMES[pred_idx])
    print("Confidence:", confidence)
    print("Uncertainty:", uncertainty)
    print("Focus ratio:", round(focus_ratio, 4))
    print("Explanation support:", explanation_support)
    print("Recommendation:", recommendation)
    print("Grad-CAM available:", gradcam_base64 is not None)
    print("----------------------------")

    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "uncertainty": uncertainty,
        "recommendation": recommendation,
        "explanation_support": explanation_support,
        "gradcam_base64": gradcam_base64,
    }