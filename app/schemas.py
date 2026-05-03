from pydantic import BaseModel
from typing import Literal


Recommendation = Literal["ACCEPT", "MONITOR", "RETAKE", "REVIEW"]
ExplanationSupport = Literal["strong", "moderate", "weak"]


class PredictionResponse(BaseModel):
    label: str
    confidence: int
    uncertainty: int
    recommendation: Recommendation
    explanation_support: ExplanationSupport
    gradcam_base64: str | None = None