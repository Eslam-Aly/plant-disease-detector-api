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


class StudyCaseResponse(BaseModel):
    participantId: str
    orderId: str
    caseId: str
    interfaceType: Literal["simple", "full"]
    selectedAction: Recommendation
    clarityScore: int
    trustScore: int
    timestamp: str


class StudyFinalResponse(BaseModel):
    participantId: str
    orderId: str
    clearerVersion: Literal["simple", "full", "no_difference"]
    moreUsefulVersion: Literal["simple", "full", "no_difference"]
    saferVersion: Literal["simple", "full", "no_difference"]
    reuseScore: int
    comment: str
    timestamp: str


class StudySubmissionRequest(BaseModel):
    participantId: str
    orderId: str
    caseResponses: list[StudyCaseResponse]
    finalResponse: StudyFinalResponse
    submittedAt: str


class StudySubmissionResponse(BaseModel):
    success: bool
    message: str
    total_submissions: int