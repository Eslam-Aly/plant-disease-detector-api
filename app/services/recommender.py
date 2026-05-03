from typing import Literal

Recommendation = Literal["ACCEPT", "MONITOR", "RETAKE", "REVIEW"]
ExplanationSupport = Literal["strong", "moderate", "weak"]


def decide_recommendation(
    confidence: int,
    uncertainty: int,
    explanation_support: ExplanationSupport,
) -> Recommendation:
    if confidence >= 75 and uncertainty <= 30 :
        return "ACCEPT"

    if confidence >= 60 and uncertainty <= 40 :
        return "MONITOR"

    if confidence >= 50 and uncertainty <= 50 :
        return "RETAKE"

    return "REVIEW"