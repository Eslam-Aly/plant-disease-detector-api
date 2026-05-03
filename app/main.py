import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    PredictionResponse,
    StudySubmissionRequest,
    StudySubmissionResponse,
)
from app.services.predictor import predict_image


RESULTS_PATH = Path(__file__).resolve().parent / "data" / "results.json"


app = FastAPI(title="Plant Disease Detector API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def load_study_results() -> list:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text("[]", encoding="utf-8")
        return []

    try:
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass

    RESULTS_PATH.write_text("[]", encoding="utf-8")
    return []


@app.post("/study/submit", response_model=StudySubmissionResponse)
async def submit_study(payload: StudySubmissionRequest) -> StudySubmissionResponse:
    submissions = load_study_results()
    submissions.append(payload.model_dump())

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(submissions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return StudySubmissionResponse(
        success=True,
        message="Study submission saved successfully.",
        total_submissions=len(submissions),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Only JPG and PNG files are supported.")

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = predict_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(**result)