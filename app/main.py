import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()
from app.schemas import (
    PredictionResponse,
    StudySubmissionRequest,
    StudySubmissionResponse,
)
from app.services.predictor import predict_image


MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "plant-disease-detector")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "study_submissions")

mongo_client = MongoClient(MONGODB_URI) if MONGODB_URI else None
mongo_db = mongo_client[MONGODB_DB_NAME] if mongo_client else None
study_collection = mongo_db[MONGODB_COLLECTION_NAME] if mongo_db is not None else None


app = FastAPI(title="Plant Disease Detector API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://plant-disease-detector-client.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/study/submit", response_model=StudySubmissionResponse)
async def submit_study(payload: StudySubmissionRequest) -> StudySubmissionResponse:
    if study_collection is None:
        raise HTTPException(
            status_code=500,
            detail="MongoDB is not configured. Please set MONGODB_URI.",
        )

    try:
        study_collection.insert_one(payload.model_dump())
        total_submissions = study_collection.count_documents({})
    except PyMongoError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save study submission: {exc}",
        ) from exc

    return StudySubmissionResponse(
        success=True,
        message="Study submission saved successfully.",
        total_submissions=total_submissions,
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