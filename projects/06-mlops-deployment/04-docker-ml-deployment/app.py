"""
FastAPI Application for Containerised ML Model Serving
=======================================================
Loads a pre-trained scikit-learn model from disk and exposes it via
a REST API with health checks, single predictions, and batch predictions.

Designed to run inside a Docker container alongside the training script.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
import pickle
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent / "models"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ml-api")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Model Serving API",
    description="Containerised model serving with FastAPI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (populated on startup)
# ---------------------------------------------------------------------------

model = None
metadata = None
start_time = None
request_count = 0


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictionInput(BaseModel):
    """Input schema for a single prediction."""
    features: List[float] = Field(
        ...,
        description="Feature values in the order defined by the model",
        min_length=1,
    )

    @validator("features", each_item=True)
    def check_finite(cls, v):
        if not np.isfinite(v):
            raise ValueError("Feature values must be finite numbers")
        return v


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    samples: List[List[float]] = Field(
        ...,
        description="List of feature vectors",
        min_length=1,
        max_length=1000,
    )


class PredictionResponse(BaseModel):
    prediction: int
    predicted_class: Optional[str]
    probabilities: Optional[Dict[str, float]]
    model_version: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    count: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
    total_requests: int


class ModelInfoResponse(BaseModel):
    algorithm: str
    dataset: str
    feature_names: List[str]
    target_names: Optional[List[str]]
    n_classes: int
    metrics: Dict
    model_hash: str
    trained_at: str


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def load_model():
    """Load the model and metadata on application startup."""
    global model, metadata, start_time
    start_time = time.time()

    model_path = MODEL_DIR / "model.pkl"
    metadata_path = MODEL_DIR / "metadata.json"

    if not model_path.exists():
        logger.warning("No model found at %s – running in degraded mode", model_path)
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", model_path)

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info("Metadata loaded: algorithm=%s, hash=%s",
                     metadata.get("algorithm"), metadata.get("model_hash"))
    else:
        metadata = {}
        logger.warning("No metadata file found")


# ---------------------------------------------------------------------------
# Middleware: request counting
# ---------------------------------------------------------------------------

@app.middleware("http")
async def count_requests(request: Request, call_next):
    global request_count
    request_count += 1
    t0 = time.time()
    response = await call_next(request)
    elapsed = (time.time() - t0) * 1000
    logger.info("%s %s completed in %.1f ms (status %d)",
                request.method, request.url.path, elapsed, response.status_code)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint for container orchestrators."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=metadata.get("model_hash", "unknown") if metadata else None,
        uptime_seconds=round(time.time() - start_time, 1),
        total_requests=request_count,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return metadata about the loaded model."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    return ModelInfoResponse(**metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(payload: PredictionInput):
    """Single-sample prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    expected_features = len(metadata.get("feature_names", []))
    if expected_features and len(payload.features) != expected_features:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_features} features, got {len(payload.features)}",
        )

    X = np.array(payload.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])

    # Probabilities (not all estimators support predict_proba)
    probabilities = None
    target_names = metadata.get("target_names")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        if target_names:
            probabilities = {name: round(float(p), 4) for name, p in zip(target_names, probs)}
        else:
            probabilities = {str(i): round(float(p), 4) for i, p in enumerate(probs)}

    predicted_class = target_names[prediction] if target_names else str(prediction)

    return PredictionResponse(
        prediction=prediction,
        predicted_class=predicted_class,
        probabilities=probabilities,
        model_version=metadata.get("model_hash", "unknown"),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(payload: BatchPredictionInput):
    """Batch prediction for multiple samples."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    expected_features = len(metadata.get("feature_names", []))
    for idx, sample in enumerate(payload.samples):
        if expected_features and len(sample) != expected_features:
            raise HTTPException(
                status_code=422,
                detail=f"Sample {idx}: expected {expected_features} features, got {len(sample)}",
            )

    X = np.array(payload.samples)
    predictions = model.predict(X).tolist()

    target_names = metadata.get("target_names")
    probs_all = None
    if hasattr(model, "predict_proba"):
        probs_all = model.predict_proba(X)

    results = []
    for i, pred in enumerate(predictions):
        pred = int(pred)
        probabilities = None
        if probs_all is not None:
            if target_names:
                probabilities = {name: round(float(p), 4) for name, p in zip(target_names, probs_all[i])}
            else:
                probabilities = {str(j): round(float(p), 4) for j, p in enumerate(probs_all[i])}

        predicted_class = target_names[pred] if target_names else str(pred)
        results.append(PredictionResponse(
            prediction=pred,
            predicted_class=predicted_class,
            probabilities=probabilities,
            model_version=metadata.get("model_hash", "unknown"),
        ))

    return BatchPredictionResponse(
        results=results,
        count=len(results),
        model_version=metadata.get("model_hash", "unknown"),
    )


@app.get("/ready", tags=["System"])
async def readiness():
    """Readiness probe: returns 200 only when the model is loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"ready": True}


# ---------------------------------------------------------------------------
# Main (for development without uvicorn CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
