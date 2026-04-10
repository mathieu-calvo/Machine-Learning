"""
Model Serving API with FastAPI

Production-ready REST API for serving ML model predictions.
Includes health checks, input validation, logging, and batch prediction.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
"""

import logging
import time
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Pydantic models for request/response validation ──────────────────────────

class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, description="Input feature vector")

class BatchPredictionRequest(BaseModel):
    instances: list[list[float]] = Field(..., min_length=1, description="List of feature vectors")

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

class BatchPredictionResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


# ── Application ──────────────────────────────────────────────────────────────

MODEL_VERSION = "1.0.0"
MODEL_PATH = "model.pkl"

ml_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global ml_model
    try:
        ml_model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(f"Model file {MODEL_PATH} not found. /predict will return errors.")
        ml_model = None
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="ML Model Serving API",
    description="Production-ready API for serving ML predictions",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and orchestrators."""
    return HealthResponse(
        status="healthy",
        model_loaded=ml_model is not None,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    X = np.array(request.features).reshape(1, -1)

    prediction = int(ml_model.predict(X)[0])
    probability = float(ml_model.predict_proba(X).max())
    latency_ms = (time.time() - start) * 1000

    logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Latency: {latency_ms:.1f}ms")

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint for multiple instances."""
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    X = np.array(request.instances)

    predictions = ml_model.predict(X).tolist()
    probabilities = ml_model.predict_proba(X).max(axis=1).tolist()
    latency_ms = (time.time() - start) * 1000

    logger.info(f"Batch prediction: {len(predictions)} instances, Latency: {latency_ms:.1f}ms")

    return BatchPredictionResponse(
        predictions=predictions,
        probabilities=probabilities,
        model_version=MODEL_VERSION,
    )
