from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator

from .train import FEATURE_COUNT


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {"level": record.levelname, "logger": record.name, "message": record.getMessage()}
        )


logger = logging.getLogger("ml_model_serving_ops")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=FEATURE_COUNT, max_length=FEATURE_COUNT)

    @field_validator("features")
    @classmethod
    def finite_features(cls, values: list[float]) -> list[float]:
        if not np.isfinite(values).all():
            raise ValueError("All features must be finite numbers.")
        return values


class PredictionResponse(BaseModel):
    model_version: str
    prediction: int
    probability: float


def create_app(artifact_dir: str | Path = "artifacts") -> FastAPI:
    artifact_dir = Path(artifact_dir)
    registry = CollectorRegistry()
    requests = Counter(
        "model_requests_total", "Prediction requests", ["status"], registry=registry
    )
    latency = Histogram(
        "model_inference_seconds", "Prediction latency", registry=registry
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model_path = artifact_dir / "model.joblib"
        metadata_path = artifact_dir / "metadata.json"
        if model_path.exists() and metadata_path.exists():
            app.state.model = joblib.load(model_path)
            app.state.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            logger.info("model_loaded version=%s", app.state.metadata["model_version"])
        else:
            app.state.model = None
            app.state.metadata = None
            logger.error("model_artifacts_missing path=%s", artifact_dir)
        yield

    app = FastAPI(title="ML Model Serving Ops", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {"status": "ready"}

    @app.get("/model")
    def model_metadata() -> dict[str, object]:
        if app.state.metadata is None:
            raise HTTPException(status_code=503, detail="Model metadata is unavailable")
        return app.state.metadata

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        if app.state.model is None:
            requests.labels(status="unavailable").inc()
            raise HTTPException(status_code=503, detail="Model is not loaded")
        started = time.perf_counter()
        try:
            probability = float(app.state.model.predict_proba([payload.features])[0, 1])
            prediction = int(probability >= 0.5)
            requests.labels(status="success").inc()
            return PredictionResponse(
                model_version=app.state.metadata["model_version"],
                prediction=prediction,
                probability=probability,
            )
        except Exception as exc:
            requests.labels(status="error").inc()
            logger.exception("prediction_failed")
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
        finally:
            latency.observe(time.perf_counter() - started)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
