"""
FastAPI application gateway for the Fraud Call Detection system.

Features:
- Versioned REST endpoints under /api/v1 (ready for Android & web clients)
- Dedicated model-service layer (isolated from training code)
- Database repository integration (SQLite / PostgreSQL)
- Authentication-ready and telemetry middleware
- Responsive Web UI dashboard served at / and /ui
- Backward-compatible legacy aliases for /health, /predict, etc.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware import (
    AuthenticationMiddleware,
    CorrelationAndTimingMiddleware,
)
from src.api.v1.endpoints import health, model_info, predict, transcribe
from src.api.v1.router import api_v1_router
from src.api.web.router import router as web_router
from src.config.settings import get_settings
from src.db.session import init_db
from src.schemas.common import HealthResponse
from src.schemas.model_info import ModelInfoResponse
from src.schemas.prediction import PredictionResponse
from src.schemas.transcription import TranscriptionResponse
from src.services.model_service import ModelService, get_model_service

logger = logging.getLogger("fraud_detection.api")
settings = get_settings()

# Legacy app_state dictionary for backwards compatibility with existing fixtures/tests
app_state: Dict[str, Any] = {
    "model": None,
    "vectorizer_config": None,
    "label_encoder": None,
    "model_config": None,
    "model_loaded": False,
    "config": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager initializing database and model service."""
    # 1. Initialize DB tables
    init_db()

    # 2. Initialize ModelService singleton
    model_svc = get_model_service(model_dir=settings.model_dir)
    app_state["model_loaded"] = model_svc.is_loaded
    app_state["config"] = settings.load_yaml_config()
    if model_svc.is_loaded:
        app_state["model"] = model_svc.model
        app_state["vectorizer_config"] = model_svc.vectorizer_config
        app_state["label_encoder"] = model_svc.label_encoder
        app_state["model_config"] = model_svc.model_config

    logger.info(
        "Application startup complete. Environment: %s, Device: %s, Model Loaded: %s",
        settings.app_env,
        settings.device,
        model_svc.is_loaded,
    )
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title=settings.app_name,
    description=(
        "Research-grade AI system for detecting fraudulent phone calls using OpenAI Whisper "
        "and Bidirectional LSTM (BiLSTM) neural networks."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware Pipeline
# ---------------------------------------------------------------------------

app.add_middleware(CorrelationAndTimingMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static Files & Web Dashboard Mounting
# ---------------------------------------------------------------------------

static_dir = settings.project_root / "src" / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(web_router)
app.include_router(api_v1_router)

# ---------------------------------------------------------------------------
# Legacy Backward-Compatible Endpoints (Aliased directly to v1 implementations)
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Legacy (v0)"])
async def legacy_health():
    """Legacy health check endpoint alias."""
    model_svc = get_model_service(settings.model_dir)
    return await health.check_health(model_svc=model_svc)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Legacy (v0)"])
async def legacy_model_info():
    """Legacy model info endpoint alias."""
    model_svc = get_model_service(settings.model_dir)
    return await model_info.get_model_info(model_svc=model_svc)


@app.post("/transcribe", response_model=TranscriptionResponse, tags=["Legacy (v0)"])
async def legacy_transcribe(
    file: UploadFile = File(..., description="Audio file"),
):
    """Legacy transcribe endpoint alias."""
    return await transcribe.transcribe_audio(file=file)


@app.post("/predict", response_model=PredictionResponse, tags=["Legacy (v0)"])
async def legacy_predict(
    file: UploadFile = File(..., description="Audio file"),
):
    """Legacy full-call predict endpoint alias."""
    from src.db.session import SessionLocal
    from src.repositories.call_repository import CallRepository

    db = SessionLocal()
    try:
        model_svc = get_model_service(settings.model_dir)
        repo = CallRepository(db)
        return await predict.predict_call(file=file, model_svc=model_svc, call_repo=repo)
    finally:
        db.close()


@app.post("/predict/segments", response_model=PredictionResponse, tags=["Legacy (v0)"])
async def legacy_predict_segments(
    file: UploadFile = File(..., description="Audio file"),
):
    """Legacy segment-level predict endpoint alias."""
    from src.db.session import SessionLocal
    from src.repositories.call_repository import CallRepository

    db = SessionLocal()
    try:
        model_svc = get_model_service(settings.model_dir)
        repo = CallRepository(db)
        return await predict.predict_call_segments(file=file, model_svc=model_svc, call_repo=repo)
    finally:
        db.close()
