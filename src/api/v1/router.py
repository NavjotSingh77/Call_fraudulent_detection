"""
API v1 Router aggregating all v1 endpoints.
"""

from fastapi import APIRouter
from src.api.v1.endpoints import health, model_info, predict, transcribe

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(model_info.router, tags=["Model Info"])
api_v1_router.include_router(transcribe.router, tags=["Transcription"])
api_v1_router.include_router(predict.router, tags=["Prediction"])
