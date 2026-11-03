"""
Health check and system diagnostics endpoint.
"""

from fastapi import APIRouter, Depends
from src.schemas.common import HealthResponse
from src.services.model_service import ModelService
from src.api.dependencies import get_model
from src.config.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health status",
    description="Returns API status, active model state, and hardware configuration.",
)
async def check_health(model_svc: ModelService = Depends(get_model)) -> HealthResponse:
    is_healthy = "healthy" if model_svc.is_loaded else "degraded"
    return HealthResponse(
        status=is_healthy,
        app_version=settings.app_version,
        model_loaded=model_svc.is_loaded,
        model_version=model_svc.get_version() if model_svc.is_loaded else None,
        hardware_device=settings.device,
    )
