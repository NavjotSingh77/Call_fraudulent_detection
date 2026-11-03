"""
Model metadata and artifact introspection endpoint.
"""

from fastapi import APIRouter, Depends
from src.schemas.model_info import ModelInfoResponse
from src.services.model_service import ModelService
from src.api.dependencies import get_model
from src.config.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata and hyperparameters",
    description="Inspects active model architecture, loaded hyperparameters, and classes.",
)
async def get_model_info(model_svc: ModelService = Depends(get_model)) -> ModelInfoResponse:
    info = model_svc.get_info()
    msg = None
    if not info["model_loaded"]:
        msg = (
            "Model artifact is not loaded or trained. "
            "To train the model, run `python main.py train` or `run_system.bat train`."
        )

    return ModelInfoResponse(
        model_loaded=info["model_loaded"],
        model_version=info.get("model_version"),
        model_format=info.get("model_format"),
        model_type=info.get("model_type", "BiLSTM"),
        classes=info.get("classes", []),
        whisper_model=settings.whisper_model,
        hyperparameters=info.get("hyperparameters", {}),
        model_metadata=info.get("model_metadata", {}),
        message=msg,
    )
