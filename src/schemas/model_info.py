"""
Pydantic schemas for model metadata and runtime information.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """Model information and metadata response."""

    model_loaded: bool = Field(description="True if model artifact is loaded and active")
    model_version: Optional[str] = Field(default=None, description="Model artifact version")
    model_format: Optional[str] = Field(default=None, description="Artifact serialization format (e.g. '.keras')")
    model_type: Optional[str] = Field(default="BiLSTM", description="Model architecture type")
    classes: List[str] = Field(default_factory=list, description="Target classification classes")
    whisper_model: str = Field(description="Speech-to-text model configured")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Loaded hyperparameter configuration",
    )
    model_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training and performance metadata saved alongside artifact",
    )
    message: Optional[str] = Field(
        default=None,
        description="Informational status message",
    )
