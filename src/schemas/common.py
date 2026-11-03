"""
Common Pydantic schemas for API responses and errors.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health and operational status response."""

    status: str = Field(description="System health status ('healthy' or 'degraded')")
    app_version: str = Field(description="API service version")
    model_loaded: bool = Field(description="Whether the BiLSTM model is loaded and ready")
    model_version: Optional[str] = Field(default=None, description="Active model artifact version")
    hardware_device: str = Field(description="Execution device: 'cpu' or 'cuda'")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ErrorResponse(BaseModel):
    """Standardized error contract for mobile and web clients."""

    error: str = Field(description="Short machine-readable error code")
    detail: str = Field(description="Human-readable explanation of error")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: Optional[str] = Field(default=None, description="Trace or request identifier")
