"""
Pydantic schemas for fraud prediction endpoints.
"""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class SegmentResult(BaseModel):
    """Segment-level prediction window result."""

    segment_id: str = Field(description="Segment identifier")
    start_time: float = Field(description="Segment start in seconds")
    end_time: float = Field(description="Segment end in seconds")
    duration: float = Field(description="Segment duration in seconds")
    fraud_probability: float = Field(description="Fraud probability for this segment [0.0, 1.0]")
    transcript_snippet: Optional[str] = Field(default=None, description="Segment transcript text")
    annotation_type: str = Field(
        default="exploratory",
        description="Annotation provenance ('exploratory' or 'verified')",
    )


class PredictionResponse(BaseModel):
    """Full-call fraud prediction report."""

    call_id: str = Field(description="Unique identifier for the analyzed call")
    language: str = Field(description="Detected or provided spoken language code")
    overall_prediction: str = Field(
        description="Classification: 'fraud', 'legitimate', or 'uncertain'"
    )
    fraud_probability: float = Field(
        description="Calibrated probability of fraud between 0.0 and 1.0"
    )
    confidence: str = Field(
        description="Confidence tier: 'high', 'medium', or 'low'"
    )
    segments: List[SegmentResult] = Field(
        default_factory=list,
        description="List of segment-level results (populated in segment mode)",
    )
    risk_indicators: List[str] = Field(
        default_factory=list,
        description="Human-readable linguistic risk factors detected",
    )
    transcript: str = Field(
        default="",
        description="Full transcription of call audio",
    )
    review_required: bool = Field(
        default=False,
        description="Flag indicating human analyst review is recommended",
    )
    model_version: str = Field(
        default="unknown",
        description="Identifier of the model used for classification",
    )
    prediction_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp when prediction was generated",
    )
    disclaimer: str = Field(
        description="Mandatory legal disclaimer regarding probabilistic nature of prediction"
    )
