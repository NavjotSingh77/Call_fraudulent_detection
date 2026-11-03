"""
Pydantic schemas for Whisper transcription endpoint.
"""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """Timestamped segment from Whisper."""

    id: Optional[int] = Field(default=None)
    start: float = Field(description="Segment start in seconds")
    end: float = Field(description="Segment end in seconds")
    text: str = Field(description="Transcribed text")


class TranscriptionResponse(BaseModel):
    """Transcription response schema."""

    call_id: str = Field(description="Unique identifier for the audio file")
    language: str = Field(description="Detected language code (e.g., 'en', 'hi', 'zh')")
    transcript: str = Field(description="Full text transcript in original spoken language")
    segments: List[TranscriptSegment] = Field(
        default_factory=list,
        description="Timestamped breakdown of transcription",
    )
    model_used: str = Field(description="Whisper model utilized")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
