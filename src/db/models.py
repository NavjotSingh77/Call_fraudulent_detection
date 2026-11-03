"""
SQLAlchemy ORM models for call records, prediction results, and segment breakdowns.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Float,
    Boolean,
    Integer,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from src.db.base import Base


class CallRecord(Base):
    """Stores full-call metadata, transcription, and prediction output."""

    __tablename__ = "call_records"

    call_id = Column(String(128), primary_key=True, index=True)
    filename = Column(String(256), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    language = Column(String(16), default="en")

    # Speech-to-text transcript
    transcript = Column(Text, nullable=True)

    # ML Inference Results
    overall_prediction = Column(String(32), index=True)  # fraud, legitimate, uncertain
    fraud_probability = Column(Float, nullable=False)
    confidence = Column(String(16))                     # high, medium, low
    review_required = Column(Boolean, default=False)
    risk_indicators_json = Column(Text, nullable=True)

    # Provenance and Model Version
    model_version = Column(String(64), nullable=True)
    client_source = Column(String(64), default="web")   # e.g., 'web', 'android', 'api'
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    segments = relationship(
        "SegmentRecord",
        back_populates="call",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class SegmentRecord(Base):
    """Stores windowed audio segment evaluation."""

    __tablename__ = "segment_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(128), ForeignKey("call_records.call_id"), index=True)
    segment_id = Column(String(128), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    transcript_snippet = Column(Text, nullable=True)
    annotation_type = Column(String(32), default="exploratory")

    call = relationship("CallRecord", back_populates="segments")
