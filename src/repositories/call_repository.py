"""
Call record and prediction repository implementation.
"""

from typing import List, Optional
import json
from sqlalchemy.orm import Session
from src.db.models import CallRecord, SegmentRecord
from src.repositories.base import IRepository


class CallRepository(IRepository[CallRecord]):
    """Concrete repository for persisting and querying call prediction results."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, call_id: str) -> Optional[CallRecord]:
        return self.db.query(CallRecord).filter(CallRecord.call_id == call_id).first()

    def get_all(self, skip: int = 0, limit: int = 100) -> List[CallRecord]:
        return (
            self.db.query(CallRecord)
            .order_by(CallRecord.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def create(self, entity: CallRecord) -> CallRecord:
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    def update(self, entity: CallRecord) -> CallRecord:
        self.db.merge(entity)
        self.db.commit()
        return entity

    def delete(self, call_id: str) -> bool:
        record = self.get(call_id)
        if record:
            self.db.delete(record)
            self.db.commit()
            return True
        return False

    def save_prediction(
        self,
        call_id: str,
        filename: Optional[str],
        duration_seconds: Optional[float],
        sample_rate: Optional[int],
        language: str,
        transcript: str,
        overall_prediction: str,
        fraud_probability: float,
        confidence: str,
        review_required: bool,
        risk_indicators: List[str],
        model_version: str,
        client_source: str = "web",
        segments: Optional[List[dict]] = None,
    ) -> CallRecord:
        """Create and commit a complete call prediction with child segment records."""
        record = CallRecord(
            call_id=call_id,
            filename=filename,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            language=language,
            transcript=transcript,
            overall_prediction=overall_prediction,
            fraud_probability=fraud_probability,
            confidence=confidence,
            review_required=review_required,
            risk_indicators_json=json.dumps(risk_indicators, ensure_ascii=False),
            model_version=model_version,
            client_source=client_source,
        )

        if segments:
            for s in segments:
                seg_rec = SegmentRecord(
                    segment_id=s.get("segment_id", f"{call_id}_seg"),
                    start_time=float(s.get("start_time", 0.0)),
                    end_time=float(s.get("end_time", 0.0)),
                    duration=float(s.get("duration", 0.0)),
                    fraud_probability=float(s.get("fraud_probability", 0.0)),
                    transcript_snippet=s.get("transcript_snippet"),
                    annotation_type=s.get("annotation_type", "exploratory"),
                )
                record.segments.append(seg_rec)

        return self.create(record)
