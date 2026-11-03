"""
Unit tests for database models and CallRepository.
"""

from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.base import Base
from src.db.models import CallRecord, SegmentRecord
from src.repositories.call_repository import CallRepository


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_call_repository_save_and_retrieve(in_memory_db):
    repo = CallRepository(in_memory_db)

    record = repo.save_prediction(
        call_id="call_test_db_001",
        filename="test_call.wav",
        duration_seconds=45.2,
        sample_rate=16000,
        language="en",
        transcript="Your bank account is compromised. Immediate wire transfer required.",
        overall_prediction="fraud",
        fraud_probability=0.912,
        confidence="high",
        review_required=False,
        risk_indicators=["Urgency language", "Financial terms"],
        model_version="BiLSTM-v1.0.0",
        client_source="unit_test",
        segments=[
            {
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 30.0,
                "duration": 30.0,
                "fraud_probability": 0.94,
                "transcript_snippet": "Your bank account is compromised.",
                "annotation_type": "exploratory",
            }
        ],
    )

    assert record.call_id == "call_test_db_001"
    assert record.overall_prediction == "fraud"
    assert len(record.segments) == 1

    # Query from database
    fetched = repo.get("call_test_db_001")
    assert fetched is not None
    assert fetched.call_id == "call_test_db_001"
    assert fetched.fraud_probability == 0.912
    assert len(fetched.segments) == 1
    assert fetched.segments[0].segment_id == "seg_001"
    assert fetched.segments[0].duration == 30.0


def test_call_repository_get_all(in_memory_db):
    repo = CallRepository(in_memory_db)
    for i in range(3):
        repo.save_prediction(
            call_id=f"call_{i}",
            filename=f"call_{i}.wav",
            duration_seconds=10.0,
            sample_rate=16000,
            language="en",
            transcript=f"Transcript {i}",
            overall_prediction="legitimate",
            fraud_probability=0.1,
            confidence="high",
            review_required=False,
            risk_indicators=[],
            model_version="1.0",
        )

    all_records = repo.get_all(skip=0, limit=10)
    assert len(all_records) == 3


def test_call_repository_delete(in_memory_db):
    repo = CallRepository(in_memory_db)
    repo.save_prediction(
        call_id="call_to_delete",
        filename="del.wav",
        duration_seconds=5.0,
        sample_rate=16000,
        language="en",
        transcript="Delete me",
        overall_prediction="legitimate",
        fraud_probability=0.05,
        confidence="high",
        review_required=False,
        risk_indicators=[],
        model_version="1.0",
    )

    deleted = repo.delete("call_to_delete")
    assert deleted is True
    assert repo.get("call_to_delete") is None
