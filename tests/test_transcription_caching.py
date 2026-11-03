"""
Unit tests for transcription caching and metadata handling.
"""

from pathlib import Path
import pytest

from src.audio.transcription import (
    TranscriptionResult,
    get_transcript_cache_path,
    load_cached_transcript,
    load_existing_transcript,
    save_transcript_cache,
)


def test_cache_path_generation(tmp_path):
    p = get_transcript_cache_path("call_123", tmp_path)
    assert p == tmp_path / "call_123_transcript.json"


def test_cache_roundtrip(tmp_path):
    tr = TranscriptionResult(
        call_id="call_test",
        audio_path="/path/to/test.wav",
        language="en",
        language_probability=0.98,
        transcript="This is a test transcript for caching validation.",
        segments=[{"id": 0, "start": 0.0, "end": 2.5, "text": "This is a test"}],
        model_used="base",
        cached=False,
    )

    saved_path = save_transcript_cache(tr, tmp_path)
    assert saved_path.exists()

    loaded = load_cached_transcript("call_test", tmp_path)
    assert loaded is not None
    assert loaded.call_id == "call_test"
    assert loaded.cached is True
    assert loaded.transcript == tr.transcript
    assert loaded.language == "en"
    assert len(loaded.segments) == 1


def test_cache_miss_returns_none(tmp_path):
    result = load_cached_transcript("non_existent_call", tmp_path)
    assert result is None


def test_load_existing_ncsu_transcript():
    sample_text = "This is an FTC enforcement case call recording."
    tr = load_existing_transcript(
        transcript_text=sample_text,
        call_id="ncsu_001",
        audio_path="data/raw/ncsu_001.wav",
        language="en",
    )
    assert tr.call_id == "ncsu_001"
    assert tr.transcript == sample_text
    assert tr.model_used == "ncsu_provided"
    assert tr.cached is True
    assert tr.error is None
