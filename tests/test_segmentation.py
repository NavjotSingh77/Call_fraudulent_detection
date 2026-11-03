"""
Unit tests for src.audio.segmentation module.
"""

from pathlib import Path
import numpy as np
import pytest

from src.audio.segmentation import (
    AudioSegment,
    get_segment_timestamps,
    load_audio_for_segmentation,
    save_segments,
    segment_audio,
)


def test_segment_timestamps_basic():
    # 90 seconds audio, 30s segment, 15s stride:
    # 0-30, 15-45, 30-60, 45-75, 60-90, 75-90
    ts = get_segment_timestamps(audio_duration=90.0, segment_duration=30.0, stride=15.0)
    assert len(ts) >= 5
    assert ts[0] == (0.0, 30.0)
    assert ts[1] == (15.0, 45.0)
    assert ts[-1][1] <= 90.0


def test_segment_timestamps_short():
    ts = get_segment_timestamps(audio_duration=5.0, segment_duration=30.0, stride=15.0)
    assert len(ts) == 1
    assert ts[0] == (0.0, 5.0)


def test_segment_timestamps_zero():
    ts = get_segment_timestamps(audio_duration=0.0, segment_duration=30.0, stride=15.0)
    assert ts == []


def test_segment_audio_basic():
    sr = 16000
    duration = 60.0
    audio = np.random.normal(0, 0.1, int(sr * duration)).astype(np.float32)

    segments = segment_audio(
        audio=audio,
        sr=sr,
        call_id="call_test_001",
        segment_duration=30.0,
        stride=15.0,
    )

    assert len(segments) >= 3
    for seg in segments:
        assert isinstance(seg, AudioSegment)
        assert seg.call_id == "call_test_001"
        assert seg.sample_rate == sr
        assert seg.annotation_type == "exploratory"
        assert seg.has_label is False
        assert seg.duration <= 30.0
        assert len(seg.audio) == int(seg.duration * sr)


def test_save_segments(tmp_path):
    sr = 16000
    audio = np.random.normal(0, 0.1, sr * 45).astype(np.float32)
    segments = segment_audio(audio, sr, "call_save", segment_duration=20.0, stride=10.0)

    out_dir = tmp_path / "segments_out"
    saved_paths = save_segments(segments, out_dir)

    assert len(saved_paths) == len(segments)
    for p in saved_paths:
        assert p.exists()
        assert p.suffix == ".wav"
