"""
Unit tests for src.audio.preprocessing module.
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.audio.preprocessing import (
    AudioLoadError,
    detect_corrupt_audio,
    detect_silence,
    get_audio_duration,
    load_audio,
    normalize_amplitude,
    preprocess_audio,
    validate_audio_format,
)


def test_validate_audio_format_valid_wav(tmp_audio_file):
    result = validate_audio_format(tmp_audio_file)
    assert result["valid"] is True
    assert result["sample_rate"] == 16000
    assert result["duration_seconds"] > 0
    assert result["error"] is None


def test_validate_audio_format_missing_file(tmp_path):
    missing_file = tmp_path / "does_not_exist.wav"
    result = validate_audio_format(missing_file)
    assert result["valid"] is False
    assert result["error"] is not None
    assert "not found" in result["error"].lower()


def test_validate_audio_format_corrupt_file(corrupt_audio_file):
    result = validate_audio_format(corrupt_audio_file)
    assert result["valid"] is False
    assert result["error"] is not None


def test_load_audio_resampling(tmp_path):
    # Create 44100Hz audio and verify resampling to 16000Hz
    sr_orig = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr_orig * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "audio_44k.wav"
    sf.write(str(path), data, sr_orig)

    audio, sr = load_audio(path, target_sr=16000, mono=True)
    assert sr == 16000
    assert len(audio) == int(duration * 16000)
    assert audio.dtype == np.float32


def test_load_audio_mono_conversion(tmp_path):
    # Create stereo audio
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    ch1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    ch2 = 0.3 * np.cos(2 * np.pi * 880 * t)
    stereo = np.column_stack([ch1, ch2]).astype(np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, sr)

    audio, sr_out = load_audio(path, target_sr=16000, mono=True)
    assert audio.ndim == 1
    assert sr_out == 16000


def test_load_audio_missing_raises_error(tmp_path):
    with pytest.raises(AudioLoadError):
        load_audio(tmp_path / "nonexistent.wav")


def test_normalize_amplitude():
    # Peak of 0.2 -> normalized to -3 dBFS (approx 0.7079)
    audio = np.array([-0.2, 0.0, 0.2, 0.1], dtype=np.float32)
    norm = normalize_amplitude(audio, target_level=-3.0)
    expected_peak = 10 ** (-3.0 / 20.0)
    assert np.isclose(np.max(np.abs(norm)), expected_peak, atol=1e-3)


def test_normalize_amplitude_silent():
    # Silent audio should remain untouched without divide by zero
    zeros = np.zeros(100, dtype=np.float32)
    norm = normalize_amplitude(zeros, target_level=-3.0)
    assert np.all(norm == 0.0)


def test_detect_silence_mostly_silent():
    sr = 16000
    silent_audio = np.random.normal(0, 1e-5, sr * 2).astype(np.float32)
    res = detect_silence(silent_audio, sr=sr, threshold_db=-60.0)
    assert res["is_mostly_silent"] is True
    assert res["silence_ratio"] > 0.9


def test_detect_silence_speech():
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    sound = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    res = detect_silence(sound, sr=sr, threshold_db=-60.0)
    assert res["is_mostly_silent"] is False
    assert res["silence_ratio"] < 0.2


def test_detect_corrupt_audio_valid(tmp_audio_file):
    res = detect_corrupt_audio(tmp_audio_file)
    assert res["is_corrupt"] is False
    assert res["reason"] is None


def test_detect_corrupt_audio_invalid(corrupt_audio_file):
    res = detect_corrupt_audio(corrupt_audio_file)
    assert res["is_corrupt"] is True
    assert res["reason"] is not None


def test_get_audio_duration(tmp_audio_file):
    dur = get_audio_duration(tmp_audio_file)
    assert abs(dur - 5.0) < 0.2


def test_preprocess_audio(tmp_audio_file, minimal_config, tmp_path):
    out_dir = tmp_path / "processed"
    res = preprocess_audio(tmp_audio_file, out_dir, minimal_config)
    assert res.get("success") is True or res.get("processed") is True
    out_path = res.get("output_path") or res.get("processed_path")
    assert out_path is not None
    assert Path(out_path).exists()
    assert res["sample_rate"] == 16000
    assert res["channels"] == 1
