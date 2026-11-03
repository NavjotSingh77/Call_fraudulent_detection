"""
Test fixtures for the Fraud Call Detection system.

Provides in-memory datasets and synthetic audio generation for testing
without requiring external network downloads or large models.
"""

from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import soundfile as sf


def create_sample_manifest() -> pd.DataFrame:
    """Create a mock manifest DataFrame with 10 records for testing."""
    data = [
        {
            "call_id": "call_001",
            "audio_path": "data/raw/call_001.wav",
            "transcript": "Your social security number has been suspended. Call immediately or face arrest warrant.",
            "language": "en",
            "label": "fraud",
            "label_status": "verified",
            "source": "ncsu_ftc",
            "recording_type": "robocall",
            "case_pdf": "pdf_files/case_01.pdf",
        },
        {
            "call_id": "call_002",
            "audio_path": "data/raw/call_002.wav",
            "transcript": "This is the IRS. You owe back taxes. Pay now with gift cards to avoid federal lawsuit.",
            "language": "en",
            "label": "fraud",
            "label_status": "verified",
            "source": "ncsu_ftc",
            "recording_type": "robocall",
            "case_pdf": "pdf_files/case_02.pdf",
        },
        {
            "call_id": "call_003",
            "audio_path": "data/raw/call_003.wav",
            "transcript": "Your bank account has been compromised. Provide your one-time password OTP to verify identity.",
            "language": "en",
            "label": "fraud",
            "label_status": "verified",
            "source": "ncsu_ftc",
            "recording_type": "robocall",
            "case_pdf": "pdf_files/case_03.pdf",
        },
        {
            "call_id": "call_004",
            "audio_path": "data/raw/call_004.wav",
            "transcript": "Final notice. Press 1 to speak with an officer before law enforcement is dispatched.",
            "language": "en",
            "label": "fraud",
            "label_status": "weak",
            "source": "ncsu_ftc",
            "recording_type": "robocall",
            "case_pdf": "pdf_files/case_04.pdf",
        },
        {
            "call_id": "call_005",
            "audio_path": "data/raw/call_005.wav",
            "transcript": "Congratulations you have won a five hundred dollar gift card claim at scam-url.com now.",
            "language": "en",
            "label": "fraud",
            "label_status": "verified",
            "source": "ncsu_ftc",
            "recording_type": "robocall",
            "case_pdf": "pdf_files/case_05.pdf",
        },
        {
            "call_id": "call_006",
            "audio_path": "data/raw/call_006.wav",
            "transcript": "Suspicious wire transfer detected on your credit card. Contact customer support immediately.",
            "language": "en",
            "label": "fraud",
            "label_status": "verified",
            "source": "ncsu_ftc",
            "recording_type": "robocall",
            "case_pdf": "pdf_files/case_06.pdf",
        },
        {
            "call_id": "call_007",
            "audio_path": "data/raw/call_007.wav",
            "transcript": "Hello, this is Dr. Smith's office confirming your dental appointment for Tuesday at 3pm.",
            "language": "en",
            "label": "legitimate",
            "label_status": "verified",
            "source": "medical_clinic",
            "recording_type": "reminder",
            "case_pdf": None,
        },
        {
            "call_id": "call_008",
            "audio_path": "data/raw/call_008.wav",
            "transcript": "Good morning, thank you for calling customer care. How can I assist you with your subscription today?",
            "language": "en",
            "label": "legitimate",
            "label_status": "verified",
            "source": "support_center",
            "recording_type": "human_dialogue",
            "case_pdf": None,
        },
        {
            "call_id": "call_009",
            "audio_path": "data/raw/call_009.wav",
            "transcript": "Your package has been delivered to your front porch. Have a nice weekend.",
            "language": "en",
            "label": "legitimate",
            "label_status": "verified",
            "source": "courier",
            "recording_type": "notification",
            "case_pdf": None,
        },
        {
            "call_id": "call_010",
            "audio_path": "data/raw/call_010.wav",
            "transcript": "Static noise and unrecognizable murmurs on line.",
            "language": "unknown",
            "label": "uncertain",
            "label_status": "unknown",
            "source": "honeypot",
            "recording_type": "unknown",
            "case_pdf": None,
        },
    ]
    return pd.DataFrame(data)


def create_synthetic_audio(duration_seconds: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a simple sine tone mixed with low-level noise."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    # 440 Hz tone + slight noise
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    noise = 0.05 * np.random.normal(0, 1, len(t))
    audio = tone + noise
    return audio.astype(np.float32)


def save_synthetic_audio(output_path: Path, duration: float = 3.0, sr: int = 16000) -> Path:
    """Generate and write a synthetic audio file in WAV format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = create_synthetic_audio(duration_seconds=duration, sample_rate=sr)
    sf.write(str(output_path), audio, sr, subtype="PCM_16")
    return output_path


def create_corrupt_audio_file(output_path: Path) -> Path:
    """Create a damaged/corrupt audio file for validation tests."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00\x00CORRUPT_DATA_GARBAGE")
    return output_path


def create_minimal_config() -> Dict[str, Any]:
    """Create minimal valid config matching config.yaml specification."""
    return {
        "project": {"name": "test-fraud-detection", "version": "1.0.0", "random_seed": 42},
        "paths": {
            "data_raw": "data/raw",
            "data_external": "data/external",
            "data_processed_audio": "data/processed/audio",
            "data_processed_transcripts": "data/processed/transcripts",
            "data_metadata": "data/metadata",
            "data_splits": "data/splits",
            "models": "models",
            "reports": "reports",
            "figures": "reports/figures",
            "evaluation": "reports/evaluation",
            "manifest": "data/metadata/manifest.csv",
        },
        "audio": {
            "target_sample_rate": 16000,
            "target_channels": 1,
            "min_duration_seconds": 1.0,
            "max_duration_seconds": 3600.0,
            "normalize_amplitude": True,
            "silence_threshold_db": -60.0,
            "noise_reduction": False,
            "segment_duration_seconds": 30,
            "segment_stride_seconds": 15,
            "supported_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
        },
        "transcription": {
            "whisper_model": "base",
            "device": "cpu",
            "language": None,
            "translate": False,
            "cache_transcripts": True,
            "batch_size": 1,
        },
        "text_preprocessing": {
            "max_sequence_length": 256,
            "vocabulary_size": 1000,
            "min_word_frequency": 1,
            "remove_stopwords": False,
        },
        "model": {
            "embedding_dim": 64,
            "lstm_units": 64,
            "dropout_rate": 0.3,
            "dense_units": 32,
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 5,
            "early_stopping_patience": 3,
            "reduce_lr_patience": 2,
            "reduce_lr_factor": 0.5,
            "num_classes": 2,
        },
        "training": {
            "test_size": 0.2,
            "val_size": 0.2,
            "stratify": True,
            "group_split": True,
            "class_weight_method": "balanced",
            "checkpoint_metric": "val_recall",
            "checkpoint_mode": "max",
        },
        "inference": {
            "confidence_threshold_high": 0.85,
            "confidence_threshold_medium": 0.65,
            "segment_aggregation": "weighted_mean",
            "review_threshold": 0.65,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "max_upload_size_mb": 100,
            "log_level": "info",
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "file": "logs/test.log",
        },
    }
