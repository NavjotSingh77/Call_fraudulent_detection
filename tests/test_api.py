"""
Unit tests for FastAPI endpoints in src.api.main.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, app_state
from src.utils.config import load_config


@pytest.fixture
def client(minimal_config):
    app_state["config"] = minimal_config
    app_state["model_loaded"] = False
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "healthy", "degraded"]
    assert "timestamp" in data
    assert "model_loaded" in data


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    # When model is not loaded yet
    assert "model_loaded" in data


def test_unsupported_file_upload(client):
    files = {"file": ("test.txt", b"plain text is not audio", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 415
    assert "Unsupported audio format" in response.json()["detail"] or "Unsupported" in response.json()["detail"]


def test_predict_with_valid_audio(client, tmp_audio_file):
    from src.services.model_service import get_model_service
    from src.config.settings import get_settings
    svc = get_model_service(get_settings().model_dir)
    orig_loaded = svc._is_loaded
    orig_predict = getattr(svc, "predict", None)

    svc._is_loaded = True
    svc.predict = lambda t: {
        "fraud_probability": 0.88,
        "confidence": "high",
        "overall_prediction": "fraud",
        "review_required": False,
        "risk_indicators": ["Urgency detected"],
        "model_version": "BiLSTM-v1.0.0",
    }
    try:
        with open(tmp_audio_file, "rb") as f:
            files = {"file": (tmp_audio_file.name, f, "audio/wav")}
            response = client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "call_id" in data
        assert "fraud_probability" in data
        assert "disclaimer" in data
    finally:
        svc._is_loaded = orig_loaded
        if orig_predict:
            svc.predict = orig_predict


def test_transcribe_endpoint(client, tmp_audio_file):
    with open(tmp_audio_file, "rb") as f:
        files = {"file": (tmp_audio_file.name, f, "audio/wav")}
        response = client.post("/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert "language" in data


def test_predict_segments_endpoint(client, tmp_audio_file):
    from src.services.model_service import get_model_service
    from src.config.settings import get_settings
    svc = get_model_service(get_settings().model_dir)
    orig_loaded = svc._is_loaded
    orig_predict = getattr(svc, "predict", None)

    svc._is_loaded = True
    svc.predict = lambda t: {
        "fraud_probability": 0.88,
        "confidence": "high",
        "overall_prediction": "fraud",
        "review_required": False,
        "risk_indicators": ["Urgency detected"],
        "model_version": "BiLSTM-v1.0.0",
    }
    try:
        with open(tmp_audio_file, "rb") as f:
            files = {"file": (tmp_audio_file.name, f, "audio/wav")}
            response = client.post("/predict/segments", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "call_id" in data
        assert "segments" in data
        assert isinstance(data["segments"], list)
    finally:
        svc._is_loaded = orig_loaded
        if orig_predict:
            svc.predict = orig_predict
