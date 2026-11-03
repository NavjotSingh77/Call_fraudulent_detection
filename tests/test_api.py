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
    assert data["status"] == "ok"
    assert "version" in data
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
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_with_valid_audio(client, tmp_audio_file):
    with open(tmp_audio_file, "rb") as f:
        files = {"file": (tmp_audio_file.name, f, "audio/wav")}
        response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "call_id" in data
    assert "fraud_probability" in data
    assert "disclaimer" in data


def test_transcribe_endpoint(client, tmp_audio_file):
    with open(tmp_audio_file, "rb") as f:
        files = {"file": (tmp_audio_file.name, f, "audio/wav")}
        response = client.post("/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert "language" in data


def test_predict_segments_endpoint(client, tmp_audio_file):
    with open(tmp_audio_file, "rb") as f:
        files = {"file": (tmp_audio_file.name, f, "audio/wav")}
        response = client.post("/predict/segments", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "call_id" in data
    assert "segments" in data
    assert isinstance(data["segments"], list)
    assert len(data["segments"]) > 0
