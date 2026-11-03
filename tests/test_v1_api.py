"""
Unit tests for API v1 endpoints and Web UI routes.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config.settings import get_settings
from src.services.model_service import get_model_service

settings = get_settings()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_v1_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "hardware_device" in data


def test_v1_model_info(client):
    resp = client.get("/api/v1/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_loaded" in data
    assert "whisper_model" in data


def test_web_ui_dashboard(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Call Fraud Detection AI" in resp.text
    assert "text/html" in resp.headers["content-type"]

    resp_ui = client.get("/ui")
    assert resp_ui.status_code == 200
    assert resp_ui.text == resp.text


def test_v1_transcribe_audio(client, tmp_audio_file):
    with open(tmp_audio_file, "rb") as f:
        files = {"file": (tmp_audio_file.name, f, "audio/wav")}
        resp = client.post("/api/v1/transcribe", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "call_id" in data
    assert "transcript" in data
    assert "language" in data


def test_correlation_and_timing_headers(client):
    resp = client.get("/api/v1/health")
    assert "X-Request-ID" in resp.headers
    assert "X-Process-Time-Ms" in resp.headers


def test_unsupported_audio_type(client):
    files = {"file": ("document.pdf", b"%PDF-1.4 dummy", "application/pdf")}
    resp = client.post("/api/v1/transcribe", files=files)
    assert resp.status_code == 415


def test_predict_model_not_ready_returns_503(client, tmp_audio_file):
    # When model is not loaded, it should never fabricate results
    model_svc = get_model_service(settings.model_dir)
    # Ensure is_loaded is False for this test
    original_state = model_svc._is_loaded
    model_svc._is_loaded = False

    try:
        with open(tmp_audio_file, "rb") as f:
            files = {"file": (tmp_audio_file.name, f, "audio/wav")}
            resp = client.post("/api/v1/predict", files=files)
        assert resp.status_code == 503
        assert "not loaded or trained" in resp.json()["detail"].lower()
    finally:
        model_svc._is_loaded = original_state
