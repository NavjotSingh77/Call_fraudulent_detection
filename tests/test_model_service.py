"""
Unit tests for the dedicated ModelService layer.
"""

from pathlib import Path
import pytest

from src.services.model_service import ModelNotReadyException, ModelService


def test_model_service_unloaded_raises_exception(tmp_path):
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    svc = ModelService(model_dir=empty_dir)

    assert svc.is_loaded is False
    with pytest.raises(ModelNotReadyException):
        svc.predict("Hello this is a test call.")


def test_model_service_validate_artifacts_exist(tmp_path):
    model_dir = tmp_path / "test_models"
    model_dir.mkdir()
    svc = ModelService(model_dir=model_dir)

    exists, missing = svc.validate_artifacts_exist()
    assert exists is False
    assert len(missing) >= 3

    # Create dummy artifacts
    (model_dir / "model.keras").touch()
    (model_dir / "vectorizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "label_encoder.pkl").write_bytes(b"dummy")
    (model_dir / "model_config.json").write_text("{}", encoding="utf-8")

    exists2, missing2 = svc.validate_artifacts_exist()
    assert exists2 is True
    assert len(missing2) == 0


def test_model_service_get_info_unloaded(tmp_path):
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    svc = ModelService(model_dir=empty_dir)
    info = svc.get_info()
    assert info["model_loaded"] is False
    assert info["model_type"] == "BiLSTM"
