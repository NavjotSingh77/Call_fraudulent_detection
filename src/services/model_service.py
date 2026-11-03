"""
Dedicated model-service layer for loading model artifacts and executing inference.

This service is isolated from training code. It operates directly on
production-saved model artifacts (.keras format preferred), vectorizer configuration,
label encoder, and metadata.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelNotReadyException(Exception):
    """Raised when inference is requested but the model artifact is not available."""


class ModelService:
    """Dedicated inference service managing model lifecycle and predictions."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.model: Optional[Any] = None
        self.vectorizer_config: Optional[Dict[str, Any]] = None
        self.label_encoder: Optional[Any] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.model_metadata: Optional[Dict[str, Any]] = None
        self.model_format: Optional[str] = None
        self._is_loaded: bool = False

        # Attempt to load artifact on service initialization
        self.load_artifacts()

    @property
    def is_loaded(self) -> bool:
        """Return True if model and all companion artifacts are loaded and ready."""
        return self._is_loaded

    def validate_artifacts_exist(self) -> Tuple[bool, List[str]]:
        """Check if all required production files exist in the model directory."""
        missing = []
        model_keras = self.model_dir / "model.keras"
        model_dir = self.model_dir / "model"

        if not (model_keras.exists() or model_dir.exists()):
            missing.append("model.keras (or model/ directory)")

        if not (self.model_dir / "vectorizer_config.json").exists():
            missing.append("vectorizer_config.json")

        if not (self.model_dir / "label_encoder.pkl").exists():
            missing.append("label_encoder.pkl")

        if not (self.model_dir / "model_config.json").exists():
            missing.append("model_config.json")

        return len(missing) == 0, missing

    def load_artifacts(self) -> bool:
        """Load the trained .keras model and companion artifacts.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        exists, missing = self.validate_artifacts_exist()
        if not exists:
            logger.warning(
                "Model artifacts not found in %s. Missing components: %s",
                self.model_dir,
                ", ".join(missing),
            )
            self._is_loaded = False
            return False

        try:
            import tensorflow as tf  # type: ignore

            model_keras = self.model_dir / "model.keras"
            model_saved_dir = self.model_dir / "model"

            if model_keras.exists():
                logger.info("Loading Keras model from %s", model_keras)
                self.model = tf.keras.models.load_model(str(model_keras))
                self.model_format = ".keras"
            elif model_saved_dir.exists():
                logger.info("Loading SavedModel from %s", model_saved_dir)
                self.model = tf.keras.models.load_model(str(model_saved_dir))
                self.model_format = "SavedModel"

            # 2. Vectorizer config
            with open(self.model_dir / "vectorizer_config.json", "r", encoding="utf-8") as f:
                self.vectorizer_config = json.load(f)

            # 3. Label encoder
            with open(self.model_dir / "label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)

            # 4. Model configuration
            with open(self.model_dir / "model_config.json", "r", encoding="utf-8") as f:
                self.model_config = json.load(f)

            # 5. Metadata (optional, created at training time)
            meta_path = self.model_dir / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.model_metadata = json.load(f)
            else:
                self.model_metadata = {
                    "version": "1.0.0",
                    "loaded_at": datetime.now(timezone.utc).isoformat(),
                }

            self._is_loaded = True
            logger.info("ModelService successfully initialized with active model.")
            return True

        except Exception as exc:
            logger.error("Failed to load model artifacts from %s: %s", self.model_dir, exc)
            self._is_loaded = False
            return False

    def predict(
        self,
        transcript: str,
        threshold_high: float = 0.75,
        threshold_medium: float = 0.45,
    ) -> Dict[str, Any]:
        """Execute inference on a transcript using the loaded model artifact.

        Raises:
            ModelNotReadyException: If model is not loaded.
        """
        if not self._is_loaded or self.model is None:
            raise ModelNotReadyException(
                "Model is not loaded or trained. Supervised predictions cannot be generated. "
                "Train the model first using: python main.py train (or run_system.bat train)"
            )

        from src.features.text_preprocessing import (
            extract_fraud_features,
            preprocess_for_model,
        )

        # 1. Feature extraction for risk indicators
        fraud_feat = extract_fraud_features(transcript)

        # 2. Text normalization
        max_seq_len = None
        if self.model_config:
            max_seq_len = (
                self.model_config.get("text_preprocessing", {}).get("max_sequence_length")
            )
        processed_text = preprocess_for_model(transcript, max_length=max_seq_len)

        # 3. Model forward pass
        raw_pred = self.model.predict([processed_text], verbose=0)
        raw_prob = float(np.squeeze(raw_pred))
        if hasattr(raw_pred, "shape") and raw_pred.shape[-1] > 1:
            raw_prob = float(raw_pred[0, -1])  # fraud class probability
        fraud_prob = float(np.clip(raw_prob, 0.0, 1.0))

        # 4. Calibration & confidence assignment
        if fraud_prob >= threshold_high:
            confidence = "high"
            overall = "fraud"
        elif fraud_prob >= threshold_medium:
            confidence = "medium"
            overall = "uncertain"
        else:
            confidence = "low" if fraud_prob > 0.25 else "high"
            overall = "legitimate"

        # 5. Risk indicators
        from src.models.inference import extract_risk_indicators

        risk_indicators = extract_risk_indicators(transcript, fraud_feat)

        review_required = overall == "uncertain" or confidence != "high"

        return {
            "fraud_probability": round(fraud_prob, 4),
            "confidence": confidence,
            "overall_prediction": overall,
            "review_required": review_required,
            "risk_indicators": risk_indicators,
            "model_version": self.get_version(),
        }

    def get_version(self) -> str:
        """Return version string of active model."""
        if self.model_metadata and "version" in self.model_metadata:
            return str(self.model_metadata["version"])
        return "BiLSTM-v1.0.0"

    def get_info(self) -> Dict[str, Any]:
        """Return runtime model metadata for health/info endpoints."""
        classes = ["legitimate", "fraud"]
        if self.label_encoder and hasattr(self.label_encoder, "classes_"):
            classes = [str(c) for c in self.label_encoder.classes_]

        return {
            "model_loaded": self._is_loaded,
            "model_version": self.get_version() if self._is_loaded else None,
            "model_format": self.model_format,
            "model_type": "BiLSTM",
            "classes": classes if self._is_loaded else [],
            "hyperparameters": self.model_config.get("model", {}) if self.model_config else {},
            "model_metadata": self.model_metadata or {},
        }


# Singleton service holder
_model_service_instance: Optional[ModelService] = None


def get_model_service(model_dir: Optional[Path] = None) -> ModelService:
    """Dependency injection provider for ModelService."""
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService(model_dir=model_dir)
    return _model_service_instance
