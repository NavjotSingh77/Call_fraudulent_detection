"""
Inference engine for fraud call classification.

Handles:
- Full-call transcript classification
- Segment-level transcript classification
- Probability calibration awareness
- Structured prediction report generation

Output explicitly states predictions are model estimates, not proof of fraud.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer string (single source of truth)
# ---------------------------------------------------------------------------

_DISCLAIMER = (
    "This prediction is a model estimate and does NOT constitute proof of fraud. "
    "Human review is recommended for consequential decisions."
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Structured output of a single-call fraud classification."""

    call_id: str
    language: str
    overall_prediction: str          # "fraud" | "legitimate" | "uncertain"
    fraud_probability: float
    confidence: str                  # "high" | "medium" | "low"
    segments: List[Dict] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)
    transcript: str = ""
    review_required: bool = False
    model_version: str = ""
    prediction_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    disclaimer: str = _DISCLAIMER


# ---------------------------------------------------------------------------
# Threshold helpers
# ---------------------------------------------------------------------------

def get_confidence_level(probability: float, config: Dict) -> str:
    """Map a fraud *probability* to a human-readable confidence label.

    Args:
        probability: Model output probability in [0, 1].
        config: Configuration dict; reads ``config["thresholds"]``.

    Returns:
        ``"high"``, ``"medium"``, or ``"low"``.
    """
    thresholds = config.get("thresholds", {})
    inf_cfg = config.get("inference", {})
    threshold_high: float = thresholds.get("threshold_high") or inf_cfg.get("confidence_threshold_high", 0.75)
    threshold_medium: float = thresholds.get("threshold_medium") or inf_cfg.get("confidence_threshold_medium", 0.45)

    if probability >= threshold_high:
        return "high"
    if probability >= threshold_medium:
        return "medium"
    return "low"


def get_prediction_label(probability: float, config: Dict) -> str:
    """Map a fraud *probability* to a prediction label.

    Args:
        probability: Model output probability in [0, 1].
        config: Configuration dict; reads ``config["thresholds"]``.

    Returns:
        ``"fraud"``, ``"uncertain"``, or ``"legitimate"``.
    """
    thresholds = config.get("thresholds", {})
    inf_cfg = config.get("inference", {})
    threshold_high: float = thresholds.get("threshold_high") or inf_cfg.get("confidence_threshold_high", 0.75)
    threshold_medium: float = thresholds.get("threshold_medium") or inf_cfg.get("confidence_threshold_medium", 0.45)

    if probability >= threshold_high:
        return "fraud"
    if probability >= threshold_medium:
        return "uncertain"
    return "legitimate"


# ---------------------------------------------------------------------------
# Segment aggregation
# ---------------------------------------------------------------------------

def aggregate_segment_predictions(
    segment_results: List[Dict],
    config: Dict,
) -> Dict:
    """Aggregate per-segment fraud probabilities into a single call-level score.

    Supported methods (configured via ``config["inference"]["aggregation_method"]``):

    - ``"weighted_mean"``: Weight each segment's probability by its
      ``duration_seconds`` field (falls back to uniform weight if missing).
    - ``"max"``: Use the maximum fraud probability across all segments.
    - ``"vote"``: Majority vote — segments are labelled fraud/not-fraud using
      ``threshold_medium``; the winning label's mean probability is returned.

    Args:
        segment_results: List of dicts, each containing at minimum
                         ``{"fraud_probability": float}`` and optionally
                         ``{"duration_seconds": float}``.
        config: Configuration dict.

    Returns:
        ``{"fraud_probability": float, "method_used": str, "num_segments": int}``
    """
    if not segment_results:
        return {"fraud_probability": 0.0, "method_used": "none", "num_segments": 0}

    inference_cfg = config.get("inference", {})
    method: str = inference_cfg.get("aggregation_method") or inference_cfg.get("segment_aggregation", "weighted_mean")
    thresholds = config.get("thresholds", {})
    threshold_medium: float = thresholds.get("threshold_medium") or inference_cfg.get("confidence_threshold_medium", 0.45)

    probabilities = [
        float(seg.get("fraud_probability", 0.0)) for seg in segment_results
    ]
    num_segments = len(probabilities)

    if method == "max":
        agg_prob = float(np.max(probabilities))

    elif method == "vote":
        labels = [p >= threshold_medium for p in probabilities]
        fraud_votes = sum(labels)
        legitimate_votes = num_segments - fraud_votes
        if fraud_votes >= legitimate_votes:
            fraud_probs = [p for p, l in zip(probabilities, labels) if l]
            agg_prob = float(np.mean(fraud_probs)) if fraud_probs else 0.5
        else:
            legit_probs = [p for p, l in zip(probabilities, labels) if not l]
            agg_prob = float(np.mean(legit_probs)) if legit_probs else 0.0

    else:  # "weighted_mean" (default)
        method = "weighted_mean"
        weights = [
            float(seg.get("duration_seconds", 1.0)) for seg in segment_results
        ]
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * num_segments
            total_weight = float(num_segments)
        agg_prob = float(
            sum(p * w for p, w in zip(probabilities, weights)) / total_weight
        )

    return {
        "fraud_probability": agg_prob,
        "method_used": method,
        "num_segments": num_segments,
    }


# ---------------------------------------------------------------------------
# Risk indicator extraction
# ---------------------------------------------------------------------------

def extract_risk_indicators(
    transcript: str,
    fraud_features,
) -> List[str]:
    """Build a human-readable list of risk indicators for the report.

    Args:
        transcript: Raw transcript text.
        fraud_features: :class:`~src.features.text_preprocessing.FraudFeatures`
                        instance (or None).

    Returns:
        List of descriptive strings.
    """
    indicators: List[str] = []

    if fraud_features is None:
        return indicators

    if fraud_features.contains_urgency_terms:
        indicators.append("Contains urgency language (e.g. 'immediately', 'final notice')")
    if fraud_features.contains_threat_terms:
        indicators.append("Contains threat language (e.g. 'arrest', 'legal action')")
    if fraud_features.contains_impersonation_terms:
        indicators.append(
            "Contains impersonation terms (e.g. 'IRS', 'Social Security Administration')"
        )
    if fraud_features.contains_financial_terms:
        indicators.append("Contains financial/payment terms (e.g. 'gift card', 'wire transfer')")
    if fraud_features.contains_otp_terms:
        indicators.append("Contains OTP / verification code requests")
    if fraud_features.contains_url:
        count = fraud_features.url_count
        indicators.append(f"Contains {count} URL{'s' if count != 1 else ''}")
    if fraud_features.contains_phone_number:
        count = fraud_features.phone_number_count
        indicators.append(f"Contains {count} phone number{'s' if count != 1 else ''}")
    if fraud_features.repeated_word_ratio > 0.1:
        indicators.append(
            f"Elevated repeated-word ratio ({fraud_features.repeated_word_ratio:.2f}) "
            "— possible scripted/pressuring speech"
        )

    return indicators


# ---------------------------------------------------------------------------
# Full transcript prediction
# ---------------------------------------------------------------------------

def predict_transcript(
    transcript: str,
    call_id: str,
    language: str,
    model,
    vectorizer_config: Dict,
    label_encoder,
    config: Dict,
    fraud_features=None,
) -> PredictionResult:
    """Run the full inference pipeline on a transcript.

    Args:
        transcript: Raw call transcript text.
        call_id: Unique identifier for the call.
        language: Detected or supplied language code.
        model: Loaded ``tf.keras.Model``.
        vectorizer_config: Dictionary returned by :func:`~src.models.bilstm.load_model_artifacts`.
        label_encoder: Fitted label encoder.
        config: Configuration dict.
        fraud_features: Optional pre-computed :class:`FraudFeatures`; extracted
                        automatically if None.

    Returns:
        Populated :class:`PredictionResult`.
    """
    # Lazy imports to keep module importable without TF
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for inference. Install with: pip install tensorflow"
        ) from exc

    from src.features.text_preprocessing import (  # type: ignore
        preprocess_for_model,
        extract_fraud_features,
    )

    # ------------------------------------------------------------------
    # 1. Feature extraction (supplementary signal)
    # ------------------------------------------------------------------
    if fraud_features is None:
        fraud_features = extract_fraud_features(transcript)

    # ------------------------------------------------------------------
    # 2. Text preprocessing
    # ------------------------------------------------------------------
    max_seq_len: Optional[int] = config.get("text_preprocessing", {}).get(
        "max_sequence_length"
    )
    processed_text = preprocess_for_model(transcript, max_length=max_seq_len)

    # ------------------------------------------------------------------
    # 3. Model prediction
    # ------------------------------------------------------------------
    try:
        raw_output = model.predict(
            [processed_text],
            verbose=0,
        )
        raw_prob = float(np.squeeze(raw_output))
        # For softmax outputs, take the fraud class probability
        if raw_output.shape[-1] > 1:
            raw_prob = float(raw_output[0, -1])  # assumes last class = fraud
        # Clip to valid probability range
        fraud_probability = float(np.clip(raw_prob, 0.0, 1.0))
    except Exception as exc:
        logger.error("Model prediction failed for call_id=%s: %s", call_id, exc)
        fraud_probability = 0.0

    # ------------------------------------------------------------------
    # 4. Threshold-based labelling & confidence
    # ------------------------------------------------------------------
    overall_prediction = get_prediction_label(fraud_probability, config)
    confidence = get_confidence_level(fraud_probability, config)
    review_required = overall_prediction in ("fraud", "uncertain")

    # ------------------------------------------------------------------
    # 5. Risk indicators
    # ------------------------------------------------------------------
    risk_indicators = extract_risk_indicators(transcript, fraud_features)

    # ------------------------------------------------------------------
    # 6. Model version
    # ------------------------------------------------------------------
    model_version: str = config.get("model", {}).get("version", "unknown")

    return PredictionResult(
        call_id=call_id,
        language=language,
        overall_prediction=overall_prediction,
        fraud_probability=fraud_probability,
        confidence=confidence,
        segments=[],
        risk_indicators=risk_indicators,
        transcript=transcript,
        review_required=review_required,
        model_version=model_version,
        prediction_timestamp=datetime.utcnow().isoformat(),
        disclaimer=_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def result_to_json(result: PredictionResult) -> Dict:
    """Serialise a :class:`PredictionResult` to a JSON-compatible dictionary.

    Args:
        result: Prediction result dataclass.

    Returns:
        Dictionary matching the project's output schema.
    """
    return {
        "call_id": result.call_id,
        "language": result.language,
        "overall_prediction": result.overall_prediction,
        "fraud_probability": round(result.fraud_probability, 6),
        "confidence": result.confidence,
        "segments": result.segments,
        "risk_indicators": result.risk_indicators,
        "transcript": result.transcript,
        "review_required": result.review_required,
        "model_version": result.model_version,
        "prediction_timestamp": result.prediction_timestamp,
        "disclaimer": result.disclaimer,
    }
