"""
Unit tests for inference, decision engine, and report generation.
"""

import pytest

from src.features.text_preprocessing import extract_fraud_features
from src.models.inference import (
    PredictionResult,
    aggregate_segment_predictions,
    extract_risk_indicators,
    get_confidence_level,
    get_prediction_label,
    result_to_json,
)


def test_confidence_level_mapping(minimal_config):
    # high: >= 0.85, medium: >= 0.65, low: < 0.65
    assert get_confidence_level(0.90, minimal_config) == "high"
    assert get_confidence_level(0.70, minimal_config) == "medium"
    assert get_confidence_level(0.40, minimal_config) == "low"


def test_prediction_label_mapping(minimal_config):
    assert get_prediction_label(0.90, minimal_config) == "fraud"
    assert get_prediction_label(0.70, minimal_config) == "uncertain"
    assert get_prediction_label(0.30, minimal_config) == "legitimate"


def test_aggregate_segment_predictions_max():
    config = {"inference": {"segment_aggregation": "max"}}
    segments = [
        {"fraud_probability": 0.2, "duration_seconds": 30.0},
        {"fraud_probability": 0.85, "duration_seconds": 30.0},
        {"fraud_probability": 0.4, "duration_seconds": 15.0},
    ]
    res = aggregate_segment_predictions(segments, config)
    assert res["fraud_probability"] == 0.85
    assert res["method_used"] == "max"
    assert res["num_segments"] == 3


def test_aggregate_segment_predictions_weighted_mean():
    config = {"inference": {"segment_aggregation": "weighted_mean"}}
    segments = [
        {"fraud_probability": 0.2, "duration_seconds": 20.0},
        {"fraud_probability": 0.8, "duration_seconds": 20.0},
    ]
    # Equal durations -> mean = 0.5
    res = aggregate_segment_predictions(segments, config)
    assert abs(res["fraud_probability"] - 0.5) < 1e-4
    assert res["method_used"] == "weighted_mean"


def test_aggregate_segment_predictions_empty():
    config = {"inference": {"segment_aggregation": "weighted_mean"}}
    res = aggregate_segment_predictions([], config)
    assert res["fraud_probability"] == 0.0
    assert res["num_segments"] == 0


def test_extract_risk_indicators():
    text = "Call immediately or face arrest by the IRS. Provide your OTP now."
    feat = extract_fraud_features(text)
    indicators = extract_risk_indicators(text, feat)
    assert len(indicators) >= 2
    assert any("urgency" in i.lower() for i in indicators)
    assert any("threat" in i.lower() for i in indicators)


def test_prediction_result_disclaimer():
    pr = PredictionResult(
        call_id="call_disc",
        language="en",
        overall_prediction="fraud",
        fraud_probability=0.92,
        confidence="high",
    )
    assert "not" in pr.disclaimer.lower() or "estimate" in pr.disclaimer.lower()


def test_result_to_json_structure():
    pr = PredictionResult(
        call_id="call_json_01",
        language="en",
        overall_prediction="suspicious",
        fraud_probability=0.87,
        confidence="medium",
        segments=[],
        risk_indicators=["Urgency detected"],
        transcript="Test transcript text.",
        review_required=True,
    )
    json_data = result_to_json(pr)
    required_keys = [
        "call_id",
        "language",
        "overall_prediction",
        "fraud_probability",
        "confidence",
        "segments",
        "risk_indicators",
        "transcript",
        "review_required",
        "disclaimer",
    ]
    for k in required_keys:
        assert k in json_data, f"Missing key in JSON output: {k}"
    assert json_data["fraud_probability"] == 0.87
