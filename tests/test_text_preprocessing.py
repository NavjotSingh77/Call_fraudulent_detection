"""
Unit tests for src.features.text_preprocessing module.
"""

import pytest

from src.features.text_preprocessing import (
    FraudFeatures,
    detect_repeated_words,
    extract_fraud_features,
    extract_phone_numbers,
    extract_urls,
    features_to_dict,
    normalize_text,
    preprocess_for_model,
)


def test_normalize_text_basic():
    text = "   HELLO   World!   "
    normalized = normalize_text(text)
    assert "hello" in normalized
    assert "world" in normalized
    assert "   " not in normalized


def test_normalize_text_preserves_numbers():
    text = "Your tax debt is $5,000. Call 1-800-555-0199 now."
    normalized = normalize_text(text)
    assert "5000" in normalized or "5,000" in normalized or "5" in normalized
    assert "tax debt" in normalized


def test_normalize_text_preserves_negations():
    text = "Do NOT ignore this call. We cannot cancel the warrant."
    normalized = normalize_text(text)
    assert "not" in normalized
    assert "cannot" in normalized or "not" in normalized


def test_extract_urls():
    text = "Click here: https://secure-bank.example.com/login and visit http://claim-prize.net now!"
    urls = extract_urls(text)
    assert len(urls) >= 2
    assert any("secure-bank.example.com" in u for u in urls)
    assert any("claim-prize.net" in u for u in urls)


def test_extract_phone_numbers():
    text = "Call us back at (800) 555-0199 or 888-123-4567 immediately."
    numbers = extract_phone_numbers(text)
    assert len(numbers) >= 2


def test_fraud_features_urgency():
    text = "Your account expires immediately. Final notice: call within 24 hours to avoid suspension."
    feat = extract_fraud_features(text)
    assert feat.contains_urgency_terms is True


def test_fraud_features_threat():
    text = "A federal warrant has been issued for your arrest. You will be taken to prison by law enforcement."
    feat = extract_fraud_features(text)
    assert feat.contains_threat_terms is True


def test_fraud_features_financial():
    text = "Purchase a Google Play or Apple gift card and execute a wire transfer to resolve back taxes."
    feat = extract_fraud_features(text)
    assert feat.contains_financial_terms is True


def test_fraud_features_otp():
    text = "Please enter your one-time password OTP to verify your account identity."
    feat = extract_fraud_features(text)
    assert feat.contains_otp_terms is True


def test_fraud_features_impersonation():
    text = "This is Officer Davis calling from the Social Security Administration and IRS."
    feat = extract_fraud_features(text)
    assert feat.contains_impersonation_terms is True


def test_fraud_features_legitimate():
    text = "Hi, your appointment with Dr. Johnson is scheduled for tomorrow at 10 AM. Reply YES to confirm."
    feat = extract_fraud_features(text)
    assert feat.contains_urgency_terms is False
    assert feat.contains_threat_terms is False
    assert feat.contains_impersonation_terms is False


def test_detect_repeated_words():
    repeated_text = "now now now call call please please"
    ratio = detect_repeated_words(repeated_text)
    assert ratio > 0.0

    normal_text = "the quick brown fox jumps over the lazy dog"
    normal_ratio = detect_repeated_words(normal_text)
    assert normal_ratio == 0.0


def test_features_to_dict():
    text = "Urgent: wire transfer $500 to IRS now."
    feat = extract_fraud_features(text)
    d = features_to_dict(feat)
    assert isinstance(d, dict)
    assert "contains_financial_terms" in d
    assert "contains_urgency_terms" in d
    assert "word_count" in d


def test_preprocess_for_model_truncation():
    long_text = "word " * 300
    truncated = preprocess_for_model(long_text, max_length=50)
    assert len(truncated.split()) <= 50
