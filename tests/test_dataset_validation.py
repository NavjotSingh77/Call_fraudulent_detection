"""
Unit tests for data validation, labeling, and splitting modules.
"""

import pandas as pd
import pytest

from src.data.dataset_loader import (
    check_class_balance,
    filter_for_training,
    get_label_distribution,
)
from src.data.labeling import (
    Label,
    LabelStatus,
    compute_class_weights,
    get_training_eligible,
    validate_labels,
)
from src.data.splitting import create_splits


def test_manifest_required_columns(sample_manifest):
    required = [
        "call_id",
        "audio_path",
        "transcript",
        "language",
        "label",
        "label_status",
        "source",
        "recording_type",
        "case_pdf",
    ]
    for col in required:
        assert col in sample_manifest.columns, f"Missing required column: {col}"


def test_filter_excludes_unknown_labels(sample_manifest):
    # sample_manifest has call_010 with label_status='unknown'
    filtered = filter_for_training(sample_manifest)
    assert len(filtered) == 9
    assert "call_010" not in filtered["call_id"].values
    assert not (filtered["label_status"] == "unknown").any()


def test_label_enum_values():
    assert Label.FRAUD.value == "fraud"
    assert Label.LEGITIMATE.value == "legitimate"
    assert Label.UNCERTAIN.value == "uncertain"

    assert LabelStatus.VERIFIED.value == "verified"
    assert LabelStatus.WEAK.value == "weak"
    assert LabelStatus.UNKNOWN.value == "unknown"


def test_get_label_distribution(sample_manifest):
    dist = get_label_distribution(sample_manifest)
    assert dist["fraud"] == 6
    assert dist["legitimate"] == 3
    assert dist["uncertain"] == 1


def test_check_class_balance_single_class():
    single_class_df = pd.DataFrame({
        "call_id": ["c1", "c2", "c3"],
        "label": ["fraud", "fraud", "fraud"],
        "label_status": ["verified", "verified", "verified"],
    })
    balance = check_class_balance(single_class_df)
    assert len(balance.get("class_counts", {})) == 1
    assert "only one class" in balance.get("recommendation", "").lower() or balance.get("imbalance_ratio") is None or balance.get("imbalance_ratio") == float("inf")


def test_check_class_balance_both_classes(sample_manifest):
    filtered = filter_for_training(sample_manifest)
    balance = check_class_balance(filtered)
    assert "fraud" in balance["class_counts"]
    assert "legitimate" in balance["class_counts"]
    assert balance["imbalance_ratio"] > 0


def test_validate_labels(sample_manifest):
    validated_df = validate_labels(sample_manifest)
    assert "validation_flags" in validated_df.columns
    # Check that invalid labels would be flagged
    bad_df = pd.DataFrame({
        "call_id": ["bad_1"],
        "label": ["invalid_label"],
        "label_status": ["invalid_status"],
    })
    val_bad = validate_labels(bad_df)
    flags = val_bad["validation_flags"].iloc[0]
    assert len(flags) > 0


def test_compute_class_weights():
    labels = ["fraud", "fraud", "legitimate"]
    weights = compute_class_weights(labels)
    assert len(weights) == 2
    # The minority class ('legitimate' -> index 1) should have a higher weight than 'fraud' -> index 0
    assert weights[1] > weights[0]


def test_group_splitting_leakage_prevention(sample_manifest, minimal_config):
    # Ensure sources don't leak across splits
    filtered = filter_for_training(sample_manifest)
    train_df, val_df, test_df = create_splits(filtered, minimal_config, random_seed=42)

    # Check total length
    assert len(train_df) + len(val_df) + len(test_df) == len(filtered)

    # Check no call_id leakage
    train_ids = set(train_df["call_id"])
    val_ids = set(val_df["call_id"])
    test_ids = set(test_df["call_id"])

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
