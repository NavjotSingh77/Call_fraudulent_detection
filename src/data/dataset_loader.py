"""
Dataset loading utilities for the Fraud Call Detection system.

Loads and validates the dataset manifest, applies train/val/test splits,
and provides iterators for training data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that every manifest is expected to expose
_REQUIRED_COLUMNS = {
    "call_id",
    "label",
    "label_status",
    "source",
}

# label_status values accepted for supervised training
_TRAINABLE_LABEL_STATUSES = {"verified", "weak"}


# ---------------------------------------------------------------------------
# 1. load_manifest
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load and lightly validate a manifest CSV file.

    Parameters
    ----------
    manifest_path:
        Absolute or relative path to ``manifest.csv``.

    Returns
    -------
    pd.DataFrame
        The parsed manifest.

    Raises
    ------
    FileNotFoundError
        If *manifest_path* does not exist.
    ValueError
        If any required columns are missing.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    logger.info(
        "Manifest loaded from %s: %d rows, columns=%s",
        manifest_path,
        len(df),
        list(df.columns),
    )
    return df


# ---------------------------------------------------------------------------
# 2. filter_for_training
# ---------------------------------------------------------------------------


def filter_for_training(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Filter manifest records to those eligible for supervised training.

    A record is included when **all** of the following hold:

    * ``label_status`` is ``"verified"`` or ``"weak"`` (not ``"unknown"``).
    * ``label`` is not null / empty.
    * At least one of ``audio_path`` or ``transcript`` is non-null.

    Parameters
    ----------
    manifest_df:
        Full manifest DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame ready for training.
    """
    original_count = len(manifest_df)

    # --- 1. Drop unknown label_status ---
    unknown_mask = ~manifest_df["label_status"].isin(_TRAINABLE_LABEL_STATUSES)
    unknown_count = unknown_mask.sum()
    if unknown_count:
        logger.info(
            "Excluding %d records with label_status='unknown' "
            "(not suitable for supervised training).",
            unknown_count,
        )

    # --- 2. Drop null labels ---
    null_label_mask = manifest_df["label"].isna() | (manifest_df["label"].astype(str).str.strip() == "")

    # --- 3. Require at least audio_path OR transcript ---
    has_audio = (
        manifest_df["audio_path"].notna()
        & (manifest_df["audio_path"].astype(str).str.strip() != "")
        if "audio_path" in manifest_df.columns
        else pd.Series(False, index=manifest_df.index)
    )
    has_transcript = (
        manifest_df["transcript"].notna()
        & (manifest_df["transcript"].astype(str).str.strip() != "")
        if "transcript" in manifest_df.columns
        else pd.Series(False, index=manifest_df.index)
    )
    no_data_mask = ~(has_audio | has_transcript)

    exclude_mask = unknown_mask | null_label_mask | no_data_mask
    filtered_df = manifest_df[~exclude_mask].copy()

    excluded_count = original_count - len(filtered_df)
    logger.info(
        "Training filter: %d → %d records (%d excluded — "
        "unknown_status=%d, null_label=%d, no_data=%d).",
        original_count,
        len(filtered_df),
        excluded_count,
        unknown_mask.sum(),
        null_label_mask.sum(),
        no_data_mask.sum(),
    )
    return filtered_df


# ---------------------------------------------------------------------------
# 3. get_label_distribution
# ---------------------------------------------------------------------------


def get_label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Return a mapping of label → record count.

    Parameters
    ----------
    df:
        Any DataFrame that contains a ``label`` column.

    Returns
    -------
    Dict[str, int]
        E.g. ``{"fraud": 120, "legitimate": 80}``.
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame does not contain a 'label' column.")
    distribution: Dict[str, int] = df["label"].value_counts().to_dict()
    logger.debug("Label distribution: %s", distribution)
    return distribution


# ---------------------------------------------------------------------------
# 4. check_class_balance
# ---------------------------------------------------------------------------


def check_class_balance(df: pd.DataFrame) -> Dict:
    """Analyse class imbalance and return a summary report.

    Parameters
    ----------
    df:
        DataFrame with a ``label`` column (already filtered for training).

    Returns
    -------
    Dict
        Keys:

        * ``class_counts``   – ``Dict[str, int]``
        * ``class_ratios``   – ``Dict[str, float]`` (each class's fraction)
        * ``imbalance_ratio``– ``float`` (max_count / min_count); ``None``
          when only one class is present
        * ``recommendation`` – Human-readable advice string
    """
    distribution = get_label_distribution(df)
    total = sum(distribution.values())

    if total == 0:
        return {
            "class_counts": distribution,
            "class_ratios": {},
            "imbalance_ratio": None,
            "recommendation": "Dataset is empty.",
        }

    class_ratios = {label: count / total for label, count in distribution.items()}

    if len(distribution) < 2:
        return {
            "class_counts": distribution,
            "class_ratios": class_ratios,
            "imbalance_ratio": None,
            "recommendation": (
                "WARNING: Only one class present in the dataset. "
                "Supervised binary classification requires at least two distinct classes. "
                "Add legitimate (non-fraud) examples before training."
            ),
        }

    counts = list(distribution.values())
    imbalance_ratio = max(counts) / min(counts)

    if imbalance_ratio <= 2.0:
        recommendation = (
            "Classes are roughly balanced (ratio ≤ 2:1). "
            "Standard training should work well."
        )
    elif imbalance_ratio <= 10.0:
        recommendation = (
            f"Moderate class imbalance (ratio ≈ {imbalance_ratio:.1f}:1). "
            "Consider using class weights or oversampling the minority class."
        )
    else:
        recommendation = (
            f"Severe class imbalance (ratio ≈ {imbalance_ratio:.1f}:1). "
            "Strongly recommend oversampling (SMOTE/augmentation), "
            "undersampling, or weighted loss functions."
        )

    result = {
        "class_counts": distribution,
        "class_ratios": class_ratios,
        "imbalance_ratio": imbalance_ratio,
        "recommendation": recommendation,
    }
    logger.info("Class balance report: %s", result)
    return result
