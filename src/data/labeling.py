"""
Label management for the Fraud Call Detection system.

Handles label assignment, validation, and quality tracking.
Never fabricates labels or metadata.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. LabelStatus enum
# ---------------------------------------------------------------------------


class LabelStatus(str, Enum):
    """Confidence / provenance tier for a record's label.

    Attributes
    ----------
    VERIFIED:
        Label is backed by authoritative evidence (e.g., FTC enforcement action).
    WEAK:
        Label is inferred heuristically or from a secondary source.
    UNKNOWN:
        Label provenance is unclear; records should be excluded from
        supervised training.
    """

    VERIFIED = "verified"
    WEAK = "weak"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# 2. Label enum
# ---------------------------------------------------------------------------


class Label(str, Enum):
    """Target class labels for the fraud-detection classifier.

    Attributes
    ----------
    FRAUD:
        The call is a fraudulent / robocall.
    LEGITIMATE:
        The call is a legitimate communication.
    UNCERTAIN:
        The label cannot be reliably assigned; treated like UNKNOWN for
        training purposes.
    """

    FRAUD = "fraud"
    LEGITIMATE = "legitimate"
    UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Helper sets for fast membership tests
# ---------------------------------------------------------------------------

_VALID_LABELS = {member.value for member in Label}
_VALID_LABEL_STATUSES = {member.value for member in LabelStatus}


# ---------------------------------------------------------------------------
# 3. validate_labels
# ---------------------------------------------------------------------------


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``label`` and ``label_status`` columns and flag anomalies.

    A record is flagged when any of the following apply:

    * ``label`` is null or not a recognised :class:`Label` value.
    * ``label_status`` is null or not a recognised :class:`LabelStatus` value.
    * ``label`` is ``Label.UNCERTAIN`` (ambiguous ground-truth).

    Parameters
    ----------
    df:
        Manifest or any DataFrame containing ``label`` and ``label_status``
        columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``validation_flags`` column (a
        semicolon-separated string of flag descriptions, or an empty
        string when the record is clean).
    """
    out = df.copy()
    flags: List[str] = []

    for _, row in out.iterrows():
        row_flags: List[str] = []

        label_val = row.get("label")
        label_status_val = row.get("label_status")

        # --- label checks ---
        if pd.isna(label_val) or str(label_val).strip() == "":
            row_flags.append("MISSING_LABEL")
        elif str(label_val).strip() not in _VALID_LABELS:
            row_flags.append(f"INVALID_LABEL:{label_val}")
        elif str(label_val).strip() == Label.UNCERTAIN.value:
            row_flags.append("UNCERTAIN_LABEL")

        # --- label_status checks ---
        if pd.isna(label_status_val) or str(label_status_val).strip() == "":
            row_flags.append("MISSING_LABEL_STATUS")
        elif str(label_status_val).strip() not in _VALID_LABEL_STATUSES:
            row_flags.append(f"INVALID_LABEL_STATUS:{label_status_val}")

        flags.append("; ".join(row_flags))

    out["validation_flags"] = flags

    flagged_count = sum(1 for f in flags if f)
    logger.info(
        "Label validation: %d / %d records have flags.", flagged_count, len(out)
    )
    return out


# ---------------------------------------------------------------------------
# 4. get_training_eligible
# ---------------------------------------------------------------------------


def get_training_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Return records suitable for supervised classification training.

    Excludes records where:

    * ``label_status`` is ``"unknown"`` (provenance insufficient).
    * ``label`` is ``"uncertain"`` (ground-truth ambiguous).
    * ``label`` is null / unrecognised.

    Parameters
    ----------
    df:
        Full manifest / labelled DataFrame.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame ready for model training.
    """
    original_count = len(df)

    # Normalise strings for comparison
    label_series = df["label"].astype(str).str.strip().str.lower()
    status_series = df["label_status"].astype(str).str.strip().str.lower()

    unknown_status_mask = status_series == LabelStatus.UNKNOWN.value
    uncertain_label_mask = label_series == Label.UNCERTAIN.value
    invalid_label_mask = ~label_series.isin(_VALID_LABELS)
    null_label_mask = df["label"].isna()

    exclude_mask = (
        unknown_status_mask
        | uncertain_label_mask
        | invalid_label_mask
        | null_label_mask
    )

    eligible_df = df[~exclude_mask].copy()

    excluded_count = original_count - len(eligible_df)
    if excluded_count:
        logger.info(
            "get_training_eligible: excluded %d records "
            "(unknown_status=%d, uncertain_label=%d, invalid_label=%d, null_label=%d). "
            "%d records remain.",
            excluded_count,
            unknown_status_mask.sum(),
            uncertain_label_mask.sum(),
            invalid_label_mask.sum(),
            null_label_mask.sum(),
            len(eligible_df),
        )
    else:
        logger.info(
            "get_training_eligible: all %d records are eligible.", len(eligible_df)
        )

    return eligible_df


# ---------------------------------------------------------------------------
# 5. compute_class_weights
# ---------------------------------------------------------------------------


def compute_class_weights(labels: List[str]) -> Dict[int, float]:
    """Compute balanced per-class weights using scikit-learn.

    Uses ``sklearn.utils.class_weight.compute_class_weight`` with
    ``class_weight="balanced"`` so that minority classes receive higher
    weight during training.

    Parameters
    ----------
    labels:
        List of string label values (e.g., ``["fraud", "fraud",
        "legitimate", …]``).

    Returns
    -------
    Dict[int, float]
        Mapping of class index (0-based, sorted alphabetically) to its
        weight.  The ordering follows the sorted unique values of *labels*.

    Raises
    ------
    ValueError
        If *labels* is empty.
    """
    if not labels:
        raise ValueError("labels list must not be empty.")

    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    unique_classes = sorted(set(labels))
    weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array(unique_classes),
        y=np.array(labels),
    )
    weights: Dict[int, float] = {
        idx: float(w) for idx, w in enumerate(weights_array)
    }
    logger.info(
        "Class weights computed for classes %s: %s", unique_classes, weights
    )
    return weights
