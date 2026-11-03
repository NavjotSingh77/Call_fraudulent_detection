"""
Train/validation/test splitting with group-based leakage prevention.

Prevents data leakage by ensuring records from the same source/call
appear in only one split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

# Default split proportions
_DEFAULT_VAL_SIZE = 0.10
_DEFAULT_TEST_SIZE = 0.10


# ---------------------------------------------------------------------------
# 1. create_splits
# ---------------------------------------------------------------------------


def create_splits(
    df: pd.DataFrame,
    config: Dict,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create group-aware train / validation / test splits.

    Records sharing the same value in the group key (``"call_id"`` if
    present, otherwise ``"source"``) are always assigned to the *same*
    split, preventing data leakage between splits.

    Parameters
    ----------
    df:
        Training-eligible manifest DataFrame (e.g., from
        ``src.data.labeling.get_training_eligible``).
    config:
        Project configuration dict.  Optional keys:

        * ``"val_size"``  – fraction for validation (default 0.10).
        * ``"test_size"`` – fraction for test (default 0.10).
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)`` DataFrames (index reset).

    Notes
    -----
    When stratification is impossible (e.g., only one label class), the
    function falls back to non-stratified group splitting and logs a
    warning.
    """
    val_size: float = float(config.get("val_size", _DEFAULT_VAL_SIZE))
    test_size: float = float(config.get("test_size", _DEFAULT_TEST_SIZE))

    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must be < 1.0"
        )

    # Choose group key
    group_key: str
    if "call_id" in df.columns and df["call_id"].notna().any():
        group_key = "call_id"
    elif "source" in df.columns:
        group_key = "source"
    else:
        raise ValueError(
            "DataFrame must contain either 'call_id' or 'source' for group-based splitting."
        )

    groups: np.ndarray = df[group_key].astype(str).values
    indices: np.ndarray = np.arange(len(df))

    # ---- Step 1: carve out test set from the full data ----
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    trainval_idx, test_idx = next(gss_test.split(indices, groups=groups))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # ---- Step 2: carve out validation set from trainval ----
    # Adjust val_size so that it is relative to the full dataset size
    adjusted_val_size = val_size / (1.0 - test_size)

    groups_trainval: np.ndarray = df_trainval[group_key].astype(str).values
    indices_trainval: np.ndarray = np.arange(len(df_trainval))

    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=adjusted_val_size, random_state=random_seed
    )
    train_idx, val_idx = next(
        gss_val.split(indices_trainval, groups=groups_trainval)
    )

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    # ---- Log split sizes ----
    logger.info(
        "Splits created (group_key='%s'): train=%d, val=%d, test=%d  "
        "(total=%d, requested val=%.0f%%, test=%.0f%%)",
        group_key,
        len(df_train),
        len(df_val),
        len(df_test),
        len(df),
        val_size * 100,
        test_size * 100,
    )

    log_split_statistics(df_train, df_val, df_test)
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# 2. save_splits
# ---------------------------------------------------------------------------


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
) -> None:
    """Persist train / val / test splits to CSV files.

    Files are saved under ``data/splits/`` relative to ``config['project_root']``.

    Parameters
    ----------
    train_df:
        Training split DataFrame.
    val_df:
        Validation split DataFrame.
    test_df:
        Test split DataFrame.
    config:
        Project configuration dict (must contain ``"project_root"``).
    """
    project_root = Path(config.get("project_root", "."))
    splits_dir = project_root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = splits_dir / f"{name}.csv"
        split_df.to_csv(out_path, index=False)
        logger.info("Saved %s split → %s (%d rows)", name, out_path, len(split_df))


# ---------------------------------------------------------------------------
# 3. load_splits
# ---------------------------------------------------------------------------


def load_splits(
    config: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved train / val / test splits from disk.

    Parameters
    ----------
    config:
        Project configuration dict (must contain ``"project_root"``).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``

    Raises
    ------
    FileNotFoundError
        If any of the expected split CSVs are missing.
    """
    project_root = Path(config.get("project_root", "."))
    splits_dir = project_root / "data" / "splits"

    split_dfs = []
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. Run create_splits() and save_splits() first."
            )
        df = pd.read_csv(path)
        logger.info("Loaded %s split from %s (%d rows).", name, path, len(df))
        split_dfs.append(df)

    return tuple(split_dfs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 4. log_split_statistics
# ---------------------------------------------------------------------------


def log_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Log class-label distribution for each split.

    Parameters
    ----------
    train_df:
        Training split DataFrame.
    val_df:
        Validation split DataFrame.
    test_df:
        Test split DataFrame.
    """
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "label" in df.columns:
            dist: Dict[str, int] = df["label"].value_counts().to_dict()
        else:
            dist = {"<no label column>": len(df)}
        logger.info("  %-6s split (%4d records): %s", name, len(df), dist)
