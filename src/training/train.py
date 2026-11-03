"""
Training pipeline for the BiLSTM Fraud Call Detection model.

Implements reproducible, group-aware training with:
- Configurable hyperparameters (no hardcoding)
- Class imbalance handling
- Training history logging
- Model checkpointing
- Validation-based threshold selection

IMPORTANT: This module trains only when verified or weak labels exist.
It will NOT fabricate training data or report imaginary metrics.
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class InsufficientDataError(Exception):
    """Raised when the dataset does not meet minimum requirements for training.

    Training requires:
    - At least 10 total samples with verified or weak labels.
    - At least 5 samples per class (fraud AND legitimate).

    To fix this, collect more labelled call transcripts and re-run the
    labelling pipeline (src/data/labeling.py) before calling run_training().
    """


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_random_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value for reproducibility.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    logger.debug("Python and NumPy seeds set to %d.", seed)

    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
        logger.debug("TensorFlow seed set to %d.", seed)
    except ImportError:
        logger.warning(
            "TensorFlow not installed; TF seed not set. "
            "Install with: pip install tensorflow"
        )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_training_data(
    df: pd.DataFrame, config: Dict
) -> Tuple[List[str], List[int], List[str]]:
    """Extract and validate texts, integer labels, and group identifiers.

    Args:
        df: DataFrame containing at least 'transcript', 'label', and 'source'
            columns (after training-eligibility filtering).
        config: Configuration dictionary (currently used for future extensions).

    Returns:
        Tuple of (texts, labels, groups) where:
        - texts: List[str] of non-null transcripts.
        - labels: List[int] with fraud→1, legitimate→0.
        - groups: List[str] of source identifiers for group-aware splitting.

    Raises:
        InsufficientDataError: If there are fewer than 10 total samples or
            fewer than 5 samples in any required class.
        ValueError: If an unexpected label value is encountered.
    """
    required_cols = {"transcript", "label", "source"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns for training: {sorted(missing)}"
        )

    # Keep only rows with non-null transcripts
    mask = df["transcript"].notna() & (df["transcript"].astype(str).str.strip() != "")
    df_clean = df[mask].copy()

    if len(df_clean) < 10:
        raise InsufficientDataError(
            f"Only {len(df_clean)} sample(s) have non-null transcripts. "
            "Training requires >= 10 total samples with verified or weak labels. "
            "Collect more labelled call recordings and run the transcription + "
            "labelling pipeline before re-attempting training."
        )

    # Encode labels: fraud → 1, legitimate → 0
    label_map = {"fraud": 1, "legitimate": 0}
    unknown_labels = set(df_clean["label"].unique()) - set(label_map.keys())
    if unknown_labels:
        raise ValueError(
            f"Encountered unexpected label value(s): {unknown_labels}. "
            "Only 'fraud' and 'legitimate' are supported."
        )

    texts: List[str] = df_clean["transcript"].astype(str).tolist()
    labels: List[int] = df_clean["label"].map(label_map).tolist()
    groups: List[str] = df_clean["source"].astype(str).tolist()

    # Class distribution check
    fraud_count = labels.count(1)
    legit_count = labels.count(0)

    logger.info(
        "Class distribution — fraud: %d, legitimate: %d (total: %d)",
        fraud_count,
        legit_count,
        len(labels),
    )

    if fraud_count < 5 or legit_count < 5:
        raise InsufficientDataError(
            f"Insufficient samples per class: fraud={fraud_count}, "
            f"legitimate={legit_count}. "
            "Each class needs at least 5 samples for training. "
            "Add more labelled examples for the minority class."
        )

    return texts, labels, groups


# ---------------------------------------------------------------------------
# Label encoder
# ---------------------------------------------------------------------------


def create_label_encoder(labels: List[str]) -> Dict:
    """Create a bidirectional label-to-index mapping from a list of string labels.

    Args:
        labels: List of string label values (may contain duplicates).

    Returns:
        Dict with keys:
        - 'label_to_index': mapping from label string → int index.
        - 'index_to_label': mapping from int index → label string.

    Example:
        >>> enc = create_label_encoder(['fraud', 'legitimate', 'fraud'])
        >>> enc['label_to_index']
        {'fraud': 0, 'legitimate': 1}
    """
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    logger.debug("Label encoder created: %s", label_to_index)
    return {
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
    }


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def compute_class_weights(labels: List[int]) -> Dict[int, float]:
    """Compute balanced class weights to mitigate class imbalance.

    Args:
        labels: List of integer class indices (e.g. [0, 1, 1, 0, ...]).

    Returns:
        Dict mapping class index → float weight (sklearn 'balanced' strategy).
    """
    from sklearn.utils.class_weight import compute_class_weight  # type: ignore

    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(labels),
    )
    weight_dict: Dict[int, float] = {
        int(cls): float(w) for cls, w in zip(classes, weights)
    }
    logger.info("Computed class weights: %s", weight_dict)
    return weight_dict


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def select_optimal_threshold(
    model,
    val_texts: List[str],
    val_labels: List[int],
    config: Dict,
) -> float:
    """Select the prediction threshold that maximises recall while keeping
    precision above 0.5 on the validation set.

    Thresholds evaluated: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8].

    Args:
        model: Trained tf.keras.Model (end-to-end, accepts raw strings).
        val_texts: Validation text inputs.
        val_labels: Validation integer labels.
        config: Configuration dictionary (used for models_dir path).

    Returns:
        Optimal threshold as a float.  Falls back to 0.5 if no threshold
        meets the precision constraint.
    """
    import tensorflow as tf  # type: ignore
    from sklearn.metrics import precision_score, recall_score  # type: ignore

    val_ds = tf.data.Dataset.from_tensor_slices(val_texts).batch(
        config.get("training", {}).get("batch_size", 32)
    )
    y_pred_proba = model.predict(val_ds, verbose=0).flatten()
    y_true = np.array(val_labels)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    best_threshold = 0.5
    best_recall = -1.0

    logger.info("Threshold selection on validation set (%d samples):", len(y_true))
    for thr in thresholds:
        y_pred = (y_pred_proba >= thr).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        logger.info(
            "  threshold=%.1f  precision=%.4f  recall=%.4f", thr, prec, rec
        )
        if prec > 0.5 and rec > best_recall:
            best_recall = rec
            best_threshold = thr

    logger.info(
        "Selected threshold=%.1f (recall=%.4f)", best_threshold, best_recall
    )

    # Persist threshold
    models_dir = Path(
        config.get("paths", {}).get("models", "models")
    )
    if not models_dir.is_absolute():
        from src.utils.config import get_project_root  # type: ignore

        models_dir = get_project_root() / models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = models_dir / "threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "threshold": best_threshold,
                "val_recall_at_threshold": round(best_recall, 6),
                "selection_criterion": "max recall with precision > 0.5",
            },
            fh,
            indent=2,
        )
    logger.info("Threshold saved to %s", threshold_path)

    return best_threshold


# ---------------------------------------------------------------------------
# Save training config
# ---------------------------------------------------------------------------


def save_training_config(
    config: Dict, training_stats: Dict, models_dir: Path
) -> None:
    """Persist training configuration and dataset statistics as a JSON file.

    Args:
        config: Full configuration dictionary.
        training_stats: Dictionary of dataset / training statistics.
        models_dir: Directory in which to write 'training_config.json'.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "random_seed": config.get("training", {}).get("random_seed", 42),
        "model_config": config.get("model", {}),
        "text_preprocessing_config": config.get("text_preprocessing", {}),
        "training_config": config.get("training", {}),
        "dataset_stats": training_stats.get("dataset_stats", {}),
        "training_stats": {
            k: v
            for k, v in training_stats.items()
            if k != "dataset_stats"
        },
    }

    out_path = models_dir / "training_config.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Training config saved to %s", out_path)


# ---------------------------------------------------------------------------
# Core training orchestration
# ---------------------------------------------------------------------------


def build_and_train_model(
    texts: List[str],
    labels: List[int],
    groups: List[str],
    config: Dict,
    models_dir: Path,
) -> Dict:
    """Full end-to-end training orchestration.

    Steps:
    1. Set random seeds.
    2. Import TensorFlow (raises clear error if missing).
    3. Build the BiLSTM model (which includes its own TextVectorization layer).
    4. Compute class weights.
    5. Group-aware train/val split via GroupShuffleSplit.
    6. Build tf.data.Dataset pipelines.
    7. Train with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger.
    8. Run threshold selection on validation set.
    9. Save model artefacts.
    10. Return training history dict.

    Args:
        texts: List of transcript strings.
        labels: List of integer labels (fraud=1, legitimate=0).
        groups: List of source identifiers for group-aware splitting.
        config: Full configuration dictionary.
        models_dir: Directory to save model artefacts.

    Returns:
        Dict with keys:
        - 'history': epoch-by-epoch metric values.
        - 'optimal_threshold': float selected on validation data.
        - 'val_size': number of validation samples.
        - 'train_size': number of training samples.
        - 'epochs_trained': number of epochs actually completed.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for training. "
            "Install it with: pip install tensorflow"
        ) from exc

    from sklearn.model_selection import GroupShuffleSplit  # type: ignore

    from src.models.bilstm import build_bilstm_model, save_model_artifacts  # type: ignore

    training_cfg = config.get("training", {})
    seed: int = training_cfg.get("random_seed", 42)
    batch_size: int = training_cfg.get("batch_size", 32)
    epochs: int = training_cfg.get("epochs", 50)
    val_size: float = training_cfg.get("val_split", 0.2)
    early_stopping_patience: int = training_cfg.get("early_stopping_patience", 5)
    reduce_lr_patience: int = training_cfg.get("reduce_lr_patience", 3)
    reduce_lr_factor: float = training_cfg.get("reduce_lr_factor", 0.5)

    # 1. Seeds
    set_random_seeds(seed)

    # 2. Group-aware train/val split
    texts_arr = np.array(texts)
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(
        splitter.split(texts_arr, labels_arr, groups=groups_arr)
    )

    train_texts = texts_arr[train_idx].tolist()
    train_labels = labels_arr[train_idx].tolist()
    val_texts = texts_arr[val_idx].tolist()
    val_labels = labels_arr[val_idx].tolist()

    logger.info(
        "Train/val split: %d training, %d validation samples.",
        len(train_texts),
        len(val_texts),
    )

    # 3. Build model (includes internal TextVectorization layer)
    model = build_bilstm_model(config, num_classes=2)

    # Adapt the internal TextVectorization layer on training texts
    # Access the layer by name and adapt it
    try:
        vec_layer = model.get_layer("text_vectorization")
        vec_layer.adapt(train_texts)
        logger.info("Internal TextVectorization layer adapted on %d training texts.", len(train_texts))
    except ValueError:
        logger.warning("Could not find 'text_vectorization' layer; skipping adapt().")

    # 4. Class weights
    class_weights = compute_class_weights(train_labels)

    # 5. Build tf.data datasets
    def make_dataset(t_list: List[str], l_list: List[int], shuffle: bool) -> "tf.data.Dataset":
        ds = tf.data.Dataset.from_tensor_slices(
            (tf.constant(t_list, dtype=tf.string), tf.constant(l_list, dtype=tf.int32))
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(t_list), seed=seed)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_texts, train_labels, shuffle=True)
    val_ds = make_dataset(val_texts, val_labels, shuffle=False)

    # 6. Callbacks
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / "best_model.keras"
    csv_log_path = models_dir / "training_history.csv"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_recall",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_recall",
            mode="max",
            patience=reduce_lr_patience,
            factor=reduce_lr_factor,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(csv_log_path),
            append=False,
        ),
    ]

    # 7. Train
    logger.info("Starting training for up to %d epochs...", epochs)
    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    history: Dict = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}
    epochs_trained = len(history.get("loss", []))
    logger.info("Training complete: %d epochs.", epochs_trained)

    # 8. Threshold selection
    # Build a text-only val dataset for threshold selection
    val_text_ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(val_texts, dtype=tf.string)
    ).batch(batch_size)
    optimal_threshold = select_optimal_threshold(model, val_texts, val_labels, config)

    # 9. Save artefacts
    # Build a standalone vectorizer for artefact saving (vocabulary from adapted layer)
    try:
        vec_layer_for_save = model.get_layer("text_vectorization")
    except ValueError:
        vec_layer_for_save = None

    label_encoder = create_label_encoder(["fraud", "legitimate"])
    saved_paths = save_model_artifacts(
        model=model,
        vectorizer=vec_layer_for_save,
        label_encoder=label_encoder,
        config=config,
        models_dir=models_dir,
    )
    logger.info("Artefacts saved: %s", {k: str(v) for k, v in saved_paths.items()})

    return {
        "history": history,
        "optimal_threshold": optimal_threshold,
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "epochs_trained": epochs_trained,
        "class_weights": class_weights,
        "checkpoint_path": str(checkpoint_path),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_training(
    config: Dict,
    manifest_path: Optional[Path] = None,
) -> Dict:
    """Main training entry point.

    Args:
        config: Full configuration dictionary (loaded from config.yaml).
        manifest_path: Optional explicit path to manifest.csv. If not provided,
            the path is read from ``config['paths']['manifest']``.

    Returns:
        Dict with keys:
        - 'status': 'success' or 'skipped'.
        - 'reason': Human-readable message (present when status='skipped').
        - 'training_result': Training output dict (present when status='success').
        - 'dataset_stats': Dataset statistics dict.
    """
    from src.data.dataset_loader import filter_for_training, load_manifest  # type: ignore
    from src.utils.config import get_project_root  # type: ignore

    # Resolve manifest path
    if manifest_path is None:
        raw_manifest_path = config.get("paths", {}).get("manifest", "data/manifest.csv")
        manifest_path = Path(raw_manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = get_project_root() / manifest_path

    logger.info("Loading manifest from %s", manifest_path)
    df = load_manifest(manifest_path)

    # Filter for training-eligible records
    df_train = filter_for_training(df)

    if df_train.empty:
        msg = (
            "No training-eligible records found in the manifest. "
            "Ensure at least some records have label_status='verified' or 'weak' "
            "and a non-null transcript."
        )
        logger.warning(msg)
        return {"status": "skipped", "reason": msg, "dataset_stats": {}}

    # Check that both classes are present
    present_labels = set(df_train["label"].dropna().unique())
    if "fraud" not in present_labels:
        msg = (
            "Cannot train: no fraud examples found in the training-eligible records. "
            "Add verified or weakly-labelled fraud transcripts before training."
        )
        logger.warning(msg)
        return {"status": "skipped", "reason": msg, "dataset_stats": {}}

    if "legitimate" not in present_labels:
        msg = (
            "Cannot train without legitimate call examples. "
            "The dataset contains only fraud-labelled records. "
            "Add verified or weakly-labelled legitimate transcripts before training."
        )
        logger.warning(msg)
        return {"status": "skipped", "reason": msg, "dataset_stats": {}}

    # Prepare data
    texts, labels, groups = prepare_training_data(df_train, config)

    dataset_stats = {
        "total_samples": len(labels),
        "fraud_samples": labels.count(1),
        "legitimate_samples": labels.count(0),
        "unique_sources": len(set(groups)),
    }

    # Resolve models directory
    raw_models_path = config.get("paths", {}).get("models", "models")
    models_dir = Path(raw_models_path)
    if not models_dir.is_absolute():
        models_dir = get_project_root() / models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train
    training_result = build_and_train_model(
        texts=texts,
        labels=labels,
        groups=groups,
        config=config,
        models_dir=models_dir,
    )

    # Persist training config + stats
    save_training_config(
        config=config,
        training_stats={"dataset_stats": dataset_stats, **training_result},
        models_dir=models_dir,
    )

    logger.info(
        "Training pipeline completed. Optimal threshold=%.2f, epochs=%d.",
        training_result["optimal_threshold"],
        training_result["epochs_trained"],
    )

    return {
        "status": "success",
        "training_result": training_result,
        "dataset_stats": dataset_stats,
    }
