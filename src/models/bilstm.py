"""
BiLSTM model architecture for fraud call transcript classification.

Architecture:
    Input Text
    → TextVectorization
    → Embedding Layer
    → Bidirectional LSTM
    → Dropout
    → Dense (ReLU)
    → Dropout
    → Output Layer (sigmoid for binary, softmax for 3-class)

All hyperparameters configurable via config.yaml.
No hardcoded hyperparameters.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ModelNotTrainedError(Exception):
    """Raised when attempting to use a model that has not been trained yet."""


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_bilstm_model(config: Dict, num_classes: int = 2):
    """Build and compile the BiLSTM model for fraud transcript classification.

    All hyperparameters are sourced from *config* — nothing is hardcoded.

    Args:
        config: Loaded configuration dictionary (e.g. from config.yaml).
        num_classes: 2 for binary (fraud / legitimate), 3 for ternary
                     (fraud / legitimate / uncertain).

    Returns:
        Compiled ``tf.keras.Model``.

    Raises:
        ImportError: If TensorFlow is not installed.
        ValueError: If num_classes is not 2 or 3.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build the BiLSTM model. "
            "Install it with: pip install tensorflow"
        ) from exc

    if num_classes not in (2, 3):
        raise ValueError(f"num_classes must be 2 or 3, got {num_classes}")

    # ------------------------------------------------------------------
    # Pull hyperparameters from config (fail-fast if keys are missing)
    # ------------------------------------------------------------------
    text_cfg = config["text_preprocessing"]
    model_cfg = config["model"]

    vocabulary_size: int = text_cfg["vocabulary_size"]
    max_sequence_length: int = text_cfg["max_sequence_length"]
    embedding_dim: int = model_cfg["embedding_dim"]
    lstm_units: int = model_cfg["lstm_units"]
    dropout_rate: float = model_cfg["dropout_rate"]
    dense_units: int = model_cfg["dense_units"]
    learning_rate: float = model_cfg["learning_rate"]

    # ------------------------------------------------------------------
    # Output layer configuration
    # ------------------------------------------------------------------
    if num_classes == 2:
        output_units = 1
        output_activation = "sigmoid"
        loss = "binary_crossentropy"
    else:  # 3 classes
        output_units = 3
        output_activation = "softmax"
        loss = "categorical_crossentropy"

    # ------------------------------------------------------------------
    # Build functional model
    # ------------------------------------------------------------------
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")

    # TextVectorization is added as a separate step; here we use an
    # IntegerLookup-compatible shape after vectorization.
    # We build with a Lambda so the vectorizer can be injected at training time,
    # OR accept integer sequences directly when the vectorizer is applied upstream.
    # For end-to-end serving, callers should embed the vectorizer via
    # create_text_vectorizer and wrap this model.

    x = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        output_mode="int",
        output_sequence_length=max_sequence_length,
        name="text_vectorization",
    )(inputs)

    x = tf.keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding",
    )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=False),
        name="bilstm",
    )(x)

    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_1")(x)

    x = tf.keras.layers.Dropout(dropout_rate / 2, name="dropout_2")(x)

    outputs = tf.keras.layers.Dense(
        output_units,
        activation=output_activation,
        name="output",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bilstm_fraud_detector")

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    logger.info(
        "BiLSTM model built: vocab=%d, seq_len=%d, emb=%d, lstm=%d, "
        "dense=%d, dropout=%.2f, lr=%.5f, classes=%d",
        vocabulary_size,
        max_sequence_length,
        embedding_dim,
        lstm_units,
        dense_units,
        dropout_rate,
        learning_rate,
        num_classes,
    )
    return model


# ---------------------------------------------------------------------------
# TextVectorization helper
# ---------------------------------------------------------------------------

def create_text_vectorizer(
    texts: List[str],
    config: Dict,
):
    """Create and adapt a TextVectorization layer from *texts*.

    Args:
        texts: Corpus of strings used to build the vocabulary.
        config: Configuration dictionary.

    Returns:
        Adapted ``tf.keras.layers.TextVectorization`` layer.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install with: pip install tensorflow"
        ) from exc

    vocabulary_size: int = config["text_preprocessing"]["vocabulary_size"]
    max_sequence_length: int = config["text_preprocessing"]["max_sequence_length"]

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        output_mode="int",
        output_sequence_length=max_sequence_length,
        name="text_vectorization",
    )
    vectorizer.adapt(texts)
    logger.info(
        "TextVectorization adapted on %d texts; vocab size=%d",
        len(texts),
        vocabulary_size,
    )
    return vectorizer


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(config: Dict, checkpoint_path: Path) -> List:
    """Build Keras training callbacks from *config*.

    Args:
        config: Configuration dictionary.  Expected keys under
                ``config["training"]``:
                - ``early_stopping_patience``
                - ``reduce_lr_patience``
                - ``reduce_lr_factor``
        checkpoint_path: File path for ModelCheckpoint.

    Returns:
        List of ``tf.keras.callbacks.Callback`` instances.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install with: pip install tensorflow"
        ) from exc

    training_cfg = config.get("training", {})
    early_stopping_patience: int = training_cfg.get("early_stopping_patience", 5)
    reduce_lr_patience: int = training_cfg.get("reduce_lr_patience", 3)
    reduce_lr_factor: float = training_cfg.get("reduce_lr_factor", 0.5)

    # CSV log sits next to the checkpoint
    csv_log_path = Path(checkpoint_path).parent / "training_history.csv"

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

    logger.info(
        "Callbacks configured: EarlyStopping(patience=%d), "
        "ReduceLROnPlateau(patience=%d, factor=%.2f)",
        early_stopping_patience,
        reduce_lr_patience,
        reduce_lr_factor,
    )
    return callbacks


# ---------------------------------------------------------------------------
# Save / load artifacts
# ---------------------------------------------------------------------------

def save_model_artifacts(
    model,
    vectorizer,
    label_encoder,
    config: Dict,
    models_dir: Path,
) -> Dict:
    """Persist the trained model and associated artefacts to *models_dir*.

    Saved files:
    - ``model/``            – SavedModel (or ``model.keras`` for Keras 3)
    - ``vectorizer_config.json`` – TextVectorization vocabulary + config
    - ``label_encoder.pkl`` – sklearn LabelEncoder (or similar)
    - ``model_config.json`` – hyperparameter config snapshot

    Args:
        model: Trained ``tf.keras.Model``.
        vectorizer: Adapted ``tf.keras.layers.TextVectorization``.
        label_encoder: Fitted label encoder.
        config: Configuration dictionary.
        models_dir: Directory to write artefacts into.

    Returns:
        Dictionary mapping artefact names to their saved ``Path``s.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}

    # 1. Model  ----------------------------------------------------------------
    try:
        model_path = models_dir / "model.keras"
        model.save(str(model_path))
        saved_paths["model"] = model_path
        logger.info("Model saved to %s", model_path)
    except Exception:
        # Fallback to SavedModel format
        model_path = models_dir / "model"
        model.save(str(model_path))
        saved_paths["model"] = model_path
        logger.info("Model saved (SavedModel format) to %s", model_path)

    # 2. TextVectorization config  ---------------------------------------------
    try:
        vocab = vectorizer.get_vocabulary()
        vec_config = {
            "vocabulary": vocab,
            "max_tokens": int(vectorizer.vocabulary_size()),
            "output_sequence_length": vectorizer._sequence_length,  # type: ignore[attr-defined]
        }
    except Exception:
        vec_config = {}
        logger.warning("Could not fully serialise vectorizer config.")

    vec_config_path = models_dir / "vectorizer_config.json"
    with open(vec_config_path, "w", encoding="utf-8") as fh:
        json.dump(vec_config, fh, ensure_ascii=False, indent=2)
    saved_paths["vectorizer_config"] = vec_config_path
    logger.info("Vectorizer config saved to %s", vec_config_path)

    # 3. Label encoder  --------------------------------------------------------
    le_path = models_dir / "label_encoder.pkl"
    with open(le_path, "wb") as fh:
        pickle.dump(label_encoder, fh)
    saved_paths["label_encoder"] = le_path
    logger.info("Label encoder saved to %s", le_path)

    # 4. Model config snapshot  ------------------------------------------------
    config_path = models_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    saved_paths["model_config"] = config_path
    logger.info("Model config snapshot saved to %s", config_path)

    # 5. Production Model Metadata  --------------------------------------------
    from datetime import datetime, timezone
    try:
        import tensorflow as tf  # type: ignore
        tf_version = tf.__version__
    except Exception:
        tf_version = "unknown"

    metadata = {
        "version": config.get("project", {}).get("version", "1.0.0"),
        "model_type": "BiLSTM",
        "framework": f"TensorFlow/{tf_version}",
        "serialization_format": ".keras" if (models_dir / "model.keras").exists() else "SavedModel",
        "trained_timestamp": datetime.now(timezone.utc).isoformat(),
        "classes": ["legitimate", "fraud"],
        "target_metric": config.get("training", {}).get("checkpoint_metric", "val_recall"),
        "vocabulary_size": config.get("text_preprocessing", {}).get("vocabulary_size", 20000),
        "embedding_dim": config.get("model", {}).get("embedding_dim", 128),
        "lstm_units": config.get("model", {}).get("lstm_units", 128),
        "disclaimer": (
            "Model predictions are probabilistic statistical estimates and do NOT "
            "constitute legal proof of fraud."
        ),
    }
    meta_path = models_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    saved_paths["model_metadata"] = meta_path
    logger.info("Model metadata saved to %s", meta_path)

    return saved_paths


def load_model_artifacts(models_dir: Path) -> Tuple:
    """Load all saved artefacts from *models_dir*.

    Args:
        models_dir: Directory containing artefacts written by
                    :func:`save_model_artifacts`.

    Returns:
        Tuple of ``(model, vectorizer_config, label_encoder, model_config)``.

    Raises:
        ModelNotTrainedError: If required artefacts are missing.
        ImportError: If TensorFlow is not installed.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install with: pip install tensorflow"
        ) from exc

    models_dir = Path(models_dir)

    # 1. Model
    model_keras = models_dir / "model.keras"
    model_dir = models_dir / "model"
    if model_keras.exists():
        model = tf.keras.models.load_model(str(model_keras))
    elif model_dir.exists():
        model = tf.keras.models.load_model(str(model_dir))
    else:
        raise ModelNotTrainedError(
            f"No model artefact found in {models_dir}. "
            "Train the model first with `python src/training/train.py`."
        )

    # 2. Vectorizer config
    vec_config_path = models_dir / "vectorizer_config.json"
    if not vec_config_path.exists():
        raise ModelNotTrainedError(f"vectorizer_config.json not found in {models_dir}.")
    with open(vec_config_path, "r", encoding="utf-8") as fh:
        vectorizer_config: Dict = json.load(fh)

    # 3. Label encoder
    le_path = models_dir / "label_encoder.pkl"
    if not le_path.exists():
        raise ModelNotTrainedError(f"label_encoder.pkl not found in {models_dir}.")
    with open(le_path, "rb") as fh:
        label_encoder = pickle.load(fh)

    # 4. Model config
    config_path = models_dir / "model_config.json"
    if not config_path.exists():
        raise ModelNotTrainedError(f"model_config.json not found in {models_dir}.")
    with open(config_path, "r", encoding="utf-8") as fh:
        model_config: Dict = json.load(fh)

    logger.info("All artefacts loaded from %s", models_dir)
    return model, vectorizer_config, label_encoder, model_config
