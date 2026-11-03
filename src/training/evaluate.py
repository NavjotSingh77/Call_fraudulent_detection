"""
Model evaluation module for the Fraud Call Detection system.

Implements comprehensive evaluation metrics:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- ROC-AUC, PR-AUC
- False positive rate, False negative rate

Clearly distinguishes:
- Internal validation performance
- Held-out test performance
- External evaluation

NEVER reports fabricated metrics. All numbers come from actual model predictions.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred_proba: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for binary classification.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for class 1 (fraud).
        threshold: Decision threshold for converting probabilities to binary classes.
        class_names: Optional names for classes [class_0, class_1].

    Returns:
        Dictionary containing all evaluation metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)
    y_pred = (y_pred_proba >= threshold).astype(int)

    if class_names is None:
        class_names = ["legitimate", "fraud"]

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC and PR-AUC require at least one sample of each class
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
        precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = float(auc(recall_pts, precision_pts))
    else:
        logger.warning("Only one class present in y_true; ROC-AUC and PR-AUC undefined.")
        roc_auc = 0.0
        pr_auc = 0.0

    clf_report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sample_count": int(len(y_true)),
        "class_counts": {
            class_names[0]: int(np.sum(y_true == 0)),
            class_names[1]: int(np.sum(y_true == 1)),
        },
        "classification_report": clf_report,
    }

    return metrics


def evaluate_on_split(
    model: Any,
    texts: List[str],
    labels: List[int],
    split_name: str,
    threshold: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run model predictions on a specific dataset split and calculate metrics.

    Args:
        model: Trained Keras model or model predicting probabilities.
        texts: Input transcript texts.
        labels: Ground truth binary labels.
        split_name: Name of split (e.g. 'val', 'test', 'external').
        threshold: Decision threshold for fraud classification.
        config: System configuration dictionary.

    Returns:
        Metrics dictionary with split_name and evaluation timestamp.
    """
    logger.info("Evaluating on split: %s (%d samples)", split_name, len(texts))
    if len(texts) == 0:
        raise ValueError(f"Split {split_name} contains 0 samples.")

    import tensorflow as tf  # type: ignore

    # Predict probabilities
    preds = model.predict(np.array(texts, dtype=object), batch_size=config.get("model", {}).get("batch_size", 32))
    if preds.ndim > 1 and preds.shape[1] == 1:
        y_pred_proba = preds.flatten()
    elif preds.ndim > 1 and preds.shape[1] > 1:
        y_pred_proba = preds[:, 1]
    else:
        y_pred_proba = preds.flatten()

    metrics = compute_metrics(
        y_true=labels,
        y_pred_proba=y_pred_proba,
        threshold=threshold,
    )
    metrics["split_name"] = split_name
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    return metrics


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Plot and save confusion matrix figure."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib or seaborn not installed; skipping confusion matrix plot.")
        return

    if class_names is None:
        class_names = ["Legitimate", "Fraud"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.info("Saved confusion matrix plot to %s", output_path)
    plt.close()


def plot_roc_curve(
    y_true: Union[np.ndarray, List[int]],
    y_pred_proba: Union[np.ndarray, List[float]],
    output_path: Optional[Path] = None,
    split_name: str = "test",
) -> Dict[str, Any]:
    """Plot and save ROC curve."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    if len(np.unique(y_true)) < 2:
        logger.warning("Cannot plot ROC curve: only one class present.")
        return {"auc": 0.0, "fpr": [], "tpr": []}

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = float(auc(fpr, tpr))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic ({split_name})")
        plt.legend(loc="lower right")
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)
            logger.info("Saved ROC curve to %s", output_path)
        plt.close()
    except ImportError:
        logger.warning("matplotlib not installed; skipping ROC curve plot.")

    return {"auc": roc_auc, "fpr": fpr.tolist(), "tpr": tpr.tolist()}


def plot_precision_recall_curve(
    y_true: Union[np.ndarray, List[int]],
    y_pred_proba: Union[np.ndarray, List[float]],
    output_path: Optional[Path] = None,
    split_name: str = "test",
) -> Dict[str, Any]:
    """Plot and save Precision-Recall curve."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    if len(np.unique(y_true)) < 2:
        logger.warning("Cannot plot PR curve: only one class present.")
        return {"auc": 0.0, "precision": [], "recall": []}

    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = float(auc(recall_pts, precision_pts))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.plot(recall_pts, precision_pts, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f"Precision-Recall Curve ({split_name})")
        plt.legend(loc="lower left")
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)
            logger.info("Saved PR curve to %s", output_path)
        plt.close()
    except ImportError:
        logger.warning("matplotlib not installed; skipping PR curve plot.")

    return {"auc": pr_auc, "precision": precision_pts.tolist(), "recall": recall_pts.tolist()}


def plot_class_distribution(df: pd.DataFrame, output_path: Optional[Path] = None) -> None:
    """Plot and save bar chart of class distribution."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if "label" not in df.columns:
            logger.warning("DataFrame missing 'label' column.")
            return

        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="label", palette="Set2")
        plt.title("Class Distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)
            logger.info("Saved class distribution plot to %s", output_path)
        plt.close()
    except ImportError:
        logger.warning("matplotlib or seaborn not installed; skipping class distribution plot.")


def generate_evaluation_report(
    metrics_dict: Dict[str, Any],
    output_path: Path,
    model_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Save comprehensive JSON evaluation report with metadata and disclaimer."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_info": model_info or {},
        "metrics": metrics_dict,
        "disclaimer": (
            "These metrics represent internal statistical performance on the evaluated dataset split. "
            "Model predictions are probabilistic estimates and should not be used as standalone legal "
            "or factual proof of fraud."
        ),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to %s", output_path)


def run_evaluation(
    config: Dict[str, Any],
    models_dir: Optional[Path] = None,
    split: str = "test",
) -> Dict[str, Any]:
    """Execute evaluation on a saved model and data split.

    Args:
        config: Loaded config dictionary.
        models_dir: Path to models directory.
        split: Split name ('test' or 'val').

    Returns:
        Evaluation metrics dictionary.
    """
    from src.models.bilstm import load_model_artifacts
    from src.utils.config import resolve_path

    if models_dir is None:
        models_dir = resolve_path(config, "paths.models")
    else:
        models_dir = Path(models_dir)

    model, vectorizer_config, label_encoder, model_config = load_model_artifacts(models_dir)

    splits_dir = resolve_path(config, "paths.data_splits")
    split_file = splits_dir / f"{split}.csv"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    df = pd.read_csv(split_file)
    valid_df = df[df["transcript"].notna() & df["label"].notna()]

    if len(valid_df) == 0:
        raise ValueError(f"No valid transcripts found in {split_file}")

    texts = valid_df["transcript"].astype(str).tolist()
    labels = valid_df["label"].map(lambda x: 1 if str(x).lower() == "fraud" else 0).tolist()

    # Load threshold if saved
    threshold_file = models_dir / "threshold.json"
    threshold = 0.5
    if threshold_file.exists():
        try:
            with open(threshold_file, "r", encoding="utf-8") as f:
                th_data = json.load(f)
                threshold = th_data.get("optimal_threshold", 0.5)
        except Exception:
            threshold = 0.5

    metrics = evaluate_on_split(
        model=model,
        texts=texts,
        labels=labels,
        split_name=split,
        threshold=threshold,
        config=config,
    )

    # Figures directory
    figures_dir = resolve_path(config, "paths.figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    y_pred_proba = model.predict(np.array(texts, dtype=object)).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)

    plot_confusion_matrix(
        y_true=labels,
        y_pred=y_pred,
        output_path=figures_dir / f"confusion_matrix_{split}.png",
    )
    plot_roc_curve(
        y_true=labels,
        y_pred_proba=y_pred_proba,
        output_path=figures_dir / f"roc_curve_{split}.png",
        split_name=split,
    )
    plot_precision_recall_curve(
        y_true=labels,
        y_pred_proba=y_pred_proba,
        output_path=figures_dir / f"pr_curve_{split}.png",
        split_name=split,
    )

    # Generate evaluation report
    eval_dir = resolve_path(config, "paths.evaluation")
    report_file = eval_dir / f"evaluation_report_{split}.json"
    generate_evaluation_report(
        metrics_dict=metrics,
        output_path=report_file,
        model_info=model_config,
    )

    logger.info("Evaluation complete for split: %s. F1: %.4f, Recall: %.4f, Precision: %.4f",
                split, metrics["f1"], metrics["recall"], metrics["precision"])
    return metrics
