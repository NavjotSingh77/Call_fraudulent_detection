"""
Unit tests for src.training.evaluate module.
"""

from pathlib import Path
import numpy as np
import pytest

from src.training.evaluate import (
    compute_metrics,
    generate_evaluation_report,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def test_compute_metrics_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.1, 0.2])
    m = compute_metrics(y_true, y_pred, threshold=0.5)

    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["false_positive_rate"] == 0.0
    assert m["false_negative_rate"] == 0.0
    assert m["tp"] == 2
    assert m["tn"] == 2


def test_compute_metrics_tradeoff():
    # True: 1, 1, 0, 0
    # Pred: 0.8, 0.4 (FN if th=0.5), 0.6 (FP if th=0.5), 0.1
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.8, 0.4, 0.6, 0.1])
    m = compute_metrics(y_true, y_pred, threshold=0.5)

    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5


def test_plot_roc_curve(tmp_path):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.7, 0.3, 0.2])
    out_file = tmp_path / "roc.png"
    res = plot_roc_curve(y_true, y_pred, output_path=out_file)
    assert res["auc"] == 1.0
    assert out_file.exists()


def test_plot_precision_recall_curve(tmp_path):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.7, 0.3, 0.2])
    out_file = tmp_path / "pr.png"
    res = plot_precision_recall_curve(y_true, y_pred, output_path=out_file)
    assert res["auc"] > 0
    assert out_file.exists()


def test_generate_evaluation_report(tmp_path):
    metrics = {"accuracy": 0.95, "f1": 0.94, "recall": 0.96}
    out_file = tmp_path / "eval_report.json"
    generate_evaluation_report(metrics, out_file, model_info={"version": "1.0"})
    assert out_file.exists()
    import json
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "disclaimer" in data
    assert data["metrics"]["f1"] == 0.94
