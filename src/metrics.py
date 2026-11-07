"""Evaluation helpers: ranking metrics, threshold metrics and out-of-fold predictions."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold


class MetricsError(Exception):
    """Raised when metric inputs are invalid or cross-validation cannot run."""


def _validate(y_true: Any, proba: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    if y.ndim != 1 or p.ndim != 1:
        raise MetricsError("y_true and proba must be 1-dimensional")
    if len(y) == 0:
        raise MetricsError("Empty input")
    if len(y) != len(p):
        raise MetricsError(f"Length mismatch: {len(y)} labels vs {len(p)} probabilities")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise MetricsError("y_true must contain both classes 0 and 1 (and nothing else)")
    if np.isnan(p).any():
        raise MetricsError("Probabilities contain NaN")
    if (p < 0).any() or (p > 1).any():
        raise MetricsError("Probabilities must be between 0 and 1")
    return y.astype(int), p


def ranking_metrics(y_true: Any, proba: Any) -> dict[str, float]:
    """PR-AUC (average precision) and ROC-AUC. These do not depend on a threshold."""
    y, p = _validate(y_true, proba)
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def threshold_metrics(y_true: Any, proba: Any, threshold: float) -> dict[str, float]:
    """Precision, recall, F1 and confusion counts when predicting churn if proba >= threshold."""
    y, p = _validate(y_true, proba)
    if not 0 <= threshold <= 1:
        raise MetricsError(f"threshold must be between 0 and 1, got {threshold!r}")
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def best_f1_threshold(y_true: Any, proba: Any) -> tuple[float, float]:
    """Return (threshold, f1) for the threshold with the highest F1 on the given predictions."""
    y, p = _validate(y_true, proba)
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def cross_val_probabilities(
    estimator: Any, X: pd.DataFrame, y: pd.Series, n_splits: int, seed: int
) -> tuple[np.ndarray, list[float]]:
    """Stratified K-fold out-of-fold churn probabilities.

    Every row is predicted by a model that never saw it. A fresh clone of `estimator` is
    fitted on each fold's training part only (so any SMOTE inside it never touches the
    held-out part).

    Returns:
        (out_of_fold_probabilities, per_fold_pr_auc_scores)

    Raises:
        MetricsError: for invalid inputs, a too-rare class, or a fit failure on a fold.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise MetricsError("X must be a DataFrame and y a Series")
    if len(X) != len(y):
        raise MetricsError(f"X has {len(X)} rows but y has {len(y)}")
    if n_splits < 2:
        raise MetricsError(f"n_splits must be at least 2, got {n_splits}")

    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    counts = y.value_counts()
    if set(counts.index) != {0, 1}:
        raise MetricsError("y must contain both classes 0 and 1")
    if counts.min() < n_splits:
        raise MetricsError(f"The rarer class has {counts.min()} rows, fewer than n_splits={n_splits}")

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    fold_scores: list[float] = []
    for number, (train_idx, test_idx) in enumerate(folds.split(X, y), start=1):
        model = clone(estimator)
        try:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
        except ValueError as exc:
            raise MetricsError(f"Fitting failed on fold {number}: {exc}") from exc
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = proba
        fold_scores.append(float(average_precision_score(y.iloc[test_idx], proba)))

    if np.isnan(oof).any():
        raise MetricsError("Internal error: some rows received no out-of-fold prediction")
    return oof, fold_scores