"""Probability calibration: metrics, a small calibrator, and a model wrapper."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.metrics import MetricsError, cross_val_probabilities

CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")
_EPS = 1e-6


class CalibrationError(Exception):
    """Raised when calibration inputs are invalid or a calibrated model cannot be fitted."""


def _check(y_true: Any, proba: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    if y.ndim != 1 or p.ndim != 1:
        raise CalibrationError("y_true and proba must be 1-dimensional")
    if len(y) == 0:
        raise CalibrationError("Empty input")
    if len(y) != len(p):
        raise CalibrationError(f"Length mismatch: {len(y)} labels vs {len(p)} probabilities")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise CalibrationError("y_true must contain only the values 0 and 1")
    if np.isnan(p).any():
        raise CalibrationError("Probabilities contain NaN")
    if (p < 0).any() or (p > 1).any():
        raise CalibrationError("Probabilities must be between 0 and 1")
    return y.astype(int), p


def brier_score(y_true: Any, proba: Any) -> float:
    """Mean squared error between predicted probability and the 0/1 outcome (lower is better)."""
    y, p = _check(y_true, proba)
    return float(np.mean((p - y) ** 2))


def calibration_table(y_true: Any, proba: Any, n_bins: int = 10) -> pd.DataFrame:
    """Per equal-width probability bin: customers, mean predicted probability, observed churn rate.

    Empty bins have count 0 and NaN rates.

    Raises:
        CalibrationError: for invalid inputs or `n_bins` below 2.
    """
    y, p = _check(y_true, proba)
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise CalibrationError(f"n_bins must be an integer >= 2, got {n_bins!r}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.digitize(p, edges[1:-1])  # p == 1.0 lands in the last bin

    rows = []
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "count": count,
                "mean_predicted": float(p[mask].mean()) if count else np.nan,
                "observed_rate": float(y[mask].mean()) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: Any, proba: Any, n_bins: int = 10) -> float:
    """Average gap between predicted probability and observed churn rate, weighted by bin size."""
    table = calibration_table(y_true, proba, n_bins)
    filled = table[table["count"] > 0]
    weights = filled["count"] / filled["count"].sum()
    return float((weights * (filled["observed_rate"] - filled["mean_predicted"]).abs()).sum())


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped))


class Calibrator:
    """Maps raw model probabilities to calibrated ones ("sigmoid" = Platt scaling, or "isotonic")."""

    def __init__(self, method: str) -> None:
        if method not in ("sigmoid", "isotonic"):
            raise CalibrationError(f"Unknown calibration method '{method}'; choose 'sigmoid' or 'isotonic'")
        self.method = method
        self._model: Any = None

    def fit(self, proba: Any, y_true: Any) -> "Calibrator":
        """Learn the mapping from probabilities and the true 0/1 outcomes."""
        y, p = _check(y_true, proba)
        if set(np.unique(y).tolist()) != {0, 1}:
            raise CalibrationError("Calibration needs both classes in y_true")
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        else:
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, proba: Any) -> np.ndarray:
        """Return calibrated probabilities (always within 0 and 1)."""
        if self._model is None:
            raise CalibrationError("This calibrator is not fitted yet")
        p = np.asarray(proba, dtype=float)
        if p.ndim != 1 or np.isnan(p).any() or (p < 0).any() or (p > 1).any():
            raise CalibrationError("Probabilities must be a 1-dimensional array of numbers between 0 and 1")
        if self.method == "sigmoid":
            out = self._model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        else:
            out = self._model.predict(p)
        return np.clip(out, 0.0, 1.0)


class CalibratedModel:
    """A model pipeline plus an optional calibrator.

    The calibrator is learned from out-of-fold predictions on the training data (every row is
    predicted by a model that never saw it), then the pipeline is refitted on all of it.
    """

    classes_ = np.array([0, 1])

    def __init__(self, base: Any, method: str = "none", n_splits: int = 5, seed: int = 42) -> None:
        if method not in CALIBRATION_METHODS:
            raise CalibrationError(f"Unknown calibration method '{method}'; choose one of {CALIBRATION_METHODS}")
        self.base = base
        self.method = method
        self.n_splits = n_splits
        self.seed = seed
        self.calibrator_: Calibrator | None = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CalibratedModel":
        """Fit the calibrator (if any) from out-of-fold predictions, then fit the base pipeline."""
        try:
            if self.method != "none":
                oof, _ = cross_val_probabilities(self.base, X, y, self.n_splits, self.seed)
                self.calibrator_ = Calibrator(self.method).fit(oof, y)
            self.base.fit(X, y)
        except (ValueError, MetricsError) as exc:
            raise CalibrationError(f"Could not fit the calibrated model: {exc}") from exc
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return an (n, 2) array of [P(stay), P(churn)]."""
        if not self.is_fitted_:
            raise CalibrationError("This model is not fitted yet")
        p = self.base.predict_proba(X)[:, 1]
        if self.calibrator_ is not None:
            p = self.calibrator_.predict(p)
        return np.column_stack([1 - p, p])


def plot_reliability(tables: dict[str, pd.DataFrame], path: str | Path, title: str) -> None:
    """Save a reliability diagram: predicted probability (x) vs observed churn rate (y)."""
    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for name, table in tables.items():
        filled = table[table["count"] > 0]
        ax.plot(filled["mean_predicted"], filled["observed_rate"], marker="o", label=name)
    ax.set_xlabel("Mean predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title(title)
    ax.legend()
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise CalibrationError(f"Cannot write the calibration plot to {path}: {exc}") from exc