"""Feature drift check with the Population Stability Index (PSI).

PSI compares how a feature is distributed in a reference crowd (the training split) and in a
current crowd (new customers). Per bin: (current% - reference%) * ln(current% / reference%),
summed over bins. 0 means identical. Missing values and unseen categories get their own bins,
so a growing share of blanks or brand-new categories counts as drift.

The 0.1 / 0.25 cut-offs are a common rule of thumb, not a law; they live in the config.
"""

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ConfigError, load_config
from src.data import DataValidationError, load_raw
from src.features import RAW_BINARY, RAW_CATEGORICAL, RAW_NUMERIC, FeatureError, add_features
from src.logger import get_logger
from src.predict import PredictionError, prepare_input
from src.split import SplitError, load_split

DRIFT_NUMERIC = [*RAW_NUMERIC, "charges_ratio", "addon_count"]
DRIFT_CATEGORICAL = [*RAW_BINARY, *RAW_CATEGORICAL, "tenure_bucket"]
EPSILON = 1e-4  # floor for an empty bin's share, so the logarithm is always defined


class DriftError(Exception):
    """Raised when drift cannot be computed or the settings are invalid."""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _check_thresholds(moderate: Any, major: Any) -> None:
    for name, value in (("moderate_threshold", moderate), ("major_threshold", major)):
        if not _is_number(value) or value <= 0:
            raise DriftError(f"'drift.{name}' must be a number > 0, got {value!r}")
    if moderate >= major:
        raise DriftError("'drift.moderate_threshold' must be smaller than 'drift.major_threshold'")


def load_drift_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the `drift` config section.

    Raises:
        DriftError: naming the setting that is wrong.
    """
    section = cfg.get("drift") if isinstance(cfg, dict) else None
    if not isinstance(section, dict):
        raise DriftError("Config is missing the 'drift' section")
    n_bins = section.get("n_bins")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise DriftError(f"'drift.n_bins' must be an integer >= 2, got {n_bins!r}")
    min_rows = section.get("min_rows")
    if not isinstance(min_rows, int) or isinstance(min_rows, bool) or min_rows < 1:
        raise DriftError(f"'drift.min_rows' must be an integer >= 1, got {min_rows!r}")
    moderate, major = section.get("moderate_threshold"), section.get("major_threshold")
    _check_thresholds(moderate, major)
    return {"n_bins": n_bins, "min_rows": min_rows, "moderate_threshold": moderate, "major_threshold": major}


def psi_from_counts(expected: Any, actual: Any, eps: float = EPSILON) -> float:
    """PSI between two count vectors (same bins). Empty bins are floored at `eps`.

    Raises:
        DriftError: for vectors of different shapes, fewer than 2 bins, negative or non-finite
            counts, or a distribution with no observations.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.ndim != 1 or a.ndim != 1 or e.shape != a.shape or e.size < 2:
        raise DriftError("The two count vectors must be 1-dimensional, equally long, with at least 2 bins")
    if not (np.isfinite(e).all() and np.isfinite(a).all()) or (e < 0).any() or (a < 0).any():
        raise DriftError("Counts must be finite and not negative")
    if e.sum() <= 0 or a.sum() <= 0:
        raise DriftError("Each distribution needs at least one observation")
    e_share = np.maximum(e / e.sum(), eps)
    a_share = np.maximum(a / a.sum(), eps)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def numeric_bin_edges(reference: pd.Series, n_bins: int) -> np.ndarray:
    """Interior bin edges from the reference's quantiles (duplicates removed).

    Raises:
        DriftError: if the reference has no values or contains infinite values.
    """
    values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise DriftError("The reference column has no values")
    if not np.isfinite(values).all():
        raise DriftError("The reference column has infinite values")
    return np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))


def _numeric_counts(series: pd.Series, edges: np.ndarray) -> np.ndarray:
    numbers = pd.to_numeric(series, errors="coerce")
    missing = int(numbers.isna().sum())
    values = numbers.dropna().to_numpy(dtype=float)
    counts = np.bincount(np.searchsorted(edges, values, side="left"), minlength=len(edges) + 1)
    return np.append(counts, missing)  # the last bin counts missing values


def psi_numeric(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """PSI for a numeric feature, using bins from the reference's quantiles plus a 'missing' bin."""
    edges = numeric_bin_edges(reference, n_bins)
    return psi_from_counts(_numeric_counts(reference, edges), _numeric_counts(current, edges))


def _label(value: Any) -> str:
    """Text label of a category value, so 0, 0.0 and '0' all count as the same category."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _categorical_counts(series: pd.Series, categories: list[str]) -> np.ndarray:
    missing = int(series.isna().sum())
    labels = series.dropna().map(_label)
    counts = [int((labels == category).sum()) for category in categories]
    unseen = len(labels) - sum(counts)
    return np.array([*counts, unseen, missing], dtype=float)  # then: unseen, missing


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature: one bin per reference category, plus 'unseen' and 'missing'."""
    categories = sorted({_label(value) for value in reference.dropna()})
    if not categories:
        raise DriftError("The reference column has no values")
    return psi_from_counts(_categorical_counts(reference, categories), _categorical_counts(current, categories))


def classify(psi_value: float, moderate: float, major: float) -> str:
    """'stable' below `moderate`, 'moderate' below `major`, otherwise 'major'."""
    _check_thresholds(moderate, major)
    if psi_value < moderate:
        return "stable"
    return "moderate" if psi_value < major else "major"


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    n_bins: int,
    moderate: float,
    major: float,
    min_rows: int,
) -> pd.DataFrame:
    """PSI and a status for every feature, largest PSI first.

    Raises:
        DriftError: for an empty reference, too few current rows, missing columns or bad settings.
    """
    if reference.empty:
        raise DriftError("The reference data is empty")
    if len(current) < min_rows:
        raise DriftError(
            f"The current data has {len(current)} rows; at least {min_rows} are needed for a reliable PSI"
        )
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise DriftError(f"n_bins must be an integer >= 2, got {n_bins!r}")
    _check_thresholds(moderate, major)
    absent = [
        col for col in [*numeric_columns, *categorical_columns] if col not in reference.columns or col not in current.columns
    ]
    if absent:
        raise DriftError(f"Columns missing from the reference or current data: {absent}")

    rows = [
        {"feature": col, "kind": "numeric", "psi": psi_numeric(reference[col], current[col], n_bins)}
        for col in numeric_columns
    ]
    rows += [
        {"feature": col, "kind": "categorical", "psi": psi_categorical(reference[col], current[col])}
        for col in categorical_columns
    ]
    table = pd.DataFrame(rows)
    table["status"] = [classify(value, moderate, major) for value in table["psi"]]
    return table.sort_values("psi", ascending=False).reset_index(drop=True)


def run_drift(
    cfg: dict[str, Any],
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    strict: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Compare new customers (or, without an input file, the validation split) with the training split.

    The test split is never opened. Writes the PSI table as a CSV.
    """
    log = logger or logging.getLogger(__name__)
    settings = load_drift_settings(cfg)
    processed = cfg["paths"]["processed_dir"]
    target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]

    train = load_split("train", processed)
    for col in (target, id_col):
        if col not in train.columns:
            raise DriftError(f"Column '{col}' not found in the train split")
    reference = add_features(train.drop(columns=[target, id_col]))

    if input_path is None:
        val = load_split("val", processed)
        for col in (target, id_col):
            if col not in val.columns:
                raise DriftError(f"Column '{col}' not found in the validation split")
        current = add_features(val.drop(columns=[target, id_col]))
        label = "validation split vs training split (a sanity check: random splits should look the same)"
        default_output = Path(cfg["paths"]["reports_dir"]) / "drift_validation_vs_train.csv"
    else:
        cleaned, _ = prepare_input(load_raw(input_path), id_col, target, strict, log)
        current = add_features(cleaned)
        label = f"{input_path} vs training split"
        default_output = Path(processed) / "drift_report.csv"

    table = compute_drift(
        reference, current, DRIFT_NUMERIC, DRIFT_CATEGORICAL,
        settings["n_bins"], settings["moderate_threshold"], settings["major_threshold"], settings["min_rows"],
    )

    out = Path(output_path) if output_path else default_output
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(out, index=False)
    except OSError as exc:
        raise DriftError(f"Cannot write the drift table to {out}: {exc}") from exc

    log.info("Drift check: %s (%d current rows)", label, len(current))
    log.info("PSI per feature (largest first):\n%s", table.round(4).to_string(index=False))
    log.info("Features by status: %s. Wrote %s", table["status"].value_counts().to_dict(), out)
    return table


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: `python -m src.drift [--input new_customers.csv]`."""
    parser = argparse.ArgumentParser(description="PSI feature-drift check against the training split.")
    parser.add_argument("--input", help="raw CSV of new customers (default: compare the validation split)")
    parser.add_argument("--output", help="where to write the drift table (CSV)")
    parser.add_argument("--strict", action="store_true", help="fail on unexpected columns in --input")
    args = parser.parse_args(argv)
    try:
        cfg = load_config()
        log = get_logger("churn.drift", log_file=cfg["paths"]["log_file"])
        run_drift(cfg, args.input, args.output, args.strict, log)
    except (
        ConfigError, SplitError, DataValidationError, FeatureError, PredictionError, DriftError
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()