"""Score new customers with the frozen final model.

Usage: python -m src.predict --input customers.csv [--output predictions.csv] [--threshold 0.17] [--strict]

The input is a CSV in the same raw format as the Telco file. It is cleaned the way the training
data was (see src/data.py), but tolerantly: blank cells and unseen category values are allowed
(the model's pipeline fills or ignores them), and the target column is never used.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration import CalibrationError
from src.config import ConfigError, load_config
from src.data import CATEGORICAL_ALLOWED, NO_SERVICE_LABELS, DataValidationError, load_raw
from src.evaluate import EvaluationError, file_sha256, load_model, load_threshold
from src.features import EXCLUDED_FEATURES, RAW_BINARY, RAW_NUMERIC, REQUIRED_COLUMNS, FeatureError
from src.logger import get_logger


class PredictionError(Exception):
    """Raised when the input cannot be scored or the scores are invalid."""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _clean_cell(value: Any) -> Any:
    """Strip spaces from text; turn empty text into NaN; leave everything else alone."""
    if isinstance(value, str):
        value = value.strip()
        return np.nan if value == "" else value
    return value


def _csv_lines(mask: pd.Series) -> list[int]:
    """CSV line numbers (the header is line 1) of the first few True rows."""
    return (np.flatnonzero(mask.to_numpy()) + 2)[:3].tolist()


def load_prediction_settings(cfg: dict[str, Any]) -> dict[str, int]:
    """Validate the `prediction` config section.

    Raises:
        PredictionError: naming the setting that is wrong.
    """
    section = cfg.get("prediction") if isinstance(cfg, dict) else None
    if not isinstance(section, dict):
        raise PredictionError("Config is missing the 'prediction' section")
    max_rows = section.get("max_rows")
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise PredictionError(f"'prediction.max_rows' must be an integer >= 1, got {max_rows!r}")
    return {"max_rows": max_rows}


def validate_threshold(threshold: Any) -> float:
    """Return the threshold as a float.

    Raises:
        PredictionError: if it is not a number between 0 and 1.
    """
    if not _is_number(threshold) or not 0 <= threshold <= 1:
        raise PredictionError(f"threshold must be a number between 0 and 1, got {threshold!r}")
    return float(threshold)


def prepare_input(
    raw: pd.DataFrame,
    id_column: str,
    target: str,
    strict: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and tidy raw customer rows. Returns (frame with id + required columns, report).

    Unlike training-time cleaning, blanks and unseen category values are allowed here: they are
    counted and warned about, and the model's pipeline fills or ignores them. Rows are never
    dropped, so the output lines up with the input.

    Raises:
        PredictionError: for no rows, duplicate column names, missing columns, blank ids, text,
            negative or infinite numbers, a SeniorCitizen outside 0/1, or (strict) unknown columns.
    """
    log = logger or logging.getLogger(__name__)
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise PredictionError("The input has no rows to score")

    df = raw.copy()
    df.columns = [str(col).strip() for col in df.columns]
    duplicated = sorted(set(df.columns[df.columns.duplicated()]))
    if duplicated:
        raise PredictionError(f"Duplicate column names in the input: {duplicated}")

    required = [id_column, *REQUIRED_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise PredictionError(f"Missing required columns: {missing}")

    known_but_ignored = {target, *EXCLUDED_FEATURES}
    unexpected = [col for col in df.columns if col not in {*required, *known_but_ignored}]
    if unexpected:
        if strict:
            raise PredictionError(f"Unexpected columns {unexpected}: remove them or run without --strict")
        log.warning("Ignoring unexpected columns: %s", unexpected)

    df = df[required].copy()
    for col in df.columns:
        df[col] = df[col].astype(object).map(_clean_cell)

    blank_ids = int(df[id_column].isna().sum())
    if blank_ids:
        raise PredictionError(f"Column '{id_column}' has {blank_ids} blank values")
    duplicate_ids = int(df[id_column].duplicated().sum())
    if duplicate_ids:
        log.warning("%d rows repeat an earlier %s; all rows are scored", duplicate_ids, id_column)

    for col in [*RAW_NUMERIC, *RAW_BINARY]:
        numbers = pd.to_numeric(df[col], errors="coerce")
        garbage = df[col].notna() & numbers.isna()
        if garbage.any():
            raise PredictionError(
                f"Column '{col}' has non-numeric values, e.g. {df.loc[garbage, col].head(3).tolist()} "
                f"(CSV lines {_csv_lines(garbage)})"
            )
        infinite = np.isinf(numbers)
        if infinite.any():
            raise PredictionError(f"Column '{col}' has infinite values (CSV lines {_csv_lines(infinite)})")
        negative = numbers < 0
        if negative.any():
            raise PredictionError(f"Column '{col}' has negative values (CSV lines {_csv_lines(negative)})")
        if col in RAW_BINARY:
            not_binary = numbers.notna() & ~numbers.isin([0, 1])
            if not_binary.any():
                raise PredictionError(f"Column '{col}' must be 0 or 1 (CSV lines {_csv_lines(not_binary)})")
        df[col] = numbers

    # Same assumption as training: no billing history yet means a total of 0.
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0

    for col, label in NO_SERVICE_LABELS.items():
        df[col] = df[col].replace({label: "No"})

    unseen: dict[str, list[str]] = {}
    for col, allowed in CATEGORICAL_ALLOWED.items():
        if col not in df.columns:
            continue
        values = df[col].dropna()
        outside = values[~values.isin(allowed)]
        if len(outside):
            unseen[col] = sorted(set(outside))
    if unseen:
        log.warning("Unseen category values (the model treats them as 'none of the known categories'): %s", unseen)

    missing_values = {col: int(n) for col, n in df[REQUIRED_COLUMNS].isna().sum().items() if n}
    if missing_values:
        log.warning("Missing values will be filled with statistics from the training data: %s", missing_values)

    report = {
        "rows": len(df),
        "missing_values": missing_values,
        "unseen_categories": unseen,
        "unexpected_columns": unexpected,
        "duplicate_ids": duplicate_ids,
    }
    return df.reset_index(drop=True), report


def score(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Churn probabilities from the model, checked for shape, NaN and range.

    Raises:
        PredictionError: if the model cannot score the data or returns invalid probabilities.
    """
    try:
        proba = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    except (FeatureError, ValueError, KeyError) as exc:
        raise PredictionError(f"The model could not score the input: {exc}") from exc
    if proba.shape != (len(features),) or np.isnan(proba).any() or (proba < 0).any() or (proba > 1).any():
        raise PredictionError("The model returned invalid probabilities (wrong length, NaN, or outside 0-1)")
    return proba


def build_predictions(ids: pd.Series, proba: Any, threshold: Any, id_column: str) -> pd.DataFrame:
    """One output row per customer: id, probability, offer flag (probability >= threshold), threshold.

    Raises:
        PredictionError: for mismatched lengths or an invalid threshold.
    """
    p = np.asarray(proba, dtype=float)
    if len(ids) != len(p):
        raise PredictionError(f"{len(ids)} ids but {len(p)} probabilities")
    cutoff = validate_threshold(threshold)
    return pd.DataFrame(
        {
            id_column: ids.to_numpy(),
            "churn_probability": np.round(p, 6),
            "flagged_for_offer": (p >= cutoff).astype(int),
            "threshold_used": cutoff,
        }
    )


def model_fingerprint_warning(model_path: Path, results_path: Path) -> str | None:
    """A warning if the model file is not the one that was evaluated on the test split, else None."""
    if not results_path.exists():
        return None
    try:
        recorded = json.loads(results_path.read_text(encoding="utf-8")).get("model_sha256")
    except (OSError, ValueError, AttributeError):
        return f"Could not read {results_path}, so the model could not be checked against the test evaluation"
    if recorded is None:
        return None
    if recorded != file_sha256(model_path):
        return (
            "The model file differs from the one evaluated on the test split (SHA-256 mismatch): "
            "the reported test metrics may not apply to it"
        )
    return None


def run_predict(
    cfg: dict[str, Any],
    input_path: str | Path,
    output_path: str | Path,
    threshold: float | None = None,
    strict: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Score a CSV of customers and write the predictions CSV.

    The threshold is read from reports/threshold_choice.json unless `threshold` is given.

    Raises:
        PredictionError / DataValidationError / EvaluationError / FeatureError: for any invalid input.
    """
    log = logger or logging.getLogger(__name__)
    settings = load_prediction_settings(cfg)
    override = validate_threshold(threshold) if threshold is not None else None
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise PredictionError("The output file would overwrite the input file: choose a different --output")

    reports = Path(cfg["paths"]["reports_dir"])
    if override is None:
        cutoff = load_threshold(reports / "threshold_choice.json")["threshold"]
    else:
        cutoff = override
        log.info("Using the threshold given on the command line: %.3f", cutoff)

    model_path = Path(cfg["paths"]["models_dir"]) / "final_model.joblib"
    model = load_model(model_path)
    warning = model_fingerprint_warning(model_path, reports / "test_results.json")
    if warning:
        log.warning(warning)

    raw = load_raw(input_path)
    if len(raw) > settings["max_rows"]:
        raise PredictionError(
            f"The input has {len(raw)} rows, more than the allowed maximum of {settings['max_rows']} "
            "(prediction.max_rows in the config)"
        )

    id_column, target = cfg["data"]["id_column"], cfg["data"]["target"]
    features, _ = prepare_input(raw, id_column, target, strict, log)
    proba = score(model, features)
    predictions = build_predictions(features[id_column], proba, cutoff, id_column)

    out = Path(output_path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(out, index=False)
    except OSError as exc:
        raise PredictionError(f"Cannot write the predictions to {out}: {exc}") from exc

    flagged = int(predictions["flagged_for_offer"].sum())
    log.info(
        "Scored %d customers: %d (%.1f%%) flagged for an offer at threshold %.3f. Wrote %s",
        len(predictions), flagged, 100.0 * flagged / len(predictions), cutoff, out,
    )
    return predictions


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: `python -m src.predict --input customers.csv`."""
    parser = argparse.ArgumentParser(description="Score customers with the frozen churn model.")
    parser.add_argument("--input", required=True, help="CSV file with the customers to score")
    parser.add_argument("--output", help="where to write the predictions (default: <processed_dir>/predictions.csv)")
    parser.add_argument("--threshold", type=float, help="override the saved threshold (a number between 0 and 1)")
    parser.add_argument("--strict", action="store_true", help="fail on unexpected columns instead of ignoring them")
    args = parser.parse_args(argv)
    try:
        cfg = load_config()
        log = get_logger("churn.predict", log_file=cfg["paths"]["log_file"])
        output = args.output or str(Path(cfg["paths"]["processed_dir"]) / "predictions.csv")
        run_predict(cfg, args.input, output, args.threshold, args.strict, log)
    except (
        ConfigError, DataValidationError, EvaluationError, FeatureError, CalibrationError, PredictionError
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()