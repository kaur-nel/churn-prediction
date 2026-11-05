"""Load, validate and clean the raw Telco churn data."""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ConfigError, load_config
from src.logger import get_logger

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

YES_NO = {"Yes", "No"}
INTERNET_DEPENDENT = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Allowed values AFTER the "No ... service" -> "No" cleanup below.
CATEGORICAL_ALLOWED: dict[str, set[str]] = {
    "gender": {"Female", "Male"},
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": YES_NO,
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{col: YES_NO for col in INTERNET_DEPENDENT},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": YES_NO,
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

# Columns where "No <service>" really just means "No".
NO_SERVICE_LABELS: dict[str, str] = {
    "MultipleLines": "No phone service",
    **{col: "No internet service" for col in INTERNET_DEPENDENT},
}


class DataValidationError(Exception):
    """Raised when the data file is missing, unreadable or fails validation."""


def _clean_cell(value: Any) -> Any:
    """Strip spaces from text; turn empty text into NaN; leave everything else alone."""
    if isinstance(value, str):
        value = value.strip()
        return np.nan if value == "" else value
    return value


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV with every column as text (we convert types ourselves).

    Raises:
        DataValidationError: if the file is missing, empty, unreadable or has no rows.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataValidationError(f"Data file not found: {csv_path.resolve()}")
    if not csv_path.is_file():
        raise DataValidationError(f"Data path is not a file: {csv_path.resolve()}")
    if csv_path.stat().st_size == 0:
        raise DataValidationError(f"Data file is empty (0 bytes): {csv_path}")

    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Data file has no content: {csv_path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise DataValidationError(f"Data file could not be read ({csv_path}): {exc}") from exc

    if df.empty:
        raise DataValidationError(f"Data file has a header but no rows: {csv_path}")
    return df


def clean_data(
    df: pd.DataFrame,
    id_column: str,
    target: str,
    positive_label: str,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and clean the raw frame. Returns (clean_frame, report_of_what_changed).

    Cleaning here is deterministic row-level tidying (no statistics are learned from the
    data), so it is safe to do before the train/validation/test split. Anything that needs
    to be *learned* (imputing, scaling, encoding) happens later inside the sklearn Pipeline.

    The target column becomes 1 for `positive_label` and 0 for the other class.

    Raises:
        DataValidationError: with a message naming the column and problem.
    """
    log = logger or logging.getLogger(__name__)
    report: dict[str, Any] = {"rows_in": len(df)}
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # --- columns -------------------------------------------------------------
    required = {id_column, target, *NUMERIC_COLUMNS, *CATEGORICAL_ALLOWED}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    extra = [col for col in df.columns if col not in required]
    if extra:
        log.warning("Dropping unexpected columns: %s", extra)
        df = df.drop(columns=extra)
    report["extra_columns_dropped"] = extra

    # --- whitespace and blanks ---------------------------------------------------
    for col in df.columns:
        df[col] = df[col].astype(object).map(_clean_cell)

    # --- text columns must not have blanks ---------------------------------------
    for col in [id_column, target, *CATEGORICAL_ALLOWED]:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            raise DataValidationError(f"Column '{col}' has {n_missing} missing values")

    # --- duplicates --------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_dropped"] = before - len(df)
    dup_mask = df[id_column].duplicated()
    if dup_mask.any():
        ids = df.loc[dup_mask, id_column].head(5).tolist()
        raise DataValidationError(f"Duplicate {id_column} values with different data, e.g. {ids}")

    # --- numeric columns -----------------------------------------------------------
    for col in NUMERIC_COLUMNS:
        raw = df[col]
        num = pd.to_numeric(raw.astype(object), errors="coerce")
        garbage = raw.notna() & num.isna()
        if garbage.any():
            examples = raw[garbage].head(5).tolist()
            raise DataValidationError(f"Column '{col}' has non-numeric values, e.g. {examples}")
        if np.isinf(num).any():
            raise DataValidationError(f"Column '{col}' has infinite values")
        df[col] = num

    for col in ("SeniorCitizen", "tenure", "MonthlyCharges"):
        n_missing = int(df[col].isna().sum())
        if n_missing:
            raise DataValidationError(f"Column '{col}' has {n_missing} missing values")

    if not df["SeniorCitizen"].isin([0, 1]).all():
        raise DataValidationError("Column 'SeniorCitizen' must contain only 0 or 1")
    if (df["tenure"] < 0).any():
        raise DataValidationError("Column 'tenure' has negative values")
    if (df["tenure"] % 1 != 0).any():
        raise DataValidationError("Column 'tenure' must contain whole months")
    for col in ("MonthlyCharges", "TotalCharges"):
        if (df[col] < 0).any():
            raise DataValidationError(f"Column '{col}' has negative values")

    # ASSUMPTION: a blank TotalCharges for a customer with tenure 0 means "not billed yet",
    # so we set it to 0. Blanks for customers with tenure > 0 are left as NaN and imputed
    # later inside the Pipeline (so nothing is learned from the test data).
    fill_mask = df["TotalCharges"].isna() & (df["tenure"] == 0)
    report["total_charges_filled_for_new_customers"] = int(fill_mask.sum())
    df.loc[fill_mask, "TotalCharges"] = 0.0
    report["total_charges_missing_left_for_pipeline"] = int(df["TotalCharges"].isna().sum())
    if report["total_charges_missing_left_for_pipeline"]:
        log.warning(
            "%d TotalCharges values are still blank (tenure > 0); the Pipeline will impute them",
            report["total_charges_missing_left_for_pipeline"],
        )

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)

    # --- inconsistent categories ----------------------------------------------------
    relabeled = 0
    for col, label in NO_SERVICE_LABELS.items():
        relabeled += int((df[col] == label).sum())
        df[col] = df[col].replace({label: "No"})
    report["no_service_labels_changed_to_no"] = relabeled

    for col, allowed in CATEGORICAL_ALLOWED.items():
        unexpected = sorted(set(df[col]) - allowed)
        if unexpected:
            raise DataValidationError(
                f"Column '{col}' has unexpected values {unexpected}; allowed: {sorted(allowed)}"
            )

    # --- target ---------------------------------------------------------------------
    labels = sorted(set(df[target]))
    if len(labels) != 2:
        raise DataValidationError(f"Target '{target}' must have exactly 2 classes, found {labels}")
    if positive_label not in labels:
        raise DataValidationError(
            f"Target '{target}' must contain the positive label {positive_label!r}; found {labels}"
        )
    df[target] = (df[target] == positive_label).astype(int)

    df = df.reset_index(drop=True)
    report["rows_out"] = len(df)
    report["churn_rate"] = round(float(df[target].mean()), 4)
    log.info("Cleaning report: %s", report)
    return df, report


def prepare_data(cfg: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load the raw CSV, clean it and save the result to `paths.processed_data`."""
    log = logger or logging.getLogger(__name__)
    raw = load_raw(cfg["paths"]["raw_data"])
    clean, _ = clean_data(
        raw,
        id_column=cfg["data"]["id_column"],
        target=cfg["data"]["target"],
        positive_label=cfg["data"]["positive_label"],
        logger=log,
    )

    out_path = Path(cfg["paths"]["processed_data"])
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(out_path, index=False)
    except OSError as exc:
        raise DataValidationError(f"Cannot write cleaned data to {out_path}: {exc}") from exc
    log.info("Saved %d clean rows to %s", len(clean), out_path)
    return clean


def main() -> None:
    """Command-line entry point: `python -m src.data`."""
    try:
        cfg = load_config()
        logger = get_logger("churn.data", log_file=cfg["paths"]["log_file"])
        prepare_data(cfg, logger)
    except (ConfigError, DataValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()