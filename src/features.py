"""Feature engineering and the preprocessing pipeline.

`add_features` only applies fixed rules (nothing is learned from data), so it is safe to run
anywhere. Everything that learns from data (imputing, scaling, encoding) lives inside the
ColumnTransformer and is fitted on the TRAIN split only.
"""

import logging
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

from src.config import ConfigError, load_config
from src.logger import get_logger
from src.split import SplitError, load_split

ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
RAW_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
RAW_BINARY = ["SeniorCitizen"]
RAW_CATEGORICAL = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *ADDON_COLUMNS,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NEW_NUMERIC = ["charges_ratio", "addon_count"]
NEW_BINARY = ["is_month_to_month", "is_electronic_check", "is_fiber_optic", "month_to_month_new"]
NEW_CATEGORICAL = ["tenure_bucket"]

# Columns that stay in the data but are deliberately NOT given to the model.
# gender: no signal in the EDA and a protected attribute; kept so fairness can be checked.
EXCLUDED_FEATURES = ["gender"]

REQUIRED_COLUMNS = [*RAW_NUMERIC, *RAW_BINARY, *RAW_CATEGORICAL]
TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]
NEW_CUSTOMER_MAX_MONTHS = 12


class FeatureError(Exception):
    """Raised when input data cannot be turned into features."""


def _strip(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the engineered feature columns added.

    Raises:
        FeatureError: if `df` is not a non-empty DataFrame, a required column is missing,
            or a numeric column has text, infinite or negative values.
    """
    if not isinstance(df, pd.DataFrame):
        raise FeatureError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise FeatureError("Cannot build features from an empty DataFrame")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise FeatureError(f"Missing required columns: {missing}")

    out = df.copy()

    for col in [*RAW_NUMERIC, *RAW_BINARY]:
        numbers = pd.to_numeric(out[col], errors="coerce")
        garbage = out[col].notna() & numbers.isna()
        if garbage.any():
            raise FeatureError(f"Column '{col}' has non-numeric values, e.g. {out.loc[garbage, col].head(3).tolist()}")
        if np.isinf(numbers).any():
            raise FeatureError(f"Column '{col}' has infinite values")
        if (numbers < 0).any():
            raise FeatureError(f"Column '{col}' has negative values")
        out[col] = numbers.astype(float)

    for col in RAW_CATEGORICAL:
        out[col] = out[col].astype(object).map(_strip)

    tenure, monthly, total = out["tenure"], out["MonthlyCharges"], out["TotalCharges"]

    # Ratio = (average bill actually paid per month) / (current monthly charge).
    # ASSUMPTION: with no billing history (tenure 0) or a 0 monthly charge, we use 1.0 ("no change").
    no_history = (tenure == 0) | (monthly == 0)
    average_paid = total / tenure.where(tenure != 0)
    out["charges_ratio"] = (average_paid / monthly.where(monthly != 0)).where(~no_history, 1.0)

    out["addon_count"] = (out[ADDON_COLUMNS] == "Yes").sum(axis=1).astype(float)
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["is_electronic_check"] = (out["PaymentMethod"] == "Electronic check").astype(int)
    out["is_fiber_optic"] = (out["InternetService"] == "Fiber optic").astype(int)
    out["month_to_month_new"] = (
        (out["is_month_to_month"] == 1) & (tenure <= NEW_CUSTOMER_MAX_MONTHS)
    ).astype(int)
    out["tenure_bucket"] = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(object)
    return out


def build_preprocessor(drop_columns: Sequence[str] = ()) -> Pipeline:
    """Build the unfitted pipeline: add features, then impute/scale/encode.

    Fit it on TRAIN data only. Columns not listed here (customerID, Churn, gender, ...)
    are dropped, so they can never reach the model.

    Args:
        drop_columns: engineered or raw feature names to leave out (e.g. ["TotalCharges"]).

    Raises:
        FeatureError: if `drop_columns` contains a name that is not a model feature.
    """
    numeric_all = [*RAW_NUMERIC, *NEW_NUMERIC]
    binary_all = [*RAW_BINARY, *NEW_BINARY]
    categorical_all = [*RAW_CATEGORICAL, *NEW_CATEGORICAL]

    unknown = set(drop_columns) - {*numeric_all, *binary_all, *categorical_all}
    if unknown:
        raise FeatureError(f"Cannot drop unknown feature(s): {sorted(unknown)}")

    drop = set(drop_columns)
    numeric = [col for col in numeric_all if col not in drop]
    binary = [col for col in binary_all if col not in drop]
    categorical = [col for col in categorical_all if col not in drop]

    columns = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    columns.set_output(transform="pandas")
    return Pipeline([("features", FunctionTransformer(add_features)), ("preprocess", columns)])


def compute_vif(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Variance Inflation Factor per column (above ~5-10 means heavy overlap with other columns).

    Raises:
        FeatureError: for missing or constant columns, or too few rows.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise FeatureError(f"Columns not found for VIF: {missing}")
    if len(columns) < 2:
        raise FeatureError("VIF needs at least 2 columns")
    data = df[list(columns)].astype(float)
    data = data.fillna(data.median())
    if len(data) <= len(columns) + 1:
        raise FeatureError("Too few rows to compute VIF")
    constant = [col for col in columns if data[col].nunique() < 2]
    if constant:
        raise FeatureError(f"Constant columns cannot be used for VIF: {constant}")

    with_const = add_constant(data, has_constant="add")
    rows = [
        (col, float(variance_inflation_factor(with_const.to_numpy(), idx)))
        for idx, col in enumerate(with_const.columns)
        if col != "const"
    ]
    return pd.DataFrame(rows, columns=["feature", "vif"]).sort_values("vif", ascending=False).reset_index(drop=True)


def main() -> None:
    """Command-line report on the TRAIN split: `python -m src.features`."""
    try:
        cfg = load_config()
        log: logging.Logger = get_logger("churn.features", log_file=cfg["paths"]["log_file"])
        target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]
        train = load_split("train", cfg["paths"]["processed_dir"])

        engineered = add_features(train)
        log.info("New feature columns: %s", [*NEW_NUMERIC, *NEW_BINARY, *NEW_CATEGORICAL])
        for col in [*NEW_BINARY, "tenure_bucket", "addon_count"]:
            table = engineered.groupby(col)[target].agg(customers="size", churn_rate="mean").round(3)
            log.info("Churn rate by %s (train only):\n%s", col, table.to_string())

        pipeline = build_preprocessor()
        matrix = pipeline.fit_transform(train.drop(columns=[target, id_col]))
        log.info("Pipeline output: %d rows x %d columns, NaN cells: %d", *matrix.shape, int(matrix.isna().sum().sum()))

        vif_columns = [*RAW_NUMERIC, *NEW_NUMERIC, *RAW_BINARY, *NEW_BINARY]
        vif = compute_vif(engineered, vif_columns)
        log.info("VIF on train (above ~5-10 = heavy overlap):\n%s", vif.round(2).to_string(index=False))
        reports = cfg["paths"]["reports_dir"]
        vif.to_csv(f"{reports}/vif_table.csv", index=False)
    except (ConfigError, SplitError, FeatureError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()