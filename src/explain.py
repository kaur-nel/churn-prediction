"""SHAP explainability for the frozen final model (logistic regression).

Explains the VALIDATION split (the test split is never opened here). SHAP values are in
log-odds. Because they are additive, they can be summed per source column ("groups"), which
keeps columns that carry the same information (e.g. Contract_Month-to-month and
is_month_to_month) from splitting one effect across several rows.

Every explanation is checked for local accuracy: base value + sum of SHAP values must equal
the model's own output.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src.business import BusinessError
from src.calibration import CalibrationError
from src.config import ConfigError, load_config
from src.evaluate import EvaluationError, load_model, load_threshold
from src.features import RAW_CATEGORICAL, FeatureError
from src.logger import get_logger
from src.split import SplitError, load_split

SAMPLE_ROLES = ("highest_risk", "nearest_threshold", "lowest_risk")
ADDITIVITY_TOLERANCE = 1e-6

# Which explanation group each model column belongs to. One-hot columns are named
# "<source column>_<value>".
EXACT_GROUPS = {
    "tenure": "tenure",
    "MonthlyCharges": "MonthlyCharges",
    "TotalCharges": "TotalCharges",
    "charges_ratio": "charges_ratio",
    "addon_count": "addon_count",
    "SeniorCitizen": "SeniorCitizen",
    "month_to_month_new": "month_to_month_new",
    "is_month_to_month": "Contract",
    "is_electronic_check": "PaymentMethod",
    "is_fiber_optic": "InternetService",
}
PREFIX_GROUPS = {**{f"{col}_": col for col in RAW_CATEGORICAL}, "tenure_bucket_": "tenure"}
# Column of the engineered data used to describe a group (default: the group's own name).
PROFILE_COLUMN = {"tenure": "tenure_bucket"}


class ExplainError(Exception):
    """Raised when the explanation cannot be produced or its inputs are invalid."""


def load_explain_settings(cfg: dict[str, Any]) -> dict[str, int]:
    """Validate the `explainability` config section.

    Raises:
        ExplainError: naming the setting that is wrong.
    """
    section = cfg.get("explainability") if isinstance(cfg, dict) else None
    if not isinstance(section, dict):
        raise ExplainError("Config is missing the 'explainability' section")
    settings = {}
    for key in ("top_n", "max_display"):
        value = section.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ExplainError(f"'explainability.{key}' must be an integer >= 1, got {value!r}")
        settings[key] = value
    return settings


def split_model(model: Any) -> tuple[Any, Any, LogisticRegression]:
    """Return (features step, preprocess step, classifier) of the saved final model.

    Raises:
        ExplainError: if the model has other steps or is not a logistic regression.
    """
    steps = getattr(getattr(model, "base", None), "named_steps", None)
    if steps is None or not {"features", "preprocess", "model"} <= set(steps):
        raise ExplainError("The saved model does not have the expected steps (features, preprocess, model)")
    classifier = steps["model"]
    if not isinstance(classifier, LogisticRegression):
        raise ExplainError(
            "This SHAP step supports logistic regression only, but the final model is "
            f"{type(classifier).__name__}. Change 'final.model' or extend src/explain.py"
        )
    return steps["features"], steps["preprocess"], classifier


def transform_features(features_step: Any, preprocess: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run raw data through the model's feature steps. Returns (engineered frame, model inputs)."""
    try:
        engineered = features_step.transform(X)
        matrix = preprocess.transform(engineered)
    except (FeatureError, ValueError, KeyError) as exc:
        raise ExplainError(f"Could not build the model inputs: {exc}") from exc
    if not isinstance(matrix, pd.DataFrame):
        raise ExplainError("The preprocessing step did not return a DataFrame")
    return engineered.reset_index(drop=True), matrix.reset_index(drop=True)


def unscale_numeric(matrix: pd.DataFrame, preprocess: Any) -> pd.DataFrame:
    """Copy of `matrix` with the standardized numeric columns converted back to real units."""
    try:
        columns = next(list(cols) for name, _, cols in preprocess.transformers_ if name == "num")
        scaler = preprocess.named_transformers_["num"].named_steps["scale"]
        means, scales = scaler.mean_, scaler.scale_
    except (AttributeError, KeyError, StopIteration) as exc:
        raise ExplainError(f"Could not read the numeric scaling from the preprocessing step: {exc}") from exc
    out = matrix.copy()
    for column, mean, scale in zip(columns, means, scales, strict=True):
        if column in out.columns:
            out[column] = out[column] * scale + mean
    return out


def compute_shap(
    classifier: LogisticRegression, background: pd.DataFrame, data: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """SHAP values (log-odds) for `data`, using `background` as the reference population.

    Returns:
        (values with shape (rows, features), base value)

    Raises:
        ExplainError: for empty or mismatched inputs, a SHAP failure, or values that do not
            add up to the model's own output.
    """
    if background.empty or data.empty:
        raise ExplainError("The background and the data to explain must not be empty")
    if list(background.columns) != list(data.columns):
        raise ExplainError("The background and the data must have the same columns")
    try:
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.LinearExplainer(classifier, masker)
        values = explainer.shap_values(data)
        base = float(np.ravel(explainer.expected_value)[0])
    except (ValueError, TypeError, AttributeError) as exc:
        raise ExplainError(f"SHAP could not explain the model: {exc}") from exc

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.shape != data.shape:
        raise ExplainError(f"Unexpected SHAP output shape {values.shape}; expected {data.shape}")

    gap = float(np.max(np.abs(base + values.sum(axis=1) - classifier.decision_function(data))))
    if gap > ADDITIVITY_TOLERANCE:
        raise ExplainError(f"SHAP values do not add up to the model's output (largest gap {gap:.2e})")
    return values, base


def group_of(column: str) -> str:
    """The explanation group of a model column.

    Raises:
        ExplainError: if no group is defined for the column.
    """
    if column in EXACT_GROUPS:
        return EXACT_GROUPS[column]
    for prefix, group in PREFIX_GROUPS.items():
        if column.startswith(prefix):
            return group
    raise ExplainError(f"No explanation group is defined for model column '{column}'")


def build_groups(columns: list[str]) -> dict[str, list[str]]:
    """Map each group to its model columns. Every column must belong to a group."""
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(group_of(column), []).append(column)
    return groups


def group_shap_values(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Sum the SHAP values of the columns in each group (one column per group)."""
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ExplainError("The SHAP values and the column names do not match")
    position = {column: i for i, column in enumerate(columns)}
    return pd.DataFrame(
        {group: values[:, [position[c] for c in cols]].sum(axis=1) for group, cols in build_groups(columns).items()}
    )


def feature_importance(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per model column, largest first."""
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ExplainError("The SHAP values and the column names do not match")
    table = pd.DataFrame({"feature": columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def group_importance(group_values: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per group, with each group's share of the total (%)."""
    if group_values.empty:
        raise ExplainError("There are no group SHAP values")
    table = pd.DataFrame({"group": group_values.columns, "mean_abs_shap": group_values.abs().mean().to_numpy()})
    total = table["mean_abs_shap"].sum()
    table["share_pct"] = 100.0 * table["mean_abs_shap"] / total if total > 0 else 0.0
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def driver_profiles(
    group_values: pd.DataFrame, engineered: pd.DataFrame, y_true: Any, groups: list[str]
) -> pd.DataFrame:
    """For each group, the effect on each value of the underlying data.

    `odds_multiplier` = exp(mean SHAP of the group): how much this group multiplies the odds of
    churn for customers with that value, compared with an average customer. `observed_churn_rate`
    is the real churn rate of those customers, so the story can be checked against the data.
    """
    y = np.asarray(y_true)
    if not len(group_values) == len(engineered) == len(y):
        raise ExplainError("The SHAP values, the data and the outcomes must have the same length")
    rows = []
    for group in groups:
        column = PROFILE_COLUMN.get(group, group)
        if group not in group_values.columns or column not in engineered.columns:
            raise ExplainError(f"Cannot profile group '{group}': a needed column is missing")
        series = engineered[column]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 8:
            labels = pd.qcut(series, 4, duplicates="drop").astype(str)
        else:
            labels = series.astype(str)
        frame = pd.DataFrame({"value": labels.to_numpy(), "shap": group_values[group].to_numpy(), "churned": y})
        part_rows = []
        for value, part in frame.groupby("value", sort=False):
            mean_shap = float(part["shap"].mean())
            part_rows.append(
                {
                    "group": group,
                    "value": value,
                    "customers": len(part),
                    "mean_shap": mean_shap,
                    "odds_multiplier": float(np.exp(mean_shap)),
                    "observed_churn_rate": float(part["churned"].mean()),
                }
            )
        rows.extend(sorted(part_rows, key=lambda row: -row["mean_shap"]))
    return pd.DataFrame(rows)


def pick_sample_customers(proba: Any, threshold: float) -> dict[str, int]:
    """Positions of 3 distinct customers chosen by a fixed rule (no cherry-picking).

    highest risk, nearest to the chosen threshold, lowest risk.

    Raises:
        ExplainError: for fewer than 3 customers, NaN scores, or all-identical scores.
    """
    p = np.asarray(proba, dtype=float)
    if p.ndim != 1 or len(p) < 3:
        raise ExplainError("At least 3 customers are needed to pick sample customers")
    if np.isnan(p).any():
        raise ExplainError("Probabilities contain NaN")
    highest, lowest = int(np.argmax(p)), int(np.argmin(p))
    if highest == lowest:
        raise ExplainError("All customers have the same predicted risk, so no samples can be picked")
    distance = np.abs(p - threshold)
    distance[[highest, lowest]] = np.inf
    return {"highest_risk": highest, "nearest_threshold": int(np.argmin(distance)), "lowest_risk": lowest}


def top_contributions(row_values: np.ndarray, columns: list[str], k: int = 5) -> list[tuple[str, float]]:
    """The k columns with the largest absolute SHAP value for one customer."""
    order = np.argsort(-np.abs(row_values))[:k]
    return [(columns[i], float(row_values[i])) for i in order]


def _save_current_figure(path: str | Path) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise ExplainError(f"Cannot write the plot to {path}: {exc}") from exc
    finally:
        plt.close("all")


def plot_summary(values: np.ndarray, data: pd.DataFrame, path: str | Path, max_display: int) -> None:
    """Beeswarm summary: each dot is one customer; colour shows the feature value (standardized)."""
    plt.switch_backend("Agg")
    shap.summary_plot(values, data, feature_names=list(data.columns), max_display=max_display, show=False)
    _save_current_figure(path)


def plot_bar(values: np.ndarray, data: pd.DataFrame, path: str | Path, max_display: int) -> None:
    """Bar chart of the mean absolute SHAP value per model column."""
    plt.switch_backend("Agg")
    shap.summary_plot(
        values, data, feature_names=list(data.columns), plot_type="bar", max_display=max_display, show=False
    )
    _save_current_figure(path)


def plot_waterfall(
    row_values: np.ndarray,
    base: float,
    row_data: Any,
    feature_names: list[str],
    path: str | Path,
    max_display: int = 10,
    title: str = "",
) -> None:
    """Waterfall for one customer: how each feature moves the score away from the average."""
    plt.switch_backend("Agg")
    explanation = shap.Explanation(
        values=np.asarray(row_values, dtype=float),
        base_values=base,
        data=np.asarray(row_data, dtype=float),
        feature_names=list(feature_names),
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    if title:
        plt.title(title)
    _save_current_figure(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_csv(path, index=False)
    except OSError as exc:
        raise ExplainError(f"Cannot write {path}: {exc}") from exc


def run_explain(cfg: dict[str, Any], logger: logging.Logger | None = None) -> dict[str, Any]:
    """Explain the frozen model on the validation split and write all reports and figures.

    Writes reports/shap_feature_importance.csv, shap_group_importance.csv, shap_driver_profiles.csv,
    shap_sample_customers.csv and figures shap_summary.png, shap_bar.png and one waterfall per sample.
    """
    log = logger or logging.getLogger(__name__)
    settings = load_explain_settings(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    figures = reports / "figures"
    threshold = load_threshold(reports / "threshold_choice.json")["threshold"]

    model = load_model(Path(cfg["paths"]["models_dir"]) / "final_model.joblib")
    features_step, preprocess, classifier = split_model(model)
    calibrated = getattr(model, "method", "none") != "none"
    if calibrated:
        log.warning("The final model is calibrated: these explanations describe the raw model score, not the calibrated one")

    processed = cfg["paths"]["processed_dir"]
    train, val = load_split("train", processed), load_split("val", processed)
    target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]
    for label, frame in (("train", train), ("val", val)):
        for col in (target, id_col):
            if col not in frame.columns:
                raise ExplainError(f"Column '{col}' not found in the {label} split")

    _, background = transform_features(features_step, preprocess, train.drop(columns=[target, id_col]))
    X_val = val.drop(columns=[target, id_col])
    engineered, matrix = transform_features(features_step, preprocess, X_val)
    proba = model.predict_proba(X_val)[:, 1]
    values, base = compute_shap(classifier, background, matrix)
    columns = list(matrix.columns)

    if not calibrated:
        from_shap = 1.0 / (1.0 + np.exp(-(base + values.sum(axis=1))))
        gap = float(np.max(np.abs(from_shap - proba)))
        if gap > ADDITIVITY_TOLERANCE:
            raise ExplainError(f"SHAP values do not reproduce the model's probabilities (largest gap {gap:.2e})")

    features = feature_importance(values, columns)
    group_values = group_shap_values(values, columns)
    groups = group_importance(group_values)
    top_groups = list(groups["group"].head(settings["top_n"]))
    profiles = driver_profiles(group_values, engineered, val[target], top_groups)

    log.info("Base value (average customer, log-odds): %.3f = churn probability %.3f", base, 1 / (1 + np.exp(-base)))
    log.info("Top groups by mean |SHAP| (share of total importance):\n%s", groups.round(3).head(10).to_string(index=False))
    log.info("Profiles of the top %d groups (odds_multiplier is vs an average customer):\n%s", len(top_groups), profiles.round(3).to_string(index=False))

    display = unscale_numeric(matrix, preprocess)
    samples = pick_sample_customers(proba, threshold)
    sample_rows = []
    for role, position in samples.items():
        contributions = top_contributions(values[position], columns, 5)
        sample_rows.append(
            {
                "role": role,
                "customer_id": val[id_col].iloc[position],
                "predicted_risk": float(proba[position]),
                "actually_churned": int(val[target].iloc[position]),
                "threshold": threshold,
                "top_contributions_log_odds": "; ".join(f"{name} {value:+.2f}" for name, value in contributions),
            }
        )
        plot_waterfall(
            values[position], base, display.iloc[position], columns, figures / f"shap_waterfall_{role}.png",
            title=f"{role}: predicted risk {proba[position]:.2f}",
        )
    sample_table = pd.DataFrame(sample_rows)
    log.info("Sample customers:\n%s", sample_table.round(3).to_string(index=False))

    try:
        reports.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExplainError(f"Cannot create {reports}: {exc}") from exc
    _write_csv(features.round(5), reports / "shap_feature_importance.csv")
    _write_csv(groups.round(5), reports / "shap_group_importance.csv")
    _write_csv(profiles.round(5), reports / "shap_driver_profiles.csv")
    _write_csv(sample_table.round(5), reports / "shap_sample_customers.csv")
    plot_summary(values, matrix, figures / "shap_summary.png", settings["max_display"])
    plot_bar(values, matrix, figures / "shap_bar.png", settings["max_display"])
    return {"features": features, "groups": groups, "profiles": profiles, "samples": sample_table, "base": base}


def main() -> None:
    """Command-line entry point: `python -m src.explain`."""
    try:
        cfg = load_config()
        log = get_logger("churn.explain", log_file=cfg["paths"]["log_file"])
        run_explain(cfg, log)
    except (
        ConfigError, SplitError, FeatureError, CalibrationError, BusinessError, EvaluationError, ExplainError
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()