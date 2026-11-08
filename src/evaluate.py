"""One-time final evaluation of the frozen model on the TEST split.

This module is the only place in the project that opens the test split. It applies the frozen
model and the threshold that was chosen earlier on the validation split
(reports/threshold_choice.json). It never tunes, re-fits or re-chooses anything.

A results file locks the evaluation: after a successful run, running it again requires the
explicit --rerun flag. Do not change the model, the threshold or the settings because of what
the test results show: that would make the test estimate optimistic.
"""

import argparse
import hashlib
import json
import logging
import math
import pickle
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score

from src.business import BusinessError, evaluate_threshold, validate_economics
from src.calibration import (
    CalibrationError,
    brier_score,
    calibration_table,
    expected_calibration_error,
    plot_reliability,
)
from src.config import ConfigError, load_config
from src.features import FeatureError
from src.logger import get_logger
from src.metrics import MetricsError, ranking_metrics, threshold_metrics
from src.split import SplitError, load_split

METRIC_NAMES = ["pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "targeted_pct", "net_per_1000"]
ECONOMICS_KEYS = ("customer_ltv", "retention_offer_cost", "offer_success_rate")
SETTING_KEYS = ["n_bootstrap", "ci_level", "calibration_bins", "fairness_columns", "min_group_positives"]


class EvaluationError(Exception):
    """Raised when the final evaluation cannot run or its inputs are invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def load_evaluation_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the `evaluation` config section.

    Raises:
        EvaluationError: naming the exact setting that is wrong.
    """
    section = cfg.get("evaluation") if isinstance(cfg, dict) else None
    if not isinstance(section, dict):
        raise EvaluationError("Config is missing the 'evaluation' section")
    missing = [key for key in SETTING_KEYS if key not in section]
    if missing:
        raise EvaluationError(f"Config section 'evaluation' is missing keys: {missing}")

    n_boot = section["n_bootstrap"]
    if not _is_int(n_boot) or n_boot < 100:
        raise EvaluationError(f"'evaluation.n_bootstrap' must be an integer >= 100, got {n_boot!r}")
    level = section["ci_level"]
    if not _is_number(level) or not 0 < level < 1:
        raise EvaluationError(f"'evaluation.ci_level' must be a number between 0 and 1 (exclusive), got {level!r}")
    bins = section["calibration_bins"]
    if not _is_int(bins) or bins < 2:
        raise EvaluationError(f"'evaluation.calibration_bins' must be an integer >= 2, got {bins!r}")
    columns = section["fairness_columns"]
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise EvaluationError(f"'evaluation.fairness_columns' must be a non-empty list of column names, got {columns!r}")
    min_pos = section["min_group_positives"]
    if not _is_int(min_pos) or min_pos < 1:
        raise EvaluationError(f"'evaluation.min_group_positives' must be an integer >= 1, got {min_pos!r}")
    return dict(section)


def load_threshold(path: Path) -> dict[str, Any]:
    """Read the threshold and money assumptions saved by `python -m src.business`.

    Raises:
        EvaluationError: if the file is missing, unreadable or its contents are invalid.
    """
    if not path.exists():
        raise EvaluationError(f"Threshold file not found: {path}. Run `python -m src.business` first")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read the threshold file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvaluationError(f"The threshold file {path} must contain a JSON object")

    threshold = data.get("threshold")
    if not _is_number(threshold) or not 0 <= threshold <= 1:
        raise EvaluationError(f"'threshold' in {path} must be a number between 0 and 1, got {threshold!r}")
    assumptions = data.get("assumptions")
    if not isinstance(assumptions, dict) or any(key not in assumptions for key in ECONOMICS_KEYS):
        raise EvaluationError(f"'assumptions' in {path} must contain {ECONOMICS_KEYS}")
    try:
        validate_economics(*(assumptions[key] for key in ECONOMICS_KEYS))
    except BusinessError as exc:
        raise EvaluationError(f"Invalid assumptions in {path}: {exc}") from exc

    return {
        "threshold": float(threshold),
        "economics": {
            "ltv": assumptions["customer_ltv"],
            "cost": assumptions["retention_offer_cost"],
            "rate": assumptions["offer_success_rate"],
        },
        "expected_net_per_1000_validation": data.get("expected_net_per_1000_customers"),
    }


def load_model(path: Path) -> Any:
    """Load the frozen final model.

    Raises:
        EvaluationError: if the file is missing or cannot be loaded.
    """
    if not path.exists():
        raise EvaluationError(f"Final model not found: {path}. Run `python -m src.final_model` first")
    try:
        return joblib.load(path)
    except (OSError, EOFError, ValueError, ImportError, AttributeError, pickle.UnpicklingError) as exc:
        raise EvaluationError(f"Could not load the final model from {path}: {exc}") from exc


def file_sha256(path: Path) -> str:
    """SHA-256 fingerprint of a file (proves which files produced the numbers)."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvaluationError(f"Cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def compute_metrics(y: Any, p: Any, threshold: float, economics: dict[str, float]) -> dict[str, float]:
    """All headline metrics at the given threshold.

    Raises:
        MetricsError / BusinessError / CalibrationError: for invalid inputs (e.g. one class only).
    """
    ranking = ranking_metrics(y, p)
    at_threshold = threshold_metrics(y, p, threshold)
    money = evaluate_threshold(y, p, threshold, economics["ltv"], economics["cost"], economics["rate"])
    return {
        "pr_auc": ranking["pr_auc"],
        "roc_auc": ranking["roc_auc"],
        "precision": at_threshold["precision"],
        "recall": at_threshold["recall"],
        "f1": at_threshold["f1"],
        "brier": brier_score(y, p),
        "targeted_pct": 100.0 * (at_threshold["tp"] + at_threshold["fp"]) / len(y),
        "net_per_1000": money["net_per_1000"],
    }


def bootstrap_metrics(
    y_true: Any,
    proba: Any,
    threshold: float,
    economics: dict[str, float],
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    """Point estimates plus percentile bootstrap confidence intervals for every metric.

    The threshold is applied as given (never re-optimised inside a resample). Resamples with only
    one class cannot be scored and are skipped.

    Returns:
        (table with metric/estimate/ci_low/ci_high, number_of_skipped_resamples)

    Raises:
        EvaluationError: for bad arguments or too few valid resamples.
    """
    if not _is_int(n_bootstrap) or n_bootstrap < 2:
        raise EvaluationError(f"n_bootstrap must be an integer >= 2, got {n_bootstrap!r}")
    if not _is_number(ci_level) or not 0 < ci_level < 1:
        raise EvaluationError(f"ci_level must be a number between 0 and 1 (exclusive), got {ci_level!r}")

    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    point = compute_metrics(y, p, threshold, economics)  # also validates the inputs

    rng = np.random.default_rng(seed)
    draws: list[dict[str, float]] = []
    skipped = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        y_sample = y[idx]
        if y_sample.min() == y_sample.max():
            skipped += 1
            continue
        draws.append(compute_metrics(y_sample, p[idx], threshold, economics))
    if len(draws) < 2:
        raise EvaluationError("Too few valid bootstrap resamples (each one needs both classes)")

    frame = pd.DataFrame(draws)
    low, high = (1 - ci_level) / 2, 1 - (1 - ci_level) / 2
    rows = [
        {
            "metric": name,
            "estimate": point[name],
            "ci_low": float(frame[name].quantile(low)),
            "ci_high": float(frame[name].quantile(high)),
        }
        for name in METRIC_NAMES
    ]
    return pd.DataFrame(rows), skipped


def fairness_table(
    frame: pd.DataFrame,
    y_true: Any,
    proba: Any,
    threshold: float,
    columns: Sequence[str],
    min_positives: int,
) -> pd.DataFrame:
    """Per-group performance at the given threshold, for each fairness column.

    `reliable` is False when a group has fewer than `min_positives` churners or non-churners,
    because its numbers are then too noisy to judge. A group with one class has an empty ROC-AUC.

    Raises:
        EvaluationError: for no columns, a missing column or mismatched lengths.
    """
    y = np.asarray(y_true)
    p = np.asarray(proba, dtype=float)
    if not len(frame) == len(y) == len(p):
        raise EvaluationError("The data, y_true and proba must have the same length")
    if not columns:
        raise EvaluationError("At least one fairness column is needed")

    rows = []
    for column in columns:
        if column not in frame.columns:
            raise EvaluationError(f"Fairness column '{column}' not found in the test data")
        labels = frame[column].astype(object).where(frame[column].notna(), "missing").astype(str).to_numpy()
        for group in sorted(set(labels)):
            mask = labels == group
            y_g, p_g = y[mask], p[mask]
            flagged = p_g >= threshold
            n = int(mask.sum())
            positives = int(y_g.sum())
            negatives = n - positives
            tp = int((flagged & (y_g == 1)).sum())
            fp = int((flagged & (y_g == 0)).sum())
            rows.append(
                {
                    "attribute": column,
                    "group": group,
                    "customers": n,
                    "positives": positives,
                    "churn_rate": positives / n,
                    "mean_predicted": float(p_g.mean()),
                    "selection_rate": float(flagged.mean()),
                    "precision": tp / int(flagged.sum()) if flagged.any() else np.nan,
                    "recall": tp / positives if positives else np.nan,
                    "false_positive_rate": fp / negatives if negatives else np.nan,
                    "roc_auc": float(roc_auc_score(y_g, p_g)) if positives and negatives else np.nan,
                    "reliable": positives >= min_positives and negatives >= min_positives,
                }
            )
    return pd.DataFrame(rows)


def fairness_gaps(table: pd.DataFrame) -> pd.DataFrame:
    """Largest difference between reliable groups, per attribute (recall, false-positive rate, selection rate).

    Raises:
        EvaluationError: for an empty table or one without the needed columns.
    """
    needed = {"attribute", "group", "recall", "false_positive_rate", "selection_rate", "reliable"}
    if table.empty or not needed <= set(table.columns):
        raise EvaluationError("The fairness table is empty or missing columns")
    rows = []
    for attribute, part in table.groupby("attribute", sort=False):
        usable = part[part["reliable"]]
        row: dict[str, Any] = {"attribute": attribute, "groups_compared": len(usable)}
        for column in ("recall", "false_positive_rate", "selection_rate"):
            row[f"{column}_gap"] = float(usable[column].max() - usable[column].min()) if len(usable) >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion(counts: dict[str, int], path: str | Path, title: str) -> None:
    """Save the confusion matrix as a picture."""
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig = Figure(figsize=(5, 4.5))
    ax = fig.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["predicted stays", "predicted churns"])
    ax.set_yticks([0, 1], labels=["actually stays", "actually churns"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=14)
    ax.set_title(title)
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise EvaluationError(f"Cannot write the confusion matrix plot to {path}: {exc}") from exc


def run_evaluation(
    cfg: dict[str, Any], logger: logging.Logger | None = None, rerun: bool = False
) -> dict[str, Any]:
    """Evaluate the frozen model once on the test split and save every report.

    Writes reports/test_metrics.csv, fairness_by_group.csv, fairness_gaps.csv, test_results.json
    and two figures. test_results.json is written last, so a failed run never locks the evaluation.

    Raises:
        EvaluationError: if the evaluation was already done (and `rerun` is False), or for any
            invalid input.
    """
    log = logger or logging.getLogger(__name__)
    reports = Path(cfg["paths"]["reports_dir"])
    results_path = reports / "test_results.json"

    previous_runs = 0
    if results_path.exists():
        if not rerun:
            raise EvaluationError(
                "The test split was already evaluated (reports/test_results.json exists). Changing the model, "
                "threshold or settings because of test results makes the estimate optimistic. If you only need to "
                "regenerate the report files, run again with --rerun"
            )
        try:
            previous_runs = int(json.loads(results_path.read_text(encoding="utf-8")).get("evaluation_count", 1))
        except (OSError, ValueError, AttributeError, TypeError):
            previous_runs = 1
        log.warning(
            "Re-running the test evaluation (run #%d). Do not change the model, threshold or settings based on "
            "these results.",
            previous_runs + 1,
        )

    settings = load_evaluation_settings(cfg)
    threshold_info = load_threshold(reports / "threshold_choice.json")
    threshold, economics = threshold_info["threshold"], threshold_info["economics"]

    model_path = Path(cfg["paths"]["models_dir"]) / "final_model.joblib"
    model = load_model(model_path)

    processed = Path(cfg["paths"]["processed_dir"])
    test = load_split("test", processed, allow_test=True)
    target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]
    for col in (target, id_col):
        if col not in test.columns:
            raise EvaluationError(f"Column '{col}' not found in the test split")

    y = test[target].to_numpy()
    proba = model.predict_proba(test.drop(columns=[target, id_col]))[:, 1]
    seed = cfg["project"]["random_seed"]

    metrics, skipped = bootstrap_metrics(
        y, proba, threshold, economics, settings["n_bootstrap"], settings["ci_level"], seed
    )
    counts = threshold_metrics(y, proba, threshold)
    confusion = {key: counts[key] for key in ("tn", "fp", "fn", "tp")}
    ece = expected_calibration_error(y, proba, settings["calibration_bins"])
    everyone = evaluate_threshold(y, proba, 0.0, economics["ltv"], economics["cost"], economics["rate"])
    fairness = fairness_table(test, y, proba, threshold, settings["fairness_columns"], settings["min_group_positives"])
    gaps = fairness_gaps(fairness)

    level_pct = settings["ci_level"] * 100
    log.info("Test split: %d customers, churn rate %.4f, threshold %.3f (chosen on validation)", len(y), y.mean(), threshold)
    for _, row in metrics.iterrows():
        log.info(
            "%-13s %10.4f   (%.0f%% CI %.4f to %.4f)", row["metric"], row["estimate"], level_pct, row["ci_low"], row["ci_high"]
        )
    log.info("Confusion matrix: %s", confusion)
    log.info("Expected calibration error: %.4f", ece)
    log.info(
        "Net per 1,000 customers: %.0f at the chosen threshold; %.0f if we target everyone; expected on validation: %s",
        metrics.set_index("metric").loc["net_per_1000", "estimate"],
        everyone["net_per_1000"],
        threshold_info["expected_net_per_1000_validation"],
    )
    log.info("Fairness by group (reliable=False means too few customers to judge):\n%s", fairness.round(3).to_string(index=False))
    log.info("Fairness gaps between reliable groups:\n%s", gaps.round(3).to_string(index=False))

    results = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "evaluation_count": previous_runs + 1,
        "test_rows": len(test),
        "test_churn_rate": float(y.mean()),
        "threshold": threshold,
        "threshold_chosen_on": "validation split (reports/threshold_choice.json)",
        "assumptions": dict(zip(ECONOMICS_KEYS, economics.values(), strict=True)),
        "model_sha256": file_sha256(model_path),
        "test_file_sha256": file_sha256(processed / "test.csv"),
        "bootstrap": {
            "n_resamples": settings["n_bootstrap"],
            "skipped_single_class": skipped,
            "ci_level": settings["ci_level"],
        },
        "metrics": {
            row["metric"]: {
                "estimate": float(row["estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            }
            for _, row in metrics.iterrows()
        },
        "confusion_matrix": confusion,
        "expected_calibration_error": ece,
        "net_per_1000_target_everyone": everyone["net_per_1000"],
        "net_per_1000_expected_on_validation": threshold_info["expected_net_per_1000_validation"],
        "note": "Money figures use placeholder assumptions, not real company data.",
    }

    try:
        reports.mkdir(parents=True, exist_ok=True)
        metrics.round(4).to_csv(reports / "test_metrics.csv", index=False)
        fairness.round(4).to_csv(reports / "fairness_by_group.csv", index=False)
        gaps.round(4).to_csv(reports / "fairness_gaps.csv", index=False)
    except OSError as exc:
        raise EvaluationError(f"Cannot write the evaluation reports to {reports}: {exc}") from exc
    plot_confusion(confusion, reports / "figures" / "confusion_matrix_test.png", "Confusion matrix (test split)")
    plot_reliability(
        {"final model": calibration_table(y, proba, settings["calibration_bins"])},
        reports / "figures" / "calibration_test.png",
        "Calibration on the test split",
    )
    try:
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")  # written last: this is the lock
    except OSError as exc:
        raise EvaluationError(f"Cannot write {results_path}: {exc}") from exc
    return {"results": results, "metrics": metrics, "fairness": fairness, "gaps": gaps}


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: `python -m src.evaluate [--rerun]`."""
    parser = argparse.ArgumentParser(description="One-time evaluation of the frozen model on the TEST split.")
    parser.add_argument("--rerun", action="store_true", help="regenerate the reports even though the test split was already evaluated")
    args = parser.parse_args(argv)
    try:
        cfg = load_config()
        log = get_logger("churn.evaluate", log_file=cfg["paths"]["log_file"])
        run_evaluation(cfg, log, rerun=args.rerun)
    except (
        ConfigError, SplitError, FeatureError, CalibrationError, BusinessError, MetricsError, EvaluationError
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()