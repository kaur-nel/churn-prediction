"""Pick the final model's calibration on the VALIDATION split, then freeze and save the model.

Reads only the train and validation splits. The test split is never opened here.
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.calibration import (
    CALIBRATION_METHODS,
    CalibratedModel,
    CalibrationError,
    brier_score,
    calibration_table,
    expected_calibration_error,
    plot_reliability,
)
from src.config import ConfigError, load_config
from src.features import FeatureError
from src.logger import get_logger
from src.metrics import MetricsError, ranking_metrics
from src.models import MODEL_NAMES, STRATEGIES, ModelError, build_pipeline
from src.split import SplitError, load_split

SETTING_KEYS = ["model", "strategy", "params", "calibration_candidates", "calibration_bins", "min_brier_gain"]


class FinalModelError(Exception):
    """Raised when the final-model settings are invalid or the selection cannot run."""


def load_final_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the `final` config section. "none" is always evaluated, and listed first.

    Raises:
        FinalModelError: naming the exact setting that is wrong.
    """
    section = cfg.get("final") if isinstance(cfg, dict) else None
    if not isinstance(section, dict):
        raise FinalModelError("Config is missing the 'final' section (model, strategy, params, ...)")
    missing = [key for key in SETTING_KEYS if key not in section]
    if missing:
        raise FinalModelError(f"Config section 'final' is missing keys: {missing}")

    if section["model"] not in MODEL_NAMES:
        raise FinalModelError(f"'final.model' must be one of {MODEL_NAMES}, got {section['model']!r}")
    if section["strategy"] not in STRATEGIES:
        raise FinalModelError(f"'final.strategy' must be one of {STRATEGIES}, got {section['strategy']!r}")
    params = section["params"]
    if not isinstance(params, dict) or not all(isinstance(key, str) for key in params):
        raise FinalModelError(f"'final.params' must be a mapping of parameter names, got {params!r}")
    candidates = section["calibration_candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise FinalModelError(f"'final.calibration_candidates' must be a non-empty list, got {candidates!r}")
    unknown = [name for name in candidates if name not in CALIBRATION_METHODS]
    if unknown:
        raise FinalModelError(
            f"'final.calibration_candidates' has unknown method(s) {unknown}; choose from {CALIBRATION_METHODS}"
        )
    bins = section["calibration_bins"]
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise FinalModelError(f"'final.calibration_bins' must be an integer >= 2, got {bins!r}")
    gain = section["min_brier_gain"]
    if isinstance(gain, bool) or not isinstance(gain, int | float) or not math.isfinite(gain) or gain < 0:
        raise FinalModelError(f"'final.min_brier_gain' must be a number >= 0, got {gain!r}")

    ordered = ["none", *[name for name in dict.fromkeys(candidates) if name != "none"]]
    return {**section, "calibration_candidates": ordered}


def choose_calibration(table: pd.DataFrame, min_brier_gain: float) -> str:
    """Pick the calibration method with the lowest Brier score, but only if it beats "none"
    by at least `min_brier_gain`; otherwise keep the plain model.

    Raises:
        FinalModelError: for an empty table, missing columns, or no "none" row.
    """
    if table.empty or not {"method", "brier"} <= set(table.columns):
        raise FinalModelError("The calibration table is empty or missing method/brier columns")
    plain = table[table["method"] == "none"]
    if plain.empty:
        raise FinalModelError("The calibration table must include the 'none' (uncalibrated) row")
    baseline = float(plain.iloc[0]["brier"])
    best = table.sort_values("brier").iloc[0]
    if best["method"] != "none" and baseline - float(best["brier"]) >= min_brier_gain:
        return str(best["method"])
    return "none"


def run_final_selection(
    cfg: dict[str, Any], logger: logging.Logger | None = None
) -> tuple[pd.DataFrame, str]:
    """Fit each calibration candidate on train, score on validation, choose, and save.

    Writes models/final_model.joblib, reports/calibration_comparison.csv,
    reports/final_model_choice.json and reports/figures/calibration_validation.png.

    Returns:
        (comparison_table, chosen_method)
    """
    log = logger or logging.getLogger(__name__)
    settings = load_final_settings(cfg)
    processed = cfg["paths"]["processed_dir"]
    train, val = load_split("train", processed), load_split("val", processed)
    target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]
    for label, frame in (("train", train), ("val", val)):
        for col in (target, id_col):
            if col not in frame.columns:
                raise FinalModelError(f"Column '{col}' not found in the {label} split")
    X_train, y_train = train.drop(columns=[target, id_col]), train[target]
    X_val, y_val = val.drop(columns=[target, id_col]), val[target]

    pos_weight = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    seed, n_splits = cfg["project"]["random_seed"], cfg["cv"]["n_splits"]
    bins = settings["calibration_bins"]

    rows: list[dict[str, Any]] = []
    tables: dict[str, pd.DataFrame] = {}
    fitted: dict[str, CalibratedModel] = {}
    for method in settings["calibration_candidates"]:
        base = build_pipeline(
            settings["model"], settings["strategy"], cfg, pos_weight=pos_weight, params=settings["params"] or None
        )
        model = CalibratedModel(base, method, n_splits=n_splits, seed=seed).fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        ranking = ranking_metrics(y_val, proba)
        rows.append(
            {
                "method": method,
                "pr_auc": ranking["pr_auc"],
                "roc_auc": ranking["roc_auc"],
                "brier": brier_score(y_val, proba),
                "ece": expected_calibration_error(y_val, proba, bins),
            }
        )
        tables[method] = calibration_table(y_val, proba, bins)
        fitted[method] = model
        log.info("calibration=%-9s pr_auc=%.4f brier=%.4f ece=%.4f", method, rows[-1]["pr_auc"], rows[-1]["brier"], rows[-1]["ece"])

    table = pd.DataFrame(rows)
    chosen = choose_calibration(table, settings["min_brier_gain"])
    log.info("Chosen calibration: %s (needs a Brier gain of at least %s over 'none')", chosen, settings["min_brier_gain"])

    reports, models_dir = Path(cfg["paths"]["reports_dir"]), Path(cfg["paths"]["models_dir"])
    choice = {
        "model": settings["model"],
        "strategy": settings["strategy"],
        "params": settings["params"],
        "calibration": chosen,
        "validation_metrics": {k: float(v) for k, v in table[table["method"] == chosen].iloc[0].items() if k != "method"},
        "train_rows": len(train),
        "val_rows": len(val),
        "seed": seed,
    }
    try:
        reports.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(reports / "calibration_comparison.csv", index=False)
        (reports / "final_model_choice.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
        joblib.dump(fitted[chosen], models_dir / "final_model.joblib")
    except OSError as exc:
        raise FinalModelError(f"Cannot write the final model or reports: {exc}") from exc
    plot_reliability(tables, reports / "figures" / "calibration_validation.png", "Calibration on the validation split")
    return table, chosen


def main() -> None:
    """Command-line entry point: `python -m src.final_model`."""
    try:
        cfg = load_config()
        log = get_logger("churn.final", log_file=cfg["paths"]["log_file"])
        table, chosen = run_final_selection(cfg, log)
        log.info("Calibration comparison (validation split):\n%s", table.round(4).to_string(index=False))
        log.info("Saved the frozen final model (calibration=%s) to %s", chosen, cfg["paths"]["models_dir"])
    except (
        ConfigError, SplitError, ModelError, MetricsError, FeatureError, CalibrationError, FinalModelError
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()