"""Tune the best models with Optuna, using cross-validation on the TRAIN split only.

After tuning, each tuned model and its default twin are scored once on the VALIDATION split
as an honesty check (the best-of-N CV score is optimistically biased). The test split is
never read here.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.compare import RunTracker, tracking_settings
from src.config import ConfigError, load_config
from src.features import FeatureError
from src.logger import get_logger
from src.metrics import MetricsError, cross_val_probabilities, ranking_metrics
from src.models import MODEL_NAMES, ModelError, build_pipeline
from src.split import SplitError, load_split

SPACE_TYPES = ("int", "float", "categorical")


class TuneError(Exception):
    """Raised when tuning cannot run or produces no usable result."""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_search_space(space: Any, model_name: str) -> None:
    """Check a model's search space from the config.

    Raises:
        TuneError: naming the model and the bad parameter.
    """
    if not isinstance(space, dict) or not space:
        raise TuneError(f"Bad search space for '{model_name}': must be a non-empty mapping")
    for name, spec in space.items():
        where = f"Bad search space for '{model_name}', parameter '{name}'"
        if not isinstance(spec, dict) or spec.get("type") not in SPACE_TYPES:
            raise TuneError(f"{where}: needs a 'type' of {SPACE_TYPES}")
        if spec["type"] == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or len(choices) < 2:
                raise TuneError(f"{where}: 'choices' must be a list of at least 2 values")
            continue
        low, high = spec.get("low"), spec.get("high")
        if not (_is_number(low) and _is_number(high)):
            raise TuneError(f"{where}: 'low' and 'high' must be numbers")
        if low >= high:
            raise TuneError(f"{where}: 'low' must be smaller than 'high'")
        if spec["type"] == "int" and not (isinstance(low, int) and isinstance(high, int)):
            raise TuneError(f"{where}: an int parameter needs whole-number bounds")
        if spec.get("log") and low <= 0:
            raise TuneError(f"{where}: a log scale needs 'low' > 0")


def suggest_params(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    """Ask Optuna for one set of parameter values inside the search space."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
        else:
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def strategy_for_model(table: pd.DataFrame, model_name: str) -> str:
    """The best imbalance strategy for `model_name` according to the Step 8 comparison table.

    Raises:
        TuneError: if the table lacks the needed columns or the model.
    """
    if not {"model", "strategy", "pr_auc"} <= set(table.columns):
        raise TuneError("The comparison table must have model, strategy and pr_auc columns")
    rows = table[table["model"] == model_name]
    if rows.empty:
        raise TuneError(f"Model '{model_name}' was not found in the comparison table; run `python -m src.compare`")
    return str(rows.sort_values("pr_auc", ascending=False).iloc[0]["strategy"])


def tune_model(
    model_name: str,
    strategy: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run an Optuna study maximising out-of-fold PR-AUC on the given (train) data.

    Returns a dict with: best_params, cv_default (PR-AUC with the config defaults on the same
    folds), cv_default_std, cv_tuned (best trial's PR-AUC) and n_completed.

    Raises:
        TuneError: for a missing/invalid search space or if no trial completes.
    """
    log = logger or logging.getLogger(__name__)
    space = cfg["tuning"]["search_space"].get(model_name)
    if space is None:
        raise TuneError(f"No search space defined for '{model_name}' in tuning.search_space")
    validate_search_space(space, model_name)

    seed, n_splits, n_trials = cfg["project"]["random_seed"], cfg["cv"]["n_splits"], cfg["tuning"]["n_trials"]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise TuneError("The target needs both classes, but only one was found")
    pos_weight = n_neg / n_pos

    def cv_score(params: dict[str, Any]) -> tuple[float, float]:
        pipeline = build_pipeline(model_name, strategy, cfg, pos_weight=pos_weight, params=params)
        oof, fold_scores = cross_val_probabilities(pipeline, X, y, n_splits, seed)
        return ranking_metrics(y, oof)["pr_auc"], float(np.std(fold_scores))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, space)
        try:
            return cv_score(params)[0]
        except MetricsError as exc:
            log.warning("%s trial %d failed and was skipped: %s", model_name, trial.number + 1, exc)
            raise optuna.TrialPruned() from exc

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        value = "skipped" if trial.value is None else f"{trial.value:.4f}"
        try:
            best = f"{study.best_value:.4f}"
        except ValueError:
            best = "n/a"
        log.info("%s trial %d/%d: pr_auc=%s (best so far %s)", model_name, trial.number + 1, n_trials, value, best)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise TuneError(f"No trial completed for '{model_name}'; check the search space and the data")

    default_score, default_std = cv_score({})
    return {
        "best_params": dict(study.best_params),
        "cv_default": default_score,
        "cv_default_std": default_std,
        "cv_tuned": float(study.best_value),
        "n_completed": len(completed),
    }


def validation_scores(
    pipeline: Any, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> dict[str, float]:
    """Fit on the train split, then return PR-AUC and ROC-AUC on the validation split."""
    try:
        pipeline.fit(X_train, y_train)
    except ValueError as exc:
        raise TuneError(f"Fitting failed: {exc}") from exc
    return ranking_metrics(y_val, pipeline.predict_proba(X_val)[:, 1])


def run_tuning(cfg: dict[str, Any], logger: logging.Logger | None = None, track: bool = True) -> pd.DataFrame:
    """Tune every model in `tuning.models`, check on validation, and save the results.

    Writes reports/tuning_results.csv and reports/best_params.json. Reads only the train
    and validation splits.
    """
    log = logger or logging.getLogger(__name__)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    models = cfg["tuning"]["models"]
    if not models:
        raise TuneError("tuning.models must name at least one model")
    unknown = [name for name in models if name not in MODEL_NAMES]
    if unknown:
        raise TuneError(f"Unknown model(s) in tuning.models: {unknown}; choose from {MODEL_NAMES}")

    reports = Path(cfg["paths"]["reports_dir"])
    comparison_path = reports / "model_comparison.csv"
    if not comparison_path.exists():
        raise TuneError(f"Model comparison table not found: {comparison_path}. Run `python -m src.compare` first")
    comparison = pd.read_csv(comparison_path)

    processed = cfg["paths"]["processed_dir"]
    train, val = load_split("train", processed), load_split("val", processed)
    target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]
    for label, frame in (("train", train), ("val", val)):
        for col in (target, id_col):
            if col not in frame.columns:
                raise TuneError(f"Column '{col}' not found in the {label} split")
    X_train, y_train = train.drop(columns=[target, id_col]), train[target]
    X_val, y_val = val.drop(columns=[target, id_col]), val[target]
    pos_weight = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)

    tracker = RunTracker(*tracking_settings(), logger=log) if track else None
    rows: list[dict[str, Any]] = []
    best_params_out: dict[str, Any] = {}

    for model_name in models:
        strategy = strategy_for_model(comparison, model_name)
        log.info("Tuning %s (strategy=%s, %d trials)...", model_name, strategy, cfg["tuning"]["n_trials"])
        result = tune_model(model_name, strategy, X_train, y_train, cfg, log)
        best = result["best_params"]

        default_val = validation_scores(
            build_pipeline(model_name, strategy, cfg, pos_weight=pos_weight), X_train, y_train, X_val, y_val
        )
        tuned_val = validation_scores(
            build_pipeline(model_name, strategy, cfg, pos_weight=pos_weight, params=best),
            X_train, y_train, X_val, y_val,
        )
        row = {
            "model": model_name,
            "strategy": strategy,
            "trials_completed": result["n_completed"],
            "cv_pr_auc_default": result["cv_default"],
            "cv_pr_auc_default_std": result["cv_default_std"],
            "cv_pr_auc_tuned": result["cv_tuned"],
            "val_pr_auc_default": default_val["pr_auc"],
            "val_pr_auc_tuned": tuned_val["pr_auc"],
            "val_roc_auc_default": default_val["roc_auc"],
            "val_roc_auc_tuned": tuned_val["roc_auc"],
        }
        rows.append(row)
        best_params_out[model_name] = {"strategy": strategy, "params": best}
        log.info("%s best parameters: %s", model_name, best)

        if tracker is not None:
            params = {"model": model_name, "strategy": strategy, "n_trials": cfg["tuning"]["n_trials"]}
            params.update({f"param.{key}": value for key, value in best.items()})
            metrics = {key: value for key, value in row.items() if key not in ("model", "strategy")}
            tracker.log_run(f"{model_name}-{strategy}-tuned", params, metrics)

    table = pd.DataFrame(rows)
    try:
        reports.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(reports / "tuning_results.csv", index=False)
        (reports / "best_params.json").write_text(json.dumps(best_params_out, indent=2), encoding="utf-8")
    except OSError as exc:
        raise TuneError(f"Cannot write tuning results to {reports}: {exc}") from exc
    return table


def main() -> None:
    """Command-line entry point: `python -m src.tune`."""
    try:
        cfg = load_config()
        log = get_logger("churn.tune", log_file=cfg["paths"]["log_file"])
        table = run_tuning(cfg, log)
        log.info("Tuning results (CV on train; val = honesty check):\n%s", table.round(3).to_string(index=False))
    except (ConfigError, SplitError, ModelError, MetricsError, FeatureError, TuneError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()