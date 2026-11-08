"""Cross-validate all four models on the TRAIN split and track every run in MLflow."""

import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import ConfigError, load_config
from src.features import FeatureError
from src.logger import get_logger
from src.metrics import (
    MetricsError,
    best_f1_threshold,
    cross_val_probabilities,
    ranking_metrics,
    threshold_metrics,
)
from src.models import MODEL_NAMES, ModelError, build_pipeline
from src.split import SplitError, load_split

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "churn-prediction"
# SMOTE was dropped after Step 7: no ranking gain over the other strategies.
COMPARE_STRATEGIES = ("baseline", "class_weight")
NON_METRIC_COLUMNS = ("model", "strategy")


class CompareError(Exception):
    """Raised when the model comparison cannot run."""


def tracking_settings() -> tuple[str, str]:
    """Return (tracking_uri, experiment_name) from the environment, with safe defaults."""
    uri = os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    name = os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT
    return uri, name


class RunTracker:
    """Small wrapper that writes one MLflow run per (model, strategy)."""

    def __init__(self, tracking_uri: str, experiment_name: str, logger: logging.Logger | None = None) -> None:
        self.log = logger or logging.getLogger(__name__)
        self.client = MlflowClient(tracking_uri=tracking_uri)
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            self.experiment_id = (
                experiment.experiment_id if experiment else self.client.create_experiment(experiment_name)
            )
        except MlflowException as exc:
            raise CompareError(f"Cannot use MLflow at {tracking_uri}: {exc}") from exc

    def log_run(self, run_name: str, params: dict[str, Any], metrics: dict[str, float]) -> bool:
        """Record one run. Returns False (and logs a warning) if MLflow fails."""
        try:
            run = self.client.create_run(self.experiment_id, run_name=run_name)
            run_id = run.info.run_id
            for key, value in params.items():
                self.client.log_param(run_id, key, str(value))
            for key, value in metrics.items():
                self.client.log_metric(run_id, key, float(value))
            self.client.set_terminated(run_id)
        except MlflowException as exc:
            self.log.warning("MLflow could not record run '%s': %s", run_name, exc)
            return False
        return True


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
    models: Sequence[str] = MODEL_NAMES,
    strategies: Sequence[str] = COMPARE_STRATEGIES,
    tracker: RunTracker | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Cross-validate every (model, strategy) pair; returns one row each, best PR-AUC first.

    PR-AUC and ROC-AUC use pooled out-of-fold predictions. `*_best` columns use the threshold
    with the highest F1 on those same predictions (optimistic: a comparison aid only).

    Raises:
        CompareError: for empty model/strategy lists or a target with only one class.
    """
    log = logger or logging.getLogger(__name__)
    if not models or not strategies:
        raise CompareError("Need at least one model and one strategy")
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise CompareError("The target needs both classes, but only one was found")
    pos_weight = n_neg / n_pos
    seed, n_splits = cfg["project"]["random_seed"], cfg["cv"]["n_splits"]

    rows = []
    for model_name in models:
        for strategy in strategies:
            pipeline = build_pipeline(model_name, strategy, cfg, pos_weight=pos_weight)
            started = time.perf_counter()
            oof, fold_scores = cross_val_probabilities(pipeline, X, y, n_splits, seed)
            seconds = time.perf_counter() - started

            ranking = ranking_metrics(y, oof)
            best_t, _ = best_f1_threshold(y, oof)
            at_best = threshold_metrics(y, oof, best_t)
            row = {
                "model": model_name,
                "strategy": strategy,
                "pr_auc": ranking["pr_auc"],
                "pr_auc_fold_mean": float(pd.Series(fold_scores).mean()),
                "pr_auc_fold_std": float(pd.Series(fold_scores).std(ddof=0)),
                "roc_auc": ranking["roc_auc"],
                "best_threshold": best_t,
                "precision_best": at_best["precision"],
                "recall_best": at_best["recall"],
                "f1_best": at_best["f1"],
                "fit_seconds": seconds,
            }
            rows.append(row)
            log.info(
                "%-13s %-12s pr_auc=%.3f (+/-%.3f) roc_auc=%.3f  %.1fs",
                model_name, strategy, row["pr_auc"], row["pr_auc_fold_std"], row["roc_auc"], seconds,
            )

            if tracker is not None:
                params: dict[str, Any] = {
                    "model": model_name,
                    "strategy": strategy,
                    "n_splits": n_splits,
                    "seed": seed,
                    "train_rows": len(y),
                    "pos_weight": round(pos_weight, 4),
                }
                params.update({f"param.{key}": value for key, value in cfg["models"][model_name].items()})
                metrics = {key: value for key, value in row.items() if key not in NON_METRIC_COLUMNS}
                tracker.log_run(f"{model_name}-{strategy}", params, metrics)

    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def top_models(table: pd.DataFrame, k: int = 2) -> list[tuple[str, str]]:
    """Best (model, strategy) for each model, ranked by PR-AUC; returns the first `k` models.

    Raises:
        CompareError: if k < 1 or the table is empty or lacks the needed columns.
    """
    if k < 1:
        raise CompareError(f"k must be at least 1, got {k}")
    if table.empty or not {"model", "strategy", "pr_auc"} <= set(table.columns):
        raise CompareError("The comparison table is empty or missing model/strategy/pr_auc columns")
    best = table.sort_values("pr_auc", ascending=False).drop_duplicates("model").head(k)
    return list(zip(best["model"], best["strategy"], strict=True))


def run_compare(
    cfg: dict[str, Any], logger: logging.Logger | None = None, track: bool = True
) -> pd.DataFrame:
    """Load ONLY the train split, compare all models and save reports/model_comparison.csv."""
    log = logger or logging.getLogger(__name__)
    train = load_split("train", cfg["paths"]["processed_dir"])
    target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]
    for col in (target, id_col):
        if col not in train.columns:
            raise CompareError(f"Column '{col}' not found in the train split")

    tracker = RunTracker(*tracking_settings(), logger=log) if track else None
    table = evaluate_models(
        train.drop(columns=[target, id_col]), train[target], cfg, tracker=tracker, logger=log
    )

    out_dir = Path(cfg["paths"]["reports_dir"])
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(out_dir / "model_comparison.csv", index=False)
    except OSError as exc:
        raise CompareError(f"Cannot write the comparison table to {out_dir}: {exc}") from exc
    return table


def main() -> None:
    """Command-line entry point: `python -m src.compare`."""
    try:
        cfg = load_config()
        log = get_logger("churn.compare", log_file=cfg["paths"]["log_file"])
        table = run_compare(cfg, log)
        log.info("Model comparison (train CV only):\n%s", table.round(3).to_string(index=False))
        log.info("Top 2 models to tune in Step 9 (model, strategy): %s", top_models(table, 2))
    except (ConfigError, SplitError, ModelError, MetricsError, FeatureError, CompareError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()