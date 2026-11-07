"""Compare ways of handling class imbalance with cross-validation on the TRAIN split only."""

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ConfigError, load_config
from src.features import FeatureError
from src.logger import get_logger
from src.metrics import MetricsError, best_f1_threshold, cross_val_probabilities, ranking_metrics, threshold_metrics
from src.models import STRATEGIES, ModelError, build_pipeline
from src.split import SplitError, load_split

# One linear model and one tree model. The other two models are compared in Step 8.
COMPARED_MODELS = ("logreg", "lightgbm")


class ImbalanceError(Exception):
    """Raised when the imbalance comparison cannot run."""


def compare_strategies(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
    models: Sequence[str] = COMPARED_MODELS,
    strategies: Sequence[str] = STRATEGIES,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Cross-validate every (model, strategy) pair and return one row of metrics per pair.

    Threshold-free metrics (PR-AUC, ROC-AUC) are the fair comparison. "t50" columns use a
    0.5 threshold; "best" columns use the threshold with the highest F1 on the same
    out-of-fold predictions (a comparison aid only; the real threshold is chosen later on
    the validation split).
    """
    log = logger or logging.getLogger(__name__)
    if not models or not strategies:
        raise ImbalanceError("Need at least one model and one strategy")
    seed, n_splits = cfg["project"]["random_seed"], cfg["cv"]["n_splits"]
    log.info("Random-guess PR-AUC equals the churn rate: %.3f", float(y.mean()))

    rows = []
    for model_name in models:
        for strategy in strategies:
            pipeline = build_pipeline(model_name, strategy, cfg)
            oof, fold_scores = cross_val_probabilities(pipeline, X, y, n_splits, seed)
            ranking = ranking_metrics(y, oof)
            at_half = threshold_metrics(y, oof, 0.5)
            best_t, _ = best_f1_threshold(y, oof)
            at_best = threshold_metrics(y, oof, best_t)
            rows.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "pr_auc": ranking["pr_auc"],
                    "pr_auc_fold_std": float(np.std(fold_scores)),
                    "roc_auc": ranking["roc_auc"],
                    "precision_t50": at_half["precision"],
                    "recall_t50": at_half["recall"],
                    "f1_t50": at_half["f1"],
                    "best_threshold": best_t,
                    "precision_best": at_best["precision"],
                    "recall_best": at_best["recall"],
                    "f1_best": at_best["f1"],
                }
            )
            log.info("%-8s %-12s pr_auc=%.3f roc_auc=%.3f", model_name, strategy, ranking["pr_auc"], ranking["roc_auc"])
    return pd.DataFrame(rows)


def run_comparison(cfg: dict[str, Any], logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load ONLY the train split, run the comparison and save reports/imbalance_comparison.csv."""
    train = load_split("train", cfg["paths"]["processed_dir"])
    target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]
    for col in (target, id_col):
        if col not in train.columns:
            raise ImbalanceError(f"Column '{col}' not found in the train split")

    table = compare_strategies(train.drop(columns=[target, id_col]), train[target], cfg, logger=logger)

    out_dir = Path(cfg["paths"]["reports_dir"])
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(out_dir / "imbalance_comparison.csv", index=False)
    except OSError as exc:
        raise ImbalanceError(f"Cannot write the comparison table to {out_dir}: {exc}") from exc
    return table


def main() -> None:
    """Command-line entry point: `python -m src.imbalance`."""
    try:
        cfg = load_config()
        log = get_logger("churn.imbalance", log_file=cfg["paths"]["log_file"])
        table = run_comparison(cfg, log)
        log.info("Comparison (train CV only):\n%s", table.round(3).to_string(index=False))
    except (ConfigError, SplitError, ModelError, MetricsError, FeatureError, ImbalanceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()