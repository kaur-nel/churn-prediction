"""Reproduce the final model from the raw data: clean, split, fit the final model, choose the threshold.

Equivalent to running, in order: python -m src.data, src.split, src.final_model and src.business.
The test split is written by the split step but never read here: the one-time test evaluation stays a
separate, deliberate step (python -m src.evaluate).

Re-fitting replaces the model file, so the command refuses to run when reports/test_results.json exists
(the current model was already evaluated on the test split) unless --force is given.
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.business import BusinessError, run_business
from src.calibration import CalibrationError
from src.config import ConfigError, load_config
from src.data import DataValidationError, prepare_data
from src.features import FeatureError
from src.final_model import FinalModelError, run_final_selection
from src.logger import get_logger
from src.metrics import MetricsError
from src.models import ModelError
from src.split import SplitError, run_split

LOCK_FILE = "test_results.json"


class TrainError(Exception):
    """Raised when training cannot start."""


def check_ready(cfg: dict[str, Any], force: bool = False) -> None:
    """Check that the raw data exists and that retraining will not replace an already-evaluated model.

    Raises:
        TrainError: if the raw CSV is missing, or the test split was already evaluated and `force` is False.
    """
    raw = Path(cfg["paths"]["raw_data"])
    if not raw.is_file():
        raise TrainError(
            f"Raw data not found: {raw}. Download 'Telco Customer Churn' (WA_Fn-UseC_-Telco-Customer-Churn.csv) "
            f"from Kaggle and save it as {raw}"
        )
    lock = Path(cfg["paths"]["reports_dir"]) / LOCK_FILE
    if lock.exists() and not force:
        raise TrainError(
            f"The test split was already evaluated ({lock} exists). Re-training would replace the evaluated model "
            "and its reported results would no longer describe it. Use --force if you really want to retrain, "
            "or run inside Docker with fresh volumes"
        )


def _timed(log: logging.Logger, name: str, step: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    result = step()
    log.info("Finished: %s (%.1fs)", name, time.perf_counter() - started)
    return result


def run_train(cfg: dict[str, Any], logger: logging.Logger | None = None, force: bool = False) -> dict[str, Any]:
    """Clean the raw data, split it, fit the final model and choose the threshold on validation data.

    Raises:
        TrainError: if training cannot start; otherwise the error of the failing step.
    """
    log = logger or logging.getLogger(__name__)
    check_ready(cfg, force)
    log.info("Training from %s", cfg["paths"]["raw_data"])

    clean = _timed(log, "clean the raw data", lambda: prepare_data(cfg, log))
    splits = _timed(log, "split into train/validation/test", lambda: run_split(cfg, log))
    _, calibration = _timed(log, "fit the final model", lambda: run_final_selection(cfg, log))
    business = _timed(log, "choose the threshold on the validation split", lambda: run_business(cfg, log))

    result = {
        "rows": len(clean),
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "calibration": calibration,
        "threshold": business["chosen"]["threshold"],
        "model_path": str(Path(cfg["paths"]["models_dir"]) / "final_model.joblib"),
    }
    log.info(
        "Done: %d customers, calibration=%s, threshold=%.4f, model saved to %s",
        result["rows"], calibration, result["threshold"], result["model_path"],
    )
    return result


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: `python -m src.train [--force]`."""
    parser = argparse.ArgumentParser(description="Reproduce the final model from the raw data.")
    parser.add_argument("--force", action="store_true", help="retrain even though the test split was already evaluated")
    args = parser.parse_args(argv)
    try:
        cfg = load_config()
        log = get_logger("churn.train", log_file=cfg["paths"]["log_file"])
        run_train(cfg, log, force=args.force)
    except (
        ConfigError, DataValidationError, SplitError, ModelError, MetricsError, FeatureError,
        CalibrationError, FinalModelError, BusinessError, TrainError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()