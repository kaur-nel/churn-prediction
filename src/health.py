"""Health check: can this installation run the project?

Checks that the configuration is valid, that the core libraries import (this catches a missing system
library such as libgomp, which LightGBM and XGBoost need on Linux), and that the output folders are writable.
With --require-model it also loads the saved model and scores one sample customer.

Exit code 0 = healthy, 1 = unhealthy; a JSON summary is printed either way. Used as the Docker HEALTHCHECK.
SHAP, MLflow and Optuna are deliberately not imported here: they are slow to load and the check runs often.
"""

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ConfigError, load_config, validate_config
from src.evaluate import EvaluationError, load_model
from src.features import FeatureError

REQUIRED_MODULES = (
    "numpy", "pandas", "scipy", "sklearn", "imblearn", "joblib", "lightgbm", "xgboost",
    "statsmodels", "matplotlib", "yaml", "dotenv",
)

# One made-up customer in the raw format, used to prove that a saved model can score.
SAMPLE_CUSTOMER = {
    "tenure": 10, "MonthlyCharges": 50.0, "TotalCharges": 500.0, "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
}


class HealthError(Exception):
    """Raised when the model cannot score the sample customer."""


def _result(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "ok": bool(ok), "detail": detail}


def check_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Is the configuration valid?"""
    try:
        validate_config(cfg)
    except ConfigError as exc:
        return _result("config", False, str(exc))
    return _result("config", True, "valid")


def check_imports() -> dict[str, Any]:
    """Do all core libraries import? (A missing system library shows up here.)"""
    failed = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except (ImportError, OSError, ValueError) as exc:
            failed.append(f"{name}: {exc}")
    if failed:
        return _result("imports", False, "; ".join(failed))
    return _result("imports", True, f"{len(REQUIRED_MODULES)} modules import")


def check_writable(cfg: dict[str, Any]) -> dict[str, Any]:
    """Can the project write to its data, model, report and log folders?"""
    paths = cfg["paths"]
    folders = [
        Path(paths["processed_dir"]), Path(paths["models_dir"]), Path(paths["reports_dir"]), Path(paths["log_file"]).parent
    ]
    problems = []
    for folder in folders:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=folder):
                pass
        except OSError as exc:
            problems.append(f"{folder}: {exc}")
    if problems:
        return _result("writable", False, "; ".join(problems))
    return _result("writable", True, f"{len(folders)} folders writable")


def score_sample(model: Any) -> float:
    """Score the sample customer with `model`.

    Raises:
        HealthError: if scoring fails or the result is not a single probability between 0 and 1.
    """
    try:
        proba = np.asarray(model.predict_proba(pd.DataFrame([SAMPLE_CUSTOMER]))[:, 1], dtype=float)
    except (ValueError, KeyError, TypeError, AttributeError, FeatureError) as exc:
        raise HealthError(f"The model could not score the sample customer: {exc}") from exc
    if proba.shape != (1,) or not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise HealthError("The model returned an invalid probability for the sample customer")
    return float(proba[0])


def check_model(cfg: dict[str, Any]) -> dict[str, Any]:
    """Does the saved final model load and score the sample customer?"""
    path = Path(cfg["paths"]["models_dir"]) / "final_model.joblib"
    try:
        model = load_model(path)
        probability = score_sample(model)
    except (EvaluationError, HealthError) as exc:
        return _result("model", False, str(exc))
    return _result("model", True, f"loaded {path.name}; the sample customer scores {probability:.3f}")


def check_health(cfg: dict[str, Any], require_model: bool = False) -> dict[str, Any]:
    """Run every check. The folder and model checks are skipped if the configuration is invalid."""
    checks = [check_config(cfg), check_imports()]
    if checks[0]["ok"]:
        checks.append(check_writable(cfg))
        if require_model:
            checks.append(check_model(cfg))
    return {"healthy": all(check["ok"] for check in checks), "checks": checks}


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: `python -m src.health [--require-model]`."""
    parser = argparse.ArgumentParser(description="Check that this installation can run the churn project.")
    parser.add_argument("--require-model", action="store_true", help="also load the saved model and score a sample")
    args = parser.parse_args(argv)
    try:
        cfg = load_config()
    except ConfigError as exc:
        report = {"healthy": False, "checks": [_result("config", False, str(exc)), check_imports()]}
    else:
        report = check_health(cfg, args.require_model)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["healthy"] else 1)


if __name__ == "__main__":
    main()