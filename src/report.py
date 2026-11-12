"""Generate docs/model_card.md and docs/report.md from the saved result files.

Every number in the documents is read from reports/*.json and reports/*.csv; nothing is typed
by hand, so the documents cannot drift away from the results. Wording that depends on the
results (ties, tuning, fairness flags) is chosen by simple, documented rules. The output contains
no "generated at" timestamp, so running the command twice gives identical files.
"""

import importlib.metadata
import json
import logging
import math
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ConfigError, load_config
from src.evaluate import EvaluationError, file_sha256
from src.features import (
    EXCLUDED_FEATURES,
    NEW_BINARY,
    NEW_CATEGORICAL,
    NEW_NUMERIC,
    RAW_BINARY,
    RAW_CATEGORICAL,
    RAW_NUMERIC,
)
from src.logger import get_logger

MODEL_LABELS = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
STRATEGY_LABELS = {
    "baseline": "no imbalance handling",
    "class_weight": "class weights",
    "smote": "SMOTE",
}

# key -> (file name, command that produces it)
JSON_INPUTS = {
    "choice": ("final_model_choice.json", "src.final_model"),
    "threshold": ("threshold_choice.json", "src.business"),
    "test": ("test_results.json", "src.evaluate"),
}
JSON_KEYS = {
    "choice": ["model", "strategy", "params", "calibration", "validation_metrics", "train_rows", "val_rows", "seed"],
    "threshold": ["threshold", "assumptions", "expected_net_per_1000_customers", "break_even_probability", "targeted_pct"],
    "test": [
        "evaluated_at_utc", "test_rows", "test_churn_rate", "metrics", "confusion_matrix",
        "expected_calibration_error", "net_per_1000_target_everyone", "model_sha256", "bootstrap",
    ],
}
METRIC_KEYS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "targeted_pct", "net_per_1000"]
ASSUMPTION_KEYS = ["customer_ltv", "retention_offer_cost", "offer_success_rate"]

# key -> (file name, command that produces it, required columns)
CSV_INPUTS = {
    "comparison": (
        "model_comparison.csv", "src.compare", ["model", "strategy", "pr_auc", "pr_auc_fold_std", "roc_auc"]
    ),
    "imbalance": ("imbalance_comparison.csv", "src.imbalance", ["model", "strategy", "pr_auc", "roc_auc"]),
    "tuning": (
        "tuning_results.csv", "src.tune",
        ["model", "strategy", "cv_pr_auc_default", "cv_pr_auc_default_std", "cv_pr_auc_tuned",
         "val_pr_auc_default", "val_pr_auc_tuned"],
    ),
    "calibration": ("calibration_comparison.csv", "src.final_model", ["method", "pr_auc", "roc_auc", "brier", "ece"]),
    "sensitivity": (
        "sensitivity_table.csv", "src.business",
        ["customer_ltv", "retention_offer_cost", "offer_success_rate", "break_even_probability",
         "best_threshold", "targeted_pct", "net_per_1000", "target_everyone_per_1000"],
    ),
    "fairness": (
        "fairness_by_group.csv", "src.evaluate",
        ["attribute", "group", "customers", "positives", "churn_rate", "mean_predicted", "selection_rate",
         "precision", "recall", "false_positive_rate", "roc_auc", "reliable"],
    ),
    "fairness_gaps": (
        "fairness_gaps.csv", "src.evaluate",
        ["attribute", "groups_compared", "recall_gap", "false_positive_rate_gap", "selection_rate_gap"],
    ),
    "shap_groups": ("shap_group_importance.csv", "src.explain", ["group", "mean_abs_shap", "share_pct"]),
    "shap_profiles": (
        "shap_driver_profiles.csv", "src.explain",
        ["group", "value", "customers", "mean_shap", "odds_multiplier", "observed_churn_rate"],
    ),
}
OPTIONAL_CSV_INPUTS = {
    "vif": ("vif_table.csv", "src.features", ["feature", "vif"]),
    "drift": ("drift_validation_vs_train.csv", "src.drift", ["feature", "kind", "psi", "status"]),
}
GAP_COLUMNS = {
    "recall_gap": "recall",
    "false_positive_rate_gap": "false-positive rate",
    "selection_rate_gap": "share flagged",
}
PACKAGES = ("scikit-learn", "pandas", "numpy", "imbalanced-learn", "xgboost", "lightgbm", "optuna", "shap", "mlflow")


class ReportError(Exception):
    """Raised when the documents cannot be generated."""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


# ---------- reading the result files ---------------------------------------------------------


def _read_json(path: Path, command: str) -> dict[str, Any]:
    if not path.exists():
        raise ReportError(f"{path.name} not found in {path.parent}: run `python -m {command}` first")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"Could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReportError(f"{path.name} must contain a JSON object")
    return data


def _read_csv(path: Path, command: str, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise ReportError(f"{path.name} not found in {path.parent}: run `python -m {command}` first")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
        raise ReportError(f"Could not read {path}: {exc}") from exc
    if frame.empty:
        raise ReportError(f"{path.name} has no rows: re-run `python -m {command}`")
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ReportError(f"{path.name} is missing columns {missing}: re-run `python -m {command}`")
    return frame


def _require_keys(mapping: Any, keys: Sequence[str], where: str) -> None:
    if not isinstance(mapping, dict):
        raise ReportError(f"{where} must be a JSON object")
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ReportError(f"{where} is missing keys {missing}: re-run the command that produces it")


def _validate_nested(data: dict[str, Any]) -> None:
    _require_keys(data["threshold"]["assumptions"], ASSUMPTION_KEYS, "threshold_choice.json 'assumptions'")
    test = data["test"]
    _require_keys(test["metrics"], METRIC_KEYS, "test_results.json 'metrics'")
    for name in METRIC_KEYS:
        _require_keys(test["metrics"][name], ["estimate", "ci_low", "ci_high"], f"test_results.json metric '{name}'")
    _require_keys(test["confusion_matrix"], ["tn", "fp", "fn", "tp"], "test_results.json 'confusion_matrix'")
    _require_keys(test["bootstrap"], ["n_resamples", "ci_level"], "test_results.json 'bootstrap'")
    if not _is_number(test["bootstrap"]["ci_level"]):
        raise ReportError("test_results.json 'bootstrap.ci_level' must be a number")
    if not isinstance(data["choice"]["validation_metrics"], dict):
        raise ReportError("final_model_choice.json 'validation_metrics' must be a JSON object")


def load_results(reports_dir: str | Path) -> dict[str, Any]:
    """Read and validate every result file the documents need.

    Optional files (VIF and drift tables) are returned as None when absent.

    Raises:
        ReportError: naming the missing or invalid file and the command that produces it.
    """
    folder = Path(reports_dir)
    data: dict[str, Any] = {}
    for key, (filename, command) in JSON_INPUTS.items():
        data[key] = _read_json(folder / filename, command)
        _require_keys(data[key], JSON_KEYS[key], filename)
    _validate_nested(data)
    for key, (filename, command, columns) in CSV_INPUTS.items():
        data[key] = _read_csv(folder / filename, command, columns)
    for key, (filename, command, columns) in OPTIONAL_CSV_INPUTS.items():
        path = folder / filename
        data[key] = _read_csv(path, command, columns) if path.exists() else None
    return data


def load_report_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the `report` config section.

    Raises:
        ReportError: naming the setting that is wrong.
    """
    section = cfg.get("report") if isinstance(cfg, dict) else None
    if not isinstance(section, dict):
        raise ReportError("Config is missing the 'report' section")
    docs_dir = section.get("docs_dir")
    if not isinstance(docs_dir, str) or not docs_dir.strip():
        raise ReportError(f"'report.docs_dir' must be a non-empty string, got {docs_dir!r}")
    limit = section.get("fairness_gap_flag")
    if not _is_number(limit) or limit <= 0:
        raise ReportError(f"'report.fairness_gap_flag' must be a number > 0, got {limit!r}")
    return {"docs_dir": docs_dir, "fairness_gap_flag": float(limit)}


# ---------- formatting -----------------------------------------------------------------------


def fmt(value: Any, digits: int = 3) -> str:
    """A number with fixed decimals; 'n/a' for missing, NaN or infinite values."""
    if value is None or isinstance(value, bool):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def pct(fraction: Any, digits: int = 1) -> str:
    """A fraction (0.25) as a percentage ('25.0%'); 'n/a' for missing values."""
    try:
        number = float(fraction)
    except (TypeError, ValueError):
        return "n/a"
    return f"{100 * number:.{digits}f}%" if math.isfinite(number) else "n/a"


def money(value: Any) -> str:
    """A whole number with thousands separators; 'n/a' for missing values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:,.0f}" if math.isfinite(number) else "n/a"


def count(value: Any) -> str:
    """A whole number with thousands separators."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """Markdown table lines. Cells are escaped so a '|' or a line break cannot break the table.

    Raises:
        ReportError: for no columns or a row of the wrong length.
    """
    if not headers:
        raise ReportError("A table needs at least one column")
    lines = ["| " + " | ".join(_cell(h) for h in headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        if len(row) != len(headers):
            raise ReportError(f"A table row has {len(row)} cells but the table has {len(headers)} columns")
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return lines


def _model_label(name: Any) -> str:
    return MODEL_LABELS.get(str(name), str(name))


def _strategy_label(name: Any) -> str:
    return STRATEGY_LABELS.get(str(name), str(name))


def _code_list(items: Sequence[str]) -> str:
    return ", ".join(f"`{item}`" for item in items) if items else "none"


def _ci(test: dict[str, Any], name: str, digits: int = 3) -> str:
    entry = test["metrics"][name]
    level = 100 * float(test["bootstrap"]["ci_level"])
    return f"{fmt(entry['estimate'], digits)} ({level:.0f}% CI {fmt(entry['ci_low'], digits)} to {fmt(entry['ci_high'], digits)})"


def _money_ci(test: dict[str, Any], name: str) -> str:
    entry = test["metrics"][name]
    level = 100 * float(test["bootstrap"]["ci_level"])
    return f"{money(entry['estimate'])} ({level:.0f}% CI {money(entry['ci_low'])} to {money(entry['ci_high'])})"


# ---------- rule-based statements (pure functions, tested separately) -------------------------


def best_per_model(comparison: pd.DataFrame) -> pd.DataFrame:
    """The best (highest PR-AUC) row for each model, best model first."""
    return comparison.sort_values("pr_auc", ascending=False).drop_duplicates("model").reset_index(drop=True)


def tied_models(comparison: pd.DataFrame) -> tuple[list[str], float]:
    """Models within one fold standard deviation of the best model's PR-AUC (a rough guide, not a test).

    Returns:
        (tied model names, best-first; the margin used)
    """
    best = best_per_model(comparison)
    top = best.iloc[0]
    margin = float(top["pr_auc_fold_std"])
    if not math.isfinite(margin):
        margin = 0.0
    tied = [str(row["model"]) for _, row in best.iterrows() if float(top["pr_auc"]) - float(row["pr_auc"]) <= margin]
    return tied, margin


def tuning_verdicts(tuning: pd.DataFrame) -> list[dict[str, Any]]:
    """Per model: validation PR-AUC before and after tuning, and whether the gain exceeds the CV noise."""
    rows = []
    for _, row in tuning.iterrows():
        delta = float(row["val_pr_auc_tuned"]) - float(row["val_pr_auc_default"])
        noise = float(row["cv_pr_auc_default_std"])
        rows.append(
            {
                "model": str(row["model"]),
                "strategy": str(row["strategy"]),
                "default": float(row["val_pr_auc_default"]),
                "tuned": float(row["val_pr_auc_tuned"]),
                "delta": delta,
                "noise": noise,
                "meaningful": bool(delta > noise),
            }
        )
    return rows


def fairness_flags(gaps: pd.DataFrame, limit: float) -> list[tuple[str, list[str]]]:
    """Attributes where any group gap (recall, false-positive rate, share flagged) exceeds `limit`."""
    flags = []
    for _, row in gaps.iterrows():
        over = [label for column, label in GAP_COLUMNS.items() if pd.notna(row[column]) and float(row[column]) > limit]
        if over:
            flags.append((str(row["attribute"]), over))
    return flags


def ordering_differs(rows: pd.DataFrame) -> bool:
    """True if ranking the values by model effect (mean SHAP) differs from ranking them by observed churn rate."""
    if len(rows) < 2:
        return False
    by_model = list(rows.sort_values("mean_shap", ascending=False, kind="stable")["value"].astype(str))
    by_data = list(rows.sort_values("observed_churn_rate", ascending=False, kind="stable")["value"].astype(str))
    return by_model != by_data


def sensitivity_summary(sensitivity: pd.DataFrame) -> dict[str, int]:
    """Scenario counts: all, offering-everyone loses money, and no campaign is profitable."""
    return {
        "scenarios": len(sensitivity),
        "everyone_loses": int((sensitivity["target_everyone_per_1000"] < 0).sum()),
        "no_campaign": int((sensitivity["targeted_pct"] <= 0).sum()),
    }


def base_ltv_rows(sensitivity: pd.DataFrame, ltv: float) -> pd.DataFrame:
    """Sensitivity rows at the base customer value (all rows if that value is not in the grid)."""
    rows = sensitivity[np.isclose(sensitivity["customer_ltv"].astype(float), float(ltv))]
    return rows if not rows.empty else sensitivity


def imbalance_spread(imbalance: pd.DataFrame) -> tuple[float, float]:
    """(largest PR-AUC spread between strategies for any one model, largest fold std in the table)."""
    pivot = imbalance.pivot_table(index="model", columns="strategy", values="pr_auc")
    spread = float((pivot.max(axis=1) - pivot.min(axis=1)).max())
    noise = float(imbalance["pr_auc_fold_std"].max()) if "pr_auc_fold_std" in imbalance.columns else float("nan")
    return spread, noise


def model_file_warning(cfg: dict[str, Any], test: dict[str, Any]) -> str | None:
    """A warning if the model file on disk is not the one evaluated on the test split, else None."""
    path = Path(cfg["paths"]["models_dir"]) / "final_model.joblib"
    if not path.exists():
        return None
    try:
        current = file_sha256(path)
    except EvaluationError as exc:
        return str(exc)
    if current != test["model_sha256"]:
        return (
            "The model file on disk differs from the one evaluated on the test split (SHA-256 mismatch): "
            "the documents describe the evaluated model, not this file"
        )
    return None


def software_versions() -> dict[str, str]:
    """Python and key package versions of the machine generating the documents."""
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


# ---------- shared sections ------------------------------------------------------------------


def _metrics_table(test: dict[str, Any]) -> list[str]:
    level = 100 * float(test["bootstrap"]["ci_level"])
    rows = []
    for key, label, digits in (
        ("pr_auc", "PR-AUC", 3),
        ("roc_auc", "ROC-AUC", 3),
        ("precision", "Precision at the threshold", 3),
        ("recall", "Recall at the threshold", 3),
        ("f1", "F1 at the threshold", 3),
        ("brier", "Brier score (lower is better)", 3),
        ("targeted_pct", "Share of customers flagged (%)", 1),
    ):
        entry = test["metrics"][key]
        rows.append([label, fmt(entry["estimate"], digits), f"{fmt(entry['ci_low'], digits)} to {fmt(entry['ci_high'], digits)}"])
    net = test["metrics"]["net_per_1000"]
    rows.append(
        ["Net benefit per 1,000 customers (placeholder money)", money(net["estimate"]), f"{money(net['ci_low'])} to {money(net['ci_high'])}"]
    )
    return markdown_table(["Metric (test split)", "Estimate", f"{level:.0f}% confidence interval"], rows)


def _confusion_table(test: dict[str, Any]) -> list[str]:
    cm = test["confusion_matrix"]
    return markdown_table(
        ["", "Predicted: stays", "Predicted: churns"],
        [["Actually stays", count(cm["tn"]), count(cm["fp"])], ["Actually churns", count(cm["fn"]), count(cm["tp"])]],
    )


def _calibration_lines(d: dict[str, Any]) -> list[str]:
    rows = [
        [str(r["method"]), fmt(r["pr_auc"], 4), fmt(r["roc_auc"], 4), fmt(r["brier"], 4), fmt(r["ece"], 4)]
        for _, r in d["calibration"].iterrows()
    ]
    return markdown_table(["Calibration method", "PR-AUC", "ROC-AUC", "Brier", "ECE"], rows)


def _fairness_lines(d: dict[str, Any], settings: dict[str, Any]) -> list[str]:
    rows = [
        [
            r["attribute"], r["group"], count(r["customers"]), count(r["positives"]), pct(int(r["positives"]) / int(r["customers"])),
            fmt(r["mean_predicted"]), pct(r["selection_rate"]), fmt(r["recall"]), fmt(r["false_positive_rate"]),
            fmt(r["roc_auc"]), "yes" if bool(r["reliable"]) else "NO (too few customers)",
        ]
        for _, r in d["fairness"].iterrows()
    ]
    lines = markdown_table(
        ["Attribute", "Group", "Customers", "Churners", "Actual churn rate", "Mean predicted risk", "Flagged",
         "Recall", "False-positive rate", "ROC-AUC", "Reliable"],
        rows,
    )
    gap_rows = [
        [r["attribute"], count(r["groups_compared"]), fmt(r["recall_gap"]), fmt(r["false_positive_rate_gap"]), fmt(r["selection_rate_gap"])]
        for _, r in d["fairness_gaps"].iterrows()
    ]
    lines += ["", "Largest difference between reliable groups:", ""]
    lines += markdown_table(
        ["Attribute", "Groups compared", "Recall gap", "False-positive-rate gap", "Share-flagged gap"], gap_rows
    )
    flags = fairness_flags(d["fairness_gaps"], settings["fairness_gap_flag"])
    lines.append("")
    if flags:
        for attribute, over in flags:
            lines.append(
                f"- **Flagged:** `{attribute}` has gaps above {fmt(settings['fairness_gap_flag'], 2)} in: {', '.join(over)}. "
                "Check the group table above to see which group is affected and whether the actual churn rates explain it."
            )
    else:
        lines.append(f"- No attribute has a gap above {fmt(settings['fairness_gap_flag'], 2)}.")
    lines.append(
        "- Groups with too few customers are marked unreliable and left out of the gaps. Fairness was checked for these "
        "attributes only; other groups and combinations were not examined."
    )
    return lines


def _driver_lines(d: dict[str, Any], top_n: int) -> list[str]:
    groups = d["shap_groups"].sort_values("mean_abs_shap", ascending=False).head(top_n)
    lines = markdown_table(
        ["Rank", "Driver (group of related columns)", "Mean |SHAP|", "Share of total importance"],
        [[i + 1, r["group"], fmt(r["mean_abs_shap"]), f"{fmt(r['share_pct'], 1)}%"] for i, (_, r) in enumerate(groups.iterrows())],
    )
    lines += ["", "What the model does with each value (odds multiplier compared with an average customer; checked against the actual churn rate):", ""]
    for group in groups["group"]:
        rows = d["shap_profiles"][d["shap_profiles"]["group"] == group].sort_values("mean_shap", ascending=False)
        for _, r in rows.iterrows():
            lines.append(
                f"- `{group}` = {r['value']}: odds of churn x{fmt(r['odds_multiplier'], 2)}; "
                f"{pct(r['observed_churn_rate'])} actually churned ({count(r['customers'])} customers)"
            )
        if not rows.empty and ordering_differs(rows):
            lines.append(
                f"  - Note: for `{group}` the model's ordering of values differs from the observed churn ordering "
                "(small groups and overlapping features can cause this; do not over-read this group)."
            )
    return lines


# ---------- the model card -------------------------------------------------------------------


def build_model_card(
    d: dict[str, Any], cfg: dict[str, Any], settings: dict[str, Any], versions: dict[str, str] | None = None
) -> str:
    """The model card as Markdown text."""
    choice, threshold, test = d["choice"], d["threshold"], d["test"]
    assumptions = threshold["assumptions"]
    drop = list(cfg.get("models", {}).get("logreg_drop_columns", [])) if choice["model"] == "logreg" else []
    params = choice["params"]
    top_n = int(cfg.get("explainability", {}).get("top_n", 5))
    churners = int(test["confusion_matrix"]["fn"]) + int(test["confusion_matrix"]["tp"])
    total = int(choice["train_rows"]) + int(choice["val_rows"]) + int(test["test_rows"])

    lines: list[str] = []
    add = lines.append
    add("# Model card: telecom customer churn model")
    add("")
    add("> Generated by `python -m src.report` from the files in `reports/`. Re-run the command instead of editing by hand.")
    add("")
    add("## Model details")
    add("")
    add(f"- **Model:** {_model_label(choice['model'])}; imbalance strategy: {_strategy_label(choice['strategy'])}")
    add(f"- **Probability calibration:** `{choice['calibration']}` (`none` means the model's own probabilities are used)")
    add(
        "- **Parameters:** library defaults plus the values in `configs/config.yaml` (no tuned values adopted)"
        if not params
        else f"- **Parameters:** `{json.dumps(params, sort_keys=True)}`"
    )
    add(f"- **Customers:** {count(total)} in total: {count(choice['train_rows'])} train, {count(choice['val_rows'])} validation, {count(test['test_rows'])} test")
    add(f"- **Random seed:** {choice['seed']}")
    add(f"- **Test-split evaluation:** {test['evaluated_at_utc']} (evaluation run number {int(test.get('evaluation_count', 1))})")
    add(f"- **Model file fingerprint (SHA-256):** `{test['model_sha256']}`")
    if versions:
        add("- **Software used to generate this card:** " + ", ".join(f"{name} {version}" for name, version in versions.items()))
    add("")
    add("## Intended use")
    add("")
    add("- Rank existing telecom customers by estimated risk of leaving, to help a retention team decide who receives a retention offer.")
    add("- Use with human oversight and a measured campaign (see `docs/report.md`), not as an automatic decision.")
    add("")
    add("## Out-of-scope use")
    add("")
    add("- Decisions about a person's access to service, pricing or credit.")
    add("- Other companies, countries or time periods without re-validation: the model learned from one public example dataset.")
    add("- Explaining *why* customers leave. The SHAP explanations describe what the model does, not causes.")
    add("- Using the money figures as forecasts: they rest on placeholder assumptions (see below).")
    add("")
    add("## Data")
    add("")
    add("- **Source:** the IBM Telco Customer Churn sample dataset (as distributed on Kaggle).")
    add(
        "- **Provenance not verified:** we do not know how it was collected, which period it covers, or whether it "
        "describes real customers. Treat it as a public example dataset."
    )
    add(f"- **Test split:** {count(test['test_rows'])} customers, {count(churners)} churners ({pct(test['test_churn_rate'])}). Splits are stratified by churn.")
    add(
        "- **Cleaning (see `src/data.py`):** spaces trimmed; a blank `TotalCharges` for a customer with tenure 0 set to 0 "
        "(an assumption); \"No internet service\" and \"No phone service\" merged into \"No\"; exact duplicate rows dropped; "
        "values outside the expected ranges or categories rejected."
    )
    add("")
    add("## Features")
    add("")
    numeric = [c for c in [*RAW_NUMERIC, *NEW_NUMERIC] if c not in drop]
    binary = [c for c in [*RAW_BINARY, *NEW_BINARY] if c not in drop]
    categorical = [c for c in [*RAW_CATEGORICAL, *NEW_CATEGORICAL] if c not in drop]
    add(f"- **Numeric:** {_code_list(numeric)}")
    add(f"- **Yes/no flags:** {_code_list(binary)}")
    add(f"- **Categories (one-hot encoded):** {_code_list(categorical)}")
    add(f"- **Left out on purpose:** {_code_list(EXCLUDED_FEATURES)} (no signal in the exploratory analysis and a protected attribute; kept in the data only for the fairness check), the customer ID and the target.")
    if drop:
        add(f"- **Dropped for this model only:** {_code_list(drop)} (nearly redundant with tenure and monthly charges; see the variance-inflation check).")
    add("- `SeniorCitizen` **is** a model feature.")
    add("- Engineered features are defined in `src/features.py`; none is computed from the target.")
    add("")
    add("## Training and selection procedure")
    add("")
    add(f"1. Stratified {cfg['split']['train_size']:.0%}/{cfg['split']['val_size']:.0%}/{cfg['split']['test_size']:.0%} split before any fitting; imputing, scaling and encoding are fitted on training data only.")
    add(f"2. Imbalance strategies ({', '.join(_strategy_label(s) for s in sorted(d['imbalance']['strategy'].unique()))}) and four model families compared with {cfg['cv']['n_splits']}-fold cross-validation on the training split.")
    add(f"3. The two best models were tuned with Optuna ({cfg['tuning']['n_trials']} trials each) and checked on the validation split.")
    add("4. Calibration methods were compared on the validation split, and the one with the best Brier score was kept only if it beat the uncalibrated model by the configured margin.")
    add("5. The threshold was chosen on the validation split by maximising expected net benefit.")
    add("6. The test split was then scored **once** with the frozen model and threshold.")
    add("")
    add("## Performance on the test split")
    add("")
    add(f"Threshold used: {fmt(test['threshold'] if 'threshold' in test else threshold['threshold'])} (chosen on validation). Confidence intervals: {int(test['bootstrap']['n_resamples'])} bootstrap resamples.")
    add("")
    lines += _metrics_table(test)
    add("")
    lines += _confusion_table(test)
    add("")
    add(f"- A model guessing at random would have a PR-AUC equal to the churn rate ({fmt(test['test_churn_rate'])}).")
    add(f"- Expected calibration error on the test split: {fmt(test['expected_calibration_error'], 4)}.")
    val = choice["validation_metrics"]
    add(
        f"- For reference, validation (used for choices, so slightly optimistic): PR-AUC {fmt(val.get('pr_auc'))}, "
        f"ROC-AUC {fmt(val.get('roc_auc'))}, Brier {fmt(val.get('brier'), 4)}, ECE {fmt(val.get('ece'), 4)}."
    )
    if int(test.get("evaluation_count", 1)) > 1:
        add(f"- **Note:** the test split was evaluated {int(test['evaluation_count'])} times. Only the first evaluation is an untouched estimate.")
    add("")
    add("### Calibration candidates (validation split)")
    add("")
    lines += _calibration_lines(d)
    add("")
    add("## Business assumptions (placeholders, not company data)")
    add("")
    lines += markdown_table(
        ["Assumption", "Value"],
        [
            ["Customer lifetime value", money(assumptions["customer_ltv"])],
            ["Cost of one retention offer", money(assumptions["retention_offer_cost"])],
            ["Offer success rate (share of would-be churners kept)", pct(assumptions["offer_success_rate"], 0)],
            ["Break-even churn probability", fmt(threshold["break_even_probability"])],
        ],
    )
    add("")
    add("## Explainability (validation split)")
    add("")
    lines += _driver_lines(d, top_n)
    add("")
    add("SHAP values are additive and are summed per group of related columns. They describe the model, not causes.")
    add("")
    add("## Fairness (test split)")
    add("")
    lines += _fairness_lines(d, settings)
    add("")
    add("## Monitoring")
    add("")
    drift_cfg = cfg.get("drift", {})
    add(
        "- Feature drift is checked with the Population Stability Index (`python -m src.drift`). Rule-of-thumb limits in the config: "
        f"noticeable shift at {fmt(drift_cfg.get('moderate_threshold'), 2)}, big shift at {fmt(drift_cfg.get('major_threshold'), 2)}."
    )
    if d["drift"] is not None:
        status = ", ".join(f"{n} {name}" for name, n in d["drift"]["status"].value_counts().items())
        worst = d["drift"].sort_values("psi", ascending=False).iloc[0]
        add(f"- Validation vs training (a sanity check on random splits): {status}; largest PSI {fmt(worst['psi'], 4)} (`{worst['feature']}`).")
    add("- Scoring new data: `python -m src.predict --input customers.csv`. There is no automatic retraining or live monitoring.")
    add("")
    if d["vif"] is not None:
        add("## Multicollinearity check (VIF)")
        add("")
        top = d["vif"].sort_values("vif", ascending=False).head(3)
        add("Highest variance-inflation factors on the training split (above about 5 to 10 means heavy overlap with other columns): " + ", ".join(f"`{r['feature']}` {fmt(r['vif'], 2)}" for _, r in top.iterrows()) + ".")
        add("")
    add("## Limitations")
    add("")
    add("- The money figures depend on three assumed numbers. The offer success rate in particular must be measured with a controlled test.")
    add("- A churn score says who is likely to leave, not who will respond to an offer.")
    add(f"- One dataset of unknown provenance; no information about when customers left. The test split holds {count(churners)} churners, so scores carry the uncertainty shown by the confidence intervals.")
    add("- The threshold was chosen on a single validation split; the exact optimum is uncertain.")
    add("- SHAP explains the model, not the world; overlapping features share credit, and small groups give noisy effects.")
    add("- Fairness was checked for the listed attributes on one test split; any flagged gap above should be reviewed before deployment.")
    add("")
    add("## Reproduce")
    add("")
    add("```")
    add("python -m src.data && python -m src.split && python -m src.final_model")
    add("python -m src.business && python -m src.evaluate && python -m src.explain && python -m src.report")
    add("```")
    add("")
    return "\n".join(lines)


# ---------- the final report -----------------------------------------------------------------


def build_report(d: dict[str, Any], cfg: dict[str, Any], settings: dict[str, Any]) -> str:
    """The final report as Markdown text."""
    choice, threshold, test = d["choice"], d["threshold"], d["test"]
    assumptions = threshold["assumptions"]
    metrics = test["metrics"]
    top_n = int(cfg.get("explainability", {}).get("top_n", 5))
    churners = int(test["confusion_matrix"]["fn"]) + int(test["confusion_matrix"]["tp"])
    total = int(choice["train_rows"]) + int(choice["val_rows"]) + int(test["test_rows"])
    everyone = float(test["net_per_1000_target_everyone"])
    net = metrics["net_per_1000"]

    lines: list[str] = []
    add = lines.append
    add("# Customer churn prediction: final report")
    add("")
    add("> Generated by `python -m src.report` from the files in `reports/`. Re-run the command instead of editing by hand.")
    add("")
    add("## Headline")
    add("")
    add(f"- **Final model:** {_model_label(choice['model'])} ({_strategy_label(choice['strategy'])}; calibration: `{choice['calibration']}`). Chosen on validation data and scored once on the test split.")
    add(f"- **Ranking quality (test):** PR-AUC {_ci(test, 'pr_auc')}, ROC-AUC {_ci(test, 'roc_auc')}. A random guess has a PR-AUC of {fmt(test['test_churn_rate'])}.")
    add(
        f"- **At the chosen threshold ({fmt(threshold['threshold'])}):** {fmt(metrics['targeted_pct']['estimate'], 1)}% of customers are flagged; "
        f"recall {_ci(test, 'recall')}, precision {_ci(test, 'precision')}."
    )
    add(
        f"- **Expected savings (placeholder assumptions):** {_money_ci(test, 'net_per_1000')} per 1,000 customers, "
        f"against {money(everyone)} if every customer received an offer and 0 for no campaign."
    )
    add("- These money figures rest on **assumed** numbers (customer value, offer cost, offer success rate), not on measured results.")
    add("")
    add("## Problem")
    add("")
    add("A telecom company loses customers (churn). Retention offers cost money, and only help customers who would have left. The question: which customers should receive an offer, and does a targeted campaign beat the alternatives?")
    add("")
    add("## Data")
    add("")
    add(f"- IBM Telco Customer Churn sample dataset (Kaggle): {count(total)} customers, split into {count(choice['train_rows'])} train, {count(choice['val_rows'])} validation and {count(test['test_rows'])} test (stratified).")
    add(f"- Test split churn rate: {pct(test['test_churn_rate'])} ({count(churners)} churners).")
    add("- Provenance, collection method and time period are not verified; treat the dataset as a public example.")
    add("")
    add("## Approach and findings")
    add("")
    add("### Class imbalance")
    add("")
    pivot = d["imbalance"].pivot_table(index="model", columns="strategy", values="pr_auc")
    strategies = list(pivot.columns)
    lines += markdown_table(
        ["Model (PR-AUC, 5-fold CV on train)", *[_strategy_label(s) for s in strategies]],
        [[_model_label(m), *[fmt(pivot.loc[m, s]) for s in strategies]] for m in pivot.index],
    )
    spread, noise = imbalance_spread(d["imbalance"])
    add("")
    if math.isfinite(noise) and spread <= noise:
        add(f"For every model, the strategies differ by at most {fmt(spread)} PR-AUC, which is within the fold-to-fold noise (up to {fmt(noise)}). No strategy clearly improved ranking quality.")
    else:
        add(f"The strategies differ by up to {fmt(spread)} PR-AUC for a single model (fold-to-fold noise up to {fmt(noise)}): at least one difference may be real. See the table.")
    add("")
    add("### Model comparison")
    add("")
    best = best_per_model(d["comparison"])
    lines += markdown_table(
        ["Model", "Best strategy", "PR-AUC", "Fold std", "ROC-AUC"],
        [[_model_label(r["model"]), _strategy_label(r["strategy"]), fmt(r["pr_auc"], 4), fmt(r["pr_auc_fold_std"], 4), fmt(r["roc_auc"], 4)] for _, r in best.iterrows()],
    )
    tied, margin = tied_models(d["comparison"])
    add("")
    add(
        f"Within one fold standard deviation ({fmt(margin)}) of the best model: {', '.join(_model_label(m) for m in tied)}. "
        "This is a rough guide, not a formal test, so these models are treated as statistically tied."
    )
    add(f"The final model is {_model_label(choice['model'])}; the choice among tied models is a judgment call (simplicity and explainability).")
    add("")
    add("### Tuning (Optuna)")
    add("")
    verdicts = tuning_verdicts(d["tuning"])
    lines += markdown_table(
        ["Model", "Validation PR-AUC (defaults)", "Validation PR-AUC (tuned)", "Change", "CV noise (std)", "Meaningful gain?"],
        [[_model_label(v["model"]), fmt(v["default"]), fmt(v["tuned"]), f"{v['delta']:+.3f}", fmt(v["noise"]), "yes" if v["meaningful"] else "no"] for v in verdicts],
    )
    add("")
    if any(v["meaningful"] for v in verdicts):
        add("Tuning improved at least one model by more than the cross-validation noise.")
    else:
        add("Tuning did not improve any model by more than the cross-validation noise, so default settings were kept.")
    add("")
    add("### Calibration")
    add("")
    lines += _calibration_lines(d)
    add("")
    add(f"The kept method is `{choice['calibration']}`. A method is adopted only if it beats the uncalibrated model's Brier score by the configured margin.")
    add("")
    add("## Results on the test split")
    add("")
    lines += _metrics_table(test)
    add("")
    lines += _confusion_table(test)
    add("")
    add(f"- Expected calibration error: {fmt(test['expected_calibration_error'], 4)}.")
    add(f"- Bootstrap: {int(test['bootstrap']['n_resamples'])} resamples.")
    add(f"- The model file fingerprint (SHA-256) is `{test['model_sha256']}`.")
    if int(test.get("evaluation_count", 1)) > 1:
        add(f"- **Note:** the test split was evaluated {int(test['evaluation_count'])} times; only the first run is an untouched estimate.")
    add("")
    add("## Business impact (placeholder assumptions)")
    add("")
    add(
        f"Assumed: customer lifetime value {money(assumptions['customer_ltv'])}, offer cost {money(assumptions['retention_offer_cost'])}, "
        f"offer success rate {pct(assumptions['offer_success_rate'], 0)}. None of these is company data."
    )
    add("")
    add(f"- An offer pays for itself when the churn probability is above {fmt(threshold['break_even_probability'])} (offer cost / (success rate x lifetime value)).")
    add(f"- The threshold {fmt(threshold['threshold'])} was chosen on the validation split by maximising net benefit.")
    if "net_per_1000_at_break_even" in threshold:
        add(f"- On validation, net benefit at the break-even probability was {money(threshold['net_per_1000_at_break_even'])} per 1,000 customers, against {money(threshold['expected_net_per_1000_customers'])} at the chosen threshold.")
    within = float(net["ci_low"]) <= float(threshold["expected_net_per_1000_customers"]) <= float(net["ci_high"])
    add(
        f"- Validation expected {money(threshold['expected_net_per_1000_customers'])}; the test result is {money(net['estimate'])}, "
        f"which is {'inside' if within else 'outside'} the test confidence interval."
    )
    if float(net["ci_low"]) > everyone:
        add(f"- The lower end of the test interval ({money(net['ci_low'])}) is above the point estimate for offering everyone ({money(everyone)}). This is an unpaired comparison, not a formal test.")
    else:
        add(f"- The test interval ({money(net['ci_low'])} to {money(net['ci_high'])}) overlaps the point estimate for offering everyone ({money(everyone)}), so the advantage of targeting is not clear from this evidence alone.")
    add("")
    add("### Sensitivity")
    add("")
    summary = sensitivity_summary(d["sensitivity"])
    add(
        f"Across {summary['scenarios']} scenarios, offering every customer loses money in {summary['everyone_loses']}, "
        f"and no campaign at all is profitable in {summary['no_campaign']}. Rows at the assumed customer value are shown; the full grid is in `reports/sensitivity_table.csv`."
    )
    add("")
    rows = base_ltv_rows(d["sensitivity"], assumptions["customer_ltv"])
    lines += markdown_table(
        ["Customer value", "Offer cost", "Success rate", "Break-even probability", "Best threshold", "Flagged", "Net / 1,000 (best)", "Net / 1,000 (everyone)"],
        [
            [money(r["customer_ltv"]), money(r["retention_offer_cost"]), pct(r["offer_success_rate"], 0), fmt(r["break_even_probability"], 4),
             fmt(r["best_threshold"], 4), f"{fmt(r['targeted_pct'], 1)}%", money(r["net_per_1000"]), money(r["target_everyone_per_1000"])]
            for _, r in rows.iterrows()
        ],
    )
    add("")
    add("## Why customers are flagged (SHAP)")
    add("")
    lines += _driver_lines(d, top_n)
    add("")
    add("SHAP explains what the model does. The associations are not proof of cause, and overlapping features share credit.")
    add("")
    add("## Fairness")
    add("")
    lines += _fairness_lines(d, settings)
    add("")
    add("## Recommendation")
    add("")
    add("This is the analyst's recommendation, and it depends on the placeholder assumptions above.")
    add("")
    add(
        f"1. **Run a controlled pilot,** not a full rollout. Flag customers with a score of at least {fmt(threshold['threshold'])} "
        f"(about {fmt(metrics['targeted_pct']['estimate'], 1)}% of customers) and keep a randomly chosen group of flagged customers untreated."
    )
    add(f"2. **Measure the real offer success rate.** The assumed {pct(assumptions['offer_success_rate'], 0)} drives every money figure; replace it with the measured value and re-run the business step.")
    flags = fairness_flags(d["fairness_gaps"], settings["fairness_gap_flag"])
    if flags:
        for attribute, over in flags:
            add(f"3. **Review the offer design for `{attribute}` groups** before launch: gaps in {', '.join(over)} exceed {fmt(settings['fairness_gap_flag'], 2)}.")
    else:
        add("3. **Keep monitoring group-level performance:** no attribute exceeded the fairness gap limit on the test split.")
    drift_cfg = cfg.get("drift", {})
    add(f"4. **Monitor drift** with `python -m src.drift` on each new batch; investigate any feature at or above PSI {fmt(drift_cfg.get('major_threshold'), 2)} and consider retraining.")
    add("5. **Do not change the model or threshold because of the test results.** Any improvement should be developed on validation data and then evaluated on fresh data.")
    add("")
    add("## Limitations and next steps")
    add("")
    add("- All money figures rest on assumptions. A churn score does not say who will respond to an offer.")
    add(f"- One dataset of unknown provenance with no time information; the test split holds {count(churners)} churners, so the confidence intervals are wide.")
    add("- The threshold was chosen on a single validation split. Net benefit is usually flat near the optimum, and the sensitivity table shows how the best threshold moves with the assumptions.")
    add("- Next steps: measure the offer effect with a randomized test, add behavioural or time-based features if available, consider uplift modelling (who responds to an offer), and define a retraining schedule.")
    add("")
    return "\n".join(lines)


# ---------- running ----------------------------------------------------------------------------


def run_report(cfg: dict[str, Any], logger: logging.Logger | None = None) -> dict[str, Path]:
    """Read the result files and write docs/model_card.md and docs/report.md.

    Raises:
        ReportError: for missing or invalid result files, invalid settings, or an unwritable folder.
    """
    log = logger or logging.getLogger(__name__)
    settings = load_report_settings(cfg)
    data = load_results(cfg["paths"]["reports_dir"])
    warning = model_file_warning(cfg, data["test"])
    if warning:
        log.warning(warning)

    card = build_model_card(data, cfg, settings, software_versions())
    report = build_report(data, cfg, settings)

    docs = Path(settings["docs_dir"])
    paths = {"model_card": docs / "model_card.md", "report": docs / "report.md"}
    try:
        docs.mkdir(parents=True, exist_ok=True)
        paths["model_card"].write_text(card, encoding="utf-8")
        paths["report"].write_text(report, encoding="utf-8")
    except OSError as exc:
        raise ReportError(f"Cannot write the documents to {docs}: {exc}") from exc
    log.info("Wrote %s and %s", paths["model_card"], paths["report"])
    return paths


def main() -> None:
    """Command-line entry point: `python -m src.report`."""
    try:
        cfg = load_config()
        log = get_logger("churn.report", log_file=cfg["paths"]["log_file"])
        run_report(cfg, log)
    except (ConfigError, EvaluationError, ReportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()