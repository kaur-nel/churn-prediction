"""Business impact: expected net benefit of a retention campaign, and the best threshold.

All money figures come from placeholder assumptions in configs/config.yaml (business section),
not from real company data. Everything is computed on the VALIDATION split with the frozen
final model. The test split is never opened here.

Money model, per customer targeted (predicted risk >= threshold):
  - we always pay `retention_offer_cost`;
  - if the customer really was going to churn, the offer works with probability
    `offer_success_rate` and saves `customer_ltv`;
  - if the customer was going to stay, the offer is wasted.
  net benefit = success_rate * LTV * (churners targeted) - cost * (customers targeted)
"""

import itertools
import json
import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.calibration import CalibrationError
from src.config import ConfigError, load_config
from src.features import FeatureError
from src.logger import get_logger
from src.split import SplitError, load_split

SENSITIVITY_KEYS = ("customer_ltv", "retention_offer_cost", "offer_success_rate")


class BusinessError(Exception):
    """Raised when business inputs are invalid or no profitable campaign exists."""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def validate_economics(customer_ltv: Any, retention_offer_cost: Any, offer_success_rate: Any) -> None:
    """Check the three money assumptions.

    Raises:
        BusinessError: naming the setting that is wrong.
    """
    if not _is_number(customer_ltv) or customer_ltv <= 0:
        raise BusinessError(f"customer_ltv must be a number > 0, got {customer_ltv!r}")
    if not _is_number(retention_offer_cost) or retention_offer_cost < 0:
        raise BusinessError(f"retention_offer_cost must be a number >= 0, got {retention_offer_cost!r}")
    if not _is_number(offer_success_rate) or not 0 <= offer_success_rate <= 1:
        raise BusinessError(f"offer_success_rate must be a number between 0 and 1, got {offer_success_rate!r}")


def break_even_probability(ltv: float, cost: float, rate: float) -> float:
    """Churn probability above which an offer is worth its cost: cost / (success_rate * LTV).

    Returns infinity when offers can never pay off (success rate 0). A value above 1 means
    even a certain churner is not worth the offer.
    """
    validate_economics(ltv, cost, rate)
    gain = ltv * rate
    return math.inf if gain == 0 else cost / gain


def load_business_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the `business` config section, including the sensitivity grid.

    Raises:
        BusinessError: naming the exact setting that is wrong.
    """
    business = cfg.get("business") if isinstance(cfg, dict) else None
    if not isinstance(business, dict):
        raise BusinessError("Config is missing the 'business' section")
    missing = [key for key in SENSITIVITY_KEYS if key not in business]
    if missing:
        raise BusinessError(f"Config section 'business' is missing keys: {missing}")
    base = {key: business[key] for key in SENSITIVITY_KEYS}
    validate_economics(**base)

    grid = business.get("sensitivity")
    if not isinstance(grid, dict):
        raise BusinessError("Config 'business.sensitivity' must be a mapping of value lists")
    clean: dict[str, list[float]] = {}
    for key in SENSITIVITY_KEYS:
        values = grid.get(key)
        if not isinstance(values, list) or not values:
            raise BusinessError(f"'business.sensitivity.{key}' must be a non-empty list of numbers")
        for value in values:
            validate_economics(**{**base, key: value})
        clean[key] = list(dict.fromkeys(values))
    return {**base, "sensitivity": clean}


def _validate_scores(y_true: Any, proba: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        y = np.asarray(y_true)
        p = np.asarray(proba, dtype=float)
    except (TypeError, ValueError) as exc:
        raise BusinessError(f"y_true and proba must be numeric arrays: {exc}") from exc
    if y.ndim != 1 or p.ndim != 1:
        raise BusinessError("y_true and proba must be 1-dimensional")
    if len(y) == 0:
        raise BusinessError("Empty input")
    if len(y) != len(p):
        raise BusinessError(f"Length mismatch: {len(y)} labels vs {len(p)} probabilities")
    labels = set(np.unique(y).tolist())
    if not labels <= {0, 1}:
        raise BusinessError("y_true must contain only the values 0 and 1")
    if labels != {0, 1}:
        raise BusinessError("y_true needs both classes (churners and non-churners)")
    if np.isnan(p).any():
        raise BusinessError("Probabilities contain NaN")
    if (p < 0).any() or (p > 1).any():
        raise BusinessError("Probabilities must be between 0 and 1")
    return y.astype(int), p


def _cumulative_counts(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """One row per possible threshold: how many customers it targets and how many are churners.

    The first row is "target nobody" (threshold NaN). Customers with identical scores are
    grouped, so a threshold never splits a tie.
    """
    order = np.argsort(-p, kind="stable")
    p_sorted, y_sorted = p[order], y[order]
    cumulative_tp = np.cumsum(y_sorted)
    group_ends = np.flatnonzero(np.append(p_sorted[1:] != p_sorted[:-1], True))
    return pd.DataFrame(
        {
            "threshold": np.concatenate([[np.nan], p_sorted[group_ends]]),
            "targeted": np.concatenate([[0], group_ends + 1]),
            "tp": np.concatenate([[0], cumulative_tp[group_ends]]),
        }
    )


def _apply_economics(counts: pd.DataFrame, n: int, total_positives: int, ltv: float, cost: float, rate: float) -> pd.DataFrame:
    out = counts.copy()
    out["fp"] = out["targeted"] - out["tp"]
    out["targeted_pct"] = 100.0 * out["targeted"] / n
    out["precision"] = out["tp"] / out["targeted"].where(out["targeted"] > 0)
    out["recall"] = out["tp"] / total_positives
    out["gross_savings"] = rate * ltv * out["tp"]
    out["offer_cost"] = cost * out["targeted"]
    out["net_benefit"] = out["gross_savings"] - out["offer_cost"]
    out["net_per_1000"] = out["net_benefit"] / n * 1000
    return out


def net_benefit_curve(y_true: Any, proba: Any, ltv: float, cost: float, rate: float) -> pd.DataFrame:
    """Net benefit for every possible threshold, from "target nobody" to "target everyone".

    Raises:
        BusinessError: for invalid scores or money assumptions.
    """
    y, p = _validate_scores(y_true, proba)
    validate_economics(ltv, cost, rate)
    return _apply_economics(_cumulative_counts(y, p), len(y), int(y.sum()), ltv, cost, rate)


def best_point(curve: pd.DataFrame) -> pd.Series:
    """The curve row with the highest net benefit (on ties, the smaller campaign).

    Raises:
        BusinessError: for an empty curve or one without a net_benefit column.
    """
    if curve.empty or "net_benefit" not in curve.columns:
        raise BusinessError("The curve is empty or has no net_benefit column")
    return curve.loc[curve["net_benefit"].idxmax()]


def choose_threshold(curve: pd.DataFrame) -> dict[str, Any]:
    """The threshold with the highest net benefit.

    Raises:
        BusinessError: if targeting nobody is the best option (no campaign is profitable).
    """
    best = best_point(curve)
    if int(best["targeted"]) == 0:
        raise BusinessError(
            "Targeting nobody gives the highest net benefit under these assumptions: no campaign is "
            "profitable. Check customer_ltv, retention_offer_cost and offer_success_rate"
        )
    return {key: (int(value) if key in ("targeted", "tp", "fp") else float(value)) for key, value in best.items()}


def evaluate_threshold(y_true: Any, proba: Any, threshold: float, ltv: float, cost: float, rate: float) -> dict[str, float]:
    """Net benefit of targeting every customer with a score >= `threshold`."""
    y, p = _validate_scores(y_true, proba)
    validate_economics(ltv, cost, rate)
    if not _is_number(threshold) or not 0 <= threshold <= 1:
        raise BusinessError(f"threshold must be a number between 0 and 1, got {threshold!r}")
    targeted_mask = p >= threshold
    targeted = int(targeted_mask.sum())
    tp = int(y[targeted_mask].sum())
    net = rate * ltv * tp - cost * targeted
    return {
        "threshold": float(threshold),
        "targeted": targeted,
        "tp": tp,
        "fp": targeted - tp,
        "net_benefit": float(net),
        "net_per_1000": float(net / len(y) * 1000),
    }


def sensitivity_table(y_true: Any, proba: Any, grid: dict[str, list[float]]) -> pd.DataFrame:
    """Best threshold and net benefit for every combination of LTV, offer cost and success rate.

    A row with best_threshold NaN and targeted_pct 0 means "no campaign is profitable".
    """
    y, p = _validate_scores(y_true, proba)
    missing = [key for key in SENSITIVITY_KEYS if key not in grid or not grid[key]]
    if missing:
        raise BusinessError(f"The sensitivity grid is missing values for: {missing}")
    counts = _cumulative_counts(y, p)
    n, positives = len(y), int(y.sum())

    rows = []
    for ltv, cost, rate in itertools.product(*(grid[key] for key in SENSITIVITY_KEYS)):
        validate_economics(ltv, cost, rate)
        curve = _apply_economics(counts, n, positives, ltv, cost, rate)
        best = best_point(curve)
        rows.append(
            {
                "customer_ltv": ltv,
                "retention_offer_cost": cost,
                "offer_success_rate": rate,
                "break_even_probability": break_even_probability(ltv, cost, rate),
                "best_threshold": float(best["threshold"]),
                "targeted_pct": float(best["targeted_pct"]),
                "net_per_1000": float(best["net_per_1000"]),
                "target_everyone_per_1000": float(curve.iloc[-1]["net_per_1000"]),
            }
        )
    return pd.DataFrame(rows)


def plot_net_benefit(curve: pd.DataFrame, best: pd.Series, path: str | Path, title: str) -> None:
    """Save net benefit per 1,000 customers against the share of customers targeted."""
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.plot(curve["targeted_pct"], curve["net_per_1000"], label="net benefit")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.scatter([best["targeted_pct"]], [best["net_per_1000"]], color="red", zorder=3, label="best threshold")
    ax.set_xlabel("Share of customers targeted (%)")
    ax.set_ylabel("Net benefit per 1,000 customers")
    ax.set_title(title)
    ax.legend()
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except OSError as exc:
        raise BusinessError(f"Cannot write the plot to {path}: {exc}") from exc


def run_business(cfg: dict[str, Any], logger: logging.Logger | None = None) -> dict[str, Any]:
    """Score the validation split with the frozen model, choose the threshold and save the reports.

    Writes reports/business_curve.csv, reports/sensitivity_table.csv, reports/threshold_choice.json
    and reports/figures/net_benefit_validation.png. Reads only the validation split.
    """
    log = logger or logging.getLogger(__name__)
    settings = load_business_settings(cfg)
    ltv, cost, rate = (settings[key] for key in SENSITIVITY_KEYS)

    val = load_split("val", cfg["paths"]["processed_dir"])
    target, id_col = cfg["data"]["target"], cfg["data"]["id_column"]
    for col in (target, id_col):
        if col not in val.columns:
            raise BusinessError(f"Column '{col}' not found in the validation split")

    model_path = Path(cfg["paths"]["models_dir"]) / "final_model.joblib"
    if not model_path.exists():
        raise BusinessError(f"Final model not found: {model_path}. Run `python -m src.final_model` first")
    try:
        model = joblib.load(model_path)
    except (OSError, EOFError, ValueError, ImportError, AttributeError, pickle.UnpicklingError) as exc:
        raise BusinessError(f"Could not load the final model from {model_path}: {exc}") from exc

    y = val[target]
    proba = model.predict_proba(val.drop(columns=[target, id_col]))[:, 1]

    curve = net_benefit_curve(y, proba, ltv, cost, rate)
    chosen = choose_threshold(curve)
    break_even = break_even_probability(ltv, cost, rate)
    at_break_even = evaluate_threshold(y, proba, min(break_even, 1.0), ltv, cost, rate)
    everyone = curve.iloc[-1]

    log.info("Assumptions (placeholders): LTV=%s, offer cost=%s, success rate=%s", ltv, cost, rate)
    log.info("Break-even churn probability (cost / (success_rate x LTV)): %.3f", break_even)
    log.info(
        "Best threshold %.3f: targets %.1f%% of customers, net benefit %.0f per 1,000 customers "
        "(precision %.3f, recall %.3f)",
        chosen["threshold"], chosen["targeted_pct"], chosen["net_per_1000"], chosen["precision"], chosen["recall"],
    )
    log.info("Net benefit at the break-even threshold: %.0f per 1,000 customers", at_break_even["net_per_1000"])
    log.info("Net benefit if we target everyone: %.0f per 1,000 customers; nobody: 0", everyone["net_per_1000"])

    sensitivity = sensitivity_table(y, proba, settings["sensitivity"])

    reports = Path(cfg["paths"]["reports_dir"])
    choice = {
        "threshold": chosen["threshold"],
        "chosen_on": f"validation split ({len(val)} customers)",
        "targeted_pct": chosen["targeted_pct"],
        "expected_net_per_1000_customers": chosen["net_per_1000"],
        "precision": chosen["precision"],
        "recall": chosen["recall"],
        "break_even_probability": break_even,
        "net_per_1000_at_break_even": at_break_even["net_per_1000"],
        "assumptions": {"customer_ltv": ltv, "retention_offer_cost": cost, "offer_success_rate": rate},
        "note": "Money figures use placeholder assumptions, not real company data.",
    }
    try:
        reports.mkdir(parents=True, exist_ok=True)
        curve.round(4).to_csv(reports / "business_curve.csv", index=False)
        sensitivity.round(4).to_csv(reports / "sensitivity_table.csv", index=False)
        (reports / "threshold_choice.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    except OSError as exc:
        raise BusinessError(f"Cannot write the business reports to {reports}: {exc}") from exc
    plot_net_benefit(curve, best_point(curve), reports / "figures" / "net_benefit_validation.png", "Net benefit (validation split)")
    return {"chosen": chosen, "curve": curve, "sensitivity": sensitivity}


def main() -> None:
    """Command-line entry point: `python -m src.business`."""
    try:
        cfg = load_config()
        log = get_logger("churn.business", log_file=cfg["paths"]["log_file"])
        result = run_business(cfg, log)
        log.info("Sensitivity table (every row is a different set of assumptions):\n%s", result["sensitivity"].round(3).to_string(index=False))
    except (ConfigError, SplitError, FeatureError, CalibrationError, BusinessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()