import json
import math

import joblib
import numpy as np
import pytest

from src.business import (
    BusinessError,
    best_point,
    break_even_probability,
    choose_threshold,
    evaluate_threshold,
    load_business_settings,
    net_benefit_curve,
    plot_net_benefit,
    run_business,
    sensitivity_table,
)
from src.calibration import CalibratedModel
from src.models import build_pipeline
from src.split import SplitError
from tests.test_models import TEST_CFG, make_xy

Y = [1, 1, 0, 0, 1, 0]
P = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1]
ECON = {"ltv": 1000.0, "cost": 50.0, "rate": 0.3}

GOOD_BUSINESS = {
    "customer_ltv": 2000.0,
    "retention_offer_cost": 50.0,
    "offer_success_rate": 0.3,
    "sensitivity": {
        "customer_ltv": [1000.0, 2000.0],
        "retention_offer_cost": [25.0, 50.0],
        "offer_success_rate": [0.2, 0.4],
    },
}


# ---------- break-even and settings --------------------------------------------------------


def test_break_even_probability():
    assert break_even_probability(1000, 50, 0.3) == pytest.approx(50 / 300)


def test_break_even_is_infinite_when_offers_never_work():
    assert math.isinf(break_even_probability(1000, 50, 0.0))


def test_valid_settings_pass_and_remove_duplicates():
    business = {**GOOD_BUSINESS, "sensitivity": {**GOOD_BUSINESS["sensitivity"], "customer_ltv": [1000.0, 1000.0]}}
    settings = load_business_settings({"business": business})
    assert settings["sensitivity"]["customer_ltv"] == [1000.0]


def test_missing_business_section_raises():
    with pytest.raises(BusinessError, match="business"):
        load_business_settings({})


def test_missing_key_raises():
    business = {key: value for key, value in GOOD_BUSINESS.items() if key != "retention_offer_cost"}
    with pytest.raises(BusinessError, match="retention_offer_cost"):
        load_business_settings({"business": business})


def test_missing_sensitivity_raises():
    business = {key: value for key, value in GOOD_BUSINESS.items() if key != "sensitivity"}
    with pytest.raises(BusinessError, match="sensitivity"):
        load_business_settings({"business": business})


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("customer_ltv", []),
        ("customer_ltv", "1000"),
        ("customer_ltv", [-5.0]),
        ("retention_offer_cost", [-1.0]),
        ("offer_success_rate", [1.5]),
        ("offer_success_rate", [True]),
    ],
)
def test_bad_sensitivity_values_raise(key, bad):
    business = {**GOOD_BUSINESS, "sensitivity": {**GOOD_BUSINESS["sensitivity"], key: bad}}
    with pytest.raises(BusinessError, match=key):
        load_business_settings({"business": business})


@pytest.mark.parametrize(
    ("ltv", "cost", "rate", "name"),
    [
        (0, 50, 0.3, "customer_ltv"),
        (-5, 50, 0.3, "customer_ltv"),
        (float("nan"), 50, 0.3, "customer_ltv"),
        ("1000", 50, 0.3, "customer_ltv"),
        (True, 50, 0.3, "customer_ltv"),
        (1000, -1, 0.3, "retention_offer_cost"),
        (1000, 50, 1.5, "offer_success_rate"),
        (1000, 50, -0.1, "offer_success_rate"),
    ],
)
def test_bad_economics_raise(ltv, cost, rate, name):
    with pytest.raises(BusinessError, match=name):
        net_benefit_curve(Y, P, ltv, cost, rate)


# ---------- the curve ----------------------------------------------------------------------


def test_curve_known_values():
    curve = net_benefit_curve(Y, P, **ECON)
    assert list(curve["targeted"]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(curve["tp"]) == [0, 1, 2, 2, 2, 3, 3]
    assert curve["net_benefit"].tolist() == pytest.approx([0, 250, 500, 450, 400, 650, 600])
    assert np.isnan(curve.loc[0, "threshold"])
    assert curve.loc[5, "threshold"] == 0.3
    assert curve.loc[5, "precision"] == pytest.approx(3 / 5)
    assert curve.loc[5, "recall"] == pytest.approx(1.0)
    assert curve.loc[5, "net_per_1000"] == pytest.approx(650 / 6 * 1000)


def test_best_point_and_choose_threshold():
    curve = net_benefit_curve(Y, P, **ECON)
    best = best_point(curve)
    assert (best["targeted"], best["threshold"], best["net_benefit"]) == (5, 0.3, 650)
    assert choose_threshold(curve)["threshold"] == 0.3


def test_ties_are_grouped():
    curve = net_benefit_curve([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.2], **ECON)
    assert list(curve["targeted"]) == [0, 3, 4]
    assert list(curve["tp"]) == [0, 2, 2]


def test_targeting_nobody_is_reported_not_invented():
    curve = net_benefit_curve(Y, P, ltv=100.0, cost=50.0, rate=0.1)  # 10 saved per churner, 50 per offer
    assert best_point(curve)["targeted"] == 0
    with pytest.raises(BusinessError, match="nobody"):
        choose_threshold(curve)


def test_free_offers_capture_every_churner():
    curve = net_benefit_curve(Y, P, ltv=1000.0, cost=0.0, rate=0.3)
    assert best_point(curve)["net_benefit"] == pytest.approx(0.3 * 1000 * 3)


def test_best_point_rejects_bad_curves():
    curve = net_benefit_curve(Y, P, **ECON)
    with pytest.raises(BusinessError, match="empty"):
        best_point(curve.iloc[0:0])
    with pytest.raises(BusinessError, match="net_benefit"):
        best_point(curve.drop(columns=["net_benefit"]))


def test_evaluate_threshold_matches_the_curve():
    result = evaluate_threshold(Y, P, 0.3, **ECON)
    assert (result["targeted"], result["tp"], result["fp"]) == (5, 3, 2)
    assert result["net_benefit"] == pytest.approx(650)


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), "0.3", True])
def test_evaluate_threshold_bad_threshold_raises(threshold):
    with pytest.raises(BusinessError, match="threshold"):
        evaluate_threshold(Y, P, threshold, **ECON)


@pytest.mark.parametrize(
    ("y", "p", "message"),
    [
        ([], [], "Empty"),
        ([0, 1], [0.5], "mismatch"),
        ([0, 2], [0.2, 0.8], "0 and 1"),
        ([0, 0], [0.2, 0.8], "both classes"),
        ([0, 1], [0.2, np.nan], "NaN"),
        ([0, 1], [0.2, 1.5], "between 0 and 1"),
        ([[0, 1]], [[0.2, 0.8]], "1-dimensional"),
    ],
)
def test_bad_scores_raise(y, p, message):
    with pytest.raises(BusinessError, match=message):
        net_benefit_curve(y, p, **ECON)


# ---------- sensitivity --------------------------------------------------------------------


def test_sensitivity_table_has_one_row_per_combination():
    table = sensitivity_table(Y, P, GOOD_BUSINESS["sensitivity"])
    assert len(table) == 8
    assert {"break_even_probability", "best_threshold", "targeted_pct", "net_per_1000"} <= set(table.columns)
    thresholds = table["best_threshold"].dropna()
    assert ((thresholds >= 0) & (thresholds <= 1)).all()


def test_higher_ltv_never_lowers_the_best_net_benefit():
    grid = {"customer_ltv": [500.0, 1000.0, 2000.0], "retention_offer_cost": [50.0], "offer_success_rate": [0.3]}
    table = sensitivity_table(Y, P, grid)
    assert table["net_per_1000"].is_monotonic_increasing


def test_unprofitable_scenario_is_reported_in_the_table():
    grid = {"customer_ltv": [100.0], "retention_offer_cost": [50.0], "offer_success_rate": [0.1]}
    row = sensitivity_table(Y, P, grid).iloc[0]
    assert row["targeted_pct"] == 0
    assert np.isnan(row["best_threshold"])


def test_sensitivity_grid_missing_values_raises():
    with pytest.raises(BusinessError, match="missing values"):
        sensitivity_table(Y, P, {"customer_ltv": [1000.0]})


def test_plot_writes_a_file(tmp_path):
    curve = net_benefit_curve(Y, P, **ECON)
    path = tmp_path / "figs" / "net.png"
    plot_net_benefit(curve, best_point(curve), path, "test")
    assert path.exists()
    assert path.stat().st_size > 0


# ---------- end to end ---------------------------------------------------------------------


def make_cfg(tmp_path, **business_overrides) -> dict:
    return {
        **TEST_CFG,
        "business": {**GOOD_BUSINESS, **business_overrides},
        "paths": {
            "processed_dir": str(tmp_path),
            "reports_dir": str(tmp_path / "reports"),
            "models_dir": str(tmp_path / "models"),
        },
        "data": {"target": "Churn", "id_column": "customerID"},
    }


def write_inputs(tmp_path) -> None:
    """Write a saved model and val.csv only: no train or test file is needed."""
    X_train, y_train = make_xy(400, 0)
    model = CalibratedModel(build_pipeline("logreg", "baseline", TEST_CFG), "none").fit(X_train, y_train)
    (tmp_path / "models").mkdir()
    joblib.dump(model, tmp_path / "models" / "final_model.joblib")

    X_val, y_val = make_xy(200, 5)
    val = X_val.copy()
    val["customerID"] = [f"val-{i}" for i in range(len(val))]
    val["Churn"] = y_val
    val.to_csv(tmp_path / "val.csv", index=False)


def test_run_business_writes_reports_using_only_the_validation_split(tmp_path):
    write_inputs(tmp_path)
    result = run_business(make_cfg(tmp_path))

    assert 0 <= result["chosen"]["threshold"] <= 1
    assert len(result["sensitivity"]) == 8
    reports = tmp_path / "reports"
    for name in ("business_curve.csv", "sensitivity_table.csv", "threshold_choice.json"):
        assert (reports / name).exists()
    assert (reports / "figures" / "net_benefit_validation.png").exists()
    saved = json.loads((reports / "threshold_choice.json").read_text(encoding="utf-8"))
    assert saved["threshold"] == pytest.approx(result["chosen"]["threshold"])
    assert saved["assumptions"]["customer_ltv"] == 2000.0


def test_run_business_without_model_raises(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "models" / "final_model.joblib").unlink()
    with pytest.raises(BusinessError, match="Final model not found"):
        run_business(make_cfg(tmp_path))


def test_run_business_without_validation_file_raises(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "val.csv").unlink()
    with pytest.raises(SplitError, match="not found"):
        run_business(make_cfg(tmp_path))


def test_run_business_stops_when_no_campaign_is_profitable(tmp_path):
    write_inputs(tmp_path)
    with pytest.raises(BusinessError, match="nobody"):
        run_business(make_cfg(tmp_path, retention_offer_cost=100000.0))