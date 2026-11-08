import joblib
import numpy as np
import pandas as pd
import pytest

from src.calibration import CalibratedModel
from src.evaluate import EvaluationError
from src.explain import (
    SAMPLE_ROLES,
    ExplainError,
    build_groups,
    compute_shap,
    driver_profiles,
    feature_importance,
    group_importance,
    group_of,
    group_shap_values,
    load_explain_settings,
    pick_sample_customers,
    plot_bar,
    plot_summary,
    plot_waterfall,
    run_explain,
    split_model,
    top_contributions,
    transform_features,
    unscale_numeric,
)
from src.features import build_preprocessor
from src.models import build_pipeline
from src.split import SplitError
from tests.test_evaluate import write_threshold
from tests.test_features import make_raw
from tests.test_models import TEST_CFG, make_xy


@pytest.fixture(scope="module")
def ctx():
    X_train, y_train = make_xy(400, 0)
    model = CalibratedModel(build_pipeline("logreg", "baseline", TEST_CFG), "none").fit(X_train, y_train)
    features_step, preprocess, classifier = split_model(model)
    _, background = transform_features(features_step, preprocess, X_train)
    X_val, y_val = make_xy(200, 5)
    engineered, matrix = transform_features(features_step, preprocess, X_val)
    values, base = compute_shap(classifier, background, matrix)
    return {
        "model": model, "preprocess": preprocess, "classifier": classifier, "background": background,
        "engineered": engineered, "matrix": matrix, "y": y_val, "values": values, "base": base,
    }


# ---------- settings ---------------------------------------------------------------------


def test_valid_settings_pass():
    assert load_explain_settings({"explainability": {"top_n": 5, "max_display": 15}}) == {"top_n": 5, "max_display": 15}


def test_missing_section_raises():
    with pytest.raises(ExplainError, match="explainability"):
        load_explain_settings({})


@pytest.mark.parametrize("key", ["top_n", "max_display"])
@pytest.mark.parametrize("bad", [0, -1, 2.5, "5", True, None])
def test_bad_settings_raise(key, bad):
    section = {"top_n": 5, "max_display": 15, key: bad}
    with pytest.raises(ExplainError, match=key):
        load_explain_settings({"explainability": section})


# ---------- the model --------------------------------------------------------------------


def test_split_model_rejects_a_tree_model():
    X, y = make_xy(200, 0)
    forest = CalibratedModel(build_pipeline("random_forest", "baseline", TEST_CFG), "none").fit(X, y)
    with pytest.raises(ExplainError, match="logistic regression"):
        split_model(forest)


def test_split_model_rejects_an_unknown_object():
    with pytest.raises(ExplainError, match="expected steps"):
        split_model(object())


# ---------- SHAP values ----------------------------------------------------------------------


def test_shap_values_add_up_to_the_model_output(ctx):
    expected = ctx["classifier"].decision_function(ctx["matrix"])
    np.testing.assert_allclose(ctx["base"] + ctx["values"].sum(axis=1), expected, atol=1e-8)


def test_shap_values_match_the_closed_form_for_a_linear_model(ctx):
    expected = ctx["classifier"].coef_[0] * (ctx["matrix"].to_numpy() - ctx["background"].to_numpy().mean(axis=0))
    np.testing.assert_allclose(ctx["values"], expected, atol=1e-8)


def test_compute_shap_rejects_bad_inputs(ctx):
    with pytest.raises(ExplainError, match="same columns"):
        compute_shap(ctx["classifier"], ctx["background"].iloc[:, :-1], ctx["matrix"])
    with pytest.raises(ExplainError, match="empty"):
        compute_shap(ctx["classifier"], ctx["background"], ctx["matrix"].iloc[0:0])


# ---------- groups ---------------------------------------------------------------------------


@pytest.mark.parametrize("drop", [[], ["TotalCharges"]])
def test_every_model_column_belongs_to_a_group(drop):
    columns = list(build_preprocessor(drop).fit_transform(make_raw(60)).columns)
    groups = build_groups(columns)
    assert sorted(c for cols in groups.values() for c in cols) == sorted(columns)


def test_group_assignment_examples():
    assert group_of("Contract_Month-to-month") == "Contract"
    assert group_of("is_month_to_month") == "Contract"
    assert group_of("tenure") == "tenure"
    assert group_of("tenure_bucket_0-6") == "tenure"
    assert group_of("OnlineSecurity_Yes") == "OnlineSecurity"
    assert group_of("addon_count") == "addon_count"


def test_unknown_column_raises():
    with pytest.raises(ExplainError, match="No explanation group"):
        group_of("mystery_feature")


def test_group_totals_equal_column_totals(ctx):
    columns = list(ctx["matrix"].columns)
    grouped = group_shap_values(ctx["values"], columns)
    np.testing.assert_allclose(grouped.sum(axis=1), ctx["values"].sum(axis=1), atol=1e-10)


def test_group_shap_values_rejects_a_mismatch(ctx):
    with pytest.raises(ExplainError, match="do not match"):
        group_shap_values(ctx["values"], list(ctx["matrix"].columns)[:-1])


def test_importance_tables_are_sorted_and_shares_add_to_100(ctx):
    columns = list(ctx["matrix"].columns)
    features = feature_importance(ctx["values"], columns)
    assert features["mean_abs_shap"].is_monotonic_decreasing
    assert len(features) == len(columns)
    groups = group_importance(group_shap_values(ctx["values"], columns))
    assert groups["mean_abs_shap"].is_monotonic_decreasing
    assert groups["share_pct"].sum() == pytest.approx(100.0)


def test_group_importance_rejects_empty():
    with pytest.raises(ExplainError, match="no group"):
        group_importance(pd.DataFrame())


# ---------- profiles -------------------------------------------------------------------------


def test_driver_profiles(ctx):
    columns = list(ctx["matrix"].columns)
    grouped = group_shap_values(ctx["values"], columns)
    profiles = driver_profiles(grouped, ctx["engineered"], ctx["y"], ["Contract", "tenure", "MonthlyCharges"])

    contract = profiles[profiles["group"] == "Contract"]
    assert contract["customers"].sum() == len(ctx["y"])
    assert set(contract["value"]) <= {"Month-to-month", "One year", "Two year"}
    assert contract["mean_shap"].is_monotonic_decreasing
    np.testing.assert_allclose(profiles["odds_multiplier"], np.exp(profiles["mean_shap"]))
    assert ((profiles["observed_churn_rate"] >= 0) & (profiles["observed_churn_rate"] <= 1)).all()
    assert set(profiles[profiles["group"] == "tenure"]["value"]) <= {"0-6", "7-12", "13-24", "25-48", "49+"}
    assert len(profiles[profiles["group"] == "MonthlyCharges"]) <= 4  # quartiles


def test_driver_profiles_reject_bad_inputs(ctx):
    grouped = group_shap_values(ctx["values"], list(ctx["matrix"].columns))
    with pytest.raises(ExplainError, match="same length"):
        driver_profiles(grouped, ctx["engineered"], ctx["y"].iloc[:10], ["Contract"])
    with pytest.raises(ExplainError, match="Cannot profile"):
        driver_profiles(grouped, ctx["engineered"], ctx["y"], ["NotAGroup"])


# ---------- samples --------------------------------------------------------------------------


def test_sample_customers_follow_the_rule():
    picks = pick_sample_customers([0.5, 0.9, 0.1, 0.2, 0.18], threshold=0.17)
    assert picks == {"highest_risk": 1, "nearest_threshold": 4, "lowest_risk": 2}
    assert tuple(picks) == SAMPLE_ROLES


def test_sample_customers_are_distinct_even_when_one_is_nearest_to_the_threshold():
    picks = pick_sample_customers([0.9, 0.5, 0.1], threshold=0.95)
    assert len(set(picks.values())) == 3


@pytest.mark.parametrize(
    ("proba", "message"),
    [
        ([0.1, 0.2], "At least 3"),
        ([0.5, 0.5, 0.5], "same predicted risk"),
        ([0.1, np.nan, 0.3], "NaN"),
    ],
)
def test_bad_sample_inputs_raise(proba, message):
    with pytest.raises(ExplainError, match=message):
        pick_sample_customers(proba, 0.2)


def test_top_contributions_are_sorted_by_size():
    top = top_contributions(np.array([0.1, -0.5, 0.3, 0.0]), ["a", "b", "c", "d"], k=2)
    assert top == [("b", -0.5), ("c", 0.3)]


# ---------- display and plots ----------------------------------------------------------------


def test_unscale_numeric_restores_real_units(ctx):
    display = unscale_numeric(ctx["matrix"], ctx["preprocess"])
    np.testing.assert_allclose(display["tenure"], ctx["engineered"]["tenure"], atol=1e-6)
    np.testing.assert_allclose(display["MonthlyCharges"], ctx["engineered"]["MonthlyCharges"], atol=1e-6)


def test_plots_are_written(ctx, tmp_path):
    columns = list(ctx["matrix"].columns)
    plot_summary(ctx["values"], ctx["matrix"], tmp_path / "summary.png", 10)
    plot_bar(ctx["values"], ctx["matrix"], tmp_path / "bar.png", 10)
    display = unscale_numeric(ctx["matrix"], ctx["preprocess"])
    plot_waterfall(ctx["values"][0], ctx["base"], display.iloc[0], columns, tmp_path / "waterfall.png", title="t")
    for name in ("summary.png", "bar.png", "waterfall.png"):
        assert (tmp_path / name).stat().st_size > 0


# ---------- end to end ----------------------------------------------------------------------


def make_cfg(tmp_path) -> dict:
    return {
        **TEST_CFG,
        "explainability": {"top_n": 3, "max_display": 8},
        "paths": {
            "processed_dir": str(tmp_path),
            "reports_dir": str(tmp_path / "reports"),
            "models_dir": str(tmp_path / "models"),
        },
        "data": {"target": "Churn", "id_column": "customerID"},
    }


def write_inputs(tmp_path) -> None:
    """Write the model, the threshold file, train.csv and val.csv ONLY (no test file)."""
    X_train, y_train = make_xy(400, 0)
    model = CalibratedModel(build_pipeline("logreg", "baseline", TEST_CFG), "none").fit(X_train, y_train)
    (tmp_path / "models").mkdir()
    joblib.dump(model, tmp_path / "models" / "final_model.joblib")
    for name, seed, n in (("train", 0, 400), ("val", 5, 200)):
        X, y = make_xy(n, seed)
        frame = X.copy()
        frame["customerID"] = [f"{name}-{i}" for i in range(len(frame))]
        frame["Churn"] = y
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / "reports").mkdir()
    write_threshold(tmp_path / "reports" / "threshold_choice.json", threshold=0.3)


def test_run_explain_writes_everything_without_touching_the_test_split(tmp_path):
    write_inputs(tmp_path)
    result = run_explain(make_cfg(tmp_path))

    reports, figures = tmp_path / "reports", tmp_path / "reports" / "figures"
    for name in (
        "shap_feature_importance.csv", "shap_group_importance.csv",
        "shap_driver_profiles.csv", "shap_sample_customers.csv",
    ):
        assert (reports / name).exists()
    for name in (
        "shap_summary.png", "shap_bar.png", "shap_waterfall_highest_risk.png",
        "shap_waterfall_nearest_threshold.png", "shap_waterfall_lowest_risk.png",
    ):
        assert (figures / name).stat().st_size > 0
    assert list(result["samples"]["role"]) == list(SAMPLE_ROLES)
    assert result["groups"]["share_pct"].sum() == pytest.approx(100.0)
    assert set(result["profiles"]["group"]) == set(result["groups"]["group"].head(3))


def test_run_explain_missing_model_raises(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "models" / "final_model.joblib").unlink()
    with pytest.raises(EvaluationError, match="Final model not found"):
        run_explain(make_cfg(tmp_path))


def test_run_explain_missing_threshold_raises(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "reports" / "threshold_choice.json").unlink()
    with pytest.raises(EvaluationError, match="not found"):
        run_explain(make_cfg(tmp_path))


def test_run_explain_missing_validation_file_raises(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "val.csv").unlink()
    with pytest.raises(SplitError, match="not found"):
        run_explain(make_cfg(tmp_path))