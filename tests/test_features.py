import numpy as np
import pandas as pd
import pytest

from src.features import (
    ADDON_COLUMNS,
    FeatureError,
    add_features,
    build_preprocessor,
    compute_vif,
)

BASE = {
    "tenure": 10,
    "MonthlyCharges": 50.0,
    "TotalCharges": 500.0,
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def row(**overrides) -> pd.DataFrame:
    return pd.DataFrame([{**BASE, **overrides}])


def make_raw(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 110, n).round(2)
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": (tenure * monthly).round(2),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
    }
    for col in ADDON_COLUMNS:
        data[col] = rng.choice(["Yes", "No"], n)
    return pd.DataFrame(data)


# ---------- add_features: values ------------------------------------------------------------


def test_addon_count():
    assert add_features(row()).loc[0, "addon_count"] == 2


def test_no_internet_service_label_is_not_counted():
    out = add_features(row(OnlineSecurity="No internet service", OnlineBackup="No internet service"))
    assert out.loc[0, "addon_count"] == 0


def test_whitespace_is_stripped_before_counting():
    out = add_features(row(OnlineSecurity="  Yes ", OnlineBackup="No"))
    assert out.loc[0, "addon_count"] == 1


def test_flags_for_a_risky_customer():
    out = add_features(row()).loc[0]
    assert (out["is_month_to_month"], out["is_electronic_check"], out["is_fiber_optic"]) == (1, 1, 0)
    assert out["month_to_month_new"] == 1


def test_month_to_month_new_is_zero_for_long_tenure():
    out = add_features(row(tenure=30, TotalCharges=1500.0))
    assert out.loc[0, "month_to_month_new"] == 0


def test_fiber_flag():
    assert add_features(row(InternetService="Fiber optic")).loc[0, "is_fiber_optic"] == 1


@pytest.mark.parametrize(
    ("tenure", "bucket"),
    [(0, "0-6"), (6, "0-6"), (7, "7-12"), (12, "7-12"), (13, "13-24"), (24, "13-24"),
     (25, "25-48"), (48, "25-48"), (49, "49+"), (72, "49+")],
)
def test_tenure_buckets(tenure, bucket):
    assert add_features(row(tenure=tenure, TotalCharges=0.0)).loc[0, "tenure_bucket"] == bucket


def test_charges_ratio_is_one_when_bill_never_changed():
    assert add_features(row()).loc[0, "charges_ratio"] == pytest.approx(1.0)


def test_charges_ratio_detects_a_changed_bill():
    assert add_features(row(TotalCharges=600.0)).loc[0, "charges_ratio"] == pytest.approx(1.2)


def test_charges_ratio_for_zero_tenure_is_one_not_infinite():
    ratio = add_features(row(tenure=0, TotalCharges=0.0)).loc[0, "charges_ratio"]
    assert ratio == 1.0


def test_charges_ratio_for_zero_monthly_charge_is_finite():
    ratio = add_features(row(MonthlyCharges=0.0)).loc[0, "charges_ratio"]
    assert np.isfinite(ratio)


def test_charges_ratio_missing_when_total_is_missing():
    assert pd.isna(add_features(row(TotalCharges=np.nan)).loc[0, "charges_ratio"])


def test_input_frame_is_not_modified():
    df = row()
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_extra_columns_are_kept_by_add_features():
    out = add_features(row(customerID="X-1", Churn=1))
    assert {"customerID", "Churn"} <= set(out.columns)


# ---------- add_features: failure paths ---------------------------------------------------------


def test_missing_column_raises():
    with pytest.raises(FeatureError, match="Contract"):
        add_features(row().drop(columns=["Contract"]))


def test_empty_frame_raises():
    with pytest.raises(FeatureError, match="empty"):
        add_features(row().iloc[0:0])


def test_non_dataframe_raises():
    with pytest.raises(FeatureError, match="DataFrame"):
        add_features([1, 2, 3])  # type: ignore[arg-type]


def test_text_in_numeric_column_raises():
    with pytest.raises(FeatureError, match="tenure"):
        add_features(row(tenure="abc"))


def test_negative_value_raises():
    with pytest.raises(FeatureError, match="MonthlyCharges"):
        add_features(row(MonthlyCharges=-5.0))


def test_infinite_value_raises():
    with pytest.raises(FeatureError, match="TotalCharges"):
        add_features(row(TotalCharges=np.inf))


# ---------- preprocessing pipeline -----------------------------------------------------------------


def test_pipeline_output_is_numeric_without_nan():
    out = build_preprocessor().fit_transform(make_raw())
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 60
    assert out.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in out.dtypes)


def test_numeric_columns_are_centered_on_train():
    out = build_preprocessor().fit_transform(make_raw())
    assert out["tenure"].mean() == pytest.approx(0, abs=1e-8)


def test_id_target_and_gender_never_reach_the_model():
    raw = make_raw()
    raw["customerID"] = [f"ID-{i}" for i in range(len(raw))]
    raw["Churn"] = 1
    raw["gender"] = "Male"
    out = build_preprocessor().fit_transform(raw)
    assert not [col for col in out.columns if col.startswith(("customerID", "Churn", "gender"))]


def test_unseen_category_is_ignored_not_an_error():
    pipeline = build_preprocessor().fit(make_raw())
    out = pipeline.transform(row(Contract="Weekly"))
    assert out.filter(like="Contract_").to_numpy().sum() == 0


def test_missing_values_at_inference_give_no_nan():
    pipeline = build_preprocessor().fit(make_raw())
    out = pipeline.transform(row(tenure=np.nan, Partner=np.nan, MonthlyCharges=np.nan, TotalCharges=np.nan))
    assert out.isna().sum().sum() == 0


def test_inference_columns_match_training_columns():
    pipeline = build_preprocessor().fit(make_raw())
    assert list(pipeline.transform(row()).columns) == list(pipeline.transform(make_raw(5)).columns)


def test_missing_column_at_inference_raises():
    pipeline = build_preprocessor().fit(make_raw())
    with pytest.raises(FeatureError, match="Contract"):
        pipeline.transform(row().drop(columns=["Contract"]))


def test_same_data_gives_same_output():
    a = build_preprocessor().fit_transform(make_raw())
    b = build_preprocessor().fit_transform(make_raw())
    pd.testing.assert_frame_equal(a, b)


def test_drop_columns_removes_the_feature():
    out = build_preprocessor(drop_columns=["TotalCharges"]).fit_transform(make_raw())
    assert "TotalCharges" not in out.columns
    assert "tenure" in out.columns


def test_dropping_an_unknown_feature_raises():
    with pytest.raises(FeatureError, match="unknown"):
        build_preprocessor(drop_columns=["NotAFeature"])


# ---------- VIF ---------------------------------------------------------------------------------------


def make_vif_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    return pd.DataFrame(
        {"a": a, "b": 2 * a + rng.normal(scale=0.01, size=200), "c": rng.normal(size=200)}
    )


def test_vif_flags_collinear_columns():
    table = compute_vif(make_vif_frame(), ["a", "b", "c"]).set_index("feature")["vif"]
    assert table["a"] > 100
    assert table["b"] > 100
    assert table["c"] < 5


def test_vif_missing_column_raises():
    with pytest.raises(FeatureError, match="not found"):
        compute_vif(make_vif_frame(), ["a", "zzz"])


def test_vif_constant_column_raises():
    frame = make_vif_frame()
    frame["k"] = 1.0
    with pytest.raises(FeatureError, match="Constant"):
        compute_vif(frame, ["a", "c", "k"])


def test_vif_too_few_rows_raises():
    with pytest.raises(FeatureError, match="Too few"):
        compute_vif(make_vif_frame().head(3), ["a", "b", "c"])


def test_vif_needs_two_columns():
    with pytest.raises(FeatureError, match="at least 2"):
        compute_vif(make_vif_frame(), ["a"])