import numpy as np
import pandas as pd
import pytest

from src.drift import (
    DRIFT_CATEGORICAL,
    DRIFT_NUMERIC,
    DriftError,
    classify,
    compute_drift,
    load_drift_settings,
    numeric_bin_edges,
    psi_categorical,
    psi_from_counts,
    psi_numeric,
    run_drift,
)
from src.features import add_features
from src.predict import PredictionError
from src.split import SplitError
from tests.test_features import make_raw
from tests.test_models import TEST_CFG, make_xy
from tests.test_predict import make_input

GOOD_DRIFT = {"n_bins": 10, "moderate_threshold": 0.1, "major_threshold": 0.25, "min_rows": 100}


# ---------- settings -----------------------------------------------------------------------


def test_valid_settings_pass():
    assert load_drift_settings({"drift": GOOD_DRIFT}) == GOOD_DRIFT


def test_missing_settings_section_raises():
    with pytest.raises(DriftError, match="drift"):
        load_drift_settings({})


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("n_bins", 1),
        ("n_bins", True),
        ("n_bins", 2.5),
        ("n_bins", "10"),
        ("min_rows", 0),
        ("min_rows", True),
        ("moderate_threshold", 0),
        ("moderate_threshold", "0.1"),
        ("moderate_threshold", True),
        ("moderate_threshold", float("nan")),
        ("major_threshold", -1),
        ("major_threshold", None),
    ],
)
def test_bad_settings_raise(key, bad):
    with pytest.raises(DriftError, match=key):
        load_drift_settings({"drift": {**GOOD_DRIFT, key: bad}})


def test_moderate_must_be_smaller_than_major():
    with pytest.raises(DriftError, match="smaller"):
        load_drift_settings({"drift": {**GOOD_DRIFT, "moderate_threshold": 0.3}})


# ---------- PSI from counts ----------------------------------------------------------------


def test_identical_distributions_have_zero_psi():
    assert psi_from_counts([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)


def test_psi_known_value():
    assert psi_from_counts([50, 50], [70, 30]) == pytest.approx(0.16946, abs=1e-4)


def test_psi_is_symmetric_and_scale_invariant():
    a, b = [50, 30, 20], [20, 30, 50]
    assert psi_from_counts(a, b) == pytest.approx(psi_from_counts(b, a))
    assert psi_from_counts(a, b) == pytest.approx(psi_from_counts([10 * x for x in a], b))


def test_an_empty_bin_does_not_break_the_psi():
    value = psi_from_counts([100, 0], [50, 50])
    assert np.isfinite(value)
    assert value > 0.25


@pytest.mark.parametrize(
    "expected, actual",
    [
        ([1, 2, 3], [1, 2]),
        ([5], [5]),
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ([1, -2, 3], [1, 2, 3]),
        ([1, np.nan, 3], [1, 2, 3]),
        ([0, 0, 0], [1, 2, 3]),
        ([1, 2, 3], [0, 0, 0]),
    ],
)
def test_bad_count_vectors_raise(expected, actual):
    with pytest.raises(DriftError):
        psi_from_counts(expected, actual)


# ---------- numeric features ---------------------------------------------------------------


def test_the_same_numeric_distribution_is_stable():
    rng = np.random.default_rng(0)
    reference, current = pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(size=5000))
    assert psi_numeric(reference, current) < 0.05


def test_a_shifted_numeric_distribution_is_a_major_drift():
    rng = np.random.default_rng(0)
    reference, current = pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(loc=1.0, size=5000))
    assert psi_numeric(reference, current) > 0.25


def test_a_constant_reference_that_changes_is_caught():
    reference, current = pd.Series([5.0] * 200), pd.Series([6.0] * 200)
    assert psi_numeric(reference, current) > 0.25


def test_a_growing_share_of_missing_values_is_drift():
    rng = np.random.default_rng(1)
    values = rng.normal(size=1000)
    current = pd.Series(values).copy()
    current[:300] = np.nan
    assert psi_numeric(pd.Series(values), current) > 0.25


def test_numeric_bin_edges_are_sorted_and_unique():
    edges = numeric_bin_edges(pd.Series([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 10)
    assert list(edges) == sorted(set(edges))


def test_numeric_bin_edges_reject_empty_or_infinite_references():
    with pytest.raises(DriftError, match="no values"):
        numeric_bin_edges(pd.Series([np.nan, np.nan]), 10)
    with pytest.raises(DriftError, match="infinite"):
        numeric_bin_edges(pd.Series([1.0, np.inf, 2.0]), 10)


# ---------- categorical features -----------------------------------------------------------


def test_the_same_category_shares_are_stable():
    reference = pd.Series(["A", "B"] * 100)
    current = pd.Series(["A", "B"] * 50)
    assert psi_categorical(reference, current) == pytest.approx(0.0)


def test_changed_category_shares_are_a_major_drift():
    reference = pd.Series(["A"] * 50 + ["B"] * 50)
    current = pd.Series(["A"] * 90 + ["B"] * 10)
    assert psi_categorical(reference, current) == pytest.approx(0.879, abs=0.01)


def test_an_unseen_category_is_drift():
    reference = pd.Series(["A"] * 50 + ["B"] * 50)
    current = pd.Series(["A"] * 45 + ["B"] * 45 + ["C"] * 10)
    assert psi_categorical(reference, current) > 0.25


def test_missing_values_in_a_categorical_feature_are_drift():
    reference = pd.Series(["A"] * 50 + ["B"] * 50)
    current = pd.Series(["A"] * 35 + ["B"] * 35 + [None] * 30)
    assert psi_categorical(reference, current) > 0.25


def test_integers_and_floats_count_as_the_same_category():
    reference, current = pd.Series([0, 1] * 100), pd.Series([0.0, 1.0] * 100)
    assert psi_categorical(reference, current) == pytest.approx(0.0, abs=1e-12)


def test_an_all_missing_reference_raises():
    with pytest.raises(DriftError, match="no values"):
        psi_categorical(pd.Series([None, None]), pd.Series(["A", "B"]))


# ---------- status and the table -----------------------------------------------------------


def test_classify_uses_the_thresholds():
    assert classify(0.05, 0.1, 0.25) == "stable"
    assert classify(0.1, 0.1, 0.25) == "moderate"
    assert classify(0.2499, 0.1, 0.25) == "moderate"
    assert classify(0.25, 0.1, 0.25) == "major"


def frame_from(n: int, seed: int) -> pd.DataFrame:
    return add_features(make_raw(n, seed))


def drift(reference, current, **overrides):
    settings = {"n_bins": 10, "moderate": 0.1, "major": 0.25, "min_rows": 100, **overrides}
    return compute_drift(reference, current, DRIFT_NUMERIC, DRIFT_CATEGORICAL, **settings)


def test_compute_drift_returns_every_feature_sorted_by_psi():
    table = drift(frame_from(400, 0), frame_from(300, 1))
    assert set(table["feature"]) == set(DRIFT_NUMERIC) | set(DRIFT_CATEGORICAL)
    assert table["psi"].is_monotonic_decreasing
    assert (table["psi"] >= 0).all()
    assert set(table["status"]) <= {"stable", "moderate", "major"}
    assert set(table["kind"]) == {"numeric", "categorical"}


def test_compute_drift_rejects_bad_inputs():
    reference, current = frame_from(400, 0), frame_from(300, 1)
    with pytest.raises(DriftError, match="at least 100"):
        drift(reference, frame_from(50, 1))
    with pytest.raises(DriftError, match="empty"):
        drift(reference.iloc[0:0], current)
    with pytest.raises(DriftError, match="missing"):
        drift(reference, current.drop(columns=["Contract"]))
    with pytest.raises(DriftError, match="smaller"):
        drift(reference, current, moderate=0.3, major=0.25)
    with pytest.raises(DriftError, match="n_bins"):
        drift(reference, current, n_bins=1)


# ---------- end to end ---------------------------------------------------------------------


def make_cfg(tmp_path) -> dict:
    return {
        **TEST_CFG,
        "drift": dict(GOOD_DRIFT),
        "paths": {"processed_dir": str(tmp_path), "reports_dir": str(tmp_path / "reports")},
        "data": {"target": "Churn", "id_column": "customerID"},
    }


def write_splits(tmp_path) -> None:
    """Write train.csv and val.csv only: the test set must not be needed."""
    for name, seed, n in (("train", 0, 400), ("val", 5, 200)):
        X, y = make_xy(n, seed)
        frame = X.copy()
        frame["customerID"] = [f"{name}-{i}" for i in range(len(frame))]
        frame["Churn"] = y
        frame.to_csv(tmp_path / f"{name}.csv", index=False)


def test_default_run_compares_validation_with_train_and_writes_a_report(tmp_path):
    write_splits(tmp_path)
    out = tmp_path / "out" / "drift.csv"
    table = run_drift(make_cfg(tmp_path), output_path=out)
    assert set(table["feature"]) == set(DRIFT_NUMERIC) | set(DRIFT_CATEGORICAL)
    assert len(pd.read_csv(out)) == len(table)


def test_a_shifted_input_file_is_flagged(tmp_path):
    write_splits(tmp_path)
    frame = make_input(150, 3)
    frame["Contract"] = "Month-to-month"
    csv = tmp_path / "new.csv"
    frame.to_csv(csv, index=False)
    table = run_drift(make_cfg(tmp_path), csv, tmp_path / "out.csv")
    assert table.set_index("feature").loc["Contract", "status"] == "major"


def test_input_with_a_missing_column_raises(tmp_path):
    write_splits(tmp_path)
    csv = tmp_path / "new.csv"
    make_input(150, 3).drop(columns=["Contract"]).to_csv(csv, index=False)
    with pytest.raises(PredictionError, match="Contract"):
        run_drift(make_cfg(tmp_path), csv, tmp_path / "out.csv")


def test_input_with_too_few_rows_raises(tmp_path):
    write_splits(tmp_path)
    csv = tmp_path / "new.csv"
    make_input(50, 3).to_csv(csv, index=False)
    with pytest.raises(DriftError, match="at least 100"):
        run_drift(make_cfg(tmp_path), csv, tmp_path / "out.csv")


def test_missing_train_file_raises(tmp_path):
    with pytest.raises(SplitError, match="not found"):
        run_drift(make_cfg(tmp_path), output_path=tmp_path / "out.csv")