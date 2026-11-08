import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    CALIBRATION_METHODS,
    CalibratedModel,
    CalibrationError,
    Calibrator,
    brier_score,
    calibration_table,
    expected_calibration_error,
    plot_reliability,
)
from src.models import build_pipeline
from tests.test_models import TEST_CFG, make_xy

# ---------- metrics ----------------------------------------------------------------------


def test_brier_known_values():
    assert brier_score([0, 1], [0.1, 0.9]) == pytest.approx(0.01)
    assert brier_score([0, 1], [1.0, 0.0]) == pytest.approx(1.0)


def test_ece_is_near_zero_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_ece_detects_overconfidence():
    y = np.array([0, 1] * 500)
    p = np.full(1000, 0.9)
    assert expected_calibration_error(y, p) == pytest.approx(0.4, abs=1e-9)


def test_calibration_table_counts_and_edge_values():
    table = calibration_table([0, 1, 1], [0.0, 1.0, 1.0], n_bins=4)
    assert list(table["count"]) == [1, 0, 0, 2]
    assert table.loc[0, "observed_rate"] == 0.0
    assert table.loc[3, "observed_rate"] == 1.0
    assert pd.isna(table.loc[1, "mean_predicted"])


@pytest.mark.parametrize("bad_bins", [1, 0, 2.5, True])
def test_bad_bin_count_raises(bad_bins):
    with pytest.raises(CalibrationError, match="n_bins"):
        calibration_table([0, 1], [0.2, 0.8], n_bins=bad_bins)


@pytest.mark.parametrize(
    ("y", "p", "message"),
    [
        ([], [], "Empty"),
        ([0, 1], [0.5], "mismatch"),
        ([0, 2], [0.2, 0.8], "0 and 1"),
        ([0, 1], [0.2, np.nan], "NaN"),
        ([0, 1], [0.2, 1.5], "between 0 and 1"),
        ([[0, 1]], [[0.2, 0.8]], "1-dimensional"),
    ],
)
def test_bad_inputs_raise(y, p, message):
    with pytest.raises(CalibrationError, match=message):
        brier_score(y, p)


# ---------- Calibrator ---------------------------------------------------------------------


def make_overconfident(n: int = 6000, seed: int = 0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.1, 0.9, n)
    y = (rng.random(n) < true_p).astype(int)
    raw = 1 / (1 + np.exp(-3 * np.log(true_p / (1 - true_p))))  # pushed toward 0 and 1
    return raw, y


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_reduces_overconfidence_on_unseen_data(method):
    raw, y = make_overconfident()
    half = len(y) // 2
    calibrated = Calibrator(method).fit(raw[:half], y[:half]).predict(raw[half:])
    before = expected_calibration_error(y[half:], raw[half:])
    after = expected_calibration_error(y[half:], calibrated)
    assert after < before / 2


def test_calibrator_unknown_method_raises():
    with pytest.raises(CalibrationError, match="Unknown"):
        Calibrator("magic")


def test_calibrator_predict_before_fit_raises():
    with pytest.raises(CalibrationError, match="not fitted"):
        Calibrator("sigmoid").predict([0.2, 0.8])


def test_calibrator_needs_both_classes():
    with pytest.raises(CalibrationError, match="both classes"):
        Calibrator("sigmoid").fit([0.1, 0.2, 0.3], [0, 0, 0])


def test_calibrator_rejects_bad_probabilities_at_predict():
    calibrator = Calibrator("isotonic").fit([0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1])
    with pytest.raises(CalibrationError, match="between 0 and 1"):
        calibrator.predict([0.5, 1.5])
    with pytest.raises(CalibrationError, match="between 0 and 1"):
        calibrator.predict([0.5, np.nan])


def test_calibrator_output_stays_within_0_and_1():
    raw, y = make_overconfident()
    for method in ("sigmoid", "isotonic"):
        out = Calibrator(method).fit(raw, y).predict(np.array([0.0, 0.5, 1.0]))
        assert ((out >= 0) & (out <= 1)).all()


# ---------- CalibratedModel ----------------------------------------------------------------


def fresh_base():
    return build_pipeline("logreg", "baseline", TEST_CFG)


def test_method_none_matches_the_plain_model():
    X, y = make_xy()
    wrapped = CalibratedModel(fresh_base(), "none").fit(X, y).predict_proba(X)
    plain = fresh_base().fit(X, y).predict_proba(X)
    np.testing.assert_allclose(wrapped, plain)


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_every_method_gives_valid_probabilities(method):
    X, y = make_xy()
    proba = CalibratedModel(fresh_base(), method, n_splits=3).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert not np.isnan(proba).any()
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_calibrated_model_is_reproducible():
    X, y = make_xy()
    a = CalibratedModel(fresh_base(), "sigmoid", n_splits=3, seed=7).fit(X, y).predict_proba(X)
    b = CalibratedModel(fresh_base(), "sigmoid", n_splits=3, seed=7).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)


def test_unknown_method_raises():
    with pytest.raises(CalibrationError, match="Unknown"):
        CalibratedModel(fresh_base(), "magic")


def test_predict_before_fit_raises():
    X, _ = make_xy()
    with pytest.raises(CalibrationError, match="not fitted"):
        CalibratedModel(fresh_base(), "none").predict_proba(X)


@pytest.mark.parametrize("method", CALIBRATION_METHODS)
def test_single_class_training_data_raises(method):
    X, y = make_xy()
    with pytest.raises(CalibrationError, match="Could not fit"):
        CalibratedModel(fresh_base(), method, n_splits=3).fit(X, y * 0)


# ---------- plot ---------------------------------------------------------------------------


def test_plot_reliability_writes_a_file(tmp_path):
    raw, y = make_overconfident(500)
    path = tmp_path / "figs" / "reliability.png"
    plot_reliability({"raw": calibration_table(y, raw, 5)}, path, "test")
    assert path.exists()
    assert path.stat().st_size > 0