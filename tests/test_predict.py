import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.calibration import CalibratedModel
from src.data import DataValidationError, load_raw
from src.evaluate import EvaluationError, file_sha256
from src.features import REQUIRED_COLUMNS
from src.models import build_pipeline
from src.predict import (
    PredictionError,
    build_predictions,
    load_prediction_settings,
    model_fingerprint_warning,
    prepare_input,
    run_predict,
    score,
)
from tests.test_evaluate import write_threshold
from tests.test_features import make_raw
from tests.test_models import TEST_CFG, make_xy

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_RAW = REPO_ROOT / "data" / "raw" / "telco_churn.csv"
REAL_CLEAN = REPO_ROOT / "data" / "processed" / "telco_clean.csv"
REAL_MODEL = REPO_ROOT / "models" / "final_model.joblib"


def make_input(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """A raw-looking, all-text frame: an id column plus every required feature column."""
    frame = make_raw(n, seed)
    frame.insert(0, "customerID", [f"ID-{i:04d}" for i in range(n)])
    return frame.astype(str)


def prep(frame: pd.DataFrame, **kwargs):
    return prepare_input(frame, "customerID", "Churn", **kwargs)


# ---------- settings and threshold ---------------------------------------------------------


def test_valid_settings_pass():
    assert load_prediction_settings({"prediction": {"max_rows": 10}}) == {"max_rows": 10}


def test_missing_settings_section_raises():
    with pytest.raises(PredictionError, match="prediction"):
        load_prediction_settings({})


@pytest.mark.parametrize("bad", [0, -1, 2.5, "5", True, None])
def test_bad_max_rows_raises(bad):
    with pytest.raises(PredictionError, match="max_rows"):
        load_prediction_settings({"prediction": {"max_rows": bad}})


# ---------- prepare_input ------------------------------------------------------------------


def test_valid_input_is_cleaned():
    features, report = prep(make_input())
    assert len(features) == 30
    assert list(features.columns) == ["customerID", *REQUIRED_COLUMNS]
    for col in ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"):
        assert pd.api.types.is_numeric_dtype(features[col])
    assert report["rows"] == 30
    assert report["missing_values"] == {}


def test_known_extra_columns_are_ignored_even_in_strict_mode():
    frame = make_input()
    frame["gender"] = "Male"
    frame["Churn"] = "Yes"
    features, report = prep(frame, strict=True)
    assert "gender" not in features.columns
    assert "Churn" not in features.columns
    assert report["unexpected_columns"] == []


def test_unknown_extra_columns_warn_or_raise_in_strict_mode():
    frame = make_input()
    frame["foo"] = "x"
    _, report = prep(frame)
    assert report["unexpected_columns"] == ["foo"]
    with pytest.raises(PredictionError, match="foo"):
        prep(frame, strict=True)


def test_missing_required_column_raises():
    with pytest.raises(PredictionError, match="Contract"):
        prep(make_input().drop(columns=["Contract"]))


def test_missing_id_column_raises():
    with pytest.raises(PredictionError, match="customerID"):
        prep(make_input().drop(columns=["customerID"]))


def test_blank_id_raises():
    frame = make_input()
    frame.loc[0, "customerID"] = " "
    with pytest.raises(PredictionError, match="customerID"):
        prep(frame)


def test_duplicate_ids_are_kept_and_reported():
    frame = make_input()
    frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    features, report = prep(frame)
    assert len(features) == 30
    assert report["duplicate_ids"] == 1


def test_empty_input_raises():
    with pytest.raises(PredictionError, match="no rows"):
        prep(make_input().iloc[0:0])


def test_duplicate_column_names_raise():
    frame = pd.concat([make_input(), make_input()[["tenure"]]], axis=1)
    with pytest.raises(PredictionError, match="Duplicate column"):
        prep(frame)


def test_column_names_with_stray_spaces_are_accepted():
    features, _ = prep(make_input().rename(columns={"tenure": " tenure "}))
    assert "tenure" in features.columns


@pytest.mark.parametrize(
    ("col", "bad"),
    [
        ("tenure", "abc"),
        ("tenure", "-1"),
        ("MonthlyCharges", "inf"),
        ("MonthlyCharges", "-5"),
        ("TotalCharges", "n/a"),
        ("SeniorCitizen", "2"),
        ("SeniorCitizen", "yes"),
    ],
)
def test_bad_numeric_values_raise_with_the_csv_line(col, bad):
    frame = make_input()
    frame.loc[3, col] = bad
    with pytest.raises(PredictionError, match=col):
        prep(frame)


def test_error_message_names_the_csv_line():
    frame = make_input()
    frame.loc[3, "tenure"] = "abc"
    with pytest.raises(PredictionError, match="CSV lines \\[5\\]"):
        prep(frame)


def test_blank_total_charges_for_a_new_customer_becomes_zero():
    frame = make_input()
    frame.loc[0, "tenure"] = "0"
    frame.loc[0, "TotalCharges"] = " "
    features, report = prep(frame)
    assert features.loc[0, "TotalCharges"] == 0.0
    assert "TotalCharges" not in report["missing_values"]


def test_other_missing_values_are_left_for_the_pipeline_and_counted():
    frame = make_input()
    frame.loc[1, "tenure"] = "12"
    frame.loc[1, "TotalCharges"] = ""
    frame.loc[2, "Partner"] = ""
    features, report = prep(frame)
    assert np.isnan(features.loc[1, "TotalCharges"])
    assert pd.isna(features.loc[2, "Partner"])
    assert report["missing_values"] == {"TotalCharges": 1, "Partner": 1}


def test_no_service_labels_become_no():
    frame = make_input()
    frame["OnlineSecurity"] = "No internet service"
    frame["MultipleLines"] = "No phone service"
    features, _ = prep(frame)
    assert (features["OnlineSecurity"] == "No").all()
    assert (features["MultipleLines"] == "No").all()


def test_unseen_category_values_are_reported_not_rejected():
    frame = make_input()
    frame.loc[0, "Contract"] = "Weekly"
    features, report = prep(frame)
    assert report["unseen_categories"] == {"Contract": ["Weekly"]}
    assert features.loc[0, "Contract"] == "Weekly"


def test_whitespace_around_text_is_stripped():
    frame = make_input()
    frame.loc[0, "Partner"] = "  Yes "
    features, _ = prep(frame)
    assert features.loc[0, "Partner"] == "Yes"


# ---------- scoring ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    X, y = make_xy(400, 0)
    return CalibratedModel(build_pipeline("logreg", "baseline", TEST_CFG), "none").fit(X, y)


def test_scores_have_the_right_shape_no_nan_and_a_valid_range(model):
    features, _ = prep(make_input(50, 3))
    proba = score(model, features)
    assert proba.shape == (50,)
    assert not np.isnan(proba).any()
    assert ((proba >= 0) & (proba <= 1)).all()


def test_missing_and_unseen_values_still_give_valid_scores(model):
    frame = make_input(10, 4)
    frame.loc[0, "tenure"] = ""
    frame.loc[1, "Contract"] = "Weekly"
    frame.loc[2, "Partner"] = ""
    features, _ = prep(frame)
    proba = score(model, features)
    assert not np.isnan(proba).any()
    assert ((proba >= 0) & (proba <= 1)).all()


def test_a_missing_column_at_scoring_time_gives_a_clear_error(model):
    features, _ = prep(make_input(10))
    with pytest.raises(PredictionError, match="Contract"):
        score(model, features.drop(columns=["Contract"]))


class FakeModel:
    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.proba, self.proba])


@pytest.mark.parametrize("bad", [np.nan, 1.5, -0.2])
def test_invalid_model_output_is_rejected(bad):
    with pytest.raises(PredictionError, match="invalid probabilities"):
        score(FakeModel(np.full(5, bad)), pd.DataFrame({"a": range(5)}))


def test_wrong_length_model_output_is_rejected():
    with pytest.raises(PredictionError, match="invalid probabilities"):
        score(FakeModel(np.full(3, 0.5)), pd.DataFrame({"a": range(5)}))


# ---------- build_predictions --------------------------------------------------------------


def test_build_predictions_flags_at_or_above_the_threshold():
    ids = pd.Series(["a", "b", "c"])
    result = build_predictions(ids, [0.1, 0.3, 0.9], 0.3, "customerID")
    assert list(result.columns) == ["customerID", "churn_probability", "flagged_for_offer", "threshold_used"]
    assert list(result["flagged_for_offer"]) == [0, 1, 1]
    assert (result["threshold_used"] == 0.3).all()


def test_build_predictions_rejects_bad_input():
    with pytest.raises(PredictionError, match="ids but"):
        build_predictions(pd.Series(["a"]), [0.1, 0.2], 0.3, "customerID")
    for bad in (1.5, -0.1, float("nan"), "0.3", True):
        with pytest.raises(PredictionError, match="threshold"):
            build_predictions(pd.Series(["a"]), [0.1], bad, "customerID")


# ---------- model fingerprint --------------------------------------------------------------


def test_matching_fingerprint_gives_no_warning(tmp_path):
    model_file = tmp_path / "m.joblib"
    model_file.write_bytes(b"model bytes")
    results = tmp_path / "r.json"
    results.write_text(json.dumps({"model_sha256": file_sha256(model_file)}), encoding="utf-8")
    assert model_fingerprint_warning(model_file, results) is None


def test_changed_model_file_gives_a_warning(tmp_path):
    model_file = tmp_path / "m.joblib"
    model_file.write_bytes(b"model bytes")
    results = tmp_path / "r.json"
    results.write_text(json.dumps({"model_sha256": "0" * 64}), encoding="utf-8")
    assert "differs" in model_fingerprint_warning(model_file, results)


def test_missing_results_file_or_hash_gives_no_warning(tmp_path):
    model_file = tmp_path / "m.joblib"
    model_file.write_bytes(b"model bytes")
    assert model_fingerprint_warning(model_file, tmp_path / "nope.json") is None
    results = tmp_path / "r.json"
    results.write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert model_fingerprint_warning(model_file, results) is None


def test_unreadable_results_file_gives_a_warning(tmp_path):
    model_file = tmp_path / "m.joblib"
    model_file.write_bytes(b"model bytes")
    results = tmp_path / "r.json"
    results.write_text("not json", encoding="utf-8")
    assert "Could not read" in model_fingerprint_warning(model_file, results)


# ---------- end to end ---------------------------------------------------------------------


def make_cfg(tmp_path, max_rows: int = 1000) -> dict:
    return {
        **TEST_CFG,
        "prediction": {"max_rows": max_rows},
        "paths": {
            "processed_dir": str(tmp_path),
            "reports_dir": str(tmp_path / "reports"),
            "models_dir": str(tmp_path / "models"),
        },
        "data": {"target": "Churn", "id_column": "customerID"},
    }


def write_inputs(tmp_path, n: int = 40) -> Path:
    """Write a saved model, a threshold file and a customers CSV (no train, validation or test file)."""
    X, y = make_xy(400, 0)
    fitted = CalibratedModel(build_pipeline("logreg", "baseline", TEST_CFG), "none").fit(X, y)
    (tmp_path / "models").mkdir()
    joblib.dump(fitted, tmp_path / "models" / "final_model.joblib")
    (tmp_path / "reports").mkdir()
    write_threshold(tmp_path / "reports" / "threshold_choice.json", threshold=0.3)
    csv = tmp_path / "customers.csv"
    make_input(n, 7).to_csv(csv, index=False)
    return csv


def test_run_predict_writes_one_scored_row_per_customer(tmp_path):
    csv = write_inputs(tmp_path)
    out = tmp_path / "out" / "predictions.csv"
    run_predict(make_cfg(tmp_path), csv, out)

    saved = pd.read_csv(out)
    assert list(saved.columns) == ["customerID", "churn_probability", "flagged_for_offer", "threshold_used"]
    assert saved["customerID"].tolist() == pd.read_csv(csv)["customerID"].tolist()
    assert saved["churn_probability"].between(0, 1).all()
    assert (saved["threshold_used"] == 0.3).all()
    assert (saved["flagged_for_offer"] == (saved["churn_probability"] >= 0.3).astype(int)).all()


def test_threshold_override_is_used(tmp_path):
    csv = write_inputs(tmp_path)
    result = run_predict(make_cfg(tmp_path), csv, tmp_path / "p.csv", threshold=0.0)
    assert (result["flagged_for_offer"] == 1).all()
    assert (result["threshold_used"] == 0.0).all()


def test_invalid_threshold_override_raises(tmp_path):
    csv = write_inputs(tmp_path)
    for bad in (1.5, float("nan")):
        with pytest.raises(PredictionError, match="threshold"):
            run_predict(make_cfg(tmp_path), csv, tmp_path / "p.csv", threshold=bad)


def test_output_cannot_overwrite_the_input(tmp_path):
    csv = write_inputs(tmp_path)
    with pytest.raises(PredictionError, match="overwrite"):
        run_predict(make_cfg(tmp_path), csv, csv)


def test_too_many_rows_raise(tmp_path):
    csv = write_inputs(tmp_path)
    with pytest.raises(PredictionError, match="maximum"):
        run_predict(make_cfg(tmp_path, max_rows=10), csv, tmp_path / "p.csv")


def test_missing_input_file_raises(tmp_path):
    write_inputs(tmp_path)
    with pytest.raises(DataValidationError, match="not found"):
        run_predict(make_cfg(tmp_path), tmp_path / "nope.csv", tmp_path / "p.csv")


def test_header_only_input_raises(tmp_path):
    write_inputs(tmp_path)
    header_only = tmp_path / "header.csv"
    header_only.write_text("customerID,tenure\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="no rows"):
        run_predict(make_cfg(tmp_path), header_only, tmp_path / "p.csv")


def test_missing_model_raises(tmp_path):
    csv = write_inputs(tmp_path)
    (tmp_path / "models" / "final_model.joblib").unlink()
    with pytest.raises(EvaluationError, match="Final model not found"):
        run_predict(make_cfg(tmp_path), csv, tmp_path / "p.csv")


def test_missing_threshold_file_raises_unless_a_threshold_is_given(tmp_path):
    csv = write_inputs(tmp_path)
    (tmp_path / "reports" / "threshold_choice.json").unlink()
    with pytest.raises(EvaluationError, match="not found"):
        run_predict(make_cfg(tmp_path), csv, tmp_path / "p.csv")
    assert len(run_predict(make_cfg(tmp_path), csv, tmp_path / "p.csv", threshold=0.3)) == 40


def test_strict_mode_rejects_unknown_columns(tmp_path):
    write_inputs(tmp_path)
    frame = make_input(40, 7)
    frame["foo"] = "x"
    csv = tmp_path / "extra.csv"
    frame.to_csv(csv, index=False)
    assert len(run_predict(make_cfg(tmp_path), csv, tmp_path / "ok.csv")) == 40
    with pytest.raises(PredictionError, match="foo"):
        run_predict(make_cfg(tmp_path), csv, tmp_path / "no.csv", strict=True)


def test_missing_prediction_section_raises(tmp_path):
    csv = write_inputs(tmp_path)
    cfg = make_cfg(tmp_path)
    del cfg["prediction"]
    with pytest.raises(PredictionError, match="prediction"):
        run_predict(cfg, csv, tmp_path / "p.csv")


@pytest.mark.skipif(
    not (REAL_RAW.exists() and REAL_CLEAN.exists() and REAL_MODEL.exists()),
    reason="real data or the saved model is not present (they are gitignored)",
)
def test_cleaning_raw_input_gives_the_same_scores_as_the_training_pipeline():
    """The raw Telco file, cleaned the inference way, must score exactly like the training-cleaned file."""
    saved_model = joblib.load(REAL_MODEL)
    features, _ = prepare_input(load_raw(REAL_RAW), "customerID", "Churn")
    from_raw = score(saved_model, features)

    clean = pd.read_csv(REAL_CLEAN)
    assert clean["customerID"].tolist() == features["customerID"].tolist()
    from_clean = saved_model.predict_proba(clean.drop(columns=["Churn", "customerID"]))[:, 1]
    np.testing.assert_allclose(from_raw, from_clean, atol=1e-9)