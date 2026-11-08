import json

import joblib
import pandas as pd
import pytest

from src.final_model import FinalModelError, choose_calibration, load_final_settings, run_final_selection
from src.split import SplitError
from tests.test_models import TEST_CFG, make_xy

GOOD_FINAL = {
    "model": "logreg",
    "strategy": "baseline",
    "params": {},
    "calibration_candidates": ["none", "sigmoid", "isotonic"],
    "calibration_bins": 5,
    "min_brier_gain": 0.0,
}


def cfg_with(**overrides) -> dict:
    return {**TEST_CFG, "final": {**GOOD_FINAL, **overrides}}


# ---------- settings ---------------------------------------------------------------------


def test_valid_settings_pass():
    assert load_final_settings(cfg_with())["model"] == "logreg"


def test_none_is_always_added_first_and_duplicates_removed():
    settings = load_final_settings(cfg_with(calibration_candidates=["sigmoid", "sigmoid"]))
    assert settings["calibration_candidates"] == ["none", "sigmoid"]


def test_missing_final_section_raises():
    with pytest.raises(FinalModelError, match="'final' section"):
        load_final_settings(TEST_CFG)


def test_missing_key_raises():
    cfg = cfg_with()
    del cfg["final"]["calibration_bins"]
    with pytest.raises(FinalModelError, match="calibration_bins"):
        load_final_settings(cfg)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("model", "svm"),
        ("strategy", "magic"),
        ("params", "C=1"),
        ("params", {1: 2}),
        ("calibration_candidates", []),
        ("calibration_candidates", "none"),
        ("calibration_candidates", ["none", "magic"]),
        ("calibration_bins", 1),
        ("calibration_bins", True),
        ("min_brier_gain", -0.1),
        ("min_brier_gain", "0.001"),
        ("min_brier_gain", True),
        ("min_brier_gain", float("nan")),
    ],
)
def test_bad_settings_raise(key, bad):
    with pytest.raises(FinalModelError, match=key):
        load_final_settings(cfg_with(**{key: bad}))


# ---------- choosing a method --------------------------------------------------------------


def make_table(none: float, sigmoid: float, isotonic: float) -> pd.DataFrame:
    return pd.DataFrame({"method": ["none", "sigmoid", "isotonic"], "brier": [none, sigmoid, isotonic]})


def test_choose_picks_a_clearly_better_method():
    assert choose_calibration(make_table(0.200, 0.195, 0.190), 0.001) == "isotonic"


def test_choose_keeps_none_when_the_gain_is_too_small():
    assert choose_calibration(make_table(0.2000, 0.1995, 0.1996), 0.001) == "none"


def test_choose_keeps_none_when_it_is_best():
    assert choose_calibration(make_table(0.19, 0.20, 0.21), 0.0) == "none"


def test_choose_needs_a_none_row_and_columns():
    with pytest.raises(FinalModelError, match="'none'"):
        choose_calibration(make_table(0.2, 0.1, 0.1).iloc[1:], 0.001)
    with pytest.raises(FinalModelError, match="empty or missing"):
        choose_calibration(make_table(0.2, 0.1, 0.1).iloc[0:0], 0.001)
    with pytest.raises(FinalModelError, match="empty or missing"):
        choose_calibration(make_table(0.2, 0.1, 0.1).drop(columns=["brier"]), 0.001)


# ---------- end to end ---------------------------------------------------------------------


def make_run_cfg(tmp_path, **final_overrides) -> dict:
    return {
        **cfg_with(**final_overrides),
        "paths": {
            "processed_dir": str(tmp_path),
            "reports_dir": str(tmp_path / "reports"),
            "models_dir": str(tmp_path / "models"),
        },
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


def test_run_final_selection_saves_a_frozen_model_without_touching_test(tmp_path):
    write_splits(tmp_path)
    table, chosen = run_final_selection(make_run_cfg(tmp_path))

    assert list(table["method"]) == ["none", "sigmoid", "isotonic"]
    assert chosen in set(table["method"])
    assert (tmp_path / "reports" / "calibration_comparison.csv").exists()
    assert (tmp_path / "reports" / "figures" / "calibration_validation.png").exists()

    choice = json.loads((tmp_path / "reports" / "final_model_choice.json").read_text(encoding="utf-8"))
    assert choice["calibration"] == chosen
    assert choice["model"] == "logreg"

    model = joblib.load(tmp_path / "models" / "final_model.joblib")
    X_val, _ = make_xy(200, 5)
    proba = model.predict_proba(X_val)
    assert proba.shape == (200, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_none_is_evaluated_even_if_not_listed(tmp_path):
    write_splits(tmp_path)
    table, _ = run_final_selection(make_run_cfg(tmp_path, calibration_candidates=["sigmoid"]))
    assert list(table["method"]) == ["none", "sigmoid"]


def test_missing_validation_file_raises(tmp_path):
    write_splits(tmp_path)
    (tmp_path / "val.csv").unlink()
    with pytest.raises(SplitError, match="not found"):
        run_final_selection(make_run_cfg(tmp_path))