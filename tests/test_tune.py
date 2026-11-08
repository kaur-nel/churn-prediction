import json

import optuna
import pandas as pd
import pytest

from src.metrics import MetricsError
from src.split import SplitError
from src.tune import (
    TuneError,
    run_tuning,
    strategy_for_model,
    suggest_params,
    tune_model,
    validate_search_space,
)
from tests.test_models import TEST_CFG, make_xy

LOGREG_SPACE = {"C": {"type": "float", "low": 0.01, "high": 10.0, "log": True}}
RF_SPACE = {
    "n_estimators": {"type": "int", "low": 10, "high": 30},
    "max_depth": {"type": "int", "low": 2, "high": 5},
}
TUNE_CFG = {
    **TEST_CFG,
    "tuning": {
        "n_trials": 3,
        "models": ["logreg"],
        "search_space": {"logreg": LOGREG_SPACE, "random_forest": RF_SPACE},
    },
}


# ---------- search space -----------------------------------------------------------------


def test_valid_search_space_passes():
    validate_search_space(LOGREG_SPACE, "logreg")
    validate_search_space(RF_SPACE, "random_forest")
    validate_search_space({"c": {"type": "categorical", "choices": ["a", "b"]}}, "x")


@pytest.mark.parametrize(
    "space",
    [
        {},
        "nope",
        {"x": 5},
        {"x": {"type": "weird", "low": 0, "high": 1}},
        {"x": {"type": "float", "low": 1, "high": 1}},
        {"x": {"type": "float", "low": 2, "high": 1}},
        {"x": {"type": "float", "low": "a", "high": 1}},
        {"x": {"type": "float", "low": True, "high": 2}},
        {"x": {"type": "int", "low": 0.5, "high": 3}},
        {"x": {"type": "float", "low": 0, "high": 1, "log": True}},
        {"x": {"type": "categorical", "choices": ["only"]}},
        {"x": {"type": "categorical"}},
    ],
)
def test_bad_search_spaces_raise(space):
    with pytest.raises(TuneError, match="search space"):
        validate_search_space(space, "logreg")


def test_suggest_params_returns_values_inside_the_space():
    space = {
        "a": {"type": "int", "low": 1, "high": 5},
        "b": {"type": "float", "low": 0.0, "high": 1.0},
        "c": {"type": "categorical", "choices": ["x", "y"]},
    }
    trial = optuna.trial.FixedTrial({"a": 3, "b": 0.5, "c": "y"})
    assert suggest_params(trial, space) == {"a": 3, "b": 0.5, "c": "y"}


# ---------- tune_model -------------------------------------------------------------------


def test_tune_model_returns_params_inside_the_search_space():
    X, y = make_xy()
    result = tune_model("logreg", "baseline", X, y, TUNE_CFG)
    assert 0.01 <= result["best_params"]["C"] <= 10.0
    assert 0 <= result["cv_default"] <= 1
    assert 0 <= result["cv_tuned"] <= 1
    assert result["n_completed"] == 3


def test_tuning_is_reproducible_with_the_same_seed():
    X, y = make_xy()
    a = tune_model("logreg", "baseline", X, y, TUNE_CFG)
    b = tune_model("logreg", "baseline", X, y, TUNE_CFG)
    assert a["best_params"] == b["best_params"]
    assert a["cv_tuned"] == pytest.approx(b["cv_tuned"])


def test_random_forest_can_be_tuned():
    X, y = make_xy()
    best = tune_model("random_forest", "class_weight", X, y, TUNE_CFG)["best_params"]
    assert 10 <= best["n_estimators"] <= 30
    assert 2 <= best["max_depth"] <= 5


def test_missing_search_space_raises():
    X, y = make_xy()
    with pytest.raises(TuneError, match="search space"):
        tune_model("xgboost", "baseline", X, y, TUNE_CFG)


def test_single_class_target_raises():
    X, y = make_xy()
    with pytest.raises(TuneError, match="both classes"):
        tune_model("logreg", "baseline", X, y * 0, TUNE_CFG)


def test_all_failing_trials_raise_instead_of_returning_a_fake_best(monkeypatch):
    def boom(*args, **kwargs):
        raise MetricsError("boom")

    monkeypatch.setattr("src.tune.cross_val_probabilities", boom)
    X, y = make_xy()
    with pytest.raises(TuneError, match="No trial"):
        tune_model("logreg", "baseline", X, y, TUNE_CFG)


# ---------- strategy lookup --------------------------------------------------------------


def make_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["logreg", "logreg", "random_forest"],
            "strategy": ["baseline", "class_weight", "class_weight"],
            "pr_auc": [0.60, 0.50, 0.65],
        }
    )


def test_strategy_for_model_picks_the_best_row():
    assert strategy_for_model(make_comparison(), "logreg") == "baseline"
    assert strategy_for_model(make_comparison(), "random_forest") == "class_weight"


def test_strategy_for_unknown_model_raises():
    with pytest.raises(TuneError, match="not found"):
        strategy_for_model(make_comparison(), "xgboost")


def test_strategy_table_without_needed_columns_raises():
    with pytest.raises(TuneError, match="columns"):
        strategy_for_model(make_comparison().drop(columns=["pr_auc"]), "logreg")


# ---------- end to end -------------------------------------------------------------------


def make_run_cfg(tmp_path, models=("logreg",)) -> dict:
    return {
        **TUNE_CFG,
        "tuning": {**TUNE_CFG["tuning"], "models": list(models)},
        "paths": {"processed_dir": str(tmp_path), "reports_dir": str(tmp_path / "reports")},
        "data": {"target": "Churn", "id_column": "customerID"},
    }


def write_inputs(tmp_path, with_comparison=True) -> None:
    """Write train.csv and val.csv (and no test.csv: the test set must not be needed)."""
    for name, seed, n in (("train", 0, 400), ("val", 5, 200)):
        X, y = make_xy(n, seed)
        frame = X.copy()
        frame["customerID"] = [f"{name}-{i}" for i in range(len(frame))]
        frame["Churn"] = y
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    if with_comparison:
        (tmp_path / "reports").mkdir()
        make_comparison().to_csv(tmp_path / "reports" / "model_comparison.csv", index=False)


def test_run_tuning_writes_results_without_touching_the_test_split(tmp_path):
    write_inputs(tmp_path)
    table = run_tuning(make_run_cfg(tmp_path), track=False)

    assert len(table) == 1
    assert {"cv_pr_auc_default", "cv_pr_auc_tuned", "val_pr_auc_default", "val_pr_auc_tuned"} <= set(table.columns)
    assert (tmp_path / "reports" / "tuning_results.csv").exists()
    saved = json.loads((tmp_path / "reports" / "best_params.json").read_text(encoding="utf-8"))
    assert saved["logreg"]["strategy"] == "baseline"
    assert "C" in saved["logreg"]["params"]


def test_run_tuning_without_comparison_table_raises(tmp_path):
    write_inputs(tmp_path, with_comparison=False)
    with pytest.raises(TuneError, match="compare"):
        run_tuning(make_run_cfg(tmp_path), track=False)


def test_run_tuning_without_validation_file_raises(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "val.csv").unlink()
    with pytest.raises(SplitError, match="not found"):
        run_tuning(make_run_cfg(tmp_path), track=False)


def test_run_tuning_unknown_model_raises(tmp_path):
    with pytest.raises(TuneError, match="Unknown"):
        run_tuning(make_run_cfg(tmp_path, models=("svm",)), track=False)


def test_run_tuning_empty_model_list_raises(tmp_path):
    with pytest.raises(TuneError, match="at least one"):
        run_tuning(make_run_cfg(tmp_path, models=()), track=False)