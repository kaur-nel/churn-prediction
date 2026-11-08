import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.compare import (
    CompareError,
    RunTracker,
    evaluate_models,
    run_compare,
    top_models,
    tracking_settings,
)
from src.models import MODEL_NAMES, ModelError
from src.split import SplitError
from tests.test_models import TEST_CFG, make_xy

TABLE_COLUMNS = {
    "model", "strategy", "pr_auc", "pr_auc_fold_mean", "pr_auc_fold_std", "roc_auc",
    "best_threshold", "precision_best", "recall_best", "f1_best", "fit_seconds",
}


def test_all_models_and_strategies_are_compared_and_sorted():
    X, y = make_xy()
    table = evaluate_models(X, y, TEST_CFG)
    assert len(table) == len(MODEL_NAMES) * 2
    assert set(table.columns) == TABLE_COLUMNS
    assert table["pr_auc"].is_monotonic_decreasing
    unit_columns = ["pr_auc", "roc_auc", "precision_best", "recall_best", "f1_best", "best_threshold"]
    assert ((table[unit_columns] >= 0) & (table[unit_columns] <= 1)).all().all()
    assert (table["fit_seconds"] >= 0).all()


def test_models_beat_random_guessing_on_signal_data():
    X, y = make_xy()
    table = evaluate_models(X, y, TEST_CFG, models=("logreg",), strategies=("baseline",))
    assert table.loc[0, "pr_auc"] > y.mean()


def test_empty_model_or_strategy_list_raises():
    X, y = make_xy()
    with pytest.raises(CompareError, match="at least one"):
        evaluate_models(X, y, TEST_CFG, models=())
    with pytest.raises(CompareError, match="at least one"):
        evaluate_models(X, y, TEST_CFG, strategies=())


def test_unknown_strategy_raises():
    X, y = make_xy()
    with pytest.raises(ModelError, match="Unknown strategy"):
        evaluate_models(X, y, TEST_CFG, models=("logreg",), strategies=("magic",))


def test_single_class_target_raises():
    X, y = make_xy()
    with pytest.raises(CompareError, match="both classes"):
        evaluate_models(X, y * 0, TEST_CFG, models=("logreg",))


# ---------- top_models -----------------------------------------------------------------------


def make_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["logreg", "logreg", "lightgbm", "random_forest"],
            "strategy": ["baseline", "class_weight", "class_weight", "baseline"],
            "pr_auc": [0.66, 0.65, 0.63, 0.60],
        }
    )


def test_top_models_returns_best_row_per_model():
    assert top_models(make_table(), 2) == [("logreg", "baseline"), ("lightgbm", "class_weight")]


def test_top_models_k_larger_than_models_returns_all():
    assert len(top_models(make_table(), 10)) == 3


def test_top_models_bad_input_raises():
    with pytest.raises(CompareError, match="at least 1"):
        top_models(make_table(), 0)
    with pytest.raises(CompareError, match="empty or missing"):
        top_models(make_table().iloc[0:0])
    with pytest.raises(CompareError, match="empty or missing"):
        top_models(make_table().drop(columns=["pr_auc"]))


# ---------- tracking -------------------------------------------------------------------------


def test_tracking_settings_defaults_and_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    assert tracking_settings() == ("sqlite:///mlflow.db", "churn-prediction")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///other.db")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "exp-x")
    assert tracking_settings() == ("sqlite:///other.db", "exp-x")


def test_runs_are_recorded_in_mlflow(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    X, y = make_xy()
    tracker = RunTracker(uri, "test-experiment")
    evaluate_models(X, y, TEST_CFG, models=("logreg",), tracker=tracker)

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("test-experiment")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 2
    assert {run.data.params["strategy"] for run in runs} == {"baseline", "class_weight"}
    assert all("pr_auc" in run.data.metrics for run in runs)


# ---------- end to end -----------------------------------------------------------------------


def make_run_cfg(tmp_path) -> dict:
    return {
        **TEST_CFG,
        "paths": {"processed_dir": str(tmp_path), "reports_dir": str(tmp_path / "reports")},
        "data": {"target": "Churn", "id_column": "customerID"},
    }


def test_run_compare_reads_only_train_and_writes_the_table(tmp_path):
    X, y = make_xy()
    train = X.copy()
    train["customerID"] = [f"ID-{i}" for i in range(len(train))]
    train["Churn"] = y
    train.to_csv(tmp_path / "train.csv", index=False)  # no val.csv or test.csv exist

    table = run_compare(make_run_cfg(tmp_path), track=False)
    assert len(table) == len(MODEL_NAMES) * 2
    assert (tmp_path / "reports" / "model_comparison.csv").exists()


def test_run_compare_without_train_file_raises(tmp_path):
    with pytest.raises(SplitError, match="not found"):
        run_compare(make_run_cfg(tmp_path), track=False)