import pytest

from src.imbalance import ImbalanceError, compare_strategies, run_comparison
from src.models import ModelError
from tests.test_models import TEST_CFG, make_xy

COLUMNS = {
    "model", "strategy", "pr_auc", "pr_auc_fold_std", "roc_auc", "precision_t50", "recall_t50",
    "f1_t50", "best_threshold", "precision_best", "recall_best", "f1_best",
}


def test_compare_strategies_returns_one_row_per_pair():
    X, y = make_xy()
    table = compare_strategies(X, y, TEST_CFG, models=("logreg",))
    assert set(table.columns) == COLUMNS
    assert list(table["strategy"]) == ["baseline", "class_weight", "smote"]
    metric_cols = [c for c in COLUMNS if c not in ("model", "strategy")]
    assert ((table[metric_cols] >= 0) & (table[metric_cols] <= 1)).all().all()


def test_baseline_beats_random_guessing_on_signal_data():
    X, y = make_xy()
    table = compare_strategies(X, y, TEST_CFG, models=("logreg",), strategies=("baseline",))
    assert table.loc[0, "pr_auc"] > y.mean()


def test_lightgbm_runs_too():
    X, y = make_xy()
    table = compare_strategies(X, y, TEST_CFG, models=("lightgbm",), strategies=("class_weight",))
    assert len(table) == 1


def test_unknown_strategy_raises():
    X, y = make_xy()
    with pytest.raises(ModelError, match="Unknown strategy"):
        compare_strategies(X, y, TEST_CFG, models=("logreg",), strategies=("magic",))


def test_empty_model_list_raises():
    X, y = make_xy()
    with pytest.raises(ImbalanceError, match="at least one"):
        compare_strategies(X, y, TEST_CFG, models=())


def make_run_cfg(tmp_path) -> dict:
    return {
        **TEST_CFG,
        "paths": {"processed_dir": str(tmp_path), "reports_dir": str(tmp_path / "reports")},
        "data": {"target": "Churn", "id_column": "customerID"},
    }


def test_run_comparison_reads_only_the_train_split(tmp_path):
    X, y = make_xy()
    train = X.copy()
    train["customerID"] = [f"ID-{i}" for i in range(len(train))]
    train["gender"] = "Male"
    train["Churn"] = y
    train.to_csv(tmp_path / "train.csv", index=False)  # no val.csv or test.csv exist

    table = run_comparison(make_run_cfg(tmp_path))
    assert len(table) == 6
    assert (tmp_path / "reports" / "imbalance_comparison.csv").exists()


def test_run_comparison_without_train_file_raises(tmp_path):
    from src.split import SplitError

    with pytest.raises(SplitError, match="not found"):
        run_comparison(make_run_cfg(tmp_path))