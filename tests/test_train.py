from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import DataValidationError
from src.final_model import FinalModelError
from src.train import TrainError, check_ready, main, run_train
from tests.test_business import GOOD_BUSINESS
from tests.test_final_model import GOOD_FINAL
from tests.test_models import TEST_CFG, make_xy


def make_cfg(root: Path) -> dict:
    return {
        **TEST_CFG,
        "business": GOOD_BUSINESS,
        "final": {**GOOD_FINAL, "calibration_candidates": ["none", "sigmoid"]},
        "split": {"train_size": 0.6, "val_size": 0.2, "test_size": 0.2},
        "paths": {
            "raw_data": str(root / "raw" / "telco.csv"),
            "processed_dir": str(root / "processed"),
            "processed_data": str(root / "processed" / "clean.csv"),
            "models_dir": str(root / "models"),
            "reports_dir": str(root / "reports"),
            "log_file": str(root / "logs" / "churn.log"),
        },
        "data": {"target": "Churn", "id_column": "customerID", "positive_label": "Yes"},
    }


def write_raw(cfg: dict, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A raw Telco-style CSV (all columns, text labels) with a real signal for the model to find."""
    X, y = make_xy(n, seed)
    raw = X.copy()
    raw.insert(0, "customerID", [f"ID-{i:05d}" for i in range(n)])
    raw["gender"] = np.where(np.arange(n) % 2 == 0, "Male", "Female")
    raw["Churn"] = np.where(y == 1, "Yes", "No")
    path = Path(cfg["paths"]["raw_data"])
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path, index=False)
    return raw


def test_run_train_reproduces_the_model_without_evaluating_the_test_split(tmp_path):
    cfg = make_cfg(tmp_path)
    write_raw(cfg)
    result = run_train(cfg)

    processed, reports = Path(cfg["paths"]["processed_dir"]), Path(cfg["paths"]["reports_dir"])
    sizes = {name: len(pd.read_csv(processed / f"{name}.csv")) for name in ("train", "val", "test")}
    assert sum(sizes.values()) == 400
    assert result["rows"] == 400
    assert result["split_rows"] == sizes
    assert result["calibration"] in {"none", "sigmoid"}
    assert 0 <= result["threshold"] <= 1
    assert (reports / "threshold_choice.json").exists()
    assert not (reports / "test_results.json").exists()

    model = joblib.load(result["model_path"])
    X_new, _ = make_xy(50, 9)
    assert model.predict_proba(X_new).shape == (50, 2)


def test_training_twice_gives_the_same_model(tmp_path):
    results, models = [], []
    for name in ("a", "b"):
        cfg = make_cfg(tmp_path / name)
        write_raw(cfg)
        results.append(run_train(cfg))
        models.append(joblib.load(results[-1]["model_path"]))
    X_new, _ = make_xy(50, 9)
    assert results[0]["threshold"] == results[1]["threshold"]
    np.testing.assert_allclose(models[0].predict_proba(X_new), models[1].predict_proba(X_new))


def test_a_missing_raw_file_says_where_to_get_it(tmp_path):
    with pytest.raises(TrainError, match="Kaggle"):
        run_train(make_cfg(tmp_path))


def test_a_raw_path_that_is_a_folder_is_rejected(tmp_path):
    cfg = make_cfg(tmp_path)
    Path(cfg["paths"]["raw_data"]).mkdir(parents=True)
    with pytest.raises(TrainError, match="not found"):
        check_ready(cfg)


def test_the_lock_blocks_retraining_unless_forced(tmp_path):
    cfg = make_cfg(tmp_path)
    write_raw(cfg)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True)
    (reports / "test_results.json").write_text("{}", encoding="utf-8")

    with pytest.raises(TrainError, match="already evaluated"):
        check_ready(cfg)
    with pytest.raises(TrainError, match="already evaluated"):
        run_train(cfg)
    check_ready(cfg, force=True)
    assert not (Path(cfg["paths"]["models_dir"]) / "final_model.joblib").exists()


def test_bad_raw_data_stops_training_with_the_data_error(tmp_path):
    cfg = make_cfg(tmp_path)
    raw = write_raw(cfg)
    raw.loc[0, "Contract"] = "Weekly"
    raw.to_csv(cfg["paths"]["raw_data"], index=False)
    with pytest.raises(DataValidationError, match="Contract"):
        run_train(cfg)


def test_a_failing_step_leaves_no_model_behind(tmp_path):
    cfg = make_cfg(tmp_path)
    write_raw(cfg)
    cfg["final"] = {**cfg["final"], "model": "svm"}
    with pytest.raises(FinalModelError, match="model"):
        run_train(cfg)
    assert not (Path(cfg["paths"]["models_dir"]) / "final_model.joblib").exists()


def test_main_reports_a_missing_config_and_exits_with_1(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "nope.yaml"))
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 1
    assert "ERROR" in capsys.readouterr().err