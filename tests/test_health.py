import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import yaml

from src import health
from src.calibration import CalibratedModel
from src.config import load_config
from src.health import (
    HealthError,
    check_config,
    check_health,
    check_imports,
    check_model,
    check_writable,
    main,
    score_sample,
)
from src.models import build_pipeline
from tests.test_models import TEST_CFG, make_xy

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_cfg(root: Path) -> dict:
    cfg = load_config(REPO_ROOT / "configs" / "config.yaml")
    cfg["paths"] = {
        **cfg["paths"],
        "processed_dir": str(root / "processed"),
        "models_dir": str(root / "models"),
        "reports_dir": str(root / "reports"),
        "log_file": str(root / "logs" / "churn.log"),
    }
    return cfg


@pytest.fixture(scope="module")
def fitted_model():
    X, y = make_xy(400, 0)
    return CalibratedModel(build_pipeline("logreg", "baseline", TEST_CFG), "none").fit(X, y)


class FakeModel:
    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.proba, self.proba])


class BrokenModel:
    def predict_proba(self, X):
        raise ValueError("boom")


# ---------- single checks --------------------------------------------------------------------


def test_a_valid_config_is_healthy(tmp_path):
    assert check_config(make_cfg(tmp_path))["ok"]


def test_an_invalid_config_is_unhealthy(tmp_path):
    cfg = make_cfg(tmp_path)
    del cfg["paths"]
    result = check_config(cfg)
    assert not result["ok"]
    assert "paths" in result["detail"]


def test_imports_succeed_here():
    assert check_imports()["ok"]


def test_a_missing_library_is_reported_by_name(monkeypatch):
    monkeypatch.setattr(health, "REQUIRED_MODULES", ("numpy", "no_such_module_xyz"))
    result = check_imports()
    assert not result["ok"]
    assert "no_such_module_xyz" in result["detail"]


def test_writable_folders_are_created_and_checked(tmp_path):
    result = check_writable(make_cfg(tmp_path))
    assert result["ok"]
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_an_unwritable_folder_is_reported(tmp_path):
    cfg = make_cfg(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a folder", encoding="utf-8")
    cfg["paths"]["models_dir"] = str(blocker)
    result = check_writable(cfg)
    assert not result["ok"]
    assert "blocker" in result["detail"]


# ---------- scoring a sample customer ------------------------------------------------------------


def test_score_sample_returns_the_probability():
    assert score_sample(FakeModel([0.3])) == pytest.approx(0.3)


@pytest.mark.parametrize("bad", [[np.nan], [1.5], [-0.1], [0.2, 0.3]])
def test_score_sample_rejects_invalid_output(bad):
    with pytest.raises(HealthError, match="invalid probability"):
        score_sample(FakeModel(bad))


def test_score_sample_wraps_a_model_failure():
    with pytest.raises(HealthError, match="could not score"):
        score_sample(BrokenModel())


# ---------- the model check --------------------------------------------------------------------


def test_a_missing_model_is_unhealthy(tmp_path):
    result = check_model(make_cfg(tmp_path))
    assert not result["ok"]
    assert "not found" in result["detail"]


def test_a_corrupt_model_file_is_unhealthy(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "final_model.joblib").write_bytes(b"this is not a model")
    assert not check_model(cfg)["ok"]


def test_a_valid_saved_model_is_healthy(tmp_path, fitted_model):
    cfg = make_cfg(tmp_path)
    (tmp_path / "models").mkdir()
    joblib.dump(fitted_model, tmp_path / "models" / "final_model.joblib")
    result = check_model(cfg)
    assert result["ok"], result["detail"]
    assert "scores" in result["detail"]


# ---------- the full check -----------------------------------------------------------------


def test_check_health_runs_every_check_in_order(tmp_path):
    report = check_health(make_cfg(tmp_path))
    assert report["healthy"]
    assert [c["check"] for c in report["checks"]] == ["config", "imports", "writable"]


def test_the_model_check_only_runs_when_required(tmp_path):
    cfg = make_cfg(tmp_path)
    assert "model" not in [c["check"] for c in check_health(cfg)["checks"]]
    report = check_health(cfg, require_model=True)
    assert not report["healthy"]
    assert [c["check"] for c in report["checks"]][-1] == "model"


def test_an_invalid_config_skips_the_dependent_checks(tmp_path):
    cfg = make_cfg(tmp_path)
    del cfg["paths"]
    report = check_health(cfg, require_model=True)
    assert not report["healthy"]
    assert [c["check"] for c in report["checks"]] == ["config", "imports"]


# ---------- the command ---------------------------------------------------------------------


def run_main(monkeypatch, capsys, config_path: Path, args: list[str]) -> tuple[int, dict]:
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    return exit_info.value.code, json.loads(capsys.readouterr().out)


def test_main_exits_0_and_prints_json_when_healthy(monkeypatch, capsys, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(make_cfg(tmp_path)), encoding="utf-8")
    code, report = run_main(monkeypatch, capsys, config_path, [])
    assert code == 0
    assert report["healthy"] is True


def test_main_exits_1_when_the_model_is_required_but_missing(monkeypatch, capsys, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(make_cfg(tmp_path)), encoding="utf-8")
    code, report = run_main(monkeypatch, capsys, config_path, ["--require-model"])
    assert code == 1
    assert report["healthy"] is False


def test_main_exits_1_when_the_config_cannot_be_loaded(monkeypatch, capsys, tmp_path):
    code, report = run_main(monkeypatch, capsys, tmp_path / "nope.yaml", [])
    assert code == 1
    assert report["checks"][0]["check"] == "config"
    assert not report["checks"][0]["ok"]