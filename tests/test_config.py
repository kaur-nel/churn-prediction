import copy
from pathlib import Path

import pytest
import yaml

from src.config import ConfigError, load_config, validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID = {
    "project": {"random_seed": 42},
    "paths": {
        "raw_data": "data/raw/x.csv",
        "processed_dir": "data/processed",
        "processed_data": "data/processed/telco_clean.csv",
        "models_dir": "models",
        "reports_dir": "reports",
        "log_file": "logs/churn.log",
    },
    "data": {"target": "Churn", "id_column": "customerID", "positive_label": "Yes"},
    "split": {"train_size": 0.6, "val_size": 0.2, "test_size": 0.2},
    "cv": {"n_splits": 5},
    "business": {"customer_ltv": 1000.0, "retention_offer_cost": 50.0, "offer_success_rate": 0.3},
    "models": {
        "logreg_drop_columns": ["TotalCharges"],
        "logreg": {"C": 1.0},
        "random_forest": {"n_estimators": 10},
        "xgboost": {"n_estimators": 10},
        "lightgbm": {"n_estimators": 10},
    },
    "tuning": {
        "n_trials": 30,
        "models": ["random_forest", "logreg"],
        "search_space": {"logreg": {"C": {"type": "float", "low": 0.001, "high": 100.0, "log": True}}},
    },
}


def make_cfg() -> dict:
    return copy.deepcopy(VALID)


def write_yaml(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_real_config_file_is_valid():
    cfg = load_config(REPO_ROOT / "configs" / "config.yaml")
    assert cfg["project"]["random_seed"] == 42
    assert cfg["data"]["positive_label"] == "Yes"  # must stay a string, not True


def test_valid_config_passes():
    validate_config(make_cfg())


def test_load_from_explicit_path(tmp_path):
    path = write_yaml(tmp_path / "c.yaml", make_cfg())
    assert load_config(path)["cv"]["n_splits"] == 5


def test_load_from_env_var(tmp_path, monkeypatch):
    path = write_yaml(tmp_path / "env.yaml", make_cfg())
    monkeypatch.setenv("CONFIG_PATH", str(path))
    assert load_config()["project"]["random_seed"] == 42


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_directory_path_raises(tmp_path):
    with pytest.raises(ConfigError, match="not a file"):
        load_config(tmp_path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_broken_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("split: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_top_level_list_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="key: value"):
        load_config(path)


@pytest.mark.parametrize("section", list(VALID))
def test_missing_section_raises(section):
    cfg = make_cfg()
    del cfg[section]
    with pytest.raises(ConfigError, match=section):
        validate_config(cfg)


def test_missing_key_raises():
    cfg = make_cfg()
    del cfg["business"]["customer_ltv"]
    with pytest.raises(ConfigError, match="customer_ltv"):
        validate_config(cfg)


def test_section_not_a_mapping_raises():
    cfg = make_cfg()
    cfg["cv"] = 5
    with pytest.raises(ConfigError, match="cv"):
        validate_config(cfg)


def test_split_not_summing_to_one_raises():
    cfg = make_cfg()
    cfg["split"]["train_size"] = 0.7
    with pytest.raises(ConfigError, match="sum"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    [
        ("project", "random_seed", -1),
        ("project", "random_seed", "42"),
        ("project", "random_seed", True),
        ("cv", "n_splits", 1),
        ("cv", "n_splits", 2.5),
        ("business", "customer_ltv", 0),
        ("business", "customer_ltv", "1000"),
        ("business", "customer_ltv", float("nan")),
        ("business", "retention_offer_cost", -5),
        ("business", "offer_success_rate", 1.5),
        ("business", "offer_success_rate", -0.1),
        ("paths", "raw_data", ""),
        ("data", "target", None),
        ("data", "positive_label", True),
        ("split", "val_size", "0.2"),
        ("split", "test_size", 0),
        ("tuning", "n_trials", 0),
        ("tuning", "n_trials", "30"),
        ("tuning", "n_trials", True),
        ("tuning", "models", []),
        ("tuning", "models", "logreg"),
        ("tuning", "search_space", "x"),
    ],
)
def test_bad_values_raise(section, key, bad_value):
    cfg = make_cfg()
    cfg[section][key] = bad_value
    with pytest.raises(ConfigError, match=key):
        validate_config(cfg)


def test_bad_logreg_drop_columns_raises():
    cfg = make_cfg()
    cfg["models"]["logreg_drop_columns"] = "TotalCharges"
    with pytest.raises(ConfigError, match="logreg_drop_columns"):
        validate_config(cfg)


def test_model_params_must_be_a_mapping():
    cfg = make_cfg()
    cfg["models"]["lightgbm"] = 5
    with pytest.raises(ConfigError, match="lightgbm"):
        validate_config(cfg)