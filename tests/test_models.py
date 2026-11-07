import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from src.models import MODEL_NAMES, STRATEGIES, ModelError, build_pipeline
from tests.test_features import make_raw

TEST_CFG = {
    "project": {"random_seed": 42},
    "cv": {"n_splits": 3},
    "models": {
        "logreg_drop_columns": ["TotalCharges"],
        "logreg": {"C": 1.0, "max_iter": 500},
        "random_forest": {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 3},
        "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
        "lightgbm": {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5, "learning_rate": 0.1},
    },
}


def make_xy(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic data where churn depends on contract and tenure, plus 10% label noise."""
    raw = make_raw(n, seed)
    rng = np.random.default_rng(seed + 1)
    signal = ((raw["Contract"] == "Month-to-month") & (raw["tenure"] < 30)).to_numpy()
    flip = rng.random(n) < 0.1
    return raw, pd.Series((signal ^ flip).astype(int), name="Churn")


def preprocessed_columns(pipeline: ImbPipeline, X: pd.DataFrame) -> list[str]:
    """Column names the model would receive (runs the 'features' and 'preprocess' steps)."""
    engineered = pipeline.named_steps["features"].fit_transform(X)
    return list(pipeline.named_steps["preprocess"].fit_transform(engineered).columns)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_and_strategy_fits_and_predicts(model_name, strategy):
    X, y = make_xy()
    pipeline = build_pipeline(model_name, strategy, TEST_CFG, pos_weight=3.0).fit(X, y)
    proba = pipeline.predict_proba(X.iloc[:10])
    assert proba.shape == (10, 2)
    assert not np.isnan(proba).any()
    assert ((proba >= 0) & (proba <= 1)).all()


def test_smote_only_present_for_smote_strategy():
    for strategy in STRATEGIES:
        pipeline = build_pipeline("logreg", strategy, TEST_CFG)
        assert isinstance(pipeline, ImbPipeline)
        assert ("smote" in dict(pipeline.steps)) == (strategy == "smote")


def test_smote_comes_after_preprocessing_and_before_the_model():
    names = [name for name, _ in build_pipeline("logreg", "smote", TEST_CFG).steps]
    assert names == ["features", "preprocess", "smote", "model"]


def test_class_weight_is_set_only_for_class_weight_strategy():
    assert build_pipeline("logreg", "class_weight", TEST_CFG).named_steps["model"].class_weight == "balanced"
    assert build_pipeline("logreg", "baseline", TEST_CFG).named_steps["model"].class_weight is None


def test_logreg_drops_total_charges_but_lightgbm_keeps_it():
    X, _ = make_xy()
    logreg_cols = preprocessed_columns(build_pipeline("logreg", "baseline", TEST_CFG), X)
    lgbm_cols = preprocessed_columns(build_pipeline("lightgbm", "baseline", TEST_CFG), X)
    assert "TotalCharges" not in logreg_cols
    assert "TotalCharges" in lgbm_cols


def test_params_override_config():
    assert build_pipeline("logreg", "baseline", TEST_CFG, params={"C": 0.1}).named_steps["model"].C == 0.1


def test_same_seed_gives_same_predictions_with_smote():
    X, y = make_xy()
    a = build_pipeline("logreg", "smote", TEST_CFG).fit(X, y).predict_proba(X)
    b = build_pipeline("logreg", "smote", TEST_CFG).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)


def test_unknown_model_raises():
    with pytest.raises(ModelError, match="Unknown model"):
        build_pipeline("svm", "baseline", TEST_CFG)


def test_unknown_strategy_raises():
    with pytest.raises(ModelError, match="Unknown strategy"):
        build_pipeline("logreg", "magic", TEST_CFG)


def test_xgboost_class_weight_needs_pos_weight():
    with pytest.raises(ModelError, match="pos_weight"):
        build_pipeline("xgboost", "class_weight", TEST_CFG)
    with pytest.raises(ModelError, match="pos_weight"):
        build_pipeline("xgboost", "class_weight", TEST_CFG, pos_weight=0)


def test_missing_config_entry_raises():
    cfg = {"project": {"random_seed": 1}, "models": {}}
    with pytest.raises(ModelError, match="missing"):
        build_pipeline("logreg", "baseline", cfg)


def test_bad_parameters_raise():
    with pytest.raises(ModelError, match="Bad parameters"):
        build_pipeline("logreg", "baseline", TEST_CFG, params={"nonsense": 1})