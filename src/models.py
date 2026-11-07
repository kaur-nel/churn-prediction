"""Model factory: builds a full pipeline (preprocessing, optional SMOTE, classifier)."""

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.features import build_preprocessor

MODEL_NAMES = ("logreg", "random_forest", "xgboost", "lightgbm")
STRATEGIES = ("baseline", "class_weight", "smote")


class ModelError(Exception):
    """Raised when a model or strategy cannot be built."""


def _make_estimator(
    model_name: str, strategy: str, params: dict[str, Any], seed: int, pos_weight: float | None
) -> Any:
    weight = "balanced" if strategy == "class_weight" else None
    try:
        if model_name == "xgboost":
            if strategy == "class_weight":
                if pos_weight is None or pos_weight <= 0:
                    raise ModelError(
                        "XGBoost has no class_weight option: pass pos_weight = (#negatives / #positives) > 0"
                    )
                params = {**params, "scale_pos_weight": pos_weight}
            return XGBClassifier(random_state=seed, eval_metric="logloss", **params)
        if model_name == "logreg":
            return LogisticRegression(random_state=seed, class_weight=weight, **params)
        if model_name == "random_forest":
            return RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight=weight, **params)
        return LGBMClassifier(random_state=seed, verbose=-1, class_weight=weight, **params)
    except TypeError as exc:
        raise ModelError(f"Bad parameters for '{model_name}': {exc}") from exc


def build_pipeline(
    model_name: str,
    strategy: str,
    cfg: dict[str, Any],
    pos_weight: float | None = None,
    params: dict[str, Any] | None = None,
) -> ImbPipeline:
    """Build an unfitted pipeline: features -> preprocess -> (SMOTE) -> classifier.

    The steps are laid out flat in ONE imblearn Pipeline (imblearn does not allow a Pipeline
    nested inside a Pipeline). SMOTE is applied only while fitting and only to the training
    data it is given. It is skipped at prediction time.

    Args:
        strategy: "baseline", "class_weight" or "smote".
        pos_weight: #negatives / #positives, needed only for XGBoost with "class_weight".
        params: overrides for the model parameters in the config (used by Optuna later).

    Raises:
        ModelError: for an unknown model/strategy, missing config entries or bad parameters.
    """
    if model_name not in MODEL_NAMES:
        raise ModelError(f"Unknown model '{model_name}'; choose one of {MODEL_NAMES}")
    if strategy not in STRATEGIES:
        raise ModelError(f"Unknown strategy '{strategy}'; choose one of {STRATEGIES}")
    try:
        model_cfg = cfg["models"]
        model_params = dict(model_cfg[model_name])
        seed = cfg["project"]["random_seed"]
    except KeyError as exc:
        raise ModelError(f"Config is missing {exc} needed to build '{model_name}'") from exc
    if params:
        model_params.update(params)

    drop = model_cfg.get("logreg_drop_columns", []) if model_name == "logreg" else []
    # Unpack the preprocessing steps ("features", "preprocess") instead of nesting a Pipeline.
    steps: list[tuple[str, Any]] = list(build_preprocessor(drop).steps)
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("model", _make_estimator(model_name, strategy, model_params, seed, pos_weight)))
    return ImbPipeline(steps)