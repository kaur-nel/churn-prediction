import numpy as np
import pytest

from src.metrics import (
    MetricsError,
    best_f1_threshold,
    cross_val_probabilities,
    ranking_metrics,
    threshold_metrics,
)
from src.models import build_pipeline
from tests.test_models import TEST_CFG, make_xy


def test_threshold_metrics_known_values():
    result = threshold_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
    assert (result["tp"], result["fn"], result["fp"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == result["recall"] == result["f1"] == 0.5


def test_threshold_is_inclusive():
    assert threshold_metrics([1, 0], [0.5, 0.1], 0.5)["tp"] == 1


def test_no_predicted_positives_gives_zero_not_a_crash():
    result = threshold_metrics([1, 1, 0, 0], [0.1, 0.1, 0.1, 0.1], 0.5)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_ranking_metrics_perfect_and_inverted():
    perfect = ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect["pr_auc"] == pytest.approx(1.0)
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert ranking_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])["roc_auc"] == pytest.approx(0.0)


def test_best_f1_threshold_on_separable_data():
    threshold, f1 = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert f1 == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.8


@pytest.mark.parametrize(
    ("y", "p", "message"),
    [
        ([], [], "Empty"),
        ([0, 1], [0.5], "mismatch"),
        ([0, 0, 0], [0.1, 0.2, 0.3], "both classes"),
        ([0, 1, 2], [0.1, 0.2, 0.3], "both classes"),
        ([0, 1], [0.2, np.nan], "NaN"),
        ([0, 1], [0.2, 1.5], "between 0 and 1"),
        ([0, 1], [-0.1, 0.5], "between 0 and 1"),
    ],
)
def test_bad_inputs_raise(y, p, message):
    with pytest.raises(MetricsError, match=message):
        ranking_metrics(y, p)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_bad_threshold_raises(threshold):
    with pytest.raises(MetricsError, match="threshold"):
        threshold_metrics([0, 1], [0.2, 0.8], threshold)


def test_cross_val_probabilities_covers_every_row():
    X, y = make_xy()
    oof, folds = cross_val_probabilities(build_pipeline("logreg", "baseline", TEST_CFG), X, y, 3, 42)
    assert len(oof) == len(y)
    assert not np.isnan(oof).any()
    assert ((oof >= 0) & (oof <= 1)).all()
    assert len(folds) == 3


def test_cross_val_probabilities_is_reproducible():
    X, y = make_xy()
    pipeline = build_pipeline("logreg", "baseline", TEST_CFG)
    a, _ = cross_val_probabilities(pipeline, X, y, 3, 42)
    b, _ = cross_val_probabilities(pipeline, X, y, 3, 42)
    np.testing.assert_allclose(a, b)


def test_cross_val_rejects_bad_setup():
    X, y = make_xy()
    pipeline = build_pipeline("logreg", "baseline", TEST_CFG)
    with pytest.raises(MetricsError, match="at least 2"):
        cross_val_probabilities(pipeline, X, y, 1, 42)
    with pytest.raises(MetricsError, match="rows"):
        cross_val_probabilities(pipeline, X, y.iloc[:10], 3, 42)
    with pytest.raises(MetricsError, match="DataFrame"):
        cross_val_probabilities(pipeline, X.to_numpy(), y, 3, 42)


def test_cross_val_rejects_a_too_rare_class():
    X, _ = make_xy()
    y = np.zeros(len(X), dtype=int)
    y[:2] = 1
    import pandas as pd

    with pytest.raises(MetricsError, match="rarer class"):
        cross_val_probabilities(build_pipeline("logreg", "baseline", TEST_CFG), X, pd.Series(y), 3, 42)


def test_smote_with_too_few_positives_gives_a_clear_error():
    import pandas as pd

    X, _ = make_xy()
    y = np.zeros(len(X), dtype=int)
    y[:6] = 1
    with pytest.raises(MetricsError, match="fold"):
        cross_val_probabilities(build_pipeline("logreg", "smote", TEST_CFG), X, pd.Series(y), 3, 42)