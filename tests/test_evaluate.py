import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.calibration import CalibratedModel
from src.evaluate import (
    EvaluationError,
    bootstrap_metrics,
    compute_metrics,
    fairness_gaps,
    fairness_table,
    load_evaluation_settings,
    load_threshold,
    run_evaluation,
)
from src.metrics import MetricsError
from src.models import build_pipeline
from src.split import SplitError
from tests.test_models import TEST_CFG, make_xy

ECON = {"ltv": 1000.0, "cost": 50.0, "rate": 0.3}
Y = np.array([1, 1, 0, 0, 1, 0])
P = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])

GOOD_EVAL = {
    "n_bootstrap": 100,
    "ci_level": 0.9,
    "calibration_bins": 5,
    "fairness_columns": ["gender", "SeniorCitizen"],
    "min_group_positives": 5,
}


# ---------- settings -----------------------------------------------------------------------


def test_valid_settings_pass():
    assert load_evaluation_settings({"evaluation": GOOD_EVAL})["n_bootstrap"] == 100


def test_missing_section_raises():
    with pytest.raises(EvaluationError, match="evaluation"):
        load_evaluation_settings({})


def test_missing_key_raises():
    section = {key: value for key, value in GOOD_EVAL.items() if key != "ci_level"}
    with pytest.raises(EvaluationError, match="ci_level"):
        load_evaluation_settings({"evaluation": section})


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("n_bootstrap", 99),
        ("n_bootstrap", True),
        ("n_bootstrap", "1000"),
        ("ci_level", 0),
        ("ci_level", 1),
        ("ci_level", 1.5),
        ("ci_level", "0.95"),
        ("ci_level", True),
        ("calibration_bins", 1),
        ("fairness_columns", []),
        ("fairness_columns", "gender"),
        ("fairness_columns", [1]),
        ("min_group_positives", 0),
    ],
)
def test_bad_settings_raise(key, bad):
    with pytest.raises(EvaluationError, match=key):
        load_evaluation_settings({"evaluation": {**GOOD_EVAL, key: bad}})


# ---------- threshold file -----------------------------------------------------------------


def write_threshold(path, **overrides) -> None:
    data = {
        "threshold": 0.3,
        "assumptions": {"customer_ltv": 1000.0, "retention_offer_cost": 50.0, "offer_success_rate": 0.3},
        "expected_net_per_1000_customers": 1234.0,
        **overrides,
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_threshold_reads_a_valid_file(tmp_path):
    path = tmp_path / "t.json"
    write_threshold(path)
    info = load_threshold(path)
    assert info["threshold"] == 0.3
    assert info["economics"] == {"ltv": 1000.0, "cost": 50.0, "rate": 0.3}


def test_missing_threshold_file_raises(tmp_path):
    with pytest.raises(EvaluationError, match="not found"):
        load_threshold(tmp_path / "nope.json")


def test_unreadable_threshold_file_raises(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(EvaluationError, match="Could not read"):
        load_threshold(path)


def test_threshold_file_must_be_an_object(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(EvaluationError, match="JSON object"):
        load_threshold(path)


@pytest.mark.parametrize("bad", [1.5, -0.1, "0.3", True, None])
def test_bad_threshold_value_raises(tmp_path, bad):
    path = tmp_path / "t.json"
    write_threshold(path, threshold=bad)
    with pytest.raises(EvaluationError, match="threshold"):
        load_threshold(path)


def test_missing_assumptions_raise(tmp_path):
    path = tmp_path / "t.json"
    write_threshold(path, assumptions={"customer_ltv": 1000.0})
    with pytest.raises(EvaluationError, match="assumptions"):
        load_threshold(path)


def test_invalid_assumptions_raise(tmp_path):
    path = tmp_path / "t.json"
    write_threshold(
        path, assumptions={"customer_ltv": 0, "retention_offer_cost": 50.0, "offer_success_rate": 0.3}
    )
    with pytest.raises(EvaluationError, match="customer_ltv"):
        load_threshold(path)


# ---------- metrics and bootstrap -----------------------------------------------------------


def test_compute_metrics_known_values():
    result = compute_metrics(Y, P, 0.5, ECON)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["brier"] == pytest.approx(0.2)
    assert result["targeted_pct"] == pytest.approx(50.0)
    assert result["net_per_1000"] == pytest.approx(75000.0)  # (0.3*1000*2 - 50*3) / 6 * 1000


def make_scores(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(size=n)
    return (rng.random(n) < p).astype(int), p


def test_bootstrap_estimate_matches_the_point_metrics():
    y, p = make_scores()
    table, _ = bootstrap_metrics(y, p, 0.4, ECON, 100, 0.9, seed=1)
    point = compute_metrics(y, p, 0.4, ECON)
    assert table.set_index("metric").loc["recall", "estimate"] == pytest.approx(point["recall"])
    assert (table["ci_low"] <= table["ci_high"]).all()


def test_wider_confidence_level_gives_wider_intervals():
    y, p = make_scores()
    narrow, _ = bootstrap_metrics(y, p, 0.4, ECON, 200, 0.5, seed=1)
    wide, _ = bootstrap_metrics(y, p, 0.4, ECON, 200, 0.95, seed=1)
    assert ((wide["ci_high"] - wide["ci_low"]) >= (narrow["ci_high"] - narrow["ci_low"]) - 1e-12).all()


def test_bootstrap_is_reproducible_and_seed_dependent():
    y, p = make_scores()
    a, _ = bootstrap_metrics(y, p, 0.4, ECON, 100, 0.9, seed=3)
    b, _ = bootstrap_metrics(y, p, 0.4, ECON, 100, 0.9, seed=3)
    c, _ = bootstrap_metrics(y, p, 0.4, ECON, 100, 0.9, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["ci_low"], c["ci_low"])


def test_single_class_resamples_are_skipped_and_counted():
    y = np.array([1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.3, 0.1, 0.4, 0.2])
    _, skipped = bootstrap_metrics(y, p, 0.5, ECON, 200, 0.9, seed=0)
    assert skipped > 0


@pytest.mark.parametrize("n_bootstrap", [1, 0, True])
def test_bad_resample_count_raises(n_bootstrap):
    y, p = make_scores()
    with pytest.raises(EvaluationError, match="n_bootstrap"):
        bootstrap_metrics(y, p, 0.4, ECON, n_bootstrap, 0.9, seed=0)


@pytest.mark.parametrize("level", [0, 1, 1.5, "0.9"])
def test_bad_ci_level_raises(level):
    y, p = make_scores()
    with pytest.raises(EvaluationError, match="ci_level"):
        bootstrap_metrics(y, p, 0.4, ECON, 100, level, seed=0)


def test_single_class_test_data_raises():
    with pytest.raises(MetricsError, match="both classes"):
        bootstrap_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5, ECON, 100, 0.9, seed=0)


# ---------- fairness -------------------------------------------------------------------------


def test_fairness_table_known_values():
    frame = pd.DataFrame({"gender": ["M", "M", "F", "F"]})
    table = fairness_table(frame, [1, 0, 1, 0], [0.9, 0.2, 0.4, 0.8], 0.5, ["gender"], 1).set_index("group")
    assert table.loc["M", "recall"] == 1.0
    assert table.loc["M", "precision"] == 1.0
    assert table.loc["M", "false_positive_rate"] == 0.0
    assert table.loc["F", "recall"] == 0.0
    assert table.loc["F", "false_positive_rate"] == 1.0
    assert table.loc["F", "selection_rate"] == 0.5
    assert table["reliable"].all()

    gaps = fairness_gaps(table.reset_index()).iloc[0]
    assert gaps["recall_gap"] == 1.0
    assert gaps["false_positive_rate_gap"] == 1.0
    assert gaps["selection_rate_gap"] == 0.0


def test_small_groups_are_marked_unreliable_and_left_out_of_gaps():
    frame = pd.DataFrame({"gender": ["M", "M", "F", "F"]})
    table = fairness_table(frame, [1, 0, 1, 0], [0.9, 0.2, 0.4, 0.8], 0.5, ["gender"], 2)
    assert not table["reliable"].any()
    gaps = fairness_gaps(table).iloc[0]
    assert gaps["groups_compared"] == 0
    assert np.isnan(gaps["recall_gap"])


def test_group_with_one_class_has_no_roc_auc_and_no_crash():
    frame = pd.DataFrame({"g": ["a", "a", "b", "b"]})
    table = fairness_table(frame, [1, 0, 0, 0], [0.9, 0.2, 0.3, 0.4], 0.5, ["g"], 1).set_index("group")
    assert not np.isnan(table.loc["a", "roc_auc"])
    assert np.isnan(table.loc["b", "roc_auc"])
    assert np.isnan(table.loc["b", "recall"])


def test_missing_values_become_their_own_group():
    frame = pd.DataFrame({"g": ["a", None, "a", None]})
    table = fairness_table(frame, [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], 0.5, ["g"], 1)
    assert set(table["group"]) == {"a", "missing"}


def test_fairness_bad_inputs_raise():
    frame = pd.DataFrame({"g": ["a", "b", "a", "b"]})
    with pytest.raises(EvaluationError, match="region"):
        fairness_table(frame, [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], 0.5, ["region"], 1)
    with pytest.raises(EvaluationError, match="same length"):
        fairness_table(frame, [1, 0, 1], [0.9, 0.1, 0.8], 0.5, ["g"], 1)
    with pytest.raises(EvaluationError, match="At least one"):
        fairness_table(frame, [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], 0.5, [], 1)


def test_fairness_gaps_bad_table_raises():
    with pytest.raises(EvaluationError, match="empty or missing"):
        fairness_gaps(pd.DataFrame())


# ---------- end to end -----------------------------------------------------------------------


def make_cfg(tmp_path, **eval_overrides) -> dict:
    return {
        **TEST_CFG,
        "evaluation": {**GOOD_EVAL, **eval_overrides},
        "paths": {
            "processed_dir": str(tmp_path),
            "reports_dir": str(tmp_path / "reports"),
            "models_dir": str(tmp_path / "models"),
        },
        "data": {"target": "Churn", "id_column": "customerID"},
    }


def write_inputs(tmp_path, threshold: float = 0.3) -> None:
    """Write the frozen model, the threshold file and test.csv ONLY (no train or validation file)."""
    X_train, y_train = make_xy(400, 0)
    model = CalibratedModel(build_pipeline("logreg", "baseline", TEST_CFG), "none").fit(X_train, y_train)
    (tmp_path / "models").mkdir()
    joblib.dump(model, tmp_path / "models" / "final_model.joblib")

    X_test, y_test = make_xy(300, 9)
    test = X_test.copy()
    test["customerID"] = [f"test-{i}" for i in range(len(test))]
    test["gender"] = np.where(np.arange(len(test)) % 2 == 0, "Male", "Female")
    test["Churn"] = y_test
    test.to_csv(tmp_path / "test.csv", index=False)

    (tmp_path / "reports").mkdir()
    write_threshold(tmp_path / "reports" / "threshold_choice.json", threshold=threshold)


def test_run_evaluation_writes_everything_using_only_the_test_split(tmp_path):
    write_inputs(tmp_path)
    outcome = run_evaluation(make_cfg(tmp_path))
    results = outcome["results"]

    assert results["test_rows"] == 300
    assert results["evaluation_count"] == 1
    for name in ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier"):
        entry = results["metrics"][name]
        assert 0 <= entry["estimate"] <= 1
        assert entry["ci_low"] <= entry["ci_high"]
    assert sum(results["confusion_matrix"].values()) == 300
    assert len(results["model_sha256"]) == 64
    assert len(results["test_file_sha256"]) == 64
    assert set(outcome["fairness"]["attribute"]) == {"gender", "SeniorCitizen"}

    reports = tmp_path / "reports"
    for name in ("test_metrics.csv", "fairness_by_group.csv", "fairness_gaps.csv", "test_results.json"):
        assert (reports / name).exists()
    for name in ("confusion_matrix_test.png", "calibration_test.png"):
        assert (reports / "figures" / name).exists()
    saved = json.loads((reports / "test_results.json").read_text(encoding="utf-8"))
    assert saved["threshold"] == 0.3


def test_the_saved_threshold_is_applied_not_re_optimised(tmp_path):
    write_inputs(tmp_path, threshold=0.0)
    results = run_evaluation(make_cfg(tmp_path))["results"]
    assert results["threshold"] == 0.0
    assert results["metrics"]["targeted_pct"]["estimate"] == 100.0
    assert results["metrics"]["recall"]["estimate"] == 1.0


def test_second_run_is_blocked_unless_rerun_is_requested(tmp_path):
    write_inputs(tmp_path)
    cfg = make_cfg(tmp_path)
    run_evaluation(cfg)
    with pytest.raises(EvaluationError, match="already evaluated"):
        run_evaluation(cfg)
    again = run_evaluation(cfg, rerun=True)
    assert again["results"]["evaluation_count"] == 2


def test_a_failed_run_does_not_lock_the_evaluation(tmp_path):
    write_inputs(tmp_path)
    with pytest.raises(EvaluationError, match="region"):
        run_evaluation(make_cfg(tmp_path, fairness_columns=["region"]))
    assert not (tmp_path / "reports" / "test_results.json").exists()


def test_missing_model_raises(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "models" / "final_model.joblib").unlink()
    with pytest.raises(EvaluationError, match="Final model not found"):
        run_evaluation(make_cfg(tmp_path))


def test_missing_threshold_file_raises_in_a_run(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "reports" / "threshold_choice.json").unlink()
    with pytest.raises(EvaluationError, match="not found"):
        run_evaluation(make_cfg(tmp_path))


def test_missing_test_file_raises(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "test.csv").unlink()
    with pytest.raises(SplitError, match="not found"):
        run_evaluation(make_cfg(tmp_path))


def test_missing_evaluation_section_raises(tmp_path):
    write_inputs(tmp_path)
    cfg = make_cfg(tmp_path)
    del cfg["evaluation"]
    with pytest.raises(EvaluationError, match="evaluation"):
        run_evaluation(cfg)