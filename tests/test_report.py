import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.report import (
    CSV_INPUTS,
    JSON_INPUTS,
    ReportError,
    base_ltv_rows,
    best_per_model,
    build_model_card,
    build_report,
    count,
    fairness_flags,
    fmt,
    imbalance_spread,
    load_report_settings,
    load_results,
    markdown_table,
    model_file_warning,
    money,
    ordering_differs,
    pct,
    run_report,
    sensitivity_summary,
    software_versions,
    tied_models,
    tuning_verdicts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"docs_dir": "docs", "fairness_gap_flag": 0.10}


# ---------- synthetic result files (made-up numbers, for testing only) --------------------------


def metric(estimate, low, high):
    return {"estimate": estimate, "ci_low": low, "ci_high": high}


def choice_json():
    return {
        "model": "logreg", "strategy": "baseline", "params": {}, "calibration": "none",
        "validation_metrics": {"pr_auc": 0.651, "roc_auc": 0.84, "brier": 0.1368, "ece": 0.024},
        "train_rows": 4225, "val_rows": 1409, "seed": 42,
    }


def threshold_json():
    return {
        "threshold": 0.171, "targeted_pct": 52.4, "expected_net_per_1000_customers": 44038.3,
        "break_even_probability": 0.1667, "net_per_1000_at_break_even": 43825.0,
        "assumptions": {"customer_ltv": 1000.0, "retention_offer_cost": 50.0, "offer_success_rate": 0.3},
    }


def make_test_json():
    return {
        "evaluated_at_utc": "2026-11-08T18:29:53+00:00", "evaluation_count": 1, "test_rows": 1409,
        "test_churn_rate": 0.2654, "threshold": 0.171, "model_sha256": "a" * 64,
        "bootstrap": {"n_resamples": 1000, "skipped_single_class": 0, "ci_level": 0.95},
        "metrics": {
            "pr_auc": metric(0.6516, 0.5985, 0.7022), "roc_auc": metric(0.8455, 0.8228, 0.8664),
            "precision": metric(0.4602, 0.4232, 0.4967), "recall": metric(0.8957, 0.8624, 0.9261),
            "f1": metric(0.608, 0.572, 0.6419), "brier": metric(0.136, 0.1249, 0.1459),
            "targeted_pct": metric(51.67, 49.04, 54.5), "net_per_1000": metric(45493.0, 38963.0, 51385.0),
        },
        "confusion_matrix": {"tn": 642, "fp": 393, "fn": 39, "tp": 335},
        "expected_calibration_error": 0.0147, "net_per_1000_target_everyone": 29631.0,
    }


def frames():
    comparison = pd.DataFrame(
        [
            ["logreg", "baseline", 0.665, 0.028, 0.850],
            ["logreg", "class_weight", 0.663, 0.029, 0.850],
            ["random_forest", "class_weight", 0.665, 0.015, 0.847],
            ["xgboost", "class_weight", 0.657, 0.012, 0.842],
            ["lightgbm", "class_weight", 0.632, 0.008, 0.830],
        ],
        columns=["model", "strategy", "pr_auc", "pr_auc_fold_std", "roc_auc"],
    )
    imbalance = pd.DataFrame(
        [
            ["logreg", "baseline", 0.665, 0.028, 0.85], ["logreg", "class_weight", 0.663, 0.029, 0.85],
            ["logreg", "smote", 0.664, 0.029, 0.85], ["lightgbm", "baseline", 0.624, 0.012, 0.83],
            ["lightgbm", "class_weight", 0.632, 0.008, 0.83], ["lightgbm", "smote", 0.634, 0.010, 0.831],
        ],
        columns=["model", "strategy", "pr_auc", "pr_auc_fold_std", "roc_auc"],
    )
    tuning = pd.DataFrame(
        [
            ["random_forest", "class_weight", 30, 0.665, 0.015, 0.665, 0.647, 0.644],
            ["logreg", "baseline", 30, 0.665, 0.028, 0.665, 0.651, 0.653],
        ],
        columns=["model", "strategy", "trials_completed", "cv_pr_auc_default", "cv_pr_auc_default_std",
                 "cv_pr_auc_tuned", "val_pr_auc_default", "val_pr_auc_tuned"],
    )
    calibration = pd.DataFrame(
        [["none", 0.6512, 0.8397, 0.1368, 0.024], ["sigmoid", 0.6512, 0.8397, 0.1368, 0.0283], ["isotonic", 0.633, 0.838, 0.1381, 0.0294]],
        columns=["method", "pr_auc", "roc_auc", "brier", "ece"],
    )
    sensitivity = pd.DataFrame(
        [
            [500.0, 100.0, 0.2, 1.0, np.nan, 0.0, 0.0, -73456.0],
            [1000.0, 50.0, 0.3, 0.167, 0.171, 52.4, 44038.0, 29631.0],
            [1000.0, 100.0, 0.2, 0.5, 0.515, 19.5, 7736.0, -46913.0],
            [1500.0, 50.0, 0.5, 0.067, 0.065, 68.3, 158552.0, 149077.0],
        ],
        columns=["customer_ltv", "retention_offer_cost", "offer_success_rate", "break_even_probability",
                 "best_threshold", "targeted_pct", "net_per_1000", "target_everyone_per_1000"],
    )
    fairness = pd.DataFrame(
        [
            ["gender", "Female", 687, 193, 0.281, 0.268, 0.523, 0.476, 0.886, 0.381, 0.842, True],
            ["gender", "Male", 722, 181, 0.251, 0.264, 0.511, 0.444, 0.906, 0.379, 0.850, True],
            ["SeniorCitizen", "0", 1187, 276, 0.233, 0.237, 0.467, 0.433, 0.870, 0.345, 0.847, True],
            ["SeniorCitizen", "1", 222, 98, 0.441, 0.419, 0.784, 0.546, 0.969, 0.637, 0.780, True],
        ],
        columns=["attribute", "group", "customers", "positives", "churn_rate", "mean_predicted", "selection_rate",
                 "precision", "recall", "false_positive_rate", "roc_auc", "reliable"],
    )
    gaps = pd.DataFrame(
        [["gender", 2, 0.02, 0.002, 0.011], ["SeniorCitizen", 2, 0.099, 0.292, 0.317]],
        columns=["attribute", "groups_compared", "recall_gap", "false_positive_rate_gap", "selection_rate_gap"],
    )
    shap_groups = pd.DataFrame(
        [["Contract", 0.601, 17.2], ["InternetService", 0.579, 16.6], ["tenure", 0.488, 14.0], ["MultipleLines", 0.2, 5.7]],
        columns=["group", "mean_abs_shap", "share_pct"],
    )
    shap_profiles = pd.DataFrame(
        [
            ["Contract", "Month-to-month", 776, 0.557, 1.746, 0.418], ["Contract", "One year", 308, -0.168, 0.845, 0.120],
            ["Contract", "Two year", 325, -1.113, 0.328, 0.040],
            ["InternetService", "Fiber optic", 640, 0.653, 1.922, 0.414], ["InternetService", "DSL", 463, -0.244, 0.784, 0.188],
            ["tenure", "0-6", 266, 0.920, 2.509, 0.519], ["tenure", "13-24", 189, 0.275, 1.317, 0.265],
            ["tenure", "7-12", 153, 0.118, 1.125, 0.399],
        ],
        columns=["group", "value", "customers", "mean_shap", "odds_multiplier", "observed_churn_rate"],
    )
    vif = pd.DataFrame([["MonthlyCharges", 14.32], ["TotalCharges", 10.74], ["tenure", 8.01]], columns=["feature", "vif"])
    drift = pd.DataFrame(
        [["charges_ratio", "numeric", 0.0139, "stable"], ["tenure", "numeric", 0.0115, "stable"]],
        columns=["feature", "kind", "psi", "status"],
    )
    return {
        "model_comparison.csv": comparison, "imbalance_comparison.csv": imbalance, "tuning_results.csv": tuning,
        "calibration_comparison.csv": calibration, "sensitivity_table.csv": sensitivity,
        "fairness_by_group.csv": fairness, "fairness_gaps.csv": gaps, "shap_group_importance.csv": shap_groups,
        "shap_driver_profiles.csv": shap_profiles, "vif_table.csv": vif, "drift_validation_vs_train.csv": drift,
    }


def write_reports(folder: Path, optional: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "final_model_choice.json").write_text(json.dumps(choice_json()), encoding="utf-8")
    (folder / "threshold_choice.json").write_text(json.dumps(threshold_json()), encoding="utf-8")
    (folder / "test_results.json").write_text(json.dumps(make_test_json()), encoding="utf-8")
    for name, frame in frames().items():
        if not optional and name in ("vif_table.csv", "drift_validation_vs_train.csv"):
            continue
        frame.to_csv(folder / name, index=False)


def make_cfg(tmp_path: Path) -> dict:
    return {
        "paths": {"reports_dir": str(tmp_path / "reports"), "models_dir": str(tmp_path / "models")},
        "models": {"logreg_drop_columns": ["TotalCharges"]},
        "tuning": {"n_trials": 30},
        "cv": {"n_splits": 5},
        "split": {"train_size": 0.6, "val_size": 0.2, "test_size": 0.2},
        "drift": {"moderate_threshold": 0.1, "major_threshold": 0.25},
        "explainability": {"top_n": 3},
        "report": {"docs_dir": str(tmp_path / "docs"), "fairness_gap_flag": 0.10},
    }


@pytest.fixture
def results(tmp_path):
    write_reports(tmp_path / "reports")
    return load_results(tmp_path / "reports")


# ---------- formatting -----------------------------------------------------------------------


def test_fmt_pct_money_and_count():
    assert fmt(0.6516) == "0.652"
    assert fmt(3.14159, 2) == "3.14"
    assert fmt(np.nan) == "n/a"
    assert fmt(np.inf) == "n/a"
    assert fmt(None) == "n/a"
    assert fmt(True) == "n/a"
    assert pct(0.2654) == "26.5%"
    assert pct(0.3, 0) == "30%"
    assert pct(np.nan) == "n/a"
    assert pct("x") == "n/a"
    assert money(45493.2576) == "45,493"
    assert money(np.nan) == "n/a"
    assert money("x") == "n/a"
    assert count(4225) == "4,225"
    assert count("x") == "n/a"


def test_markdown_table_escapes_cells_and_checks_row_length():
    lines = markdown_table(["a", "b"], [["x|y", "line\nbreak"]])
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| x\\|y | line break |"
    with pytest.raises(ReportError, match="cells"):
        markdown_table(["a", "b"], [["only one"]])
    with pytest.raises(ReportError, match="at least one column"):
        markdown_table([], [])


# ---------- loading --------------------------------------------------------------------------


def test_load_results_reads_everything(results):
    assert results["test"]["test_rows"] == 1409
    assert len(results["comparison"]) == 5
    assert results["vif"] is not None
    assert results["drift"] is not None


def test_optional_files_may_be_missing(tmp_path):
    write_reports(tmp_path / "reports", optional=False)
    data = load_results(tmp_path / "reports")
    assert data["vif"] is None
    assert data["drift"] is None


@pytest.mark.parametrize("filename", [spec[0] for spec in JSON_INPUTS.values()] + [spec[0] for spec in CSV_INPUTS.values()])
def test_a_missing_required_file_names_the_command(tmp_path, filename):
    write_reports(tmp_path / "reports")
    (tmp_path / "reports" / filename).unlink()
    with pytest.raises(ReportError, match=r"python -m src\."):
        load_results(tmp_path / "reports")


def test_invalid_json_raises(tmp_path):
    write_reports(tmp_path / "reports")
    (tmp_path / "reports" / "test_results.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ReportError, match="Could not read"):
        load_results(tmp_path / "reports")


def make_test_json_that_is_not_an_object_raises(tmp_path):
    write_reports(tmp_path / "reports")
    (tmp_path / "reports" / "test_results.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ReportError, match="JSON object"):
        load_results(tmp_path / "reports")


def test_missing_json_key_raises(tmp_path):
    write_reports(tmp_path / "reports")
    data = make_test_json()
    del data["model_sha256"]
    (tmp_path / "reports" / "test_results.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportError, match="model_sha256"):
        load_results(tmp_path / "reports")


def test_missing_nested_metric_raises(tmp_path):
    write_reports(tmp_path / "reports")
    data = make_test_json()
    del data["metrics"]["brier"]
    (tmp_path / "reports" / "test_results.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportError, match="brier"):
        load_results(tmp_path / "reports")


def test_missing_assumption_raises(tmp_path):
    write_reports(tmp_path / "reports")
    data = threshold_json()
    del data["assumptions"]["customer_ltv"]
    (tmp_path / "reports" / "threshold_choice.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportError, match="customer_ltv"):
        load_results(tmp_path / "reports")


def test_non_numeric_ci_level_raises(tmp_path):
    write_reports(tmp_path / "reports")
    data = make_test_json()
    data["bootstrap"]["ci_level"] = "high"
    (tmp_path / "reports" / "test_results.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportError, match="ci_level"):
        load_results(tmp_path / "reports")


def test_empty_csv_raises(tmp_path):
    write_reports(tmp_path / "reports")
    (tmp_path / "reports" / "model_comparison.csv").write_text("", encoding="utf-8")
    with pytest.raises(ReportError, match="Could not read"):
        load_results(tmp_path / "reports")


def test_header_only_csv_raises(tmp_path):
    write_reports(tmp_path / "reports")
    (tmp_path / "reports" / "tuning_results.csv").write_text("model,strategy\n", encoding="utf-8")
    with pytest.raises(ReportError, match="no rows"):
        load_results(tmp_path / "reports")


def test_csv_missing_a_column_raises(tmp_path):
    write_reports(tmp_path / "reports")
    frames()["model_comparison.csv"].drop(columns=["pr_auc"]).to_csv(tmp_path / "reports" / "model_comparison.csv", index=False)
    with pytest.raises(ReportError, match="pr_auc"):
        load_results(tmp_path / "reports")


# ---------- settings -------------------------------------------------------------------------


def test_valid_settings_pass():
    assert load_report_settings({"report": {"docs_dir": "docs", "fairness_gap_flag": 0.1}}) == SETTINGS


def test_missing_settings_section_raises():
    with pytest.raises(ReportError, match="report"):
        load_report_settings({})


@pytest.mark.parametrize(
    ("key", "bad"),
    [("docs_dir", ""), ("docs_dir", "  "), ("docs_dir", 5), ("fairness_gap_flag", 0), ("fairness_gap_flag", -1),
     ("fairness_gap_flag", "0.1"), ("fairness_gap_flag", True), ("fairness_gap_flag", float("nan"))],
)
def test_bad_settings_raise(key, bad):
    section = {"docs_dir": "docs", "fairness_gap_flag": 0.1, key: bad}
    with pytest.raises(ReportError, match=key):
        load_report_settings({"report": section})


# ---------- rule-based statements ------------------------------------------------------------


def test_best_per_model_keeps_one_row_per_model_best_first(results):
    best = best_per_model(results["comparison"])
    assert list(best["model"]) == ["logreg", "random_forest", "xgboost", "lightgbm"] or list(best["model"])[0] in ("logreg", "random_forest")
    assert best["model"].is_unique
    assert best["pr_auc"].is_monotonic_decreasing


def test_tied_models_uses_one_fold_std_of_the_best(results):
    tied, margin = tied_models(results["comparison"])
    assert set(tied) == {"logreg", "random_forest", "xgboost"}
    assert "lightgbm" not in tied
    assert margin in (0.028, 0.015)


def test_tied_models_with_an_invalid_margin_returns_only_the_best():
    table = pd.DataFrame([["a", "baseline", 0.7, np.nan, 0.8], ["b", "baseline", 0.69, 0.01, 0.8]],
                         columns=["model", "strategy", "pr_auc", "pr_auc_fold_std", "roc_auc"])
    tied, margin = tied_models(table)
    assert tied == ["a"]
    assert margin == 0.0


def test_tuning_verdicts_require_a_gain_above_the_noise(results):
    verdicts = {v["model"]: v for v in tuning_verdicts(results["tuning"])}
    assert not verdicts["random_forest"]["meaningful"]
    assert verdicts["random_forest"]["delta"] == pytest.approx(-0.003)
    assert not verdicts["logreg"]["meaningful"]
    big_gain = results["tuning"].copy()
    big_gain.loc[1, "val_pr_auc_tuned"] = 0.70
    assert {v["model"]: v for v in tuning_verdicts(big_gain)}["logreg"]["meaningful"]


def test_fairness_flags_use_the_limit(results):
    flags = dict(fairness_flags(results["fairness_gaps"], 0.10))
    assert set(flags) == {"SeniorCitizen"}
    assert "false-positive rate" in flags["SeniorCitizen"]
    assert "share flagged" in flags["SeniorCitizen"]
    assert "recall" not in flags["SeniorCitizen"]  # 0.099 is not above 0.10
    assert fairness_flags(results["fairness_gaps"], 0.5) == []
    assert {a for a, _ in fairness_flags(results["fairness_gaps"], 0.01)} == {"gender", "SeniorCitizen"}


def test_fairness_flags_ignore_missing_gaps():
    gaps = pd.DataFrame([["g", 1, np.nan, np.nan, np.nan]],
                        columns=["attribute", "groups_compared", "recall_gap", "false_positive_rate_gap", "selection_rate_gap"])
    assert fairness_flags(gaps, 0.1) == []


def test_ordering_differs_detects_a_changed_ranking(results):
    profiles = results["shap_profiles"]
    assert not ordering_differs(profiles[profiles["group"] == "Contract"])
    assert ordering_differs(profiles[profiles["group"] == "tenure"])
    assert not ordering_differs(profiles[profiles["group"] == "tenure"].head(1))


def test_sensitivity_summary_counts(results):
    assert sensitivity_summary(results["sensitivity"]) == {"scenarios": 4, "everyone_loses": 2, "no_campaign": 1}


def test_base_ltv_rows_filter_and_fallback(results):
    rows = base_ltv_rows(results["sensitivity"], 1000.0)
    assert set(rows["customer_ltv"]) == {1000.0}
    assert len(base_ltv_rows(results["sensitivity"], 777.0)) == 4


def test_imbalance_spread(results):
    spread, noise = imbalance_spread(results["imbalance"])
    assert spread == pytest.approx(0.010)
    assert noise == pytest.approx(0.029)


def test_model_file_warning(tmp_path):
    cfg = make_cfg(tmp_path)
    test = make_test_json()
    assert model_file_warning(cfg, test) is None  # no model file
    (tmp_path / "models").mkdir()
    model_file = tmp_path / "models" / "final_model.joblib"
    model_file.write_bytes(b"model bytes")
    assert "differs" in model_file_warning(cfg, test)
    from src.evaluate import file_sha256

    test["model_sha256"] = file_sha256(model_file)
    assert model_file_warning(cfg, test) is None


def test_software_versions_has_python_and_packages():
    versions = software_versions()
    assert "python" in versions
    assert "scikit-learn" in versions


# ---------- the documents --------------------------------------------------------------------


def documents(tmp_path, results):
    cfg = make_cfg(tmp_path)
    return build_model_card(results, cfg, SETTINGS), build_report(results, cfg, SETTINGS)


def test_documents_contain_the_numbers_from_the_result_files(tmp_path, results):
    card, report = documents(tmp_path, results)
    for text in (card, report):
        assert "0.652" in text  # test PR-AUC
        assert fmt(0.5985) in text and "0.702" in text  # its confidence interval
        assert "45,493" in text  # net benefit per 1,000
        assert "a" * 64 in text  # model fingerprint
        assert "0.171" in text  # threshold
    assert "29,631" in report  # offering everyone (only the report compares the strategies)
    assert "4,225" in card and "1,409" in card
    assert "Logistic regression" in report


def test_documents_never_show_nan_or_none(tmp_path, results):
    for text in documents(tmp_path, results):
        assert not re.search(r"\bnan\b", text, flags=re.IGNORECASE)
        # a Python None leaking into a table cell or sentence (the English words "None of ..." are fine)
        assert not re.search(r"\|\s*None\s*\||[:=]\s*None\b|\(None\)", text)


def test_documents_disclose_the_placeholder_assumptions_and_limitations(tmp_path, results):
    card, report = documents(tmp_path, results)
    assert "placeholder" in card.lower() and "placeholder" in report.lower()
    assert "## Limitations" in card
    assert "## Limitations and next steps" in report
    assert "SeniorCitizen" in report


def test_generated_wording_follows_the_rules(tmp_path, results):
    _, report = documents(tmp_path, results)
    assert "did not improve any model by more than the cross-validation noise" in report
    assert "**Review the offer design for `SeniorCitizen` groups**" in report
    assert "`gender` groups" not in report
    assert "statistically tied" in report
    assert "within the fold-to-fold noise" in report
    assert "inside the test confidence interval" in report
    assert "2 scenarios" not in report and "Across 4 scenarios" in report
    assert "does not clearly" not in report


def test_wording_changes_when_the_numbers_change(tmp_path):
    reports = tmp_path / "reports"
    write_reports(reports)
    big_gain = frames()["tuning_results.csv"]
    big_gain.loc[1, "val_pr_auc_tuned"] = 0.70
    big_gain.to_csv(reports / "tuning_results.csv", index=False)
    data = load_results(reports)
    report = build_report(data, make_cfg(tmp_path), SETTINGS)
    assert "Tuning improved at least one model" in report


def test_interval_overlap_wording(tmp_path):
    reports = tmp_path / "reports"
    write_reports(reports)
    data = make_test_json()
    data["net_per_1000_target_everyone"] = 50000.0
    (reports / "test_results.json").write_text(json.dumps(data), encoding="utf-8")
    report = build_report(load_results(reports), make_cfg(tmp_path), SETTINGS)
    assert "overlaps the point estimate for offering everyone" in report
    assert "is above the point estimate for offering everyone" not in report


def test_repeated_evaluation_is_disclosed(tmp_path):
    reports = tmp_path / "reports"
    write_reports(reports)
    data = make_test_json()
    data["evaluation_count"] = 3
    (reports / "test_results.json").write_text(json.dumps(data), encoding="utf-8")
    card, report = documents(tmp_path, load_results(reports))
    assert "evaluated 3 times" in card
    assert "evaluated 3 times" in report


def test_optional_sections_are_left_out_when_files_are_missing(tmp_path):
    write_reports(tmp_path / "reports", optional=False)
    data = load_results(tmp_path / "reports")
    card = build_model_card(data, make_cfg(tmp_path), SETTINGS)
    assert "Multicollinearity check (VIF)" not in card
    assert "Validation vs training" not in card


def test_optional_sections_appear_when_files_exist(tmp_path, results):
    card, _ = documents(tmp_path, results)
    assert "Multicollinearity check (VIF)" in card
    assert "`MonthlyCharges` 14.32" in card
    assert "Validation vs training" in card


def test_feature_section_reflects_the_dropped_columns(tmp_path, results):
    card, _ = documents(tmp_path, results)
    features = card.split("## Features")[1].split("## Training")[0]
    assert "Dropped for this model only" in features
    numeric_line = next(line for line in features.splitlines() if line.startswith("- **Numeric:**"))
    assert "`TotalCharges`" not in numeric_line
    assert "`gender`" in features  # listed as left out on purpose


def test_documents_are_reproducible(tmp_path, results):
    first = documents(tmp_path, results)
    second = documents(tmp_path, results)
    assert first == second


# ---------- running --------------------------------------------------------------------------


def test_run_report_writes_both_documents(tmp_path):
    write_reports(tmp_path / "reports")
    paths = run_report(make_cfg(tmp_path))
    assert paths["model_card"].read_text(encoding="utf-8").startswith("# Model card")
    assert paths["report"].read_text(encoding="utf-8").startswith("# Customer churn prediction")
    run_report(make_cfg(tmp_path))  # running again overwrites cleanly


def test_run_report_without_results_raises(tmp_path):
    with pytest.raises(ReportError, match="not found"):
        run_report(make_cfg(tmp_path))


def test_run_report_with_an_unwritable_docs_folder_raises(tmp_path):
    write_reports(tmp_path / "reports")
    blocker = tmp_path / "docs"
    blocker.write_text("I am a file, not a folder", encoding="utf-8")
    with pytest.raises(ReportError, match="Cannot write"):
        run_report(make_cfg(tmp_path))


def test_run_report_with_bad_settings_raises(tmp_path):
    write_reports(tmp_path / "reports")
    cfg = make_cfg(tmp_path)
    cfg["report"]["fairness_gap_flag"] = 0
    with pytest.raises(ReportError, match="fairness_gap_flag"):
        run_report(cfg)


@pytest.mark.skipif(
    not all((REPO_ROOT / "reports" / spec[0]).exists() for spec in [*JSON_INPUTS.values(), *[(v[0],) for v in CSV_INPUTS.values()]]),
    reason="the real result files are not present",
)
def test_real_results_produce_documents_that_match_the_result_files(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    cfg = load_config()
    cfg["report"]["docs_dir"] = str(tmp_path / "docs")
    paths = run_report(cfg)
    report = paths["report"].read_text(encoding="utf-8")
    test = json.loads((REPO_ROOT / "reports" / "test_results.json").read_text(encoding="utf-8"))
    assert fmt(test["metrics"]["pr_auc"]["estimate"]) in report
    assert money(test["metrics"]["net_per_1000"]["estimate"]) in report
    assert test["model_sha256"] in paths["model_card"].read_text(encoding="utf-8")