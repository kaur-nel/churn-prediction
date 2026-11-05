from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config
from src.data import DataValidationError, clean_data, load_raw, prepare_data

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = REPO_ROOT / "data" / "raw" / "telco_churn.csv"


def make_raw(n: int = 6) -> pd.DataFrame:
    """A small, valid, all-text frame shaped like the raw Telco file."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "customerID": f"ID-{i:04d}",
                "gender": "Female" if i % 2 else "Male",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": str(12 + i),
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": "50.5",
                "TotalCharges": str(round(50.5 * (12 + i), 2)),
                "Churn": "Yes" if i % 3 == 0 else "No",
            }
        )
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame):
    return clean_data(df, "customerID", "Churn", "Yes")


# ---------- loading ----------------------------------------------------------


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_raw(tmp_path / "nope.csv")


def test_load_raw_directory(tmp_path):
    with pytest.raises(DataValidationError, match="not a file"):
        load_raw(tmp_path)


def test_load_raw_zero_byte_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(DataValidationError, match="empty"):
        load_raw(path)


def test_load_raw_header_only(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text("customerID,gender\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="no rows"):
        load_raw(path)


def test_load_raw_handles_bom(tmp_path):
    path = tmp_path / "bom.csv"
    make_raw().to_csv(path, index=False, encoding="utf-8-sig")
    assert "customerID" in load_raw(path).columns


# ---------- happy path and cleaning --------------------------------------------


def test_valid_data_cleans():
    df, report = clean(make_raw())
    assert len(df) == 6
    assert set(df["Churn"].unique()) == {0, 1}
    assert df["tenure"].dtype.kind == "i"
    assert report["rows_out"] == 6


def test_whitespace_is_stripped():
    raw = make_raw()
    raw.loc[0, "gender"] = "  Female "
    df, _ = clean(raw)
    assert df.loc[0, "gender"] == "Female"


def test_blank_total_charges_for_new_customer_becomes_zero():
    raw = make_raw()
    raw.loc[0, "tenure"] = "0"
    raw.loc[0, "TotalCharges"] = " "
    df, report = clean(raw)
    assert df.loc[0, "TotalCharges"] == 0.0
    assert report["total_charges_filled_for_new_customers"] == 1


def test_blank_total_charges_for_old_customer_stays_missing():
    raw = make_raw()
    raw.loc[1, "TotalCharges"] = " "
    df, report = clean(raw)
    assert pd.isna(df.loc[1, "TotalCharges"])
    assert report["total_charges_missing_left_for_pipeline"] == 1


def test_no_service_labels_become_no():
    raw = make_raw()
    raw["OnlineSecurity"] = "No internet service"
    raw["MultipleLines"] = "No phone service"
    df, report = clean(raw)
    assert (df["OnlineSecurity"] == "No").all()
    assert (df["MultipleLines"] == "No").all()
    assert report["no_service_labels_changed_to_no"] == 12


def test_exact_duplicate_rows_are_dropped():
    raw = make_raw()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    df, report = clean(raw)
    assert len(df) == 6
    assert report["duplicate_rows_dropped"] == 1


def test_same_id_with_different_data_raises():
    raw = make_raw()
    clash = raw.iloc[[0]].copy()
    clash["tenure"] = "99"
    with pytest.raises(DataValidationError, match="Duplicate"):
        clean(pd.concat([raw, clash], ignore_index=True))


# ---------- schema ---------------------------------------------------------------


def test_missing_column_raises():
    with pytest.raises(DataValidationError, match="tenure"):
        clean(make_raw().drop(columns=["tenure"]))


def test_extra_column_is_dropped():
    raw = make_raw()
    raw["junk"] = "x"
    df, report = clean(raw)
    assert "junk" not in df.columns
    assert report["extra_columns_dropped"] == ["junk"]


# ---------- bad values -------------------------------------------------------------


@pytest.mark.parametrize(
    ("col", "bad"),
    [
        ("tenure", "-1"),
        ("tenure", "abc"),
        ("tenure", "12.5"),
        ("MonthlyCharges", "-5"),
        ("MonthlyCharges", "inf"),
        ("MonthlyCharges", ""),
        ("TotalCharges", "-1"),
        ("TotalCharges", "n/a"),
        ("SeniorCitizen", "2"),
        ("SeniorCitizen", "yes"),
    ],
)
def test_bad_numeric_values_raise(col, bad):
    raw = make_raw()
    raw.loc[0, col] = bad
    with pytest.raises(DataValidationError, match=col):
        clean(raw)


@pytest.mark.parametrize(
    ("col", "bad"),
    [
        ("gender", "Robot"),
        ("Contract", "Weekly"),
        ("InternetService", "Satellite"),
        ("Partner", "maybe"),
        ("gender", ""),
        ("customerID", ""),
    ],
)
def test_bad_category_values_raise(col, bad):
    raw = make_raw()
    raw.loc[0, col] = bad
    with pytest.raises(DataValidationError, match=col):
        clean(raw)


# ---------- target -----------------------------------------------------------------


def test_three_target_classes_raise():
    raw = make_raw()
    raw.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError, match="Churn"):
        clean(raw)


def test_single_target_class_raises():
    raw = make_raw()
    raw["Churn"] = "No"
    with pytest.raises(DataValidationError, match="Churn"):
        clean(raw)


def test_missing_positive_label_raises():
    raw = make_raw()
    raw["Churn"] = ["A", "B", "A", "B", "A", "B"]
    with pytest.raises(DataValidationError, match="positive label"):
        clean(raw)


# ---------- end to end -----------------------------------------------------------------


def test_prepare_data_writes_clean_file(tmp_path):
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "out" / "clean.csv"
    make_raw().to_csv(raw_path, index=False)
    cfg = {
        "paths": {"raw_data": str(raw_path), "processed_data": str(out_path)},
        "data": {"target": "Churn", "id_column": "customerID", "positive_label": "Yes"},
    }
    df = prepare_data(cfg)
    assert out_path.exists()
    assert pd.read_csv(out_path).shape == df.shape


@pytest.mark.skipif(not REAL_CSV.exists(), reason="raw Telco CSV not present (it is gitignored)")
def test_real_telco_file_cleans():
    cfg = load_config(REPO_ROOT / "configs" / "config.yaml")
    df, report = clean_data(
        load_raw(REAL_CSV),
        id_column=cfg["data"]["id_column"],
        target=cfg["data"]["target"],
        positive_label=cfg["data"]["positive_label"],
    )
    assert df["Churn"].isin([0, 1]).all()
    assert df["customerID"].is_unique
    assert report["rows_out"] == len(df)