import numpy as np
import pandas as pd
import pytest

from src.split import SplitError, load_split, run_split, save_splits, split_data


def make_df(n: int = 200, pos_rate: float = 0.25) -> pd.DataFrame:
    n_pos = int(n * pos_rate)
    return pd.DataFrame(
        {
            "customerID": [f"ID-{i:04d}" for i in range(n)],
            "tenure": np.arange(n),
            "Churn": [1] * n_pos + [0] * (n - n_pos),
        }
    )


def do_split(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    return split_data(df, "Churn", "customerID", 0.6, 0.2, 0.2, seed)


def test_sizes_are_60_20_20():
    splits = do_split(make_df())
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (120, 40, 40)


def test_no_row_lost_and_no_overlap():
    splits = do_split(make_df())
    all_ids = pd.concat([s["customerID"] for s in splits.values()])
    assert len(all_ids) == 200
    assert all_ids.is_unique


def test_churn_rate_is_preserved_in_every_split():
    for frame in do_split(make_df()).values():
        assert frame["Churn"].mean() == pytest.approx(0.25, abs=0.02)


def test_same_seed_gives_same_split():
    a, b = do_split(make_df(), seed=7), do_split(make_df(), seed=7)
    pd.testing.assert_frame_equal(a["train"], b["train"])


def test_different_seed_gives_different_split():
    a, b = do_split(make_df(), seed=1), do_split(make_df(), seed=2)
    assert set(a["train"]["customerID"]) != set(b["train"]["customerID"])


def test_sizes_not_summing_to_one_raise():
    with pytest.raises(SplitError, match="sum"):
        split_data(make_df(), "Churn", "customerID", 0.7, 0.2, 0.2, 42)


def test_empty_data_raises():
    with pytest.raises(SplitError, match="empty"):
        do_split(make_df().iloc[0:0])


def test_missing_target_column_raises():
    with pytest.raises(SplitError, match="Churn"):
        do_split(make_df().drop(columns=["Churn"]))


def test_missing_id_column_raises():
    with pytest.raises(SplitError, match="customerID"):
        do_split(make_df().drop(columns=["customerID"]))


def test_blank_target_raises():
    df = make_df()
    df["Churn"] = df["Churn"].astype(float)
    df.loc[0, "Churn"] = np.nan
    with pytest.raises(SplitError, match="missing"):
        do_split(df)


def test_non_binary_target_raises():
    df = make_df()
    df.loc[0, "Churn"] = 2
    with pytest.raises(SplitError, match="0 and 1"):
        do_split(df)


def test_single_class_raises():
    df = make_df()
    df["Churn"] = 0
    with pytest.raises(SplitError, match="0 and 1"):
        do_split(df)


def test_duplicate_ids_raise():
    df = make_df()
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    with pytest.raises(SplitError, match="Duplicate"):
        do_split(df)


def test_too_few_rows_in_rarer_class_raise():
    with pytest.raises(SplitError, match="rarer class"):
        do_split(make_df(n=10, pos_rate=0.1))


def test_test_set_is_locked_by_default(tmp_path):
    save_splits(do_split(make_df()), tmp_path)
    with pytest.raises(SplitError, match="locked"):
        load_split("test", tmp_path)
    assert len(load_split("test", tmp_path, allow_test=True)) == 40
    assert len(load_split("train", tmp_path)) == 120


def test_unknown_split_name_raises(tmp_path):
    with pytest.raises(SplitError, match="Unknown"):
        load_split("holdout", tmp_path)


def test_missing_split_file_raises(tmp_path):
    with pytest.raises(SplitError, match="not found"):
        load_split("val", tmp_path)


def make_cfg(tmp_path) -> dict:
    return {
        "paths": {
            "processed_data": str(tmp_path / "clean.csv"),
            "processed_dir": str(tmp_path / "out"),
        },
        "data": {"target": "Churn", "id_column": "customerID"},
        "split": {"train_size": 0.6, "val_size": 0.2, "test_size": 0.2},
        "project": {"random_seed": 42},
    }


def test_run_split_writes_three_files(tmp_path):
    cfg = make_cfg(tmp_path)
    make_df().to_csv(cfg["paths"]["processed_data"], index=False)
    run_split(cfg)
    for name in ("train", "val", "test"):
        assert (tmp_path / "out" / f"{name}.csv").exists()


def test_run_split_missing_clean_file_raises(tmp_path):
    with pytest.raises(SplitError, match="not found"):
        run_split(make_cfg(tmp_path))