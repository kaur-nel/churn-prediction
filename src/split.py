"""Create stratified train/validation/test splits and save them to disk."""

import logging
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ConfigError, load_config
from src.logger import get_logger

SPLIT_NAMES = ("train", "val", "test")


class SplitError(Exception):
    """Raised when the data cannot be split or a split file cannot be loaded."""


def split_data(
    df: pd.DataFrame,
    target: str,
    id_column: str,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Split into stratified train/val/test frames (same churn rate in each).

    Nothing is fitted here, and this must run before any preprocessing is learned.

    Raises:
        SplitError: for empty data, missing columns, a bad target, duplicate ids,
            sizes that do not sum to 1, or classes too small to stratify.
    """
    log = logger or logging.getLogger(__name__)

    if df.empty:
        raise SplitError("Cannot split an empty DataFrame")
    for col in (target, id_column):
        if col not in df.columns:
            raise SplitError(f"Column '{col}' not found in the data")
    if df[target].isna().any():
        raise SplitError(f"Target '{target}' has missing values")
    if set(df[target].unique()) != {0, 1}:
        raise SplitError(f"Target '{target}' must contain exactly the values 0 and 1")
    if df[id_column].duplicated().any():
        raise SplitError(f"Duplicate values in '{id_column}'")
    if not math.isclose(train_size + val_size + test_size, 1.0, abs_tol=1e-6):
        raise SplitError("Split sizes must sum to 1.0")

    rarer_class = int(df[target].value_counts().min())
    if rarer_class * min(train_size, val_size, test_size) < 2:
        raise SplitError(f"Too few rows in the rarer class ({rarer_class}) to stratify")

    try:
        rest, test = train_test_split(
            df, test_size=test_size, stratify=df[target], random_state=seed
        )
        train, val = train_test_split(
            rest,
            test_size=val_size / (train_size + val_size),
            stratify=rest[target],
            random_state=seed,
        )
    except ValueError as exc:
        raise SplitError(f"Could not split the data: {exc}") from exc

    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    id_sets = {name: set(frame[id_column]) for name, frame in splits.items()}
    if sum(len(ids) for ids in id_sets.values()) != len(df) or len(set.union(*id_sets.values())) != len(df):
        raise SplitError("Internal error: splits overlap or lost rows")

    for name, frame in splits.items():
        log.info("%-5s rows=%d churn_rate=%.4f", name, len(frame), frame[target].mean())
    return splits


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    """Write train.csv, val.csv and test.csv into `out_dir`."""
    folder = Path(out_dir)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in splits.items():
            frame.to_csv(folder / f"{name}.csv", index=False)
    except OSError as exc:
        raise SplitError(f"Cannot write split files to {folder}: {exc}") from exc


def load_split(name: str, processed_dir: str | Path, allow_test: bool = False) -> pd.DataFrame:
    """Load one split. The test set is locked unless `allow_test=True`.

    Raises:
        SplitError: for an unknown name, a locked test set, or a missing file.
    """
    if name not in SPLIT_NAMES:
        raise SplitError(f"Unknown split '{name}'; choose one of {SPLIT_NAMES}")
    if name == "test" and not allow_test:
        raise SplitError("The test set is locked. Pass allow_test=True for the final evaluation only")
    path = Path(processed_dir) / f"{name}.csv"
    if not path.exists():
        raise SplitError(f"Split file not found: {path}. Run `python -m src.split` first")
    return pd.read_csv(path)


def run_split(cfg: dict[str, Any], logger: logging.Logger | None = None) -> dict[str, pd.DataFrame]:
    """Read the cleaned data, split it and save the three files."""
    clean_path = Path(cfg["paths"]["processed_data"])
    if not clean_path.exists():
        raise SplitError(f"Clean data not found: {clean_path}. Run `python -m src.data` first")
    df = pd.read_csv(clean_path)
    splits = split_data(
        df,
        target=cfg["data"]["target"],
        id_column=cfg["data"]["id_column"],
        train_size=cfg["split"]["train_size"],
        val_size=cfg["split"]["val_size"],
        test_size=cfg["split"]["test_size"],
        seed=cfg["project"]["random_seed"],
        logger=logger,
    )
    save_splits(splits, cfg["paths"]["processed_dir"])
    return splits


def main() -> None:
    """Command-line entry point: `python -m src.split`."""
    try:
        cfg = load_config()
        logger = get_logger("churn.split", log_file=cfg["paths"]["log_file"])
        run_split(cfg, logger)
    except (ConfigError, SplitError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()