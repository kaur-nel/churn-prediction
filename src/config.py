"""Load and validate the project configuration (configs/config.yaml)."""

import math
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = "configs/config.yaml"

REQUIRED_KEYS: dict[str, list[str]] = {
    "project": ["random_seed"],
    "paths": ["raw_data", "processed_dir", "models_dir", "reports_dir", "log_file"],
    "data": ["target", "id_column", "positive_label"],
    "split": ["train_size", "val_size", "test_size"],
    "cv": ["n_splits"],
    "business": ["customer_ltv", "retention_offer_cost", "offer_success_rate"],
}


class ConfigError(Exception):
    """Raised when the configuration file is missing, unreadable or invalid."""


def _is_int(value: Any) -> bool:
    """True for real integers (bool is excluded, even though True == 1 in Python)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """True for finite ints/floats (bool, NaN and infinity are excluded)."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_config(cfg: dict[str, Any]) -> None:
    """Check that the config has every section/key and that values make sense.

    Raises:
        ConfigError: with a message naming the exact section.key that is wrong.
    """
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a mapping, got {type(cfg).__name__}")

    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg:
            raise ConfigError(f"Missing config section: '{section}'")
        block = cfg[section]
        if not isinstance(block, dict):
            raise ConfigError(
                f"Config section '{section}' must be a mapping, got {type(block).__name__}"
            )
        missing = [key for key in keys if key not in block]
        if missing:
            raise ConfigError(f"Config section '{section}' is missing keys: {missing}")

    seed = cfg["project"]["random_seed"]
    if not _is_int(seed) or seed < 0:
        raise ConfigError(f"'project.random_seed' must be an integer >= 0, got {seed!r}")

    for section in ("paths", "data"):
        for key in REQUIRED_KEYS[section]:
            value = cfg[section][key]
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"'{section}.{key}' must be a non-empty string, got {value!r}")

    total = 0.0
    for key in REQUIRED_KEYS["split"]:
        value = cfg["split"][key]
        if not _is_number(value) or not 0 < value < 1:
            raise ConfigError(
                f"'split.{key}' must be a number between 0 and 1 (exclusive), got {value!r}"
            )
        total += value
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ConfigError(f"'split' sizes must sum to 1.0, got {total:.4f}")

    n_splits = cfg["cv"]["n_splits"]
    if not _is_int(n_splits) or n_splits < 2:
        raise ConfigError(f"'cv.n_splits' must be an integer >= 2, got {n_splits!r}")

    business = cfg["business"]
    ltv = business["customer_ltv"]
    if not _is_number(ltv) or ltv <= 0:
        raise ConfigError(f"'business.customer_ltv' must be a number > 0, got {ltv!r}")
    cost = business["retention_offer_cost"]
    if not _is_number(cost) or cost < 0:
        raise ConfigError(f"'business.retention_offer_cost' must be a number >= 0, got {cost!r}")
    rate = business["offer_success_rate"]
    if not _is_number(rate) or not 0 <= rate <= 1:
        raise ConfigError(f"'business.offer_success_rate' must be between 0 and 1, got {rate!r}")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read, parse and validate the YAML config.

    The file is chosen in this order: the `path` argument, the CONFIG_PATH
    environment variable (also read from a local .env file), then the default
    `configs/config.yaml`. Run commands from the repository root, because paths
    inside the config are relative to it.

    Raises:
        ConfigError: if the file is missing, unreadable, not valid YAML, or invalid.
    """
    load_dotenv()
    config_path = Path(path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path.resolve()}")
    if not config_path.is_file():
        raise ConfigError(f"Config path is not a file: {config_path.resolve()}")

    try:
        with config_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML ({config_path}): {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Config file could not be read ({config_path}): {exc}") from exc

    if cfg is None:
        raise ConfigError(f"Config file is empty: {config_path}")
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config file must contain key: value pairs, got {type(cfg).__name__}")

    validate_config(cfg)
    return cfg