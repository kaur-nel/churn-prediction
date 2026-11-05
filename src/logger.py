"""Logging setup: one place that builds loggers writing to console and (optionally) a file."""

import logging
import os
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_HANDLER_FLAG = "_churn_handler"


class LoggerSetupError(Exception):
    """Raised when the log file cannot be created or written."""


def _resolve_level(level: str | int | None) -> int:
    """Pick the log level: argument first, then LOG_LEVEL env var, then INFO."""
    raw = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    name = str(raw).strip().upper()
    levels = logging.getLevelNamesMapping()
    if name not in levels:
        raise ValueError(f"Invalid log level {raw!r}. Choose one of: {sorted(levels)}")
    return levels[name]


def shutdown_logger(logger: logging.Logger) -> None:
    """Remove and close the handlers that get_logger() added to this logger."""
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_FLAG, False):
            logger.removeHandler(handler)
            handler.close()


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    level: str | int | None = None,
) -> logging.Logger:
    """Return a logger that writes to the console and, if given, to `log_file`.

    Calling it again with the same name replaces the old handlers, so lines
    are never printed twice. Missing log folders are created.

    Raises:
        ValueError: if `level` is not a valid logging level.
        LoggerSetupError: if the log file cannot be created.
    """
    resolved_level = _resolve_level(level)
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(path, encoding="utf-8"))
        except OSError as exc:
            raise LoggerSetupError(f"Cannot write log file {path}: {exc}") from exc

    logger = logging.getLogger(name)
    shutdown_logger(logger)
    logger.setLevel(resolved_level)
    logger.propagate = False
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_FLAG, True)
        logger.addHandler(handler)
    return logger