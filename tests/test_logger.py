import logging

import pytest

from src.logger import LoggerSetupError, get_logger, shutdown_logger


@pytest.fixture
def make_logger():
    created = []

    def _make(name, **kwargs):
        logger = get_logger(name, **kwargs)
        created.append(logger)
        return logger

    yield _make
    for logger in created:
        shutdown_logger(logger)


def test_writes_to_file_and_creates_folders(make_logger, tmp_path):
    log_file = tmp_path / "nested" / "dir" / "app.log"
    logger = make_logger("t_file", log_file=log_file)
    logger.info("hello file")
    assert "hello file" in log_file.read_text(encoding="utf-8")


def test_writes_to_console(make_logger, capsys):
    logger = make_logger("t_console")
    logger.info("hello console")
    assert "hello console" in capsys.readouterr().out


def test_level_filters_messages(make_logger, tmp_path):
    log_file = tmp_path / "lvl.log"
    logger = make_logger("t_level", log_file=log_file, level="WARNING")
    logger.info("quiet message")
    logger.warning("loud message")
    text = log_file.read_text(encoding="utf-8")
    assert "loud message" in text
    assert "quiet message" not in text


def test_no_duplicate_handlers_on_repeat_calls(make_logger, tmp_path):
    log_file = tmp_path / "dup.log"
    make_logger("t_dup", log_file=log_file)
    logger = make_logger("t_dup", log_file=log_file)
    assert len(logger.handlers) == 2  # one console + one file, not four


def test_level_from_env_var(make_logger, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert make_logger("t_env").level == logging.DEBUG


def test_argument_beats_env_var(make_logger, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert make_logger("t_arg", level="ERROR").level == logging.ERROR


def test_invalid_level_raises():
    with pytest.raises(ValueError, match="Invalid log level"):
        get_logger("t_bad_level", level="LOUD")


def test_unwritable_log_path_raises(tmp_path):
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("I am a file, not a folder", encoding="utf-8")
    with pytest.raises(LoggerSetupError, match="Cannot write log file"):
        get_logger("t_blocked", log_file=blocker / "app.log")