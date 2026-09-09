from __future__ import annotations

import logging
from pathlib import Path

from src.logging_utils import VERBOSE_LEVEL_NUM, make_logger


def test_make_logger_without_console_attaches_only_file_handler(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=False, console=False)

    assert log_path == tmp_path / "log.txt"
    assert len(logger.handlers) == 1
    assert logger.handlers[0].level == logging.INFO


def test_make_logger_with_console_attaches_file_and_console_handlers(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=False, console=True)

    assert log_path == tmp_path / "log.txt"
    assert len(logger.handlers) == 2
    levels = sorted(handler.level for handler in logger.handlers)
    assert levels == [logging.INFO, logging.INFO]


def test_make_logger_verbose_sets_logger_level_to_verbose(tmp_path) -> None:
    logger, _ = make_logger(tmp_path, verbose=True, console=False)

    assert logger.level == VERBOSE_LEVEL_NUM


def test_make_logger_nonverbose_sets_logger_level_to_info(tmp_path) -> None:
    logger, _ = make_logger(tmp_path, verbose=False, console=False)

    assert logger.level == logging.INFO


def test_make_logger_reconfiguration_does_not_accumulate_handlers(tmp_path) -> None:
    logger1, path1 = make_logger(tmp_path, verbose=False, console=False)
    assert len(logger1.handlers) == 1

    logger2, path2 = make_logger(tmp_path, verbose=True, console=True)

    assert logger1 is logger2
    assert path1 == tmp_path / "log.txt"
    assert path2 == tmp_path / "log.txt"
    assert len(logger2.handlers) == 2


def test_make_logger_scenario_id_uses_scenario_log_path_and_creates_logs_dir(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=False, scenario_id=11, console=False)

    assert log_path == tmp_path / "logs" / "s11.txt"
    assert (tmp_path / "logs").exists()
    for handler in logger.handlers:
        handler.flush()


def test_make_logger_writes_verbose_message_to_file_when_verbose_enabled(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=True, console=False)

    logger.verbose("verbose line for file")
    for handler in logger.handlers:
        handler.flush()

    text = log_path.read_text(encoding="utf-8")
    assert "verbose line for file" in text


def test_make_logger_suppresses_verbose_message_when_verbose_disabled(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=False, console=False)

    logger.verbose("this should not appear")
    logger.info("this should appear")
    for handler in logger.handlers:
        handler.flush()

    text = log_path.read_text(encoding="utf-8")
    assert "this should appear" in text
    assert "this should not appear" not in text