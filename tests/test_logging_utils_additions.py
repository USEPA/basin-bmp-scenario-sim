from __future__ import annotations

from pathlib import Path

from src.logging_utils import make_logger


def test_make_logger_without_scenario_id_writes_top_level_log_file(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=False, console=False)

    assert log_path == tmp_path / "log.txt"
    assert log_path.parent == tmp_path
    logger.handlers[0].flush()


def test_make_logger_with_scenario_id_writes_logs_subdirectory_file(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=False, console=False, scenario_id=7)

    assert log_path == tmp_path / "logs" / "s7.txt"
    assert log_path.parent == tmp_path / "logs"
    logger.handlers[0].flush()


def test_make_logger_creates_parent_directories_for_log_path(tmp_path) -> None:
    outputs = tmp_path / "nested" / "outputs"

    logger, log_path = make_logger(outputs, verbose=False, console=False, scenario_id=3)

    assert outputs.exists()
    assert (outputs / "logs").exists()
    assert log_path.exists()
    logger.handlers[0].flush()


def test_make_logger_writes_emitted_message_to_file(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=False, console=False)

    logger.info("hello batch 5")
    for handler in logger.handlers:
        handler.flush()

    text = log_path.read_text(encoding="utf-8")
    assert "hello batch 5" in text


def test_make_logger_scenario_log_writes_to_expected_file(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=False, console=False, scenario_id=2)

    logger.warning("scenario specific message")
    for handler in logger.handlers:
        handler.flush()

    text = log_path.read_text(encoding="utf-8")
    assert "scenario specific message" in text
    assert log_path.name == "s2.txt"


def test_make_logger_returns_path_objects_under_requested_output_dir(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=True, console=False)

    assert isinstance(log_path, Path)
    assert tmp_path in log_path.parents or log_path.parent == tmp_path
    logger.handlers[0].flush()