from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import run_model
from src.constants import CFG_OUTPUTS, CFG_RANDOM_SEED, CFG_VERBOSE


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", str(message)))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", str(message)))

    def log(self, *args, **kwargs):
        return None

    def verbose(self, *args, **kwargs):
        return None


def test_main_uses_config_path_as_path_object_in_read_config(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}
    logger = DummyLogger()

    monkeypatch.setattr("sys.argv", ["run_model.py", str(tmp_path / "config.yaml")])

    def fake_read_config(path):
        seen["path"] = path
        return {CFG_OUTPUTS: str(tmp_path / "out"), CFG_VERBOSE: False, CFG_RANDOM_SEED: 1}

    monkeypatch.setattr(run_model, "read_config", fake_read_config)
    monkeypatch.setattr(run_model, "normalize_config", lambda cfg: dict(cfg))
    monkeypatch.setattr(run_model, "validate_config", lambda cfg: None)
    monkeypatch.setattr(
        run_model,
        "make_logger",
        lambda outputs_dir, *, verbose, console: (logger, Path(outputs_dir) / "log.txt"),
    )
    monkeypatch.setattr(run_model, "load_and_validate_all", lambda cfg, logger: {})
    monkeypatch.setattr(run_model, "Model", lambda cfg, data, logger: SimpleNamespace(run_all_scenarios=lambda: []))
    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)

    run_model.main()

    assert seen["path"] == tmp_path / "config.yaml"


def test_main_passes_outputs_dir_path_to_make_summary_plots(tmp_path, monkeypatch) -> None:
    logger = DummyLogger()
    seen: dict[str, object] = {}

    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_RANDOM_SEED: 7,
        CFG_VERBOSE: False,
    }

    monkeypatch.setattr("sys.argv", ["run_model.py", str(tmp_path / "config.yaml")])
    monkeypatch.setattr(run_model, "read_config", lambda path: dict(cfg))
    monkeypatch.setattr(run_model, "normalize_config", lambda raw: dict(raw))
    monkeypatch.setattr(run_model, "validate_config", lambda cfg: None)
    monkeypatch.setattr(
        run_model,
        "make_logger",
        lambda outputs_dir, *, verbose, console: (logger, Path(outputs_dir) / "log.txt"),
    )
    monkeypatch.setattr(run_model, "load_and_validate_all", lambda cfg, logger: {"data": True})
    monkeypatch.setattr(
        run_model,
        "Model",
        lambda cfg, data, logger: SimpleNamespace(run_all_scenarios=lambda: [{"scenario": 1}]),
    )

    def fake_make_summary_plots(cfg_arg, data_arg, scenario_records, outputs_dir, logger_arg):
        seen["cfg"] = cfg_arg
        seen["data"] = data_arg
        seen["scenario_records"] = scenario_records
        seen["outputs_dir"] = outputs_dir
        seen["logger_same"] = logger_arg is logger

    monkeypatch.setattr(run_model, "make_summary_plots", fake_make_summary_plots)

    run_model.main()

    assert seen["scenario_records"] == [{"scenario": 1}]
    assert seen["data"] == {"data": True}
    assert seen["outputs_dir"] == tmp_path / "outputs"
    assert seen["logger_same"] is True


def test_main_creates_outputs_directory_before_logger_setup(tmp_path, monkeypatch) -> None:
    logger = DummyLogger()
    outputs_dir = tmp_path / "nested" / "outputs"
    seen: dict[str, object] = {}

    cfg = {
        CFG_OUTPUTS: str(outputs_dir),
        CFG_RANDOM_SEED: 7,
        CFG_VERBOSE: False,
    }

    monkeypatch.setattr("sys.argv", ["run_model.py", str(tmp_path / "config.yaml")])
    monkeypatch.setattr(run_model, "read_config", lambda path: dict(cfg))
    monkeypatch.setattr(run_model, "normalize_config", lambda raw: dict(raw))
    monkeypatch.setattr(run_model, "validate_config", lambda cfg: None)

    def fake_make_logger(outputs_dir_arg, *, verbose, console):
        seen["dir_exists_when_logger_called"] = Path(outputs_dir_arg).exists()
        return logger, Path(outputs_dir_arg) / "log.txt"

    monkeypatch.setattr(run_model, "make_logger", fake_make_logger)
    monkeypatch.setattr(run_model, "load_and_validate_all", lambda cfg, logger: {})
    monkeypatch.setattr(run_model, "Model", lambda cfg, data, logger: SimpleNamespace(run_all_scenarios=lambda: []))
    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)

    run_model.main()

    assert outputs_dir.exists()
    assert seen["dir_exists_when_logger_called"] is True


def test_main_logs_config_and_completion_messages(tmp_path, monkeypatch) -> None:
    logger = DummyLogger()
    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_RANDOM_SEED: 7,
        CFG_VERBOSE: True,
    }

    monkeypatch.setattr("sys.argv", ["run_model.py", str(tmp_path / "config.yaml")])
    monkeypatch.setattr(run_model, "read_config", lambda path: dict(cfg))
    monkeypatch.setattr(run_model, "normalize_config", lambda raw: dict(raw))
    monkeypatch.setattr(run_model, "validate_config", lambda cfg: None)
    monkeypatch.setattr(
        run_model,
        "make_logger",
        lambda outputs_dir, *, verbose, console: (logger, Path(outputs_dir) / "log.txt"),
    )
    monkeypatch.setattr(run_model, "load_and_validate_all", lambda cfg, logger: {})
    monkeypatch.setattr(run_model, "Model", lambda cfg, data, logger: SimpleNamespace(run_all_scenarios=lambda: []))
    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)

    run_model.main()

    info_messages = [msg for level, msg in logger.messages if level == "info"]
    assert any("Starting model run" in msg for msg in info_messages)
    assert any("Config:" in msg for msg in info_messages)
    assert any("Logging to:" in msg for msg in info_messages)
    assert any("Model run complete" in msg for msg in info_messages)


def test_main_preserves_cli_seed_override_into_model_cfg(tmp_path, monkeypatch) -> None:
    logger = DummyLogger()
    seen: dict[str, object] = {}

    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_RANDOM_SEED: 111,
        CFG_VERBOSE: False,
    }

    monkeypatch.setattr(
        "sys.argv",
        ["run_model.py", str(tmp_path / "config.yaml"), "--seed", "999"],
    )
    monkeypatch.setattr(run_model, "read_config", lambda path: dict(cfg))
    monkeypatch.setattr(run_model, "normalize_config", lambda raw: dict(raw))
    monkeypatch.setattr(run_model, "validate_config", lambda cfg: None)
    monkeypatch.setattr(
        run_model,
        "make_logger",
        lambda outputs_dir, *, verbose, console: (logger, Path(outputs_dir) / "log.txt"),
    )

    def fake_load_and_validate_all(cfg_arg, logger_arg):
        seen["cfg_after_overrides"] = dict(cfg_arg)
        return {}

    monkeypatch.setattr(run_model, "load_and_validate_all", fake_load_and_validate_all)
    monkeypatch.setattr(
        run_model,
        "Model",
        lambda cfg_arg, data, logger_arg: SimpleNamespace(run_all_scenarios=lambda: []),
    )
    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)

    run_model.main()

    assert seen["cfg_after_overrides"][CFG_RANDOM_SEED] == 999