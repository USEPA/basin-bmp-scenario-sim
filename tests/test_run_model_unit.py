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

    def log(self, *args, **kwargs) -> None:
        return None

    def verbose(self, *args, **kwargs) -> None:
        return None


class DummyModel:
    def __init__(self, cfg, data, logger) -> None:
        self.cfg = cfg
        self.data = data
        self.logger = logger
        self.run_called = False

    def run_all_scenarios(self):
        self.run_called = True
        return [{"scenario": 1, "result": "ok"}]


def test_parse_args_parses_required_config(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_model.py", "config.yaml"])

    args = run_model.parse_args()

    assert args.config == "config.yaml"
    assert args.outputs is None
    assert args.seed is None
    assert args.quiet is False


def test_parse_args_parses_optional_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_model.py",
            "config.yaml",
            "--outputs",
            "custom_outputs",
            "--seed",
            "123",
            "--quiet",
        ],
    )

    args = run_model.parse_args()

    assert args.config == "config.yaml"
    assert args.outputs == "custom_outputs"
    assert args.seed == 123
    assert args.quiet is True


def test_parse_args_version_exits_cleanly(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["run_model.py", "--version"])

    with pytest.raises(SystemExit) as excinfo:
        run_model.parse_args()

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "basin-bmp-sim 0.1.0" in captured.out


def test_main_prints_error_and_exits_on_missing_file(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["run_model.py", "missing.yaml"])
    monkeypatch.setattr(run_model, "read_config", lambda path: (_ for _ in ()).throw(FileNotFoundError("no such file")))

    with pytest.raises(SystemExit) as excinfo:
        run_model.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR: no such file" in captured.out


def test_main_prints_error_and_exits_on_validation_failure(monkeypatch, capsys) -> None:
    cfg = {CFG_OUTPUTS: "outputs", CFG_VERBOSE: False}

    monkeypatch.setattr("sys.argv", ["run_model.py", "bad.yaml"])
    monkeypatch.setattr(run_model, "read_config", lambda path: {"raw": "cfg"})
    monkeypatch.setattr(run_model, "normalize_config", lambda raw: cfg)
    monkeypatch.setattr(run_model, "validate_config", lambda normalized: (_ for _ in ()).throw(ValueError("invalid config")))

    with pytest.raises(SystemExit) as excinfo:
        run_model.main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR: invalid config" in captured.out


def test_main_applies_cli_overrides_creates_outputs_and_runs_pipeline(tmp_path, monkeypatch) -> None:
    normalized_cfg = {
        CFG_OUTPUTS: str(tmp_path / "from_cfg"),
        CFG_RANDOM_SEED: 111,
        CFG_VERBOSE: True,
    }
    logger = DummyLogger()
    data_obj = {"loaded": "data"}
    calls: dict[str, object] = {}

    def fake_read_config(path):
        calls["read_config_path"] = path
        return {"raw": "cfg"}

    def fake_normalize_config(raw):
        calls["normalize_input"] = raw
        return dict(normalized_cfg)

    def fake_validate_config(cfg):
        calls["validated_cfg"] = dict(cfg)

    def fake_make_logger(outputs_dir, *, verbose, console, scenario_id=None):
        calls["make_logger"] = {
            "outputs_dir": outputs_dir,
            "verbose": verbose,
            "console": console,
            "scenario_id": scenario_id,
        }
        return logger, Path(outputs_dir) / "log.txt"

    def fake_load_and_validate_all(cfg, logger_arg):
        calls["load_and_validate_all"] = {
            "cfg": dict(cfg),
            "logger_is_same": logger_arg is logger,
        }
        return data_obj

    class CapturingModel(DummyModel):
        def __init__(self, cfg, data, logger_arg) -> None:
            super().__init__(cfg, data, logger_arg)
            calls["model_init"] = {
                "cfg": dict(cfg),
                "data": data,
                "logger_is_same": logger_arg is logger,
            }

        def run_all_scenarios(self):
            calls["run_all_scenarios_called"] = True
            return [{"scenario": 1, "metric": 42.0}]

    def fake_make_summary_plots(cfg, data, scenario_records, outputs_dir, logger_arg):
        calls["make_summary_plots"] = {
            "cfg": dict(cfg),
            "data": data,
            "scenario_records": scenario_records,
            "outputs_dir": outputs_dir,
            "logger_is_same": logger_arg is logger,
        }

    override_outputs = tmp_path / "override_outputs"

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_model.py",
            str(tmp_path / "config.yaml"),
            "--outputs",
            str(override_outputs),
            "--seed",
            "999",
        ],
    )
    monkeypatch.setattr(run_model, "read_config", fake_read_config)
    monkeypatch.setattr(run_model, "normalize_config", fake_normalize_config)
    monkeypatch.setattr(run_model, "validate_config", fake_validate_config)
    monkeypatch.setattr(run_model, "make_logger", fake_make_logger)
    monkeypatch.setattr(run_model, "load_and_validate_all", fake_load_and_validate_all)
    monkeypatch.setattr(run_model, "Model", CapturingModel)
    monkeypatch.setattr(run_model, "make_summary_plots", fake_make_summary_plots)

    run_model.main()

    assert Path(calls["read_config_path"]) == tmp_path / "config.yaml"
    assert calls["normalize_input"] == {"raw": "cfg"}

    validated_cfg = calls["validated_cfg"]
    assert validated_cfg[CFG_OUTPUTS] == str(tmp_path / "from_cfg")
    assert validated_cfg[CFG_RANDOM_SEED] == 111

    logger_call = calls["make_logger"]
    assert logger_call["outputs_dir"] == override_outputs
    assert logger_call["verbose"] is True
    assert logger_call["console"] is True
    assert logger_call["scenario_id"] is None

    assert override_outputs.exists()
    assert override_outputs.is_dir()

    loaded_cfg = calls["load_and_validate_all"]["cfg"]
    assert loaded_cfg[CFG_OUTPUTS] == str(override_outputs)
    assert loaded_cfg[CFG_RANDOM_SEED] == 999
    assert calls["load_and_validate_all"]["logger_is_same"] is True

    model_cfg = calls["model_init"]["cfg"]
    assert model_cfg[CFG_OUTPUTS] == str(override_outputs)
    assert model_cfg[CFG_RANDOM_SEED] == 999
    assert calls["model_init"]["data"] == data_obj
    assert calls["model_init"]["logger_is_same"] is True

    assert calls["run_all_scenarios_called"] is True

    plot_call = calls["make_summary_plots"]
    assert plot_call["cfg"][CFG_OUTPUTS] == str(override_outputs)
    assert plot_call["cfg"][CFG_RANDOM_SEED] == 999
    assert plot_call["data"] == data_obj
    assert plot_call["scenario_records"] == [{"scenario": 1, "metric": 42.0}]
    assert plot_call["outputs_dir"] == override_outputs
    assert plot_call["logger_is_same"] is True

    info_messages = [msg for level, msg in logger.messages if level == "info"]
    assert any("Starting model run" in msg for msg in info_messages)
    assert any(f"Config: {tmp_path / 'config.yaml'}" in msg for msg in info_messages)
    assert any(f"Logging to: {override_outputs / 'log.txt'}" in msg for msg in info_messages)
    assert any("Model run complete" in msg for msg in info_messages)


def test_main_passes_console_false_when_quiet(tmp_path, monkeypatch) -> None:
    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_RANDOM_SEED: 123,
        CFG_VERBOSE: False,
    }
    calls: dict[str, object] = {}
    logger = DummyLogger()

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_model.py",
            str(tmp_path / "config.yaml"),
            "--quiet",
        ],
    )
    monkeypatch.setattr(run_model, "read_config", lambda path: {"raw": True})
    monkeypatch.setattr(run_model, "normalize_config", lambda raw: dict(cfg))
    monkeypatch.setattr(run_model, "validate_config", lambda normalized: None)

    def fake_make_logger(outputs_dir, *, verbose, console, scenario_id=None):
        calls["make_logger"] = {
            "outputs_dir": outputs_dir,
            "verbose": verbose,
            "console": console,
            "scenario_id": scenario_id,
        }
        return logger, Path(outputs_dir) / "log.txt"

    monkeypatch.setattr(run_model, "make_logger", fake_make_logger)
    monkeypatch.setattr(run_model, "load_and_validate_all", lambda cfg, logger_arg: {})
    monkeypatch.setattr(run_model, "Model", lambda cfg, data, logger_arg: SimpleNamespace(run_all_scenarios=lambda: []))
    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)

    run_model.main()

    logger_call = calls["make_logger"]
    assert logger_call["outputs_dir"] == tmp_path / "outputs"
    assert logger_call["verbose"] is False
    assert logger_call["console"] is False
    assert logger_call["scenario_id"] is None


def test_main_uses_config_verbose_flag_for_file_logging_not_quiet_flag(tmp_path, monkeypatch) -> None:
    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_RANDOM_SEED: 5,
        CFG_VERBOSE: True,
    }
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_model.py",
            str(tmp_path / "config.yaml"),
            "--quiet",
        ],
    )
    monkeypatch.setattr(run_model, "read_config", lambda path: {"raw": True})
    monkeypatch.setattr(run_model, "normalize_config", lambda raw: dict(cfg))
    monkeypatch.setattr(run_model, "validate_config", lambda normalized: None)
    monkeypatch.setattr(run_model, "load_and_validate_all", lambda cfg, logger: {})
    monkeypatch.setattr(run_model, "Model", lambda cfg, data, logger: SimpleNamespace(run_all_scenarios=lambda: []))
    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)

    def fake_make_logger(outputs_dir, *, verbose, console, scenario_id=None):
        calls["verbose"] = verbose
        calls["console"] = console
        return DummyLogger(), Path(outputs_dir) / "log.txt"

    monkeypatch.setattr(run_model, "make_logger", fake_make_logger)

    run_model.main()

    assert calls["verbose"] is True
    assert calls["console"] is False


def test_main_without_cli_overrides_uses_config_values(tmp_path, monkeypatch) -> None:
    cfg = {
        CFG_OUTPUTS: str(tmp_path / "cfg_outputs"),
        CFG_RANDOM_SEED: 321,
        CFG_VERBOSE: False,
    }
    calls: dict[str, object] = {}
    logger = DummyLogger()

    monkeypatch.setattr("sys.argv", ["run_model.py", str(tmp_path / "config.yaml")])
    monkeypatch.setattr(run_model, "read_config", lambda path: {"raw": "cfg"})
    monkeypatch.setattr(run_model, "normalize_config", lambda raw: dict(cfg))
    monkeypatch.setattr(run_model, "validate_config", lambda normalized: None)

    def fake_make_logger(outputs_dir, *, verbose, console, scenario_id=None):
        calls["outputs_dir"] = outputs_dir
        return logger, Path(outputs_dir) / "log.txt"

    def fake_load_and_validate_all(cfg_arg, logger_arg):
        calls["cfg_seen"] = dict(cfg_arg)
        return {}

    monkeypatch.setattr(run_model, "make_logger", fake_make_logger)
    monkeypatch.setattr(run_model, "load_and_validate_all", fake_load_and_validate_all)
    monkeypatch.setattr(run_model, "Model", lambda cfg, data, logger_arg: SimpleNamespace(run_all_scenarios=lambda: []))
    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)

    run_model.main()

    assert calls["outputs_dir"] == tmp_path / "cfg_outputs"
    assert calls["cfg_seen"][CFG_OUTPUTS] == str(tmp_path / "cfg_outputs")
    assert calls["cfg_seen"][CFG_RANDOM_SEED] == 321