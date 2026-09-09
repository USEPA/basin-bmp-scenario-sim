from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.constants import (
    CFG_BMP_COST,
    CFG_OUTLET_MEAN,
    CFG_OUTLET_TARGET,
    COL_MEAN,
    COL_OID,
    COL_POLLUTANT,
    COL_TARGET,
    DATA_OUTLET_LOC,
    DATA_OUTLET_MEAN,
    DATA_OUTLET_TARGET,
    DATA_POLLUTANTS,
    DIR_OUTLET_TRAJECTORIES,
    FILE_ALL_SCENARIOS_PARQUET,
    XAXIS_COST,
    YAXIS_MEAN,
    YAXIS_TARGET,
    YAXIS_TOTAL,
)
from src.plotting import _build_denominator_maps, make_summary_plots


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def log(self, *args, **kwargs):
        return None

    def verbose(self, message, *args, **kwargs):
        self.messages.append(("verbose", str(message)))

    def info(self, message, *args, **kwargs):
        self.messages.append(("info", str(message)))

    def warning(self, message, *args, **kwargs):
        self.messages.append(("warning", str(message)))


def test_build_denominator_maps_converts_numeric_values_and_preserves_nan_float_entries() -> None:
    data = {
        DATA_OUTLET_TARGET: pd.DataFrame(
            [
                {COL_OID: "1", COL_POLLUTANT: "TN", COL_TARGET: 10.0},
                {COL_OID: "2", COL_POLLUTANT: "TN", COL_TARGET: "bad"},
            ]
        ),
        DATA_OUTLET_MEAN: pd.DataFrame(
            [
                {COL_OID: "1", COL_POLLUTANT: "TN", COL_MEAN: 20.0},
                {COL_OID: "2", COL_POLLUTANT: "TN", COL_MEAN: None},
            ]
        ),
    }

    target_map, mean_map = _build_denominator_maps(data)

    assert target_map == {("1", "TN"): pytest.approx(10.0)}
    assert mean_map[("1", "TN")] == pytest.approx(20.0)
    assert ("2", "TN") in mean_map
    assert pd.isna(mean_map[("2", "TN")])


def test_make_summary_plots_uses_in_memory_records_without_loading_canonical_table(
    tmp_path, monkeypatch
) -> None:
    logger = DummyLogger()
    cfg = {CFG_BMP_COST: "bmp_cost.csv"}
    data = {
        DATA_POLLUTANTS: ["TN"],
        DATA_OUTLET_LOC: pd.DataFrame({COL_OID: [1]}),
    }
    scenario_records = {
        ("TN", "1", XAXIS_COST, YAXIS_TOTAL): [
            (1, 100.0, 5.0),
            (1, 200.0, 4.0),
        ]
    }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("load_trajectory_records should not be called when in-memory records are provided")

    monkeypatch.setattr("src.plotting.load_trajectory_records", fail_if_called)

    make_summary_plots(cfg, data, scenario_records, tmp_path, logger)

    plots_dir = tmp_path / "plots"
    assert plots_dir.exists()
    assert any(p.name.startswith("plot_") for p in plots_dir.iterdir())


def test_make_summary_plots_loads_canonical_table_when_records_missing(tmp_path, monkeypatch) -> None:
    logger = DummyLogger()
    cfg = {CFG_BMP_COST: "bmp_cost.csv"}
    data = {
        DATA_POLLUTANTS: ["TN"],
        DATA_OUTLET_LOC: pd.DataFrame({COL_OID: [1]}),
    }
    canonical = tmp_path / DIR_OUTLET_TRAJECTORIES / FILE_ALL_SCENARIOS_PARQUET
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("placeholder", encoding="utf-8")

    called: dict[str, object] = {}

    def fake_load_trajectory_records(path: Path):
        called["path"] = Path(path)
        return {
            ("TN", "1", XAXIS_COST, YAXIS_TOTAL): [
                (1, 100.0, 5.0),
                (1, 150.0, 4.0),
            ]
        }

    monkeypatch.setattr("src.plotting.load_trajectory_records", fake_load_trajectory_records)

    make_summary_plots(cfg, data, None, tmp_path, logger)

    assert called["path"] == canonical
    assert any("Loaded canonical trajectory table for plotting" in msg for level, msg in logger.messages if level == "verbose")
    assert any(p.name.startswith("plot_") for p in (tmp_path / "plots").iterdir())


def test_make_summary_plots_warns_and_continues_when_canonical_load_fails(tmp_path, monkeypatch) -> None:
    logger = DummyLogger()
    cfg = {CFG_BMP_COST: "bmp_cost.csv"}
    data = {
        DATA_POLLUTANTS: ["TN"],
        DATA_OUTLET_LOC: pd.DataFrame({COL_OID: [1]}),
    }
    canonical = tmp_path / DIR_OUTLET_TRAJECTORIES / FILE_ALL_SCENARIOS_PARQUET
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("placeholder", encoding="utf-8")

    def boom(path: Path):
        raise RuntimeError("bad parquet")

    monkeypatch.setattr("src.plotting.load_trajectory_records", boom)

    make_summary_plots(cfg, data, None, tmp_path, logger)

    warnings = [msg for level, msg in logger.messages if level == "warning"]
    assert any("Failed to load canonical trajectory table" in msg for msg in warnings)
    assert (tmp_path / "plots").exists()


def test_make_summary_plots_supports_target_and_mean_axes(tmp_path) -> None:
    logger = DummyLogger()
    cfg = {
        CFG_BMP_COST: "bmp_cost.csv",
        CFG_OUTLET_TARGET: "outlet_target.csv",
        CFG_OUTLET_MEAN: "outlet_mean.csv",
    }
    data = {
        DATA_POLLUTANTS: ["TN"],
        DATA_OUTLET_LOC: pd.DataFrame({COL_OID: [1]}),
        DATA_OUTLET_TARGET: pd.DataFrame(
            [{COL_OID: "1", COL_POLLUTANT: "TN", COL_TARGET: 10.0}]
        ),
        DATA_OUTLET_MEAN: pd.DataFrame(
            [{COL_OID: "1", COL_POLLUTANT: "TN", COL_MEAN: 20.0}]
        ),
    }
    scenario_records = {
        ("TN", "1", XAXIS_COST, YAXIS_TOTAL): [(1, 100.0, 5.0), (1, 200.0, 4.0)],
        ("TN", "1", XAXIS_COST, YAXIS_TARGET): [(1, 100.0, 0.5), (1, 200.0, 0.4)],
        ("TN", "1", XAXIS_COST, YAXIS_MEAN): [(1, 100.0, 0.25), (1, 200.0, 0.20)],
    }

    make_summary_plots(cfg, data, scenario_records, tmp_path, logger)

    plot_names = {p.name for p in (tmp_path / "plots").iterdir()}
    assert any("cost" in name and "total" in name for name in plot_names)
    assert any("cost" in name and "target" in name for name in plot_names)
    assert any("cost" in name and "mean" in name for name in plot_names)