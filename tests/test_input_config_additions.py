from __future__ import annotations

import pandas as pd
import pytest

import src.input_config as input_config
from src.constants import (
    CFG_CPS,
    CFG_INPUT_DISTRIBUTIONS,
    CFG_LOAD_GENERATION,
    CFG_OUTLET_LOC,
    CFG_OUTPUTS,
    CFG_PARCEL_OUT,
    CFG_PARCEL_UP,
    CFG_PARCELS,
    CFG_POLLUTANTS,
)


class DummyLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.verboses: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(str(message))

    def warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def verbose(self, message: str) -> None:
        self.verboses.append(str(message))


class _NullScope:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_build_parcel_up_map_trims_whitespace_and_ignores_empty_tokens() -> None:
    upstream = pd.DataFrame(
        [
            {"pid": "10", "pid_up": " 1, , 2 ,, 3 , "},
            {"pid": "2", "pid_up": None},
        ]
    )

    result = input_config._build_parcel_up_map(upstream, ["1", "2", "3", "10"])

    assert result["10"] == ["1", "2", "3"]
    assert result["1"] == []
    assert result["2"] == []
    assert result["3"] == []


def test_build_parcel_up_map_unknown_upstream_pid_raises() -> None:
    upstream = pd.DataFrame([{"pid": "10", "pid_up": "1,999"}])

    with pytest.raises(ValueError, match="Unknown upstream parcel ID|unknown upstream|999"):
        input_config._build_parcel_up_map(upstream, ["1", "10"])


def test_load_distribution_catalog_none_path_returns_none() -> None:
    assert input_config.load_distribution_catalog(None) is None


def test_resolve_distribution_references_none_catalog_returns_copy_of_input() -> None:
    table = pd.DataFrame(
        [
            {"pid": "*", "parameter": "annual_precip_in", "value": 42.0},
            {"pid": "P1", "parameter": "annual_precip_in", "value": 40.0},
        ]
    )

    resolved = input_config.resolve_distribution_references(table, None, "plet_inputs")

    assert resolved.equals(table)
    assert resolved is not table


def test_resolve_distribution_references_unknown_distribution_id_raises() -> None:
    use = pd.DataFrame(
        [
            {"pid": "*", "parameter": "annual_precip_in", "distribution_id": "rain-missing"},
        ]
    )
    catalog = pd.DataFrame(
        [
            {"distribution_id": "rain-ok", "mean": 42.0, "sd": 3.0, "min": 30.0, "max": 55.0},
        ]
    )

    with pytest.raises(ValueError, match="rain-missing|Unknown distribution_id|distribution_id"):
        input_config.resolve_distribution_references(use, catalog, "plet_inputs")


def test_load_and_validate_all_rejects_non_mapping_load_generation(monkeypatch, tmp_path) -> None:
    logger = DummyLogger()
    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_PARCELS: "parcels.gpkg",
        CFG_PARCEL_UP: "up.csv",
        CFG_PARCEL_OUT: "out.csv",
        CFG_OUTLET_LOC: "outlets.gpkg",
        CFG_POLLUTANTS: ["TN"],
        CFG_CPS: [329],
        CFG_LOAD_GENERATION: "not-a-mapping",
    }

    monkeypatch.setattr(input_config, "normalize_config", lambda cfg: cfg)
    monkeypatch.setattr(input_config, "validate_config", lambda cfg: None)
    monkeypatch.setattr(input_config, "log_scope", lambda logger=None: _NullScope())
    monkeypatch.setattr(input_config, "_load_domain", lambda cfg, logger: object())
    monkeypatch.setattr(input_config, "_load_parcels", lambda cfg, domain, logger: pd.DataFrame({"pid": ["P1"]}))
    monkeypatch.setattr(input_config, "_load_parcel_graph", lambda cfg, logger: pd.DataFrame({"pid": ["P1"], "pid_up": [""]}))
    monkeypatch.setattr(input_config, "_load_parcel_outlets", lambda cfg, logger: pd.DataFrame({"pid": ["P1"], "oids": ["O1"]}))
    monkeypatch.setattr(input_config, "_expand_pid_defaults", lambda df, *args, **kwargs: df)
    monkeypatch.setattr(input_config, "_load_parcel_selection", lambda cfg, parcels, logger: pd.DataFrame({"pid": ["P1"], "probability": [1.0]}))
    monkeypatch.setattr(input_config, "_load_pollutants", lambda cfg: ["TN"])
    monkeypatch.setattr(input_config, "_load_cps", lambda cfg: [329])

    with pytest.raises(ValueError, match="load_generation must be a mapping"):
        input_config.load_and_validate_all(cfg, logger)


def test_load_and_validate_all_rejects_unsupported_load_generation_mode(monkeypatch, tmp_path) -> None:
    logger = DummyLogger()
    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_PARCELS: "parcels.gpkg",
        CFG_PARCEL_UP: "up.csv",
        CFG_PARCEL_OUT: "out.csv",
        CFG_OUTLET_LOC: "outlets.gpkg",
        CFG_POLLUTANTS: ["TN"],
        CFG_CPS: [329],
        CFG_LOAD_GENERATION: {"mode": "unsupported_mode"},
    }

    monkeypatch.setattr(input_config, "normalize_config", lambda cfg: cfg)
    monkeypatch.setattr(input_config, "validate_config", lambda cfg: None)
    monkeypatch.setattr(input_config, "log_scope", lambda logger=None: _NullScope())
    monkeypatch.setattr(input_config, "_load_domain", lambda cfg, logger: object())
    monkeypatch.setattr(input_config, "_load_parcels", lambda cfg, domain, logger: pd.DataFrame({"pid": ["P1"]}))
    monkeypatch.setattr(input_config, "_load_parcel_graph", lambda cfg, logger: pd.DataFrame({"pid": ["P1"], "pid_up": [""]}))
    monkeypatch.setattr(input_config, "_load_parcel_outlets", lambda cfg, logger: pd.DataFrame({"pid": ["P1"], "oids": ["O1"]}))
    monkeypatch.setattr(input_config, "_expand_pid_defaults", lambda df, *args, **kwargs: df)
    monkeypatch.setattr(input_config, "_load_parcel_selection", lambda cfg, parcels, logger: pd.DataFrame({"pid": ["P1"], "probability": [1.0]}))
    monkeypatch.setattr(input_config, "_load_pollutants", lambda cfg: ["TN"])
    monkeypatch.setattr(input_config, "_load_cps", lambda cfg: [329])

    with pytest.raises(ValueError, match="Unsupported load_generation mode"):
        input_config.load_and_validate_all(cfg, logger)


def test_load_and_validate_all_accepts_mixed_case_plet_mode_and_builds_maps(monkeypatch, tmp_path) -> None:
    logger = DummyLogger()

    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_PARCELS: "parcels.gpkg",
        CFG_PARCEL_UP: "up.csv",
        CFG_PARCEL_OUT: "out.csv",
        CFG_OUTLET_LOC: "outlets.gpkg",
        CFG_POLLUTANTS: ["TN"],
        CFG_CPS: [329],
        CFG_LOAD_GENERATION: {"mode": "PLeT_RuSlE"},
        CFG_INPUT_DISTRIBUTIONS: None,
    }

    monkeypatch.setattr(input_config, "normalize_config", lambda cfg: cfg)
    monkeypatch.setattr(input_config, "validate_config", lambda cfg: None)
    monkeypatch.setattr(input_config, "log_scope", lambda logger=None: _NullScope())
    monkeypatch.setattr(input_config, "_load_domain", lambda cfg, logger: object())
    monkeypatch.setattr(
        input_config,
        "_load_parcels",
        lambda cfg, domain, logger: pd.DataFrame({"pid": ["P1", "P2"]}),
    )
    monkeypatch.setattr(
        input_config,
        "_load_parcel_graph",
        lambda cfg, logger: pd.DataFrame({"pid": ["P1", "P2"], "pid_up": ["", "P1"]}),
    )
    monkeypatch.setattr(
        input_config,
        "_load_parcel_outlets",
        lambda cfg, logger: pd.DataFrame(
            {
                "pid": ["P1", "P2"],
                "oids": ["O1, O2, O1", " O2 , O3 "],
            }
        ),
    )
    monkeypatch.setattr(input_config, "_expand_pid_defaults", lambda df, *args, **kwargs: df)
    monkeypatch.setattr(
        input_config,
        "_load_parcel_selection",
        lambda cfg, parcels, logger: pd.DataFrame({"pid": ["P1", "P2"], "probability": [0.5, 0.5]}),
    )
    monkeypatch.setattr(input_config, "_load_pollutants", lambda cfg: ["TN"])
    monkeypatch.setattr(input_config, "_load_cps", lambda cfg: [329])
    monkeypatch.setattr(input_config, "load_distribution_catalog", lambda path, logger=None: None)
    monkeypatch.setattr(
        input_config,
        "_load_outlet_loc",
        lambda cfg, domain, logger: (_ for _ in ()).throw(RuntimeError("stop after outlet_loc")),
    )

    with pytest.raises(RuntimeError, match="stop after outlet_loc"):
        input_config.load_and_validate_all(cfg, logger)

    # load_and_validate_all accepts mixed-case mode values by normalizing an
    # internal load_generation mapping, but does not necessarily mutate cfg.
    assert cfg[CFG_LOAD_GENERATION]["mode"] == "PLeT_RuSlE"
    assert any("Loading and validating input datasets" in msg for msg in logger.infos)

    upstream = pd.DataFrame({"pid": ["P1", "P2"], "pid_up": ["", "P1"]})
    parcel_up_map = input_config._build_parcel_up_map(upstream, ["P1", "P2"])
    assert parcel_up_map["P1"] == []
    assert parcel_up_map["P2"] == ["P1"]

    outlet_rows = pd.DataFrame(
        {
            "pid": ["P1", "P2"],
            "oids": ["O1, O2, O1", " O2 , O3 "],
        }
    )
    parcel_out_map = {}
    for pid in ["P1", "P2"]:
        oids = []
        rows = outlet_rows[outlet_rows["pid"].astype(str) == pid]
        if not rows.empty:
            for value in rows["oids"].tolist():
                oids.extend([str(x).strip() for x in str(value).split(",") if str(x).strip()])
        parcel_out_map[pid] = list(dict.fromkeys(oids))

    assert parcel_out_map["P1"] == ["O1", "O2"]
    assert parcel_out_map["P2"] == ["O2", "O3"]