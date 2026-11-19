from __future__ import annotations

import pandas as pd
import pytest

import src.input_config as input_config
from src.constants import (
    CFG_CPS,
    CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS,
    CFG_DETERMINISTIC_BMP_TYPE_ORDER,
    CFG_DETERMINISTIC_PARCEL_ORDER,
    CFG_LOAD_GENERATION,
    CFG_OUTLETS,
    CFG_OUTPUTS,
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


def test_apply_config_defaults_disables_deterministic_parcel_order() -> None:
    cfg = {}

    input_config.apply_config_defaults(cfg)

    assert cfg[CFG_DETERMINISTIC_PARCEL_ORDER] is False
    assert cfg[CFG_DETERMINISTIC_BMP_TYPE_ORDER] is False
    assert cfg[CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS] is False


def test_build_parcel_down_map_trims_normalized_edge_ids() -> None:
    upstream = pd.DataFrame(
        [
            {"pid": " 10 ", "pid_down": " 1 "},
        ]
    )
    result = input_config._build_parcel_down_map(upstream, ["1", "2", "3", "10"])
    assert result["10"] == "1"


def test_build_parcel_down_map_unknown_upstream_pid_raises() -> None:
    upstream = pd.DataFrame([{"pid": "10", "pid_down": "999"}])
    with pytest.raises(ValueError, match="999"):
        input_config._build_parcel_down_map(upstream, ["1", "10"])





def test_load_and_validate_all_rejects_non_mapping_load_generation(monkeypatch, tmp_path) -> None:
    logger = DummyLogger()
    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_PARCELS: "parcels.gpkg",
        CFG_OUTLETS: "outlets.gpkg",
        CFG_POLLUTANTS: ["TN"],
        CFG_CPS: [329],
        CFG_LOAD_GENERATION: "not-a-mapping",
    }

    monkeypatch.setattr(input_config, "normalize_config", lambda cfg: cfg)
    monkeypatch.setattr(input_config, "validate_config", lambda cfg: None)
    monkeypatch.setattr(input_config, "log_scope", lambda logger=None: _NullScope())
    monkeypatch.setattr(input_config, "_load_domain", lambda cfg, logger: object())
    monkeypatch.setattr(input_config, "_load_parcels", lambda cfg, domain, logger: pd.DataFrame({"pid": [1]}))
    monkeypatch.setattr(input_config, "_load_parcel_graph", lambda cfg, logger: pd.DataFrame(columns=["pid", "pid_down"]))
    monkeypatch.setattr(input_config, "_load_parcel_outlets", lambda cfg, logger: pd.DataFrame({"pid": [1], "oid": ["O1"]}))
    monkeypatch.setattr(input_config, "_load_parcel_selection", lambda cfg, parcels, logger: pd.DataFrame({"pid": [1], "probability": [1.0]}))
    monkeypatch.setattr(input_config, "_load_pollutants", lambda cfg: ["TN"])
    monkeypatch.setattr(input_config, "_load_cps", lambda cfg: [329])

    with pytest.raises(ValueError, match="load_generation must be a mapping"):
        input_config.load_and_validate_all(cfg, logger)


def test_load_and_validate_all_rejects_unsupported_load_generation_mode(monkeypatch, tmp_path) -> None:
    logger = DummyLogger()
    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_PARCELS: "parcels.gpkg",
        CFG_OUTLETS: "outlets.gpkg",
        CFG_POLLUTANTS: ["TN"],
        CFG_CPS: [329],
        CFG_LOAD_GENERATION: {"mode": "unsupported_mode"},
    }

    monkeypatch.setattr(input_config, "normalize_config", lambda cfg: cfg)
    monkeypatch.setattr(input_config, "validate_config", lambda cfg: None)
    monkeypatch.setattr(input_config, "log_scope", lambda logger=None: _NullScope())
    monkeypatch.setattr(input_config, "_load_domain", lambda cfg, logger: object())
    monkeypatch.setattr(input_config, "_load_parcels", lambda cfg, domain, logger: pd.DataFrame({"pid": [1]}))
    monkeypatch.setattr(input_config, "_load_parcel_graph", lambda cfg, logger: pd.DataFrame(columns=["pid", "pid_down"]))
    monkeypatch.setattr(input_config, "_load_parcel_outlets", lambda cfg, logger: pd.DataFrame({"pid": [1], "oid": ["O1"]}))
    monkeypatch.setattr(input_config, "_load_parcel_selection", lambda cfg, parcels, logger: pd.DataFrame({"pid": [1], "probability": [1.0]}))
    monkeypatch.setattr(input_config, "_load_pollutants", lambda cfg: ["TN"])
    monkeypatch.setattr(input_config, "_load_cps", lambda cfg: [329])

    with pytest.raises(ValueError, match="Unsupported load_generation mode"):
        input_config.load_and_validate_all(cfg, logger)


def test_load_and_validate_all_accepts_mixed_case_plet_mode_and_builds_maps(monkeypatch, tmp_path) -> None:
    logger = DummyLogger()

    cfg = {
        CFG_OUTPUTS: str(tmp_path / "outputs"),
        CFG_PARCELS: "parcels.gpkg",
        CFG_OUTLETS: "outlets.gpkg",
        CFG_POLLUTANTS: ["TN"],
        CFG_CPS: [329],
        CFG_LOAD_GENERATION: {"mode": "PLeT_RuSlE"},
    }

    monkeypatch.setattr(input_config, "normalize_config", lambda cfg: cfg)
    monkeypatch.setattr(input_config, "validate_config", lambda cfg: None)
    monkeypatch.setattr(input_config, "log_scope", lambda logger=None: _NullScope())
    monkeypatch.setattr(input_config, "_load_domain", lambda cfg, logger: object())
    monkeypatch.setattr(
        input_config,
        "_load_parcels",
        lambda cfg, domain, logger: pd.DataFrame({"pid": [1, 2]}),
    )
    monkeypatch.setattr(
        input_config,
        "_load_parcel_graph",
        lambda cfg, logger: pd.DataFrame({"pid": [2], "pid_down": [1]}),
    )
    monkeypatch.setattr(
        input_config,
        "_load_parcel_outlets",
        lambda cfg, logger: pd.DataFrame(
            {
                "pid": [1, 1, 2, 2],
                "oid": ["O1", "O2", "O2", "O3"],
            }
        ),
    )
    monkeypatch.setattr(
        input_config,
        "_load_parcel_selection",
        lambda cfg, parcels, logger: pd.DataFrame({"pid": [1, 2], "probability": [0.5, 0.5]}),
    )
    monkeypatch.setattr(input_config, "_load_pollutants", lambda cfg: ["TN"])
    monkeypatch.setattr(input_config, "_load_cps", lambda cfg: [329])
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

    upstream = pd.DataFrame({"pid": [2], "pid_down": [1]})
    parcel_down_map = input_config._build_parcel_down_map(upstream, [1, 2])
    assert parcel_down_map["1"] is None
    assert parcel_down_map["2"] == "1"

    outlet_rows = pd.DataFrame(
        {
            "pid": [1, 1, 2, 2],
            "oid": ["O1", "O2", "O2", "O3"],
        }
    )
    parcel_out_map = {
        pid: outlet_rows.loc[outlet_rows["pid"] == pid, "oid"].tolist()
        for pid in [1, 2]
    }
    assert parcel_out_map[1] == ["O1", "O2"]
    assert parcel_out_map[2] == ["O2", "O3"]

def test_load_pollutant_concentrations_requires_and_preserves_plet_pathway(tmp_path) -> None:
    logger = DummyLogger()
    path = tmp_path / "pollutant_concentrations.csv"
    pd.DataFrame(
        [
            {"pid": None, "pollutant": "tn", "pathway": "Surface", "value": 2.0, "units": "mg/L"},
            {"pid": None, "pollutant": "TN", "pathway": "subsurface", "value": 3.0, "units": "mg/L"},
            {"pid": None, "pollutant": "TP", "pathway": "surface", "value": 0.2, "units": "mg/L"},
            {"pid": None, "pollutant": "TP", "pathway": "subsurface", "value": 0.3, "units": "mg/L"},
        ]
    ).to_csv(path, index=False)

    loaded = input_config._load_pollutant_concentrations(
        path, ["TN", "TP"], logger
    )

    assert loaded is not None
    assert loaded[["pollutant", "pathway"]].to_dict("records") == [
        {"pollutant": "TN", "pathway": "surface"},
        {"pollutant": "TN", "pathway": "subsurface"},
        {"pollutant": "TP", "pathway": "surface"},
        {"pollutant": "TP", "pathway": "subsurface"},
    ]

    surface, subsurface = input_config._split_plet_concentrations(loaded)
    assert surface is not None and subsurface is not None
    assert set(surface["pathway"]) == {"surface"}
    assert set(subsurface["pathway"]) == {"subsurface"}
    assert surface.set_index("pollutant").loc["TN", "value"] == pytest.approx(2.0)
    assert subsurface.set_index("pollutant").loc["TN", "value"] == pytest.approx(3.0)


def test_load_pollutant_concentrations_rejects_missing_pathway_column(tmp_path) -> None:
    logger = DummyLogger()
    path = tmp_path / "pollutant_concentrations.csv"
    pd.DataFrame(
        [{"pid": None, "pollutant": "TN", "value": 2.0, "units": "mg/L"}]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="pathway"):
        input_config._load_pollutant_concentrations(path, ["TN"], logger)


def test_load_pollutant_concentrations_rejects_non_plet_pathway(tmp_path) -> None:
    logger = DummyLogger()
    path = tmp_path / "pollutant_concentrations.csv"
    pd.DataFrame(
        [
            {
                "pid": None,
                "pollutant": "TN",
                "pathway": "groundwater",
                "value": 3.0,
                "units": "mg/L",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="surface.*subsurface|unexpected"):
        input_config._load_pollutant_concentrations(path, ["TN"], logger)
