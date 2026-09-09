"""Integration regressions at PLET/RUSLE-to-scenario boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.constants import OUTPUT_IMPACTED_PIDS
from src.input_config import _build_parcel_up_map
from src.input_validation import validate_plet_runtime_inputs
from src.model import _bmp_impacted_parcel_indices


def test_comma_separated_upstream_ids_are_all_resolved() -> None:
    """Protect the historical multi-upstream ``pid_up`` parsing regression."""
    upstream = pd.DataFrame([{"pid": "10", "pid_up": "1, 2,3"}])

    parcel_up_map = _build_parcel_up_map(upstream, ["1", "2", "3", "10"])

    assert parcel_up_map["10"] == ["1", "2", "3"]
    assert parcel_up_map["1"] == []
    assert parcel_up_map["2"] == []
    assert parcel_up_map["3"] == []


def test_wetland_impacted_parcels_use_full_hydrologic_universe() -> None:
    """Ensure upstream wetland parcels are retained for downstream accounting."""
    ctx = SimpleNamespace(
        parcel_ids=["A", "B", "C"],
        pid_to_index={"A": 0, "B": 1, "C": 2},
    )
    bmp_rec = {OUTPUT_IMPACTED_PIDS: "C,A,B"}

    impacted = _bmp_impacted_parcel_indices(ctx, 2, bmp_rec)

    assert impacted == [2, 0, 1]


def test_wetland_baseline_mass_inputs_can_preserve_originating_parcels() -> None:
    """Document the parcel-resolved information needed for wetland routing tests."""
    ctx = SimpleNamespace(
        parcel_ids=["A", "B", "C"],
        pid_to_index={"A": 0, "B": 1, "C": 2},
        parcel_area_ha=np.asarray([1.0, 2.0, 3.0]),
    )
    bmp_rec = {OUTPUT_IMPACTED_PIDS: "C,A,B"}
    pre = np.asarray([[10.0], [20.0], [30.0]])
    post = np.asarray([[8.0], [15.0], [27.0]])

    origin_removed_mass_rates = {}
    for idx in _bmp_impacted_parcel_indices(ctx, 2, bmp_rec):
        origin_removed_mass_rates[ctx.parcel_ids[idx]] = (
            (pre[idx, 0] - post[idx, 0]) * ctx.parcel_area_ha[idx]
        )

    assert origin_removed_mass_rates == {"C": 9.0, "A": 2.0, "B": 10.0}


def _patch_loader_preamble(monkeypatch) -> None:
    """Bypass unrelated file I/O so load-generation config errors are isolated."""
    import src.input_config as input_config

    monkeypatch.setattr(input_config, "normalize_config", lambda cfg: cfg)
    monkeypatch.setattr(input_config, "validate_config", lambda cfg: None)
    monkeypatch.setattr(input_config, "_load_domain", lambda cfg, logger: object())
    monkeypatch.setattr(
        input_config,
        "_load_parcels",
        lambda cfg, domain, logger: pd.DataFrame({"pid": ["P1"]}),
    )
    monkeypatch.setattr(
        input_config,
        "_load_parcel_graph",
        lambda cfg, logger: pd.DataFrame({"pid": ["P1"], "pid_up": [np.nan]}),
    )
    monkeypatch.setattr(
        input_config,
        "_load_parcel_outlets",
        lambda cfg, logger: pd.DataFrame({"pid": ["P1"], "oids": ["O1"]}),
    )
    monkeypatch.setattr(input_config, "_expand_pid_defaults", lambda df, *args, **kwargs: df)
    monkeypatch.setattr(
        input_config,
        "_load_parcel_selection",
        lambda cfg, parcels, logger: pd.DataFrame(
            {"pid": ["P1"], "probability": [1.0]}
        ),
    )
    monkeypatch.setattr(input_config, "_load_pollutants", lambda cfg: ["TN"])
    monkeypatch.setattr(input_config, "_load_cps", lambda cfg: [340])
    monkeypatch.setattr(input_config, "load_distribution_catalog", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        input_config,
        "_load_outlet_loc",
        lambda cfg, domain, logger: pd.DataFrame({"oid": ["O1"]}),
    )
    monkeypatch.setattr(
        input_config,
        "_load_optional_outlet_stats",
        lambda *args, **kwargs: None,
    )


def test_removed_pathway_mode_key_is_rejected(monkeypatch) -> None:
    """Ensure stale PLET configuration cannot silently imply old behavior."""
    import logging

    import pytest

    from src.input_config import load_and_validate_all

    _patch_loader_preamble(monkeypatch)
    cfg = {
        "load_generation": {
            "mode": "plet_rusle",
            "pathway_mode": "derive_from_plet",
        }
    }

    with pytest.raises(ValueError, match="load_generation.pathway_mode has been removed"):
        load_and_validate_all(cfg, logging.getLogger("test_removed_pathway_mode"))


def test_unknown_load_generation_mode_is_rejected(monkeypatch) -> None:
    """Ensure misspelled/unsupported load-generation modes fail explicitly."""
    import logging

    import pytest

    from src.input_config import load_and_validate_all

    _patch_loader_preamble(monkeypatch)
    cfg = {"load_generation": {"mode": "plet_rusel"}}

    with pytest.raises(ValueError, match="Unsupported load_generation mode"):
        load_and_validate_all(cfg, logging.getLogger("test_bad_load_mode"))


@pytest.mark.parametrize(
    "label", ["watershed_area_mi2", "watershed_area_sqmi", "Watershed Area SQ MI"]
)
def test_removed_watershed_area_rusle_input_is_rejected(label) -> None:
    """Ensure a stale watershed-area row cannot be silently ignored at 100% delivery."""
    rusle_inputs = pd.DataFrame(
        [
            {"pid": "*", "parameter": "r", "value": 100.0},
            {"pid": "*", "parameter": label, "value": 5.0},
        ]
    )

    with pytest.raises(ValueError, match="no longer supported"):
        validate_plet_runtime_inputs(
            pd.DataFrame(columns=["pid", "parameter", "value"]),
            rusle_inputs,
            None,
            None,
            ["P1"],
            ["TN"],
        )
