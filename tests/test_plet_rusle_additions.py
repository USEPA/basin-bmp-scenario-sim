from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.sampling as sampling
from src.plet_rusle import (
    PLET_HSG_VALUES,
    PLET_LAND_COVERS,
    PLET_PATHWAY_NAMES,
    _sample_parameter_table,
    _sample_plet_hydrology,
    initialize_plet_rusle_state,
    plet_hydrology_from_classifications,
)


class Ctx:
    def __init__(self, seed: int = 1) -> None:
        self.rng = np.random.default_rng(seed)

    def _trunc_normal(self, *args, **kwargs):
        return sampling._trunc_normal(self, *args, **kwargs)

    def _piecewise_quantile_sample(self, *args, **kwargs):
        return sampling._piecewise_quantile_sample(self, *args, **kwargs)

    def _sample_from_stats(self, *args, **kwargs):
        return sampling._sample_from_stats(self, *args, **kwargs)


def hydrology_rows(default_cn: float = 70.0, default_inf: float = 0.25) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for land_cover in PLET_LAND_COVERS:
        for hsg in PLET_HSG_VALUES:
            rows.append({"land_cover": land_cover, "hsg": hsg, "parameter": "cn", "value": default_cn})
            rows.append(
                {
                    "land_cover": land_cover,
                    "hsg": hsg,
                    "parameter": "infiltration_fraction",
                    "value": default_inf,
                }
            )
    return pd.DataFrame(rows)


def test_plet_pathway_names_are_surface_and_subsurface_only() -> None:
    assert tuple(PLET_PATHWAY_NAMES) == ("surface", "subsurface")


@pytest.mark.parametrize(
    ("land_cover", "expected"),
    [
        ("developed", "urban"),
        ("crop", "cropland"),
        ("row_crop", "cropland"),
        ("pasture", "pastureland"),
        ("woods", "forest"),
        ("userdefined", "user_defined"),
    ],
)
def test_plet_hydrology_from_classifications_normalizes_land_cover_aliases(land_cover, expected) -> None:
    params = plet_hydrology_from_classifications(land_cover, "b")
    assert params["land_cover"] == expected
    assert params["hsg"] == "B"
    assert "cn" in params
    assert "infiltration_fraction" in params


def test_sample_plet_hydrology_uses_cached_value_for_same_pid_and_pair() -> None:
    ctx = Ctx(seed=123)
    table = hydrology_rows()
    mask_cn = (table.land_cover == "cropland") & (table.hsg == "B") & (table.parameter == "cn")
    mask_inf = (table.land_cover == "cropland") & (table.hsg == "B") & (table.parameter == "infiltration_fraction")
    table.loc[mask_cn, "value"] = np.nan
    table.loc[mask_cn, ["mean", "sd", "min", "max"]] = [78.0, 2.0, 70.0, 86.0]
    table.loc[mask_inf, "value"] = np.nan
    table.loc[mask_inf, ["mean", "sd", "min", "max"]] = [0.30, 0.03, 0.20, 0.40]
    ctx.load_generation = {"_hydrology_lookup_table": table}

    cache: dict[object, object] = {}
    first = _sample_plet_hydrology(ctx, "cropland", "B", pid="P1", cache=cache)
    second = _sample_plet_hydrology(ctx, "cropland", "B", pid="P1", cache=cache)

    assert first == second
    assert 70.0 <= first["cn"] <= 86.0
    assert 0.20 <= first["infiltration_fraction"] <= 0.40


def test_wildcard_parameter_distributions_are_independent_without_sample_group() -> None:
    ctx = Ctx(seed=7)
    table = pd.DataFrame([{"pid": "*", "parameter": "annual_precip_in", "mean": 42.0, "sd": 3.0}])

    values = _sample_parameter_table(ctx, table, ["P1", "P2"], cache_prefix="plet")

    assert values[0]["annual_precip_in"] != values[1]["annual_precip_in"]


def test_sample_group_explicitly_shares_parameter_draw() -> None:
    ctx = Ctx(seed=7)
    table = pd.DataFrame(
        [
            {
                "pid": "*",
                "parameter": "annual_precip_in",
                "mean": 42.0,
                "sd": 3.0,
                "sample_group": "watershed_year",
            }
        ]
    )

    values = _sample_parameter_table(ctx, table, ["P1", "P2"], cache_prefix="plet")

    assert values[0]["annual_precip_in"] == values[1]["annual_precip_in"]


def test_sample_parameter_table_exact_pid_override_wins_over_wildcard_default() -> None:
    ctx = Ctx(seed=7)
    table = pd.DataFrame(
        [
            {"pid": "*", "parameter": "annual_precip_in", "value": 42.0},
            {"pid": "P2", "parameter": "annual_precip_in", "value": 55.0},
            {"pid": "*", "parameter": "rain_days", "value": 120.0},
            {"pid": "*", "parameter": "rain_correction_fraction", "value": 0.9},
            {"pid": "*", "parameter": "runoff_day_fraction", "value": 0.35},
            {"pid": "*", "parameter": "land_cover", "value": "cropland"},
            {"pid": "*", "parameter": "hsg", "value": "B"},
        ]
    )

    values = _sample_parameter_table(ctx, table, ["P1", "P2"], cache_prefix="plet")

    assert values[0]["annual_precip_in"] == pytest.approx(42.0)
    assert values[1]["annual_precip_in"] == pytest.approx(55.0)
    assert values[0]["land_cover"] == "cropland"
    assert values[1]["hsg"] == "B"


def test_initialize_plet_rusle_state_returns_baseline_and_surface_subsurface_state() -> None:
    ctx = Ctx(seed=5)
    ctx.parcel_selection_ids = ["P1"]
    ctx.pollutants = ["TN"]
    ctx.plet_inputs = pd.DataFrame(
        [
            {"pid": "*", "parameter": "annual_precip_in", "value": 42.0},
            {"pid": "*", "parameter": "rain_days", "value": 120.0},
            {"pid": "*", "parameter": "rain_correction_fraction", "value": 0.9},
            {"pid": "*", "parameter": "runoff_day_fraction", "value": 0.35},
            {"pid": "*", "parameter": "land_cover", "value": "cropland"},
            {"pid": "*", "parameter": "hsg", "value": "B"},
        ]
    )
    ctx.rusle_inputs = None
    ctx.pollutant_concentrations = pd.DataFrame([{"pid": "*", "pollutant": "TN", "value": 3.0}])
    ctx.groundwater_concentrations = pd.DataFrame([{"pid": "*", "pollutant": "TN", "value": 5.5}])

    table = hydrology_rows()
    table.loc[(table.land_cover == "cropland") & (table.hsg == "B") & (table.parameter == "cn"), "value"] = 61.0
    table.loc[
        (table.land_cover == "cropland") & (table.hsg == "B") & (table.parameter == "infiltration_fraction"),
        "value",
    ] = 0.41
    ctx.load_generation = {"_hydrology_lookup_table": table}

    baseline, state = initialize_plet_rusle_state(ctx)

    assert tuple(PLET_PATHWAY_NAMES) == ("surface", "subsurface")
    assert baseline.shape == (1, 1)
    assert state.pathway_load_rates.shape == (1, 1, 2)
    assert state.parameters[0]["cn"] == pytest.approx(61.0)
    assert state.parameters[0]["infiltration_fraction"] == pytest.approx(0.41)


def test_initialize_plet_rusle_state_preserves_one_parameter_mapping_per_selected_parcel() -> None:
    ctx = Ctx(seed=9)
    ctx.parcel_selection_ids = ["P1", "P2"]
    ctx.pollutants = ["TN"]
    ctx.plet_inputs = pd.DataFrame(
        [
            {"pid": "*", "parameter": "annual_precip_in", "value": 42.0},
            {"pid": "*", "parameter": "rain_days", "value": 120.0},
            {"pid": "*", "parameter": "rain_correction_fraction", "value": 0.9},
            {"pid": "*", "parameter": "runoff_day_fraction", "value": 0.35},
            {"pid": "*", "parameter": "land_cover", "value": "cropland"},
            {"pid": "*", "parameter": "hsg", "value": "B"},
            {"pid": "P2", "parameter": "annual_precip_in", "value": 50.0},
        ]
    )
    ctx.rusle_inputs = None
    ctx.pollutant_concentrations = pd.DataFrame([{"pid": "*", "pollutant": "TN", "value": 3.0}])
    ctx.groundwater_concentrations = pd.DataFrame([{"pid": "*", "pollutant": "TN", "value": 5.5}])
    ctx.load_generation = {"_hydrology_lookup_table": hydrology_rows()}

    baseline, state = initialize_plet_rusle_state(ctx)

    assert tuple(PLET_PATHWAY_NAMES) == ("surface", "subsurface")
    assert baseline.shape == (2, 1)
    assert state.pathway_load_rates.shape == (2, 1, 2)
    assert len(state.parameters) == 2
    assert state.parameters[0]["annual_precip_in"] == pytest.approx(42.0)
    assert state.parameters[1]["annual_precip_in"] == pytest.approx(50.0)