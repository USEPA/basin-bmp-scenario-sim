from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.sampling as sampling
from src.input_config import (
    load_distribution_catalog,
    resolve_distribution_references,
    _expand_pollutant_load_rate_defaults,
    _load_plet_hydrology_lookup,
    _load_plet_parameter_table,
)
from src.input_validation import validate_numeric_distribution_rows
from src.plet_rusle import (
    PLET_HSG_VALUES,
    PLET_LAND_COVERS,
    _sample_parameter_table,
    _sample_plet_hydrology,
    initialize_plet_rusle_state,
)


class Logger:
    def verbose(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass


class Ctx:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)

    def _trunc_normal(self, *args, **kwargs):
        return sampling._trunc_normal(self, *args, **kwargs)

    def _piecewise_quantile_sample(self, *args, **kwargs):
        return sampling._piecewise_quantile_sample(self, *args, **kwargs)

    def _sample_from_stats(self, *args, **kwargs):
        return sampling._sample_from_stats(self, *args, **kwargs)


def hydrology_rows(default_cn=70.0, default_inf=0.25):
    rows = []
    for land_cover in PLET_LAND_COVERS:
        for hsg in PLET_HSG_VALUES:
            rows.append({"land_cover": land_cover, "hsg": hsg, "parameter": "cn", "value": default_cn})
            rows.append({"land_cover": land_cover, "hsg": hsg, "parameter": "infiltration_fraction", "value": default_inf})
    return pd.DataFrame(rows)


def test_numeric_schema_rejects_mixed_fixed_and_distribution():
    with pytest.raises(ValueError, match="mixes fixed value"):
        validate_numeric_distribution_rows(pd.DataFrame([{"value": 1, "mean": 1, "sd": 0.1}]), "x")


def test_distribution_catalog_reference_expands(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame([{"distribution_id": "rain", "mean": 42, "sd": 3, "min": 30, "max": 55}]).to_csv(path, index=False)
    catalog = load_distribution_catalog(path)
    use = pd.DataFrame([{"pid": "*", "parameter": "annual_precip_in", "distribution_id": "rain"}])
    resolved = resolve_distribution_references(use, catalog, "plet_inputs")
    assert resolved.loc[0, "mean"] == 42
    assert resolved.loc[0, "sd"] == 3
    assert resolved.loc[0, "min"] == 30
    assert resolved.loc[0, "max"] == 55


def test_hydrology_lookup_requires_full_pair_parameter_coverage(tmp_path):
    table = hydrology_rows().iloc[:-1]
    path = tmp_path / "hydrology.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="must define cn and infiltration_fraction"):
        _load_plet_hydrology_lookup(path, Logger())


def test_hydrology_lookup_accepts_stochastic_cn_and_infiltration(tmp_path):
    table = hydrology_rows()
    mask_cn = (table.land_cover == "cropland") & (table.hsg == "B") & (table.parameter == "cn")
    mask_inf = (table.land_cover == "cropland") & (table.hsg == "B") & (table.parameter == "infiltration_fraction")
    table.loc[mask_cn, "value"] = np.nan
    table.loc[mask_cn, ["mean", "sd", "min", "max"]] = [78, 2, 70, 86]
    table.loc[mask_inf, "value"] = np.nan
    table.loc[mask_inf, ["mean", "sd", "min", "max"]] = [0.30, 0.03, 0.20, 0.40]
    path = tmp_path / "hydrology.csv"
    table.to_csv(path, index=False)
    loaded = _load_plet_hydrology_lookup(path, Logger())
    ctx = Ctx(seed=123)
    ctx.load_generation = {"_hydrology_lookup_table": loaded}
    result = _sample_plet_hydrology(ctx, "cropland", "B", pid="P1", cache={})
    assert 70 <= result["cn"] <= 86
    assert 0.20 <= result["infiltration_fraction"] <= 0.40


def test_wildcard_parameter_distributions_are_independent_without_sample_group():
    ctx = Ctx(seed=7)
    table = pd.DataFrame([{"pid": "*", "parameter": "annual_precip_in", "mean": 42.0, "sd": 3.0}])
    values = _sample_parameter_table(ctx, table, ["P1", "P2"], cache_prefix="plet")
    assert values[0]["annual_precip_in"] != values[1]["annual_precip_in"]


def test_sample_group_explicitly_shares_parameter_draw():
    ctx = Ctx(seed=7)
    table = pd.DataFrame([{"pid": "*", "parameter": "annual_precip_in", "mean": 42.0, "sd": 3.0, "sample_group": "watershed_year"}])
    values = _sample_parameter_table(ctx, table, ["P1", "P2"], cache_prefix="plet")
    assert values[0]["annual_precip_in"] == values[1]["annual_precip_in"]


def test_pollutant_load_rate_wildcard_defaults_expand_with_exact_override():
    table = pd.DataFrame([
        {"pid": "*", "pollutant": "TN", "pathway": "surface", "value": 10.0},
        {"pid": "P2", "pollutant": "TN", "pathway": "surface", "value": 20.0},
    ])
    out = _expand_pollutant_load_rate_defaults(table, ["P1", "P2"], ["TN"])
    got = dict(zip(out.pid, out.value))
    assert got == {"P1": 10.0, "P2": 20.0}


def test_generic_mean_sd_respects_row_min_max():
    ctx = Ctx(seed=22)
    for _ in range(100):
        value = sampling._sample_from_stats(ctx, {"mean": 0, "sd": 100, "min": 4, "max": 6})
        assert 4 <= value <= 6


def test_signed_efficiency_still_allows_negative_values():
    ctx = Ctx(seed=1)
    assert sampling._sample_from_stats(ctx, {"value": -0.25}, kind="efficiency") == -0.25


def test_plet_initializer_uses_configured_hydrology_table():
    ctx = Ctx(seed=5)
    ctx.parcel_selection_ids = ["P1"]
    ctx.pollutants = ["TN"]
    ctx.plet_inputs = pd.DataFrame([
        {"pid": "*", "parameter": "annual_precip_in", "value": 42.0},
        {"pid": "*", "parameter": "rain_days", "value": 120.0},
        {"pid": "*", "parameter": "rain_correction_fraction", "value": 0.9},
        {"pid": "*", "parameter": "runoff_day_fraction", "value": 0.35},
        {"pid": "*", "parameter": "land_cover", "value": "cropland"},
        {"pid": "*", "parameter": "hsg", "value": "B"},
    ])
    ctx.rusle_inputs = None
    ctx.pollutant_concentrations = pd.DataFrame([{"pid": "*", "pollutant": "TN", "value": 3.0}])
    ctx.groundwater_concentrations = pd.DataFrame([{"pid": "*", "pollutant": "TN", "value": 5.5}])
    table = hydrology_rows()
    table.loc[(table.land_cover == "cropland") & (table.hsg == "B") & (table.parameter == "cn"), "value"] = 61.0
    table.loc[(table.land_cover == "cropland") & (table.hsg == "B") & (table.parameter == "infiltration_fraction"), "value"] = 0.41
    ctx.load_generation = {"_hydrology_lookup_table": table}
    _, state = initialize_plet_rusle_state(ctx)
    assert state.parameters[0]["cn"] == 61.0
    assert state.parameters[0]["infiltration_fraction"] == 0.41
    assert state.pathway_load_rates.shape == (1, 1, 2)


def test_distribution_catalog_rejects_blank_id(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame([{"distribution_id": None, "value": 1.0}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="blank distribution_id"):
        load_distribution_catalog(path)


def test_numeric_schema_rejects_mixed_percentile_and_normal_forms():
    table = pd.DataFrame([
        {"mean": 10.0, "sd": 2.0, "min": 4.0, "p50": 10.0, "max": 16.0}
    ])
    with pytest.raises(ValueError, match="mixes percentile and normal"):
        validate_numeric_distribution_rows(table, "x")


def test_plet_classifications_reject_distribution_statistics(tmp_path):
    path = tmp_path / "plet_inputs.csv"
    pd.DataFrame([
        {"pid": "*", "parameter": "annual_precip_in", "value": 42.0},
        {"pid": "*", "parameter": "land_cover", "value": "cropland", "mean": 1.0},
        {"pid": "*", "parameter": "hsg", "value": "B"},
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="classifications and must use only a fixed value"):
        _load_plet_parameter_table(path, ["P1"], Logger())
