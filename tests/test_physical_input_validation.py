from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    CFG_BMP_COST,
    CFG_BMP_EFFICIENCY,
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_BUFFER_DEPTH_FT,
    CFG_N_SCENARIOS,
    CFG_PARALLEL,
    CFG_POLLUTANT_LOAD_RATE,
    LOAD_CONCENTRATIONS,
)
from src.input_validation import validate_config, validate_stats_rows
from src.plet_rusle import plet_runoff_depth_in, rusle_sediment_load_rate_kg_ha_yr
from src import sampling


class DummySampler:
    _trunc_normal = sampling._trunc_normal
    _piecewise_quantile_sample = sampling._piecewise_quantile_sample
    _sample_from_stats = sampling._sample_from_stats

    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)


def _one(**values):
    return pd.DataFrame([values])


def test_default_load_rate_rejects_negative_fixed_value():
    with pytest.raises(ValueError, match="allowed physical domain"):
        validate_stats_rows(_one(value=-0.01), CFG_POLLUTANT_LOAD_RATE)


def test_default_load_rate_rejects_negative_explicit_support():
    with pytest.raises(ValueError, match="allowed physical domain"):
        validate_stats_rows(_one(mean=2.0, sd=1.0, min=-0.1, max=5.0), CFG_POLLUTANT_LOAD_RATE)


def test_negative_bmp_efficiency_is_valid_but_efficiency_above_one_is_not():
    validate_stats_rows(_one(value=-0.25), CFG_BMP_EFFICIENCY)
    with pytest.raises(ValueError, match="allowed physical domain"):
        validate_stats_rows(_one(value=1.01), CFG_BMP_EFFICIENCY)


def test_bmp_cost_and_concentration_are_nonnegative():
    with pytest.raises(ValueError):
        validate_stats_rows(_one(value=-1.0), CFG_BMP_COST)
    with pytest.raises(ValueError):
        validate_stats_rows(_one(value=-1.0), LOAD_CONCENTRATIONS)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("ia_ratio", -0.001),
        ("ia_ratio", 0.201),
        ("cn", 0.0),
        ("cn", 100.01),
        ("infiltration_fraction", -0.01),
        ("infiltration_fraction", 1.01),
        ("sdr", 1.01),
        ("annual_precip_in", -0.01),
        ("r", -0.01),
        ("sediment_n_pct", 100.01),
    ],
)
def test_plet_rusle_parameter_domains_are_enforced(parameter, value):
    with pytest.raises(ValueError, match="allowed physical domain"):
        validate_stats_rows(_one(parameter=parameter, value=value), "plet_rusle_parameter")


def test_ia_ratio_boundaries_are_valid():
    validate_stats_rows(_one(parameter="ia_ratio", value=0.0), "plet_inputs")
    validate_stats_rows(_one(parameter="ia_ratio", value=0.20), "plet_inputs")


def test_bounded_normal_is_truncated_not_clipped_after_sampling():
    sampler = DummySampler(seed=8)
    samples = [
        sampler._sample_from_stats({"mean": 0.05, "sd": 0.50}, kind="fraction")
        for _ in range(200)
    ]
    assert all(0.0 <= value <= 1.0 for value in samples)
    assert len(set(round(value, 8) for value in samples)) > 20


def test_invalid_fixed_value_is_rejected_by_sampler_instead_of_clipped():
    sampler = DummySampler()
    with pytest.raises(ValueError, match="allowed physical domain"):
        sampler._sample_from_stats({"value": -3.0}, kind="nonnegative")
    with pytest.raises(ValueError, match="allowed physical domain"):
        sampler._sample_from_stats({"value": 1.2}, kind="efficiency")


def test_config_rejects_negative_limits_and_nonintegral_counts():
    base = {
        CFG_N_SCENARIOS: 10,
        CFG_BUFFER_DEPTH_FT: 35.0,
        CFG_BMP_FAIL_RATE: 0.0,
        CFG_BMP_FAIL_REDUCTION: 0.25,
        CFG_BMP_LIMIT_N: None,
        CFG_BMP_LIMIT_USD: None,
        CFG_PARALLEL: {"n_jobs": 1},
    }
    validate_config(dict(base))

    bad = dict(base)
    bad[CFG_BMP_LIMIT_USD] = -1.0
    with pytest.raises(ValueError):
        validate_config(bad)

    bad = dict(base)
    bad[CFG_BMP_LIMIT_N] = 1.5
    with pytest.raises(ValueError):
        validate_config(bad)


def test_plet_runoff_rejects_invalid_inputs_instead_of_sanitizing():
    with pytest.raises(ValueError, match="ia_ratio"):
        plet_runoff_depth_in(40.0, 100.0, 1.0, 0.5, 75.0, ia_ratio=0.25)
    with pytest.raises(ValueError, match="annual_precip_in"):
        plet_runoff_depth_in(-1.0, 100.0, 1.0, 0.5, 75.0, ia_ratio=0.1)


def test_rusle_rejects_negative_factor_instead_of_clipping():
    params = {
        "r": -1.0,
        "k": 0.2,
        "ls": 1.0,
        "c": 0.1,
        "p": 1.0,
        "sdr": 0.5,
    }
    with pytest.raises(ValueError, match="r="):
        rusle_sediment_load_rate_kg_ha_yr(params)


def test_delivery_ratio_nan_is_rejected():
    from src.input_config import _complete_delivery_ratio_defaults

    table = pd.DataFrame(
        [{
            "pid": "1",
            "oid": "A",
            "sdr_f_to_s": np.nan,
            "sdr_s_to_o": 1.0,
            "ndr_f_to_s": 1.0,
            "ndr_s_to_o": 1.0,
        }]
    )
    with pytest.raises(ValueError, match="allowed physical domain"):
        _complete_delivery_ratio_defaults(table, {"1": ["A"]})


def test_aggregate_pathway_fraction_nan_is_rejected():
    from src.constants import CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS
    from src.input_config import _resolve_aggregate_pathway_fractions

    cfg = {
        CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS: {
            "surface": np.nan,
            "shallow subsurface": 1.0,
        }
    }
    with pytest.raises(ValueError, match="must be finite"):
        _resolve_aggregate_pathway_fractions(
            cfg, {}, ["surface", "shallow subsurface"]
        )
