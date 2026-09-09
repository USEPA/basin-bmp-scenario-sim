import math
import logging
import numpy as np
import pandas as pd

from src.model import Model
from src.constants import (
    DATA_RANDOM_SEED,
    DATA_PARCELS,
    DATA_POLLUTANTS,
    DATA_CPS,
    DATA_PARCEL_OUT_MAP,
    DATA_PARCEL_UP_MAP,
    DATA_OUTLET_LOC,
    DATA_PARCEL_P,
    DATA_POLLUTANT_LOAD_RATE,
    DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE,
    DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS,
    DATA_BMP_EFFICIENCY,
    DATA_LOAD_GENERATION,
)


def _minimal_cfg():
    return {
        "outputs": "./outputs",
        "verbose": False,
        "bmp_sel_prob_via_costs": False,
    }


def _minimal_data():
    parcels = pd.DataFrame(
        {
            "pid": ["P1"],
            "area_ha": [1.0],
            "perim_m": [100.0],
        }
    )

    outlet_loc = pd.DataFrame(
        {
            "oid": ["O1"],
        }
    )

    parcel_p = pd.DataFrame(
        {
            "pid": ["P1"],
            "probability": [1.0],
        }
    )

    pollutant_load_rate = pd.DataFrame(
        {
            "pid": ["P1"],
            "pollutant": ["TN"],
            "value": [100.0],
        }
    )

    bmp_eff = pd.DataFrame(
        {
            "cps": [340, 340, 340],
            "pollutant": ["TN", "TN", "TN"],
            "pathway": ["surface", "shallow subsurface", "deep subsurface"],
            "value": [0.5, 0.5, 0.5],
        }
    )

    return {
        DATA_RANDOM_SEED: 123,
        DATA_PARCELS: parcels,
        DATA_POLLUTANTS: ["TN"],
        DATA_CPS: [340],
        DATA_PARCEL_OUT_MAP: {"P1": ["O1"]},
        DATA_PARCEL_UP_MAP: {"P1": []},
        DATA_OUTLET_LOC: outlet_loc,
        DATA_PARCEL_P: parcel_p,
        DATA_POLLUTANT_LOAD_RATE: pollutant_load_rate,
        DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE: True,
        DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS: {
            "surface": 0.5,
            "shallow subsurface": 0.25,
            "deep subsurface": 0.25,
        },
        DATA_BMP_EFFICIENCY: bmp_eff,
        DATA_LOAD_GENERATION: {"mode": "statistical"},
    }


def _make_model():
    logger = logging.getLogger("test_serial_stacking_live")
    return Model(_minimal_cfg(), _minimal_data(), logger)


def _set_current_pathways(model, surface, shallow, deep):
    model.pathway_names = ["surface", "shallow subsurface", "deep subsurface"]
    model.current_pathway_load_rates = np.array(
        [[[float(surface), float(shallow), float(deep)]]],
        dtype=float,
    )


def test_apply_pathway_reduction_two_full_50_percent_reductions_produce_75_percent_total_reduction():
    model = _make_model()
    _set_current_pathways(model, surface=50.0, shallow=25.0, deep=25.0)

    baseline = model._get_current_total_load_rate(0, 0, 100.0)
    assert math.isclose(baseline, 100.0, rel_tol=0.0, abs_tol=1e-12)

    eff = {
        "surface": 0.50,
        "shallow subsurface": 0.50,
        "deep subsurface": 0.50,
    }

    model._apply_pathway_reduction(0, 0, 1.0, eff)
    after_first = model._get_current_total_load_rate(0, 0, 100.0)

    model._apply_pathway_reduction(0, 0, 1.0, eff)
    after_second = model._get_current_total_load_rate(0, 0, 100.0)

    assert math.isclose(after_first, 50.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_second, 25.0, rel_tol=0.0, abs_tol=1e-12)


def test_apply_pathway_reduction_second_application_acts_on_remaining_load_not_original_load():
    model = _make_model()
    _set_current_pathways(model, surface=50.0, shallow=25.0, deep=25.0)

    eff_1 = {
        "surface": 0.40,
        "shallow subsurface": 0.40,
        "deep subsurface": 0.40,
    }
    eff_2 = {
        "surface": 0.25,
        "shallow subsurface": 0.25,
        "deep subsurface": 0.25,
    }

    model._apply_pathway_reduction(0, 0, 1.0, eff_1)
    after_first = model._get_current_total_load_rate(0, 0, 100.0)

    model._apply_pathway_reduction(0, 0, 1.0, eff_2)
    after_second = model._get_current_total_load_rate(0, 0, 100.0)

    expected = 100.0 * (1.0 - 0.40) * (1.0 - 0.25)

    assert math.isclose(after_first, 60.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_second, expected, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(after_second, 45.0, rel_tol=0.0, abs_tol=1e-12)


def test_apply_pathway_reduction_partial_treatment_fraction_only_reduces_treated_portion():
    model = _make_model()
    _set_current_pathways(model, surface=50.0, shallow=25.0, deep=25.0)

    eff = {
        "surface": 0.40,
        "shallow subsurface": 0.40,
        "deep subsurface": 0.40,
    }

    model._apply_pathway_reduction(0, 0, 0.50, eff)
    updated = model._get_current_total_load_rate(0, 0, 100.0)

    assert math.isclose(updated, 80.0, rel_tol=0.0, abs_tol=1e-12)


def test_apply_pathway_reduction_negative_efficiency_increases_load():
    model = _make_model()
    _set_current_pathways(model, surface=50.0, shallow=25.0, deep=25.0)

    eff = {
        "surface": -0.20,
        "shallow subsurface": -0.20,
        "deep subsurface": -0.20,
    }

    model._apply_pathway_reduction(0, 0, 1.0, eff)
    updated = model._get_current_total_load_rate(0, 0, 100.0)

    assert math.isclose(updated, 120.0, rel_tol=0.0, abs_tol=1e-12)


def test_apply_pathway_reduction_failure_adjusted_efficiency_scales_before_application():
    model = _make_model()
    _set_current_pathways(model, surface=50.0, shallow=25.0, deep=25.0)

    sampled_eff = 0.50
    failure_reduction = 0.25
    effective_eff = sampled_eff * failure_reduction

    eff = {
        "surface": effective_eff,
        "shallow subsurface": effective_eff,
        "deep subsurface": effective_eff,
    }

    model._apply_pathway_reduction(0, 0, 1.0, eff)
    updated = model._get_current_total_load_rate(0, 0, 100.0)

    assert math.isclose(effective_eff, 0.125, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(updated, 87.5, rel_tol=0.0, abs_tol=1e-12)


def test_get_pathway_load_rates_returns_tracked_live_pathways():
    model = _make_model()
    _set_current_pathways(model, surface=40.0, shallow=35.0, deep=25.0)

    by_path = model._get_pathway_load_rates(0, 0, 100.0)

    assert set(by_path.keys()) == {"surface", "shallow subsurface", "deep subsurface"}
    assert math.isclose(by_path["surface"], 40.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(by_path["shallow subsurface"], 35.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(by_path["deep subsurface"], 25.0, rel_tol=0.0, abs_tol=1e-12)


def test_get_current_total_load_rate_sums_current_tracked_pathways():
    model = _make_model()
    _set_current_pathways(model, surface=10.0, shallow=20.0, deep=30.0)

    total = model._get_current_total_load_rate(0, 0, 999.0)

    assert math.isclose(total, 60.0, rel_tol=0.0, abs_tol=1e-12)