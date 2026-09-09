"""Deterministic mass-accounting tests for PLET/RUSLE pathway loads."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import INCH_OVER_HA_TO_LITERS
from src.plet_rusle import (
    calculate_plet_pathway_load_rates,
    plet_annual_infiltration_in,
    plet_annual_surface_runoff_in,
    rusle_sediment_load_rate_kg_ha_yr,
)


def _parameters() -> dict[str, float]:
    return {
        "annual_precip_in": 40.0,
        "rain_days": 100.0,
        "rain_correction_fraction": 0.80,
        "runoff_day_fraction": 0.25,
        "cn": 80.0,
        "ia_ratio": 0.20,
        "infiltration_fraction": 0.30,
        "runoff_multiplier": 1.0,
        "groundwater_multiplier": 1.0,
        "r": 100.0,
        "k": 0.20,
        "ls": 1.5,
        "c": 0.10,
        "p": 0.50,
        "sdr": 0.40,
        "sediment_multiplier": 1.0,
        "sediment_delivery_multiplier": 1.0,
        "sediment_n_pct": 1.0,
        "sediment_p_pct": 0.5,
        "enrichment_ratio": 2.0,
        "load_multiplier_tn": 1.0,
        "load_multiplier_tp": 1.0,
        "load_multiplier_tss": 1.0,
    }


def test_pathway_loads_match_independent_mass_calculation() -> None:
    """Verify dissolved, sediment-bound, and subsurface terms independently."""
    params = _parameters()
    runoff_conc = {"TN": 2.0, "TP": 0.2, "TSS": 999.0}
    groundwater_conc = {"TN": 3.0, "TP": 0.3}

    loads = calculate_plet_pathway_load_rates(
        params,
        runoff_conc,
        groundwater_conc,
        ["TN", "TP", "TSS"],
    )

    annual_runoff_in = plet_annual_surface_runoff_in(params)[3]
    runoff_l_ha = annual_runoff_in * INCH_OVER_HA_TO_LITERS
    infiltration_l_ha = (
        plet_annual_infiltration_in(params) * INCH_OVER_HA_TO_LITERS
    )
    sediment = rusle_sediment_load_rate_kg_ha_yr(params)

    expected_surface_tn = (
        2.0 * runoff_l_ha / 1_000_000.0
        + sediment * 0.01 * 2.0
    )
    expected_surface_tp = (
        0.2 * runoff_l_ha / 1_000_000.0
        + sediment * 0.005 * 2.0
    )
    expected_subsurface_tn = 3.0 * infiltration_l_ha / 1_000_000.0
    expected_subsurface_tp = 0.3 * infiltration_l_ha / 1_000_000.0

    assert loads.shape == (3, 2)
    assert loads[0, 0] == pytest.approx(expected_surface_tn)
    assert loads[0, 1] == pytest.approx(expected_subsurface_tn)
    assert loads[1, 0] == pytest.approx(expected_surface_tp)
    assert loads[1, 1] == pytest.approx(expected_subsurface_tp)

    # With complete RUSLE inputs, TSS is the RUSLE sediment load and the runoff
    # TSS concentration is intentionally not added a second time.
    assert loads[2, 0] == pytest.approx(sediment)
    assert loads[2, 1] == pytest.approx(0.0)

    expected_totals = np.asarray(
        [
            expected_surface_tn + expected_subsurface_tn,
            expected_surface_tp + expected_subsurface_tp,
            sediment,
        ]
    )
    assert np.sum(loads, axis=1) == pytest.approx(expected_totals)


def test_sediment_bound_nutrient_term_is_percent_times_enrichment() -> None:
    """Isolate the sediment-bound nutrient calculation from dissolved runoff."""
    params = _parameters()
    sediment = rusle_sediment_load_rate_kg_ha_yr(params)

    loads = calculate_plet_pathway_load_rates(
        params,
        {"TN": 0.0, "TP": 0.0},
        {"TN": 0.0, "TP": 0.0},
        ["TN", "TP"],
    )

    assert loads[0, 0] == pytest.approx(sediment * 0.01 * 2.0)
    assert loads[1, 0] == pytest.approx(sediment * 0.005 * 2.0)
    assert np.allclose(loads[:, 1], 0.0)


def test_doubling_runoff_concentration_changes_only_surface_dissolved_load() -> None:
    """Check that runoff concentration does not leak into subsurface load."""
    params = _parameters()
    groundwater = {"TN": 3.0}

    base = calculate_plet_pathway_load_rates(
        params, {"TN": 2.0}, groundwater, ["TN"]
    )
    doubled = calculate_plet_pathway_load_rates(
        params, {"TN": 4.0}, groundwater, ["TN"]
    )

    annual_runoff_in = plet_annual_surface_runoff_in(params)[3]
    dissolved_increment = 2.0 * annual_runoff_in * INCH_OVER_HA_TO_LITERS / 1_000_000.0
    assert doubled[0, 0] - base[0, 0] == pytest.approx(dissolved_increment)
    assert doubled[0, 1] == pytest.approx(base[0, 1])


def test_doubling_groundwater_concentration_doubles_only_subsurface_load() -> None:
    """Check that groundwater concentration affects only the subsurface pathway."""
    params = _parameters()

    base = calculate_plet_pathway_load_rates(
        params, {"TN": 2.0}, {"TN": 3.0}, ["TN"]
    )
    doubled = calculate_plet_pathway_load_rates(
        params, {"TN": 2.0}, {"TN": 6.0}, ["TN"]
    )

    assert doubled[0, 0] == pytest.approx(base[0, 0])
    assert doubled[0, 1] == pytest.approx(2.0 * base[0, 1])
