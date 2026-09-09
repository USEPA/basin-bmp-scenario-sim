"""Deterministic reference tests for core PLET/RUSLE equations."""

from __future__ import annotations

import pytest

from src.constants import TON_PER_ACRE_TO_KG_PER_HA
from src.plet_rusle import (
    plet_annual_infiltration_in,
    plet_annual_surface_runoff_in,
    plet_runoff_depth_in,
    rusle_sediment_load_rate_kg_ha_yr,
)


def test_plet_runoff_matches_hand_calculation() -> None:
    """Verify representative-event rainfall, storage, abstraction, and runoff."""
    annual_precip_in = 40.0
    rain_days = 100.0
    rain_correction_fraction = 0.80
    runoff_day_fraction = 0.25
    cn = 80.0
    ia_ratio = 0.20

    runoff_days = rain_days * runoff_day_fraction
    expected_event_rainfall = annual_precip_in * rain_correction_fraction / runoff_days
    retention_in = (1000.0 / cn) - 10.0
    initial_abstraction_in = ia_ratio * retention_in
    expected_event_runoff = (
        (expected_event_rainfall - initial_abstraction_in) ** 2
        / (expected_event_rainfall - initial_abstraction_in + retention_in)
    )
    expected_annual_runoff = expected_event_runoff * runoff_days

    event_rainfall, event_runoff, annual_runoff = plet_runoff_depth_in(
        annual_precip_in,
        rain_days,
        rain_correction_fraction,
        runoff_day_fraction,
        cn,
        ia_ratio,
    )

    assert retention_in == pytest.approx(2.5)
    assert initial_abstraction_in == pytest.approx(0.5)
    assert event_rainfall == pytest.approx(1.28)
    assert event_rainfall == pytest.approx(expected_event_rainfall)
    assert event_runoff == pytest.approx(0.18548780487804878)
    assert event_runoff == pytest.approx(expected_event_runoff)
    assert annual_runoff == pytest.approx(4.637195121951219)
    assert annual_runoff == pytest.approx(expected_annual_runoff)


def test_surface_runoff_multiplier_applies_after_cn_runoff() -> None:
    """Verify the runoff multiplier scales annual runoff but not event runoff."""
    params = {
        "annual_precip_in": 40.0,
        "rain_days": 100.0,
        "rain_correction_fraction": 0.80,
        "runoff_day_fraction": 0.25,
        "cn": 80.0,
        "ia_ratio": 0.20,
        "runoff_multiplier": 1.5,
    }

    event_rainfall, event_runoff, annual_storm_runoff, annual_total_runoff = (
        plet_annual_surface_runoff_in(params)
    )

    assert event_rainfall == pytest.approx(1.28)
    assert event_runoff == pytest.approx(0.18548780487804878)
    assert annual_storm_runoff == pytest.approx(4.637195121951219)
    assert annual_total_runoff == pytest.approx(6.955792682926829)


def test_annual_infiltration_includes_rain_correction_fraction() -> None:
    """Verify AR * Rcor * infiltration fraction * multiplier."""
    params = {
        "annual_precip_in": 40.0,
        "rain_correction_fraction": 0.80,
        "infiltration_fraction": 0.30,
        "groundwater_multiplier": 1.25,
    }

    expected = 40.0 * 0.80 * 0.30 * 1.25
    assert plet_annual_infiltration_in(params) == pytest.approx(expected)
    assert expected == pytest.approx(12.0)


@pytest.mark.parametrize("sdr", [0.0, 0.4, 1.0])
def test_rusle_sediment_load_rate_matches_hand_calculation(sdr: float) -> None:
    """Verify R*K*LS*C*P, SDR, unit conversion, and delivery multipliers."""
    params = {
        "r": 100.0,
        "k": 0.20,
        "ls": 1.5,
        "c": 0.10,
        "p": 0.50,
        "sdr": sdr,
        "sediment_multiplier": 1.20,
        "sediment_delivery_multiplier": 0.80,
    }

    gross_ton_ac_yr = 100.0 * 0.20 * 1.5 * 0.10 * 0.50
    expected = (
        gross_ton_ac_yr
        * sdr
        * TON_PER_ACRE_TO_KG_PER_HA
        * 1.20
        * 0.80
    )

    assert gross_ton_ac_yr == pytest.approx(1.5)
    assert rusle_sediment_load_rate_kg_ha_yr(params) == pytest.approx(expected)
