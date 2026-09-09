"""Regression tests for Element 24 unit enforcement and conversion."""

from types import SimpleNamespace

import pytest

from src.input_distributions import stats_from_row
from src.input_units import canonical_cost_unit, convert_cost_value
from src.sampling import _sample_from_stats


def test_plet_precipitation_mm_per_year_converts_to_inches_per_year() -> None:
    stats = stats_from_row(
        {
            "pid": "*",
            "parameter": "annual_precip_in",
            "mean": 1016.0,
            "sd": 254.0,
            "min": 508.0,
            "max": 1524.0,
            "units": "mm/year",
        }
    )
    assert stats == pytest.approx({"mean": 40.0, "sd": 10.0, "min": 20.0, "max": 60.0})


def test_concentration_micrograms_per_liter_converts_to_mg_per_liter() -> None:
    stats = stats_from_row(
        {"pid": "*", "pollutant": "TN", "value": 2500.0, "units": "ug/L"},
        {"pid", "pollutant", "sample_group", "distribution_id", "units"},
    )
    assert stats["value"] == pytest.approx(2.5)


def test_concentration_rejects_load_rate_units() -> None:
    with pytest.raises(ValueError, match="dimensionally incompatible"):
        stats_from_row(
            {"pid": "*", "pollutant": "TN", "value": 2.0, "units": "lb/ac/yr"},
            {"pid", "pollutant", "sample_group", "distribution_id", "units"},
        )


def test_sediment_fraction_converts_to_percent() -> None:
    stats = stats_from_row(
        {"pid": "*", "parameter": "sediment_n_pct", "value": 0.0012, "units": "fraction"}
    )
    assert stats["value"] == pytest.approx(0.12)


def test_bmp_efficiency_percent_converts_to_fraction() -> None:
    stats = stats_from_row(
        {"cps": 340, "pollutant": "TN", "value": 35.0, "units": "%"}
    )
    assert stats["value"] == pytest.approx(0.35)


def test_runtime_load_rate_sampling_converts_lb_per_acre_year() -> None:
    dummy = SimpleNamespace()
    sampled = _sample_from_stats(
        dummy,
        {"value": 10.0, "units": "lb/ac/yr"},
        kind="load_rate",
    )
    expected = 10.0 * 0.45359237 / 0.40468564224
    assert sampled == pytest.approx(expected)


def test_runtime_efficiency_sampling_converts_percent() -> None:
    dummy = SimpleNamespace()
    assert _sample_from_stats(dummy, {"value": -20.0, "units": "%"}, kind="efficiency") == pytest.approx(-0.2)


def test_runtime_load_rate_rejects_concentration_units() -> None:
    dummy = SimpleNamespace()
    with pytest.raises(ValueError, match="dimensionally incompatible"):
        _sample_from_stats(dummy, {"value": 2.0, "units": "mg/L"}, kind="load_rate")


def test_distribution_reference_cannot_change_catalog_scale() -> None:
    # Catalog validation/extraction registers the distribution's numeric scale.
    stats_from_row(
        {"distribution_id": "rain", "mean": 1000.0, "sd": 100.0, "units": "mm/year"}
    )
    with pytest.raises(ValueError, match="may not reinterpret"):
        stats_from_row({"distribution_id": "rain", "units": "in/year"})


def test_distribution_reference_allows_same_scale_alias() -> None:
    stats_from_row(
        {"distribution_id": "rain_alias", "mean": 1000.0, "sd": 100.0, "units": "mm/year"}
    )
    assert stats_from_row({"distribution_id": "rain_alias", "units": "mm/yr"}) == {}


def test_cost_area_units_convert_to_usd_per_hectare() -> None:
    canonical, factor = canonical_cost_unit("$/acre")
    assert canonical == "usd/ha"
    assert factor == pytest.approx(1.0 / 0.40468564224)
    assert convert_cost_value(100.0, "$/acre") == pytest.approx(247.1053814671653)


def test_cost_length_units_convert_to_usd_per_meter() -> None:
    canonical, factor = canonical_cost_unit("USD/ft")
    assert canonical == "usd/m"
    assert factor == pytest.approx(1.0 / 0.3048)


def test_unknown_cost_unit_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported or dimensionally incompatible"):
        canonical_cost_unit("usd/acre-year")


def test_rusle_r_does_not_guess_convert_other_unit_systems() -> None:
    with pytest.raises(ValueError, match="dimensionally incompatible"):
        stats_from_row(
            {
                "pid": "*",
                "parameter": "r",
                "value": 180.0,
                "units": "MJ mm ha-1 h-1 yr-1",
            }
        )


def test_rusle_r_accepts_explicit_canonical_convention() -> None:
    stats = stats_from_row(
        {"pid": "*", "parameter": "r", "value": 180.0, "units": "rusle-r-us-customary"}
    )
    assert stats["value"] == pytest.approx(180.0)


def test_rusle_r_accepts_current_example_index_convention() -> None:
    stats = stats_from_row(
        {"pid": "*", "parameter": "r", "value": 175.0, "units": "index"}
    )
    assert stats["value"] == pytest.approx(175.0)


def test_rusle_k_accepts_current_plet_customary_label() -> None:
    stats = stats_from_row(
        {
            "pid": "*",
            "parameter": "k",
            "value": 0.32,
            "units": "ton acre hour/(acre foot ton inch)",
        }
    )
    assert stats["value"] == pytest.approx(0.32)
