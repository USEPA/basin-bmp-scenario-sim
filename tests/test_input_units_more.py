from __future__ import annotations

import pytest

from src.input_units import canonical_cost_unit, convert_cost_value


def test_cost_project_unit_aliases_share_canonical_form() -> None:
    canonical1, factor1 = canonical_cost_unit("usd/project")
    canonical2, factor2 = canonical_cost_unit("USD per project")
    canonical3, factor3 = canonical_cost_unit("$/project")

    assert canonical1 == "usd/project"
    assert canonical2 == "usd/project"
    assert canonical3 == "usd/project"
    assert factor1 == pytest.approx(1.0)
    assert factor2 == pytest.approx(1.0)
    assert factor3 == pytest.approx(1.0)


def test_cost_area_unit_aliases_share_canonical_form() -> None:
    canonical1, factor1 = canonical_cost_unit("usd/ha")
    canonical2, factor2 = canonical_cost_unit("USD per ha")
    canonical3, factor3 = canonical_cost_unit("$/hectare")

    assert canonical1 == "usd/ha"
    assert canonical2 == "usd/ha"
    assert canonical3 == "usd/ha"
    assert factor1 == pytest.approx(1.0)
    assert factor2 == pytest.approx(1.0)
    assert factor3 == pytest.approx(1.0)


def test_cost_length_unit_aliases_share_canonical_form() -> None:
    canonical1, factor1 = canonical_cost_unit("usd/m")
    canonical2, factor2 = canonical_cost_unit("USD per meter")
    canonical3, factor3 = canonical_cost_unit("$/m")

    assert canonical1 == "usd/m"
    assert canonical2 == "usd/m"
    assert canonical3 == "usd/m"
    assert factor1 == pytest.approx(1.0)
    assert factor2 == pytest.approx(1.0)
    assert factor3 == pytest.approx(1.0)


def test_convert_cost_value_is_identity_for_canonical_project_units() -> None:
    assert convert_cost_value(750.0, "usd/project") == pytest.approx(750.0)
    assert convert_cost_value(750.0, "USD per project") == pytest.approx(750.0)


def test_convert_cost_value_is_identity_for_canonical_area_units() -> None:
    assert convert_cost_value(125.0, "usd/ha") == pytest.approx(125.0)
    assert convert_cost_value(125.0, "USD per ha") == pytest.approx(125.0)


def test_convert_cost_value_is_identity_for_canonical_length_units() -> None:
    assert convert_cost_value(30.0, "usd/m") == pytest.approx(30.0)
    assert convert_cost_value(30.0, "USD per meter") == pytest.approx(30.0)


def test_cost_unit_parsing_is_case_and_spacing_tolerant() -> None:
    canonical_a, factor_a = canonical_cost_unit("  USD / ACRE ")
    canonical_b, factor_b = canonical_cost_unit("usd per acre")

    assert canonical_a == "usd/ha"
    assert canonical_b == "usd/ha"
    assert factor_a == pytest.approx(factor_b)


def test_convert_cost_value_for_usd_per_foot_matches_expected_meter_conversion() -> None:
    expected = 100.0 / 0.3048
    assert convert_cost_value(100.0, "usd/ft") == pytest.approx(expected)
    assert convert_cost_value(100.0, "USD per foot") == pytest.approx(expected)


def test_convert_cost_value_for_usd_per_acre_matches_expected_hectare_conversion() -> None:
    expected = 250.0 / 0.40468564224
    assert convert_cost_value(250.0, "usd/acre") == pytest.approx(expected)
    assert convert_cost_value(250.0, "USD per acre") == pytest.approx(expected)


def test_unknown_cost_project_time_unit_still_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported or dimensionally incompatible"):
        canonical_cost_unit("usd/project/year")


def test_unknown_cost_length_time_unit_still_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported or dimensionally incompatible"):
        canonical_cost_unit("usd/m/year")