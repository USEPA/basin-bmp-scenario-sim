from __future__ import annotations

import math
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from src.bmp import (
    _apply_pathway_reduction,
    _get_current_total_load_rate,
    _get_pathway_load_rates,
)


def _make_model() -> SimpleNamespace:
    model = SimpleNamespace(
        current_pathway_load_rates=np.array([[[0.0, 0.0, 0.0]]], dtype=float),
        pathway_names=["surface", "shallow subsurface", "deep subsurface"],
    )
    model._get_pathway_load_rates = MethodType(_get_pathway_load_rates, model)
    model._get_current_total_load_rate = MethodType(_get_current_total_load_rate, model)
    model._apply_pathway_reduction = MethodType(_apply_pathway_reduction, model)
    return model


def _set_current_pathways(
    model: SimpleNamespace,
    *,
    surface: float,
    shallow: float,
    deep: float,
) -> None:
    model.current_pathway_load_rates[0, 0, :] = np.array([surface, shallow, deep], dtype=float)


def test_apply_pathway_reduction_zero_treatment_fraction_leaves_loads_unchanged() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=10.0, shallow=20.0, deep=30.0)

    removed = model._apply_pathway_reduction(
        0,
        0,
        0.0,
        {
            "surface": 0.5,
            "shallow subsurface": 0.5,
            "deep subsurface": 0.5,
        },
    )

    np.testing.assert_allclose(model.current_pathway_load_rates[0, 0, :], np.array([10.0, 20.0, 30.0]))
    assert removed == pytest.approx(0.0)


def test_apply_pathway_reduction_zero_efficiencies_leave_loads_unchanged() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=12.0, shallow=8.0, deep=5.0)

    removed = model._apply_pathway_reduction(
        0,
        0,
        1.0,
        {
            "surface": 0.0,
            "shallow subsurface": 0.0,
            "deep subsurface": 0.0,
        },
    )

    np.testing.assert_allclose(model.current_pathway_load_rates[0, 0, :], np.array([12.0, 8.0, 5.0]))
    assert removed == pytest.approx(0.0)


def test_apply_pathway_reduction_full_treatment_fraction_applies_per_pathway_scaling() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=50.0, shallow=30.0, deep=20.0)

    removed = model._apply_pathway_reduction(
        0,
        0,
        1.0,
        {
            "surface": 0.20,
            "shallow subsurface": 0.50,
            "deep subsurface": 1.00,
        },
    )

    np.testing.assert_allclose(
        model.current_pathway_load_rates[0, 0, :],
        np.array([40.0, 15.0, 0.0]),
    )
    assert removed == pytest.approx(45.0)


def test_apply_pathway_reduction_partial_treatment_fraction_scales_removed_amount() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=40.0, shallow=20.0, deep=10.0)

    removed = model._apply_pathway_reduction(
        0,
        0,
        0.25,
        {
            "surface": 0.40,
            "shallow subsurface": 0.40,
            "deep subsurface": 0.40,
        },
    )

    np.testing.assert_allclose(
        model.current_pathway_load_rates[0, 0, :],
        np.array([36.0, 18.0, 9.0]),
    )
    assert removed == pytest.approx(7.0)


def test_apply_pathway_reduction_negative_efficiency_increases_only_targeted_pathway() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=10.0, shallow=5.0, deep=1.0)

    removed = model._apply_pathway_reduction(
        0,
        0,
        0.5,
        {
            "surface": -0.40,
            "shallow subsurface": 0.0,
            "deep subsurface": 0.0,
        },
    )

    np.testing.assert_allclose(
        model.current_pathway_load_rates[0, 0, :],
        np.array([12.0, 5.0, 1.0]),
    )
    assert removed == pytest.approx(-2.0)


def test_apply_pathway_reduction_missing_pathway_key_raises_keyerror() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=30.0, shallow=20.0, deep=10.0)

    with pytest.raises(KeyError, match="shallow subsurface"):
        model._apply_pathway_reduction(
            0,
            0,
            1.0,
            {
                "surface": 0.10,
                "deep subsurface": 0.50,
            },
        )


def test_apply_pathway_reduction_ignores_extra_efficiency_keys_not_in_active_pathways() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=10.0, shallow=10.0, deep=10.0)

    removed = model._apply_pathway_reduction(
        0,
        0,
        1.0,
        {
            "surface": 0.10,
            "shallow subsurface": 0.20,
            "deep subsurface": 0.30,
            "not-a-real-pathway": 1.00,
        },
    )

    np.testing.assert_allclose(
        model.current_pathway_load_rates[0, 0, :],
        np.array([9.0, 8.0, 7.0]),
    )
    assert removed == pytest.approx(6.0)


def test_apply_pathway_reduction_preserves_mass_balance_relative_to_total_load() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=25.0, shallow=15.0, deep=10.0)

    before = model._get_current_total_load_rate(0, 0, 999.0)
    removed = model._apply_pathway_reduction(
        0,
        0,
        0.5,
        {
            "surface": 0.20,
            "shallow subsurface": 0.40,
            "deep subsurface": 0.60,
        },
    )
    after = model._get_current_total_load_rate(0, 0, 999.0)

    assert before == pytest.approx(50.0)
    assert after == pytest.approx(before - removed)


def test_apply_pathway_reduction_with_adverse_effect_produces_negative_removed_mass_balance() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=20.0, shallow=0.0, deep=0.0)

    before = model._get_current_total_load_rate(0, 0, 999.0)
    removed = model._apply_pathway_reduction(
        0,
        0,
        1.0,
        {
            "surface": -0.25,
            "shallow subsurface": 0.0,
            "deep subsurface": 0.0,
        },
    )
    after = model._get_current_total_load_rate(0, 0, 999.0)

    assert removed == pytest.approx(-5.0)
    assert after == pytest.approx(25.0)
    assert after == pytest.approx(before - removed)


def test_get_pathway_load_rates_returns_live_pathway_mapping() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=4.0, shallow=5.5, deep=6.25)

    mapping = model._get_pathway_load_rates(0, 0, 999.0)

    assert mapping == {
        "surface": pytest.approx(4.0),
        "shallow subsurface": pytest.approx(5.5),
        "deep subsurface": pytest.approx(6.25),
    }


def test_get_current_total_load_rate_sums_tracked_pathways_not_fallback_total() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=1.25, shallow=2.50, deep=3.75)

    total = model._get_current_total_load_rate(0, 0, 999.0)

    assert math.isclose(total, 7.5, rel_tol=0.0, abs_tol=1e-12)


def test_apply_pathway_reduction_can_zero_out_all_pathways() -> None:
    model = _make_model()
    _set_current_pathways(model, surface=3.0, shallow=4.0, deep=5.0)

    removed = model._apply_pathway_reduction(
        0,
        0,
        1.0,
        {
            "surface": 1.0,
            "shallow subsurface": 1.0,
            "deep subsurface": 1.0,
        },
    )

    np.testing.assert_allclose(model.current_pathway_load_rates[0, 0, :], np.zeros(3))
    assert removed == pytest.approx(12.0)