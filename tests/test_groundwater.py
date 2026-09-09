from __future__ import annotations

import logging
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from src.bmp import (
    _apply_pathway_reduction,
    _get_current_total_load_rate,
    _get_pathway_load_rates,
    _simulate_infield,
)
from src.constants import OUTPUT_REMOVED, OUTPUT_TREATED
from src.plet_rusle import (
    INCH_OVER_HA_TO_LITERS,
    plet_annual_infiltration_in,
)


def _plet_parameters() -> dict[str, float]:
    return {
        "annual_precip_in": 10.0,
        "rain_days": 100.0,
        "rain_correction_fraction": 0.5,
        "runoff_day_fraction": 0.5,
        "cn": 70.0,
        "ia_ratio": 0.2,
        "infiltration_fraction": 0.2,
    }


def test_plet_infiltration_uses_rain_corrected_precipitation() -> None:
    parameters = _plet_parameters() | {"groundwater_multiplier": 1.25}
    assert plet_annual_infiltration_in(parameters) == pytest.approx(
        10.0 * 0.5 * 0.2 * 1.25
    )



def test_infield_bmp_does_not_treat_or_reduce_protected_groundwater() -> None:
    model = SimpleNamespace(
        logger=logging.getLogger("test-groundwater"),
        pollutants=["TN"],
        parcel_area_ha=np.array([2.0]),
        current_pathway_load_rates=np.array([[[10.0, 0.0, 0.0]]]),
        current_untreated_groundwater_load_rates=np.array([[4.0]]),
    )
    model._get_pathway_load_rates = MethodType(_get_pathway_load_rates, model)
    model._get_current_total_load_rate = MethodType(_get_current_total_load_rate, model)
    model._apply_pathway_reduction = MethodType(_apply_pathway_reduction, model)

    load_rates = np.array([[14.0]])
    outputs = {
        OUTPUT_TREATED: np.zeros(1, dtype=float),
        OUTPUT_REMOVED: np.zeros(1, dtype=float),
    }
    efficiency = [
        {
            "surface": 0.5,
            "shallow subsurface": 0.5,
            "deep subsurface": 0.5,
        }
    ]

    _simulate_infield(model, 0, efficiency, load_rates, {}, outputs)

    assert model.current_pathway_load_rates[0, 0] == pytest.approx([5.0, 0.0, 0.0])
    assert model.current_untreated_groundwater_load_rates[0, 0] == pytest.approx(4.0)
    assert load_rates[0, 0] == pytest.approx(9.0)
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(20.0)
    assert outputs[OUTPUT_REMOVED][0] == pytest.approx(10.0)
