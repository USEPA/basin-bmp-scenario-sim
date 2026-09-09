"""Regression tests for cost-dependent BMP selection validation."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    COL_CPS,
    COL_PROBABILITY,
    COL_UNIT,
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    DATA_BMP_COST,
    DATA_CPS,
)
from src.cost import _estimate_costs_for_probabilities, _select_cost_rate_median


class DummyLogger:
    """Minimal logger implementing the interface used by cost helpers."""

    def log(self, *args, **kwargs):
        return None

    def verbose(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None


def _model(cost_df: pd.DataFrame | None, cps: list[int]) -> SimpleNamespace:
    model = SimpleNamespace(
        logger=DummyLogger(),
        data={
            DATA_BMP_COST: cost_df,
            DATA_CPS: cps,
            DATA_AVG_AREA_HA: 2.0,
            DATA_AVG_PERIM_M: 100.0,
        },
    )
    model._select_cost_rate_median = lambda row, cps=None: _select_cost_rate_median(
        model, row, cps=cps
    )
    return model


def test_cost_based_selection_requires_cost_table() -> None:
    model = _model(None, [340, 590])

    with pytest.raises(
        ValueError,
        match="Cost-based BMP selection requires BMP cost data for every configured CPS",
    ):
        _estimate_costs_for_probabilities(model)


def test_cost_based_selection_rejects_missing_cps_cost_entry() -> None:
    costs = pd.DataFrame(
        [{COL_CPS: 340, COL_UNIT: "usd/project", "value": 100.0}]
    )
    model = _model(costs, [340, 590])

    with pytest.raises(
        ValueError,
        match=r"missing cost entries for cps=\[590\]",
    ):
        _estimate_costs_for_probabilities(model)


def test_cost_based_selection_rejects_cost_row_without_usable_value() -> None:
    costs = pd.DataFrame(
        [
            {COL_CPS: 340, COL_UNIT: "usd/project", "value": 100.0},
            {
                COL_CPS: 590,
                COL_UNIT: "usd/project",
                "value": np.nan,
                "mean": np.nan,
                "min": np.nan,
                "max": np.nan,
            },
        ]
    )
    model = _model(costs, [340, 590])

    with pytest.raises(ValueError, match="Could not determine finite cost rate for cps=590"):
        _estimate_costs_for_probabilities(model)


def test_cost_based_selection_allows_blank_value_when_distribution_is_defined() -> None:
    costs = pd.DataFrame(
        [
            {
                COL_CPS: 340,
                COL_UNIT: "usd/project",
                "value": np.nan,
                "min": 100.0,
                "max": 200.0,
            },
            {COL_CPS: 590, COL_UNIT: "usd/project", "value": 300.0},
        ]
    )
    model = _model(costs, [340, 590])

    probabilities = _estimate_costs_for_probabilities(model)

    assert probabilities[COL_CPS].tolist() == [340, 590]
    assert probabilities[COL_PROBABILITY].sum() == pytest.approx(1.0)
    # Representative costs are 150 and 300, so inverse-cost weighting is 2:1.
    assert probabilities.loc[
        probabilities[COL_CPS] == 340, COL_PROBABILITY
    ].iloc[0] == pytest.approx(2.0 / 3.0)


def test_cost_based_selection_returns_finite_probabilities_with_complete_costs() -> None:
    costs = pd.DataFrame(
        [
            {COL_CPS: 340, COL_UNIT: "usd/ha", "value": 100.0},
            {COL_CPS: 590, COL_UNIT: "usd/project", "value": 500.0},
        ]
    )
    model = _model(costs, [340, 590])

    probabilities = _estimate_costs_for_probabilities(model)

    assert probabilities[COL_CPS].tolist() == [340, 590]
    assert np.all(np.isfinite(probabilities[COL_PROBABILITY]))
    assert probabilities[COL_PROBABILITY].sum() == pytest.approx(1.0)
