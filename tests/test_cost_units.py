import math
import pandas as pd

from src.cost import _get_bmp_cost, _estimate_costs_for_probabilities
from src.constants import (
    COL_CPS,
    COL_PROBABILITY,
    COL_UNIT,
    CFG_BUFFER_DEPTH_FT,
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    DATA_BMP_COST,
    DATA_CPS,
)


class DummyLogger:
    def log(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def verbose(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class DummyModel:
    def __init__(self, bmp_cost_df, *, avg_area_ha=2.0, avg_perim_m=100.0, buffer_depth_ft=35.0):
        self.logger = DummyLogger()
        self.cfg = {
            CFG_BUFFER_DEPTH_FT: buffer_depth_ft,
        }
        self.data = {
            DATA_BMP_COST: bmp_cost_df,
            DATA_AVG_AREA_HA: float(avg_area_ha),
            DATA_AVG_PERIM_M: float(avg_perim_m),
            DATA_CPS: sorted(pd.to_numeric(bmp_cost_df[COL_CPS]).astype(int).unique().tolist()),
        }

    def _sample_from_stats(self, stats, kind=None):
        if "value" in stats:
            return float(stats["value"])
        if "p50" in stats:
            return float(stats["p50"])
        if "mean" in stats:
            return float(stats["mean"])
        raise AssertionError(f"Unexpected stats for deterministic test: {stats}")

    def _select_cost_rate_median(self, row, cps=None):
        from src.cost import _select_cost_rate_median
        return _select_cost_rate_median(self, row, cps=cps)


def test_get_bmp_cost_uses_canonical_area_unit_without_conversion_error():
    df = pd.DataFrame(
        [
            {
                COL_CPS: 340,
                COL_UNIT: "usd/ha",
                "value": 100.0,
            }
        ]
    )
    model = DummyModel(df)

    total = _get_bmp_cost(model, cps=340, quantity=2.0)

    assert math.isclose(total, 200.0, rel_tol=0.0, abs_tol=1e-9)


def test_get_bmp_cost_uses_canonical_project_unit_without_area_scaling():
    df = pd.DataFrame(
        [
            {
                COL_CPS: 590,
                COL_UNIT: "usd/project",
                "value": 750.0,
            }
        ]
    )
    model = DummyModel(df)

    total = _get_bmp_cost(model, cps=590, quantity=5.0)

    assert math.isclose(total, 750.0, rel_tol=0.0, abs_tol=1e-9)


def test_get_bmp_cost_uses_buffer_length_conversion_for_usd_per_m():
    df = pd.DataFrame(
        [
            {
                COL_CPS: 412,
                COL_UNIT: "usd/m",
                "value": 10.0,
            }
        ]
    )
    model = DummyModel(df, buffer_depth_ft=35.0)

    total = _get_bmp_cost(model, cps=412, quantity=1.0)

    depth_m = 35.0 * 0.3048
    expected_length_m = 10000.0 / depth_m
    expected_total = 10.0 * expected_length_m

    assert math.isclose(total, expected_total, rel_tol=0.0, abs_tol=1e-9)


def test_get_bmp_cost_selection_fallback_uses_average_area_for_area_units_when_quantity_nonpositive():
    df = pd.DataFrame(
        [
            {
                COL_CPS: 340,
                COL_UNIT: "usd/ha",
                "value": 100.0,
            }
        ]
    )
    model = DummyModel(df, avg_area_ha=3.5)

    total = _get_bmp_cost(model, cps=340, quantity=0.0)

    assert math.isclose(total, 350.0, rel_tol=0.0, abs_tol=1e-9)


def test_estimate_costs_for_probabilities_prefers_lower_total_cost():
    df = pd.DataFrame(
        [
            {
                COL_CPS: 340,
                COL_UNIT: "usd/ha",
                "value": 100.0,
            },
            {
                COL_CPS: 590,
                COL_UNIT: "usd/project",
                "value": 500.0,
            },
        ]
    )
    model = DummyModel(df, avg_area_ha=2.0)

    probs = _estimate_costs_for_probabilities(model)

    prob_by_cps = {
        int(row[COL_CPS]): float(row[COL_PROBABILITY])
        for _, row in probs.iterrows()
    }

    assert prob_by_cps[340] > prob_by_cps[590]
    assert math.isclose(sum(prob_by_cps.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_estimate_costs_for_probabilities_uses_representative_total_cost_for_area_units():
    df = pd.DataFrame(
        [
            {
                COL_CPS: 340,
                COL_UNIT: "usd/ha",
                "value": 100.0,
            },
            {
                COL_CPS: 329,
                COL_UNIT: "usd/ha",
                "value": 200.0,
            },
        ]
    )
    model = DummyModel(df, avg_area_ha=2.0)

    probs = _estimate_costs_for_probabilities(model)

    prob_by_cps = {
        int(row[COL_CPS]): float(row[COL_PROBABILITY])
        for _, row in probs.iterrows()
    }

    assert prob_by_cps[340] > prob_by_cps[329]
    assert math.isclose(sum(prob_by_cps.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_cost_unit_alias_usd_per_ha_behaves_like_canonical_usd_per_ha():
    df = pd.DataFrame(
        [
            {
                COL_CPS: 340,
                COL_UNIT: "usd per ha",
                "value": 100.0,
            }
        ]
    )
    model = DummyModel(df)

    total = _get_bmp_cost(model, cps=340, quantity=2.0)

    assert math.isclose(total, 200.0, rel_tol=0.0, abs_tol=1e-9)


def test_cost_unit_alias_usd_per_project_behaves_like_canonical_usd_per_project():
    df = pd.DataFrame(
        [
            {
                COL_CPS: 590,
                COL_UNIT: "usd per project",
                "value": 750.0,
            }
        ]
    )
    model = DummyModel(df)

    total = _get_bmp_cost(model, cps=590, quantity=2.0)

    assert math.isclose(total, 750.0, rel_tol=0.0, abs_tol=1e-9)