from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    CFG_BMP_COST,
    COL_CPS,
    COL_OID,
    COL_UNIT,
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    DATA_BMP_COST,
    DATA_OUTLET_LOC,
    DATA_POLLUTANTS,
    XAXIS_COST,
    YAXIS_TOTAL,
)
from src.cost import _get_bmp_cost, _select_cost_rate_median
from src.plotting import _build_line_segments, make_summary_plots


class DummyLogger:
    def __init__(self) -> None:
        self.messages = []

    def log(self, *args, **kwargs):
        return None

    def verbose(self, message, *args, **kwargs):
        self.messages.append(("verbose", str(message)))

    def info(self, message, *args, **kwargs):
        self.messages.append(("info", str(message)))

    def warning(self, message, *args, **kwargs):
        self.messages.append(("warning", str(message)))


def _dummy_cost_model(cost_df: pd.DataFrame):
    captured = {}

    def sample_from_stats(stats, kind=None):
        captured["stats"] = dict(stats)
        if "value" in stats:
            return float(stats["value"])
        if "min" in stats and "max" in stats:
            return (float(stats["min"]) + float(stats["max"])) / 2.0
        raise AssertionError(f"Unexpected stats passed to sampler: {stats}")

    model = SimpleNamespace(
        logger=DummyLogger(),
        data={
            DATA_BMP_COST: cost_df,
            DATA_AVG_AREA_HA: 1.0,
            DATA_AVG_PERIM_M: 100.0,
        },
        cfg={},
        _sample_from_stats=sample_from_stats,
    )
    return model, captured


def test_get_bmp_cost_ignores_blank_value_and_uses_distribution_bounds() -> None:
    cost_df = pd.DataFrame(
        [
            {
                COL_CPS: 340,
                COL_UNIT: "usd/ha",
                "value": np.nan,
                "mean": np.nan,
                "sd": np.nan,
                "min": 91.0,
                "max": 185.0,
            }
        ]
    )
    model, captured = _dummy_cost_model(cost_df)

    cost = _get_bmp_cost(model, 340, quantity=1.0)

    assert captured["stats"] == {"min": 91.0, "max": 185.0}
    assert cost == pytest.approx(138.0)
    assert np.isfinite(cost)


def test_select_cost_rate_median_ignores_blank_value() -> None:
    row = pd.Series(
        {
            COL_CPS: 340,
            COL_UNIT: "usd/ha",
            "value": np.nan,
            "p50": np.nan,
            "mean": np.nan,
            "min": 100.0,
            "max": 200.0,
        }
    )
    model, _ = _dummy_cost_model(pd.DataFrame([row]))

    representative = _select_cost_rate_median(model, row, cps=340)

    assert representative == pytest.approx(150.0)


def test_build_line_segments_preserves_simulation_record_order() -> None:
    # Deliberately use decreasing x values. The old plotting code sorted by x,
    # which changed simulation order. The plot should follow record/step order.
    by_scenario = {1: [(10.0, 1.0), (5.0, 2.0), (12.0, 3.0)]}

    lines = _build_line_segments(
        by_scenario,
        pollutant="TN",
        oid="1",
        x_axis="cost",
        y_axis="total",
    )

    assert lines == [
        [(0.0, 0.0), (10.0, 1.0)],
        [(10.0, 1.0), (5.0, 2.0)],
        [(5.0, 2.0), (12.0, 3.0)],
    ]


def test_build_line_segments_rejects_nonfinite_trajectory_values() -> None:
    by_scenario = {7: [(100.0, 1.0), (np.nan, 2.0)]}

    with pytest.raises(
        ValueError,
        match=r"scenario=7.*pollutant=TN.*x_axis=cost.*x_value=nan",
    ):
        _build_line_segments(
            by_scenario,
            pollutant="TN",
            oid="1",
            x_axis="cost",
            y_axis="total",
        )


def test_make_summary_plots_writes_cost_plot_for_finite_records(tmp_path) -> None:
    logger = DummyLogger()
    cfg = {CFG_BMP_COST: "bmp_cost.csv"}
    data = {
        DATA_POLLUTANTS: ["TN"],
        DATA_OUTLET_LOC: pd.DataFrame({COL_OID: [1]}),
    }
    records = {
        ("TN", "1", XAXIS_COST, YAXIS_TOTAL): [
            (1, 100.0, 5.0),
            (1, 250.0, 9.0),
        ]
    }

    make_summary_plots(cfg, data, records, tmp_path, logger)

    plot_file = tmp_path / "plots" / "plot_TN_oid1_xcost_ytotal.jpg"
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0
