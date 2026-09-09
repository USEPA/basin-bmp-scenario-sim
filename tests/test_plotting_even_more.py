from __future__ import annotations

import pandas as pd
import pytest

from src.constants import (
    CFG_BMP_COST,
    CFG_OUTLET_MEAN,
    CFG_OUTLET_TARGET,
    COL_MEAN,
    COL_OID,
    COL_POLLUTANT,
    COL_TARGET,
    DATA_OUTLET_LOC,
    DATA_OUTLET_MEAN,
    DATA_OUTLET_TARGET,
    DATA_POLLUTANTS,
    XAXIS_COST,
    YAXIS_MEAN,
    YAXIS_TARGET,
    YAXIS_TOTAL,
)
from src.plotting import _build_line_segments, make_summary_plots


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def log(self, *args, **kwargs):
        return None

    def verbose(self, message, *args, **kwargs):
        self.messages.append(("verbose", str(message)))

    def info(self, message, *args, **kwargs):
        self.messages.append(("info", str(message)))

    def warning(self, message, *args, **kwargs):
        self.messages.append(("warning", str(message)))


def test_build_line_segments_orders_scenarios_by_scenario_id() -> None:
    by_scenario = {
        5: [(1.0, 10.0)],
        2: [(2.0, 20.0)],
    }

    lines = _build_line_segments(
        by_scenario,
        pollutant="TN",
        oid="1",
        x_axis="cost",
        y_axis="total",
    )

    assert lines == [
        [(0.0, 0.0), (2.0, 20.0)],
        [(0.0, 0.0), (1.0, 10.0)],
    ]


def test_build_line_segments_single_scenario_multiple_points_preserves_internal_order() -> None:
    by_scenario = {
        3: [(4.0, 1.0), (2.0, 2.0), (8.0, 3.0)],
    }

    lines = _build_line_segments(
        by_scenario,
        pollutant="TN",
        oid="9",
        x_axis="cost",
        y_axis="total",
    )

    assert lines == [
        [(0.0, 0.0), (4.0, 1.0)],
        [(4.0, 1.0), (2.0, 2.0)],
        [(2.0, 2.0), (8.0, 3.0)],
    ]


def test_make_summary_plots_warns_once_for_nonpositive_target_denominator_across_multiple_scenarios(tmp_path) -> None:
    logger = DummyLogger()
    cfg = {
        CFG_BMP_COST: "bmp_cost.csv",
        CFG_OUTLET_TARGET: "outlet_target.csv",
    }
    data = {
        DATA_POLLUTANTS: ["TN"],
        DATA_OUTLET_LOC: pd.DataFrame({COL_OID: [1]}),
        DATA_OUTLET_TARGET: pd.DataFrame(
            [{COL_OID: "1", COL_POLLUTANT: "TN", COL_TARGET: -5.0}]
        ),
    }
    scenario_records = {
        ("TN", "1", XAXIS_COST, YAXIS_TOTAL): [
            (1, 100.0, 5.0),
            (2, 150.0, 4.0),
        ],
        ("TN", "1", XAXIS_COST, YAXIS_TARGET): [
            (1, 100.0, 0.5),
            (2, 150.0, 0.4),
        ],
    }

    make_summary_plots(cfg, data, scenario_records, tmp_path, logger)

    warnings = [msg for level, msg in logger.messages if level == "warning"]
    target_warnings = [
        msg for msg in warnings if "missing or nonpositive outlet target denominator" in msg
    ]
    assert len(target_warnings) == 1


def test_make_summary_plots_warns_once_for_nonpositive_mean_denominator_across_multiple_scenarios(tmp_path) -> None:
    logger = DummyLogger()
    cfg = {
        CFG_BMP_COST: "bmp_cost.csv",
        CFG_OUTLET_MEAN: "outlet_mean.csv",
    }
    data = {
        DATA_POLLUTANTS: ["TN"],
        DATA_OUTLET_LOC: pd.DataFrame({COL_OID: [1]}),
        DATA_OUTLET_MEAN: pd.DataFrame(
            [{COL_OID: "1", COL_POLLUTANT: "TN", COL_MEAN: 0.0}]
        ),
    }
    scenario_records = {
        ("TN", "1", XAXIS_COST, YAXIS_TOTAL): [
            (1, 100.0, 5.0),
            (2, 150.0, 4.0),
        ],
        ("TN", "1", XAXIS_COST, YAXIS_MEAN): [
            (1, 100.0, 0.5),
            (2, 150.0, 0.4),
        ],
    }

    make_summary_plots(cfg, data, scenario_records, tmp_path, logger)

    warnings = [msg for level, msg in logger.messages if level == "warning"]
    mean_warnings = [
        msg for msg in warnings if "missing or nonpositive outlet mean denominator" in msg
    ]
    assert len(mean_warnings) == 1


def test_make_summary_plots_with_only_total_axis_does_not_emit_denominator_warnings(tmp_path) -> None:
    logger = DummyLogger()
    cfg = {CFG_BMP_COST: "bmp_cost.csv"}
    data = {
        DATA_POLLUTANTS: ["TN"],
        DATA_OUTLET_LOC: pd.DataFrame({COL_OID: [1]}),
    }
    scenario_records = {
        ("TN", "1", XAXIS_COST, YAXIS_TOTAL): [
            (1, 100.0, 5.0),
            (1, 200.0, 4.0),
        ],
    }

    make_summary_plots(cfg, data, scenario_records, tmp_path, logger)

    warnings = [msg for level, msg in logger.messages if level == "warning"]
    assert not any("denominator" in msg for msg in warnings)