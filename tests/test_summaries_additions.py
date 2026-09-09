from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.summaries import BMPSummaryCollector, _compute_statistics


def test_compute_statistics_returns_nan_summary_for_empty_input() -> None:
    stats = _compute_statistics(np.asarray([], dtype=float))

    assert stats["count"] == 0
    assert np.isnan(stats["mean"])
    assert np.isnan(stats["std"])
    assert np.isnan(stats["min"])
    assert np.isnan(stats["p25"])
    assert np.isnan(stats["p50"])
    assert np.isnan(stats["p75"])
    assert np.isnan(stats["max"])


def test_compute_statistics_ignores_nonfinite_values() -> None:
    stats = _compute_statistics(np.asarray([1.0, np.nan, 3.0, np.inf, 5.0], dtype=float))

    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["p50"] == pytest.approx(3.0)
    assert stats["max"] == pytest.approx(5.0)


def test_summary_dataframe_uses_nan_for_undefined_mass_ratios() -> None:
    collector = BMPSummaryCollector(["TN"], scenario_id=1)
    collector.add_bmp_record(
        {
            "cps": 340,
            "pid": "A",
            "failed": False,
            "baseline_mass_TN_kg": 0.0,
            "treated_baseline_mass_TN_kg": 0.0,
            "removed_mass_TN_kg": 0.0,
            "cost_usd": 5.0,
        }
    )

    summary = collector.generate_summary_dataframe().iloc[0]

    assert summary["baseline_mass_TN_kg_total"] == pytest.approx(0.0)
    assert summary["treated_baseline_mass_TN_kg_total"] == pytest.approx(0.0)
    assert summary["removed_mass_TN_kg_total"] == pytest.approx(0.0)
    assert np.isnan(summary["treatment_exposure_fraction_TN"])
    assert np.isnan(summary["realized_efficiency_TN"])
    assert np.isnan(summary["overall_reduction_fraction_TN"])


def test_summary_dataframe_counts_failures_with_failures_count_column() -> None:
    collector = BMPSummaryCollector(["TN"], scenario_id=2)
    collector.add_bmp_record(
        {
            "cps": 340,
            "pid": "A",
            "failed": False,
            "baseline_mass_TN_kg": 10.0,
            "treated_baseline_mass_TN_kg": 10.0,
            "removed_mass_TN_kg": 2.0,
            "cost_usd": 1.0,
        }
    )
    collector.add_bmp_record(
        {
            "cps": 340,
            "pid": "B",
            "failed": True,
            "baseline_mass_TN_kg": 20.0,
            "treated_baseline_mass_TN_kg": 20.0,
            "removed_mass_TN_kg": 1.0,
            "cost_usd": 1.0,
        }
    )

    summary = collector.generate_summary_dataframe().iloc[0]

    assert summary["bmp_count"] == pytest.approx(2.0)
    assert summary["failures_count"] == pytest.approx(1.0)


def test_summary_dataframe_does_not_emit_pid_rollup_columns() -> None:
    collector = BMPSummaryCollector(["TN"], scenario_id=3)
    collector.add_bmp_record(
        {
            "cps": 340,
            "pid": "A",
            "failed": False,
            "baseline_mass_TN_kg": 10.0,
            "treated_baseline_mass_TN_kg": 10.0,
            "removed_mass_TN_kg": 1.0,
            "cost_usd": 1.0,
        }
    )
    collector.add_bmp_record(
        {
            "cps": 340,
            "pid": "B",
            "failed": False,
            "baseline_mass_TN_kg": 20.0,
            "treated_baseline_mass_TN_kg": 10.0,
            "removed_mass_TN_kg": 2.0,
            "cost_usd": 1.0,
        }
    )

    summary = collector.generate_summary_dataframe().iloc[0]

    assert "pid_count" not in summary.index
    assert "pid_mean" not in summary.index
    assert summary["bmp_count"] == pytest.approx(2.0)


def test_generate_rollup_summary_uses_all_cps_label() -> None:
    collector = BMPSummaryCollector(["TN"], scenario_id=4)
    collector.add_bmp_record(
        {
            "cps": 340,
            "pid": "A",
            "failed": False,
            "baseline_mass_TN_kg": 10.0,
            "treated_baseline_mass_TN_kg": 5.0,
            "removed_mass_TN_kg": 2.0,
            "cost_usd": 1.0,
        }
    )
    collector.add_bmp_record(
        {
            "cps": 412,
            "pid": "B",
            "failed": False,
            "baseline_mass_TN_kg": 20.0,
            "treated_baseline_mass_TN_kg": 10.0,
            "removed_mass_TN_kg": 4.0,
            "cost_usd": 2.0,
        }
    )

    rollup = collector.generate_rollup_summary()

    assert rollup["scenario"] == 4
    assert rollup["cps"] == 0
    assert rollup["cps_name"] == "All CPS"
    assert rollup["bmp_count"] == pytest.approx(2.0)
    assert rollup["failures_count"] == pytest.approx(0.0)


def test_generate_summary_dataframe_is_empty_before_any_records() -> None:
    collector = BMPSummaryCollector(["TN"], scenario_id=5)

    summary_df = collector.generate_summary_dataframe()

    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.empty