from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.model import (
    _add_mass_metrics_to_bmp_record,
    _baseline_mass_before_bmp_kg,
)
from src.summaries import BMPSummaryCollector


def test_wetland_baseline_mass_includes_all_impacted_hydrologic_parcels() -> None:
    ctx = SimpleNamespace(
        parcel_ids=["A", "B", "C"],
        pid_to_index={"A": 0, "B": 1, "C": 2},
        parcel_area_ha=[2.0, 3.0, 5.0],
    )
    # Current pre-BMP areal load rates (kg/ha/yr) immediately before this annual BMP step.
    pre_bmp_load_rates = np.asarray(
        [
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 3.0],
        ],
        dtype=float,
    )
    bmp_rec = {"pid": "C", "impacted_pids": "C,A,B"}

    mass = _baseline_mass_before_bmp_kg(ctx, pre_bmp_load_rates, 2, bmp_rec)

    # Annual rate * ha * 1 year => kg over the modeled annual timestep.
    assert mass[0] == pytest.approx(10 * 2 + 20 * 3 + 30 * 5)
    assert mass[1] == pytest.approx(1 * 2 + 2 * 3 + 3 * 5)


def test_bmp_record_gets_explicit_mass_metrics() -> None:
    rec = {}
    _add_mass_metrics_to_bmp_record(
        rec,
        ["TN"],
        np.asarray([100.0]),
        np.asarray([60.0]),
        np.asarray([18.0]),
    )

    assert rec["baseline_mass_TN_kg"] == pytest.approx(100.0)
    assert rec["treated_baseline_mass_TN_kg"] == pytest.approx(60.0)
    assert rec["removed_mass_TN_kg"] == pytest.approx(18.0)
    assert rec["treatment_exposure_fraction_TN"] == pytest.approx(0.60)
    assert rec["realized_efficiency_TN"] == pytest.approx(0.30)
    assert rec["overall_reduction_fraction_TN"] == pytest.approx(0.18)
    assert rec["mass_timestep_years"] == pytest.approx(1.0)


def test_signed_removed_mass_produces_signed_efficiency() -> None:
    rec = {}
    _add_mass_metrics_to_bmp_record(
        rec,
        ["TP"],
        np.asarray([50.0]),
        np.asarray([25.0]),
        np.asarray([-5.0]),
    )
    assert rec["realized_efficiency_TP"] == pytest.approx(-0.20)
    assert rec["overall_reduction_fraction_TP"] == pytest.approx(-0.10)


def test_summary_ratios_are_mass_weighted_not_mean_of_per_bmp_efficiencies() -> None:
    collector = BMPSummaryCollector(["TN"], scenario_id=1)
    records = [
        {
            "cps": 340,
            "pid": "A",
            "failed": False,
            "baseline_mass_TN_kg": 10.0,
            "treated_baseline_mass_TN_kg": 5.0,
            "removed_mass_TN_kg": 4.0,  # 80% realized efficiency
            "cost_usd": 1.0,
        },
        {
            "cps": 340,
            "pid": "B",
            "failed": False,
            "baseline_mass_TN_kg": 100.0,
            "treated_baseline_mass_TN_kg": 100.0,
            "removed_mass_TN_kg": 20.0,  # 20% realized efficiency
            "cost_usd": 1.0,
        },
    ]
    for rec in records:
        collector.add_bmp_record(rec)

    summary = collector.generate_summary_dataframe().iloc[0]

    assert summary["baseline_mass_TN_kg_total"] == pytest.approx(110.0)
    assert summary["treated_baseline_mass_TN_kg_total"] == pytest.approx(105.0)
    assert summary["removed_mass_TN_kg_total"] == pytest.approx(24.0)
    assert summary["treatment_exposure_fraction_TN"] == pytest.approx(105.0 / 110.0)
    assert summary["realized_efficiency_TN"] == pytest.approx(24.0 / 105.0)
    assert summary["overall_reduction_fraction_TN"] == pytest.approx(24.0 / 110.0)

    # The old unweighted mean would be (0.8 + 0.2) / 2 = 0.5; do not report it.
    assert summary["realized_efficiency_TN"] != pytest.approx(0.5)
    assert not any(str(column).startswith("efficiency_TN_") for column in summary.index)


def test_rollup_uses_sum_then_ratio_across_bmp_types() -> None:
    collector = BMPSummaryCollector(["TN"], scenario_id=1)
    for cps, baseline, treated, removed in (
        (340, 10.0, 10.0, 5.0),
        (590, 90.0, 45.0, 9.0),
    ):
        collector.add_bmp_record(
            {
                "cps": cps,
                "pid": str(cps),
                "failed": False,
                "baseline_mass_TN_kg": baseline,
                "treated_baseline_mass_TN_kg": treated,
                "removed_mass_TN_kg": removed,
            }
        )

    rollup = collector.generate_rollup_summary()
    assert rollup["treatment_exposure_fraction_TN"] == pytest.approx(55.0 / 100.0)
    assert rollup["realized_efficiency_TN"] == pytest.approx(14.0 / 55.0)
    assert rollup["overall_reduction_fraction_TN"] == pytest.approx(14.0 / 100.0)
