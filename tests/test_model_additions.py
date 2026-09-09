from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.constants import OUTPUT_IMPACTED_PIDS
from src.model import (
    _add_mass_metrics_to_bmp_record,
    _baseline_mass_before_bmp_kg,
    _bmp_impacted_parcel_indices,
)


def test_bmp_impacted_parcel_indices_defaults_to_focal_parcel_when_field_missing() -> None:
    ctx = SimpleNamespace(
        parcel_ids=["A", "B", "C"],
        pid_to_index={"A": 0, "B": 1, "C": 2},
    )

    impacted = _bmp_impacted_parcel_indices(ctx, 1, {})

    assert impacted == [1]


def test_bmp_impacted_parcel_indices_defaults_to_focal_parcel_when_field_blank() -> None:
    ctx = SimpleNamespace(
        parcel_ids=["A", "B", "C"],
        pid_to_index={"A": 0, "B": 1, "C": 2},
    )

    impacted = _bmp_impacted_parcel_indices(ctx, 2, {OUTPUT_IMPACTED_PIDS: ""})

    assert impacted == [2]


def test_bmp_impacted_parcel_indices_preserves_list_order_from_record() -> None:
    ctx = SimpleNamespace(
        parcel_ids=["A", "B", "C"],
        pid_to_index={"A": 0, "B": 1, "C": 2},
    )

    impacted = _bmp_impacted_parcel_indices(ctx, 2, {OUTPUT_IMPACTED_PIDS: "C,A,B"})

    assert impacted == [2, 0, 1]


def test_baseline_mass_before_bmp_defaults_to_focal_parcel_only_when_no_impacted_list() -> None:
    ctx = SimpleNamespace(
        parcel_ids=["A", "B"],
        pid_to_index={"A": 0, "B": 1},
        parcel_area_ha=[2.0, 5.0],
    )
    pre_bmp_load_rates = np.asarray(
        [
            [10.0, 1.0],
            [20.0, 2.0],
        ],
        dtype=float,
    )

    mass = _baseline_mass_before_bmp_kg(ctx, pre_bmp_load_rates, 1, {})

    assert mass[0] == pytest.approx(20.0 * 5.0)
    assert mass[1] == pytest.approx(2.0 * 5.0)


def test_add_mass_metrics_to_bmp_record_handles_zero_treated_mass_without_division_error() -> None:
    rec = {}
    _add_mass_metrics_to_bmp_record(
        rec,
        ["TN"],
        np.asarray([100.0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
    )

    assert rec["baseline_mass_TN_kg"] == pytest.approx(100.0)
    assert rec["treated_baseline_mass_TN_kg"] == pytest.approx(0.0)
    assert rec["removed_mass_TN_kg"] == pytest.approx(0.0)
    assert rec["treatment_exposure_fraction_TN"] == pytest.approx(0.0)
    assert rec["overall_reduction_fraction_TN"] == pytest.approx(0.0)
    assert rec["realized_efficiency_TN"] is None
    assert rec["mass_timestep_years"] == pytest.approx(1.0)


def test_add_mass_metrics_to_bmp_record_handles_zero_baseline_mass_without_division_error() -> None:
    rec = {}
    _add_mass_metrics_to_bmp_record(
        rec,
        ["TN"],
        np.asarray([0.0]),
        np.asarray([0.0]),
        np.asarray([0.0]),
    )

    assert rec["baseline_mass_TN_kg"] == pytest.approx(0.0)
    assert rec["treated_baseline_mass_TN_kg"] == pytest.approx(0.0)
    assert rec["removed_mass_TN_kg"] == pytest.approx(0.0)
    assert rec["treatment_exposure_fraction_TN"] is None
    assert rec["realized_efficiency_TN"] is None
    assert rec["overall_reduction_fraction_TN"] is None
    assert rec["mass_timestep_years"] == pytest.approx(1.0)


def test_add_mass_metrics_to_bmp_record_supports_multiple_pollutants() -> None:
    rec = {}
    _add_mass_metrics_to_bmp_record(
        rec,
        ["TN", "TP"],
        np.asarray([100.0, 50.0]),
        np.asarray([60.0, 25.0]),
        np.asarray([18.0, 5.0]),
    )

    assert rec["baseline_mass_TN_kg"] == pytest.approx(100.0)
    assert rec["treated_baseline_mass_TN_kg"] == pytest.approx(60.0)
    assert rec["removed_mass_TN_kg"] == pytest.approx(18.0)
    assert rec["treatment_exposure_fraction_TN"] == pytest.approx(0.60)
    assert rec["realized_efficiency_TN"] == pytest.approx(0.30)
    assert rec["overall_reduction_fraction_TN"] == pytest.approx(0.18)

    assert rec["baseline_mass_TP_kg"] == pytest.approx(50.0)
    assert rec["treated_baseline_mass_TP_kg"] == pytest.approx(25.0)
    assert rec["removed_mass_TP_kg"] == pytest.approx(5.0)
    assert rec["treatment_exposure_fraction_TP"] == pytest.approx(0.50)
    assert rec["realized_efficiency_TP"] == pytest.approx(0.20)
    assert rec["overall_reduction_fraction_TP"] == pytest.approx(0.10)


def test_add_mass_metrics_to_bmp_record_preserves_signed_removed_mass_for_multiple_pollutants() -> None:
    rec = {}
    _add_mass_metrics_to_bmp_record(
        rec,
        ["TN", "TP"],
        np.asarray([40.0, 50.0]),
        np.asarray([20.0, 25.0]),
        np.asarray([-4.0, 5.0]),
    )

    assert rec["realized_efficiency_TN"] == pytest.approx(-0.20)
    assert rec["overall_reduction_fraction_TN"] == pytest.approx(-0.10)
    assert rec["realized_efficiency_TP"] == pytest.approx(0.20)
    assert rec["overall_reduction_fraction_TP"] == pytest.approx(0.10)