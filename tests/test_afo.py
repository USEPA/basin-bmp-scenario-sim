"""AFO spatial allocation, PLET feedlot loading, and compartment BMP tests."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.afo import load_afo_inputs, apply_afo_baseline, _feedlot_runoff_kg
from src.bmp import _get_pathway_load_rates, _apply_pathway_reduction
from src.plet_rusle import plet_annual_surface_runoff_in


def _parcels():
    return gpd.GeoDataFrame(
        {"pid": [1, 2, 3, 4], "area_ha": [1., 1., 1., 1.]},
        geometry=[box(x*100, 0, (x+1)*100, 100) for x in range(4)],
        crs="EPSG:32617",
    )


def _package(path, allocation="explicit", selection="nearest", **overrides):
    op = dict(afo_id="A1", pid=1, feedlot_area_ha=0.2, paved_pct=30.,
              allocation=allocation, selection=selection, manure_months=6.,
              n_fields_value=2.)
    op.update(overrides)
    with sqlite3.connect(path) as conn:
        pd.DataFrame([op]).to_sql("afo_operations", conn, index=False)
        pd.DataFrame([{"afo_id": "A1", "species": "beef_steer", "count": 10,
                      "factor_tn": 1., "factor_tp": 1., "factor_bod": 1.}]).to_sql("afo_animals", conn, index=False)
        pd.DataFrame([{"afo_id": "A1", "pid": 1}, {"afo_id": "A1", "pid": 3}]
                     if allocation == "explicit" else [], columns=["afo_id", "pid"]).to_sql("afo_manure_fields", conn, index=False)
        pd.DataFrame([{"land_cover": "cropland", "pollutant": "TN", "value": 8.},
                      {"land_cover": "cropland", "pollutant": "TP", "value": .8}]).to_sql("afo_manure_concentration_increments", conn, index=False)
    return path


def _state():
    runoff_parameters = dict(land_cover="cropland", hsg="B", avg_rain_in=1.,
                             rain_days=100., runoff_day_fraction=0.3,
                             cn=80., infiltration_fraction=0.3,
                             ia_ratio=0., runoff_multiplier=1.,
                             load_multiplier_tn=1., load_multiplier_tp=1.)
    parameters = [dict(runoff_parameters) for _ in range(4)]
    concentrations = [dict(TN=2., TP=0.2) for _ in range(4)]
    return SimpleNamespace(
        parcel_ids=["1", "2", "3", "4"], parameters=parameters,
        concentrations=concentrations,
        pathway_load_rates=np.array([[[3., 2.], [1., 0.5]]] * 4),
        baseline_pathway_load_rates=None,
        baseline_concentrations=[],
    )


def _ctx(inputs, seed=4):
    return SimpleNamespace(
        afo_inputs=inputs, pollutants=["TN", "TP"], rng=np.random.default_rng(seed),
        _sample_from_stats=lambda stats, kind=None: float(stats.get("value", stats.get("mean", 0.))),
        pathway_names=["surface", "subsurface"],
    )


def test_feedlot_exact_plet_event_cn_and_density():
    parameters = _state().parameters[0]
    op = {"afo_id": "A1", "feedlot_area_ha": 0.2, "paved_pct": 30.}
    animals = [{"afo_id": "A1", "count": 10., "factor_tn": 1., "factor_tp": 1.}]
    tn = _feedlot_runoff_kg(op, animals, parameters, ["TN"])["TN"]
    expected_runoff = plet_annual_surface_runoff_in({**parameters, "cn": 92.})[3]
    pack = min(10 / (0.2 / 0.40468564224), 100) / 100
    expected = expected_runoff * 254000. * .2 * (pack * 1500) / 1e6
    assert tn == pytest.approx(expected)
    assert _feedlot_runoff_kg({**op, "paved_pct": 75.}, animals, parameters, ["TN"])["TN"] > tn


def test_explicit_allocation_applies_manure_and_feedlot_without_subsurface_increase(tmp_path):
    data = load_afo_inputs(_package(tmp_path / "afo.gpkg"), _parcels(), ["TN", "TP"])
    state = _state()
    context = _ctx(data)
    allocation = apply_afo_baseline(context, state)
    assert {(x["afo_id"], x["receiving_pid"]) for x in allocation} == {("A1", "1"), ("A1", "3")}
    assert state.concentrations[2]["TN"] == pytest.approx(6.)  # 2+(6/12)*8
    assert state.concentrations[1]["TN"] == 2.
    assert state.pathway_load_rates[2, 0, 0] > 3.
    assert state.pathway_load_rates[2, 0, 1] == pytest.approx(2.)
    assert state.pathway_load_rates[0, 0, 1] == pytest.approx(1.6)  # 20% feedlot replaces land
    assert state.pathway_load_rates[0, 0, 0] == pytest.approx(
        state.baseline_pathway_load_rates[0, 0, 0]
    )
    assert context.current_afo_feedlot_pathway_rates[0, 0, 0] > 0.
    assert context.current_afo_feedlot_pathway_rates[2, 0, 0] == 0.


def test_manure_increment_is_independent_of_background_concentration(tmp_path):
    """Positive manure effect even when the increment is below the background."""
    package = _package(tmp_path / "afo.gpkg")
    with sqlite3.connect(package) as conn:
        conn.execute(
            "UPDATE afo_manure_concentration_increments SET value=1 WHERE pollutant='TN'"
        )
    data = load_afo_inputs(package, _parcels(), ["TN", "TP"])
    state = _state()
    state.concentrations[2]["TN"] = 20.0  # greater than sampled increment
    apply_afo_baseline(_ctx(data), state)
    assert state.concentrations[2]["TN"] == pytest.approx(20.5)
    assert state.concentrations[0]["TN"] == pytest.approx(2.4)  # host 80% eligible
    assert state.pathway_load_rates[2, 0, 1] == pytest.approx(2.)  # no manure subsurface


def test_zero_manure_increment_and_reject_negative(tmp_path):
    package = _package(tmp_path / "afo.gpkg")
    with sqlite3.connect(package) as conn:
        conn.execute(
            "UPDATE afo_manure_concentration_increments SET value=0 WHERE pollutant='TN'"
        )
    data = load_afo_inputs(package, _parcels(), ["TN", "TP"])
    state = _state()
    apply_afo_baseline(_ctx(data), state)
    assert state.concentrations[2]["TN"] == pytest.approx(2.)
    with sqlite3.connect(package) as conn:
        conn.execute(
            "UPDATE afo_manure_concentration_increments SET value=-0.25 WHERE pollutant='TN'"
        )
    with pytest.raises(ValueError, match="afo_manure_concentration_increments"):
        load_afo_inputs(package, _parcels(), ["TN", "TP"])


def test_negative_sampled_manure_increment_rejected(tmp_path):
    data = load_afo_inputs(_package(tmp_path / "afo.gpkg"), _parcels(), ["TN", "TP"])
    context = _ctx(data)
    context._sample_from_stats = lambda stats, kind=None: -0.1
    with pytest.raises(ValueError, match="sampled manure concentration increment"):
        apply_afo_baseline(context, _state())


def test_old_absolute_manured_table_is_not_accepted(tmp_path):
    package = _package(tmp_path / "afo.gpkg")
    with sqlite3.connect(package) as conn:
        conn.execute(
            "ALTER TABLE afo_manure_concentration_increments "
            "RENAME TO afo_manured_concentrations"
        )
    with pytest.raises(ValueError, match="afo_manure_concentration_increments"):
        load_afo_inputs(package, _parcels(), ["TN", "TP"])


def test_sampled_nearest_and_random_assignment(tmp_path):
    data = load_afo_inputs(_package(tmp_path / "afo.gpkg", allocation="sampled"), _parcels(), ["TN", "TP"])
    assignments = apply_afo_baseline(_ctx(data), _state())
    assert [a["receiving_pid"] for a in assignments] == ["1", "2"]
    # The host is included and the random choice is reproducible for a seed.
    data["operations"][0]["selection"] = "random"
    a = apply_afo_baseline(_ctx(data, seed=8), _state())
    b = apply_afo_baseline(_ctx(data, seed=8), _state())
    assert a == b
    assert len(a) == 2 and a[0]["receiving_pid"] == "1"
    assert a[1]["receiving_pid"] in {"2", "3", "4"}


def test_invalid_host_duplicate_and_unsuitable_receiving_parcel(tmp_path):
    package = _package(tmp_path / "afo.gpkg")
    with sqlite3.connect(package) as conn:
        conn.execute("UPDATE afo_operations SET pid=99")
    with pytest.raises(ValueError, match="unknown host"):
        load_afo_inputs(package, _parcels(), ["TN", "TP"])
    package2 = _package(tmp_path / "afo2.gpkg")
    data = load_afo_inputs(package2, _parcels(), ["TN", "TP"])
    state = _state()
    state.parameters[2]["land_cover"] = "forest"
    with pytest.raises(ValueError, match="lacks eligible"):
        apply_afo_baseline(_ctx(data), state)


def test_feedlot_bmp_reduces_only_feedlot_and_field_bmp_preserves_it(tmp_path):
    data = load_afo_inputs(_package(tmp_path / "afo.gpkg"), _parcels(), ["TN", "TP"])
    state = _state()
    context = _ctx(data)
    apply_afo_baseline(context, state)
    context.current_pathway_load_rates = state.pathway_load_rates
    context.current_bmp_targets_feedlot = True
    feedlot_before = context.current_afo_feedlot_pathway_rates[0, 0, 0]
    field_before = context.current_pathway_load_rates[0, 0, 0] - feedlot_before
    assert _get_pathway_load_rates(context, 0, 0, 0)["surface"] == pytest.approx(feedlot_before)
    _apply_pathway_reduction(context, 0, 0, 1., {"surface": .5, "subsurface": 0.})
    assert context.current_afo_feedlot_pathway_rates[0, 0, 0] == pytest.approx(feedlot_before * .5)
    assert context.current_pathway_load_rates[0, 0, 0] == pytest.approx(field_before + feedlot_before*.5)
    context.current_bmp_targets_feedlot = False
    _apply_pathway_reduction(context, 0, 0, 1., {"surface": .5, "subsurface": 0.})
    assert context.current_afo_feedlot_pathway_rates[0, 0, 0] == pytest.approx(feedlot_before * .5)
    assert context.current_pathway_load_rates[0, 0, 0] == pytest.approx(field_before*.5 + feedlot_before*.5)
