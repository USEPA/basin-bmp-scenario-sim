"""PR #20 review follow-up: climate, sediment caveat, schema, and identifiers."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.afo import (
    STAT_FEEDLOT_RAIN_DAYS, STAT_FEEDLOT_RAIN_CORRECTION,
    STAT_FEEDLOT_RUNOFF_DAY_FRACTION, _feedlot_runoff_kg,
    load_afo_inputs, mass_afo_realization,
)
from src.plet_rusle import plet_annual_surface_runoff_in
from test_afo import _ctx, _package, _parcels
from test_afo_mass import _mass_package, _load


def test_statistical_feedlot_uses_plet_cn_with_annual_precip(tmp_path):
    inputs = _load(_mass_package(tmp_path), mode="statistical")
    ctx = _ctx(inputs)
    ctx.parcel_ids = ["1", "2", "3", "4"]
    ctx.current_pathway_load_rates = np.zeros((4, 2, 2))
    mass_afo_realization(ctx)
    op = inputs["operations"][0]
    animals = [dict(a, count=a["count"]) for a in inputs["animals"]]
    expected = _feedlot_runoff_kg(
        op, animals,
        {"annual_precip_in": 1000 / 25.4,
         "rain_days": STAT_FEEDLOT_RAIN_DAYS,
         "rain_correction_fraction": STAT_FEEDLOT_RAIN_CORRECTION,
         "runoff_day_fraction": STAT_FEEDLOT_RUNOFF_DAY_FRACTION},
        ["TN", "TP"],
    )
    assert ctx.current_afo_feedlot_pathway_rates[0, 0, 0] == pytest.approx(
        expected["TN"] / inputs["area_ha"]["1"])
    # With zero initial abstraction, CN-based runoff cannot exceed annual P.
    for paved, cn in ((0, 91), (30, 92), (50, 93), (100, 94)):
        forcing = {"annual_precip_in": 1000 / 25.4,
                   "rain_days": STAT_FEEDLOT_RAIN_DAYS,
                   "rain_correction_fraction": STAT_FEEDLOT_RAIN_CORRECTION,
                   "runoff_day_fraction": STAT_FEEDLOT_RUNOFF_DAY_FRACTION,
                   "cn": cn}
        assert 0 <= plet_annual_surface_runoff_in(forcing)[3] <= 1000 / 25.4


def test_obsolete_statistical_feedlot_runoff_input_fails_early(tmp_path):
    path = _mass_package(tmp_path)
    with sqlite3.connect(path) as con:
        con.execute('ALTER TABLE afo_hydrology ADD COLUMN feedlot_runoff_mm REAL')
    with pytest.raises(ValueError, match="no longer accepts feedlot_runoff_mm"):
        _load(path, mode="statistical")


def test_statistical_rain_days_and_precipitation_distribution(tmp_path):
    path = _mass_package(tmp_path)
    with sqlite3.connect(path) as con:
        con.execute('ALTER TABLE afo_hydrology ADD COLUMN rain_days_value REAL')
        con.execute('ALTER TABLE afo_hydrology ADD COLUMN annual_precip_mm_value REAL')
        con.execute('UPDATE afo_hydrology SET rain_days_value=80, '
                    'annual_precip_mm=NULL, annual_precip_mm_value=1200')
    inputs = _load(path, mode="statistical")
    ctx = _ctx(inputs)
    ctx.parcel_ids = ["1", "2", "3", "4"]
    ctx.current_pathway_load_rates = np.zeros((4, 2, 2))
    mass_afo_realization(ctx)
    animals = [dict(a, count=a["count"]) for a in inputs["animals"]]
    expected = _feedlot_runoff_kg(
        inputs["operations"][0], animals,
        {"annual_precip_in": 1200 / 25.4, "rain_days": 80,
         "rain_correction_fraction": STAT_FEEDLOT_RAIN_CORRECTION,
         "runoff_day_fraction": STAT_FEEDLOT_RUNOFF_DAY_FRACTION}, ["TN", "TP"])
    assert ctx.current_afo_feedlot_pathway_rates[0, 1, 0] == pytest.approx(
        expected["TP"] / inputs["area_ha"]["1"])


def test_no_feedlot_sediment_source_displaces_background_tss(tmp_path):
    """Explicit regression documenting the known TSS limitation (not a fix)."""
    inputs = _load(_mass_package(tmp_path), mode="statistical")
    ctx = _ctx(inputs)
    ctx.pollutants = ["TN", "TP", "TSS"]
    ctx.parcel_ids = ["1", "2", "3", "4"]
    ctx.current_pathway_load_rates = np.ones((4, 3, 2))
    ctx.current_pathway_load_rates[:, 2, :] = [6, 4]
    mass_afo_realization(ctx)
    assert np.all(ctx.current_afo_feedlot_pathway_rates[:, 2, :] == 0)
    assert ctx.current_pathway_load_rates[0, 2, 0] == pytest.approx(6 * 0.8)
    assert ctx.current_pathway_load_rates[0, 2, 1] == pytest.approx(4 * 0.8)
    assert ctx.current_pathway_load_rates[2, 2, 0] == pytest.approx(6)


def test_missing_landcover_pollutant_increment_rejected_on_input(tmp_path):
    path = _package(tmp_path / "afo.gpkg")
    with sqlite3.connect(path) as con:
        con.execute("DELETE FROM afo_manure_concentration_increments "
                    "WHERE land_cover='pastureland' AND pollutant='TP'")
    with pytest.raises(ValueError, match="missing required.*combinations.*pastureland.*TP"):
        load_afo_inputs(path, _parcels(), ["TN", "TP"])


def test_whitespace_ids_consistent_across_all_mass_tables(tmp_path):
    path = _mass_package(tmp_path)
    with sqlite3.connect(path) as con:
        for table in ("afo_animals", "afo_manure_management", "afo_manure_fields"):
            con.execute(f"UPDATE {table} SET afo_id=' A1 '")
    inputs = _load(path, mode="statistical")
    assert {r["afo_id"] for r in inputs["management"]} == {"A1"}
    assert {r["afo_id"] for r in inputs["fields"]} == {"A1"}
    assert {r["afo_id"] for r in inputs["animals"]} == {"A1"}


def test_canonical_receiving_pid_duplicates_rejected(tmp_path):
    path = _package(tmp_path / "afo.gpkg")
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE afo_manure_fields")
        pd.DataFrame([{"afo_id": "A1", "pid": "1"},
                      {"afo_id": " A1 ", "pid": "1.0"}]).to_sql(
            "afo_manure_fields", con, index=False, dtype={"pid": "TEXT"})
    with pytest.raises(ValueError, match="Duplicate operation-to-receiving-parcel"):
        load_afo_inputs(path, _parcels(), ["TN", "TP"])
