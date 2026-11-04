"""APLE-derived P / APLE-inspired N, both baseline modes and mass ledgers."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.afo import (
    _draw, animal_production, manure_surface_export_kg,
    load_afo_inputs, mass_afo_realization, reference_excretion_factors,
)
from test_afo import _parcels, _state, _ctx, _package


def _mass_package(tmp_path, *, weight=False):
    path = _package(tmp_path / 'afo.gpkg')
    with sqlite3.connect(path) as con:
        pd.DataFrame([dict(afo_id='A1', season='spring', solids_pct=25,
                           wep_fraction=0.4, f_n=0.2,
                           collection_fraction=0.8, export_fraction=0.1,
                           n_retention_fraction=0.7, p_retention_fraction=0.9,
                           incorporated_fraction=0.2)]).to_sql('afo_manure_management', con, index=False)
        if weight:
            con.execute("ALTER TABLE afo_manure_fields ADD COLUMN weight REAL")
            con.execute("UPDATE afo_manure_fields SET weight=2 WHERE pid=1")
            con.execute("UPDATE afo_manure_fields SET weight=1 WHERE pid=3")
        pd.DataFrame([dict(pid=1, eligible=1), dict(pid=3, eligible=1)]).to_sql('afo_field_eligibility', con, index=False)
        pd.DataFrame([dict(pid=None, annual_runoff_mm=200, annual_precip_mm=1000,
                           feedlot_runoff_mm=300)]).to_sql('afo_hydrology', con, index=False)
    return path


def _load(path, mode='plet_rusle'):
    return load_afo_inputs(path, _parcels(), ['TN', 'TP'], method='mass_based', mode=mode,
                           pathways=['surface', 'subsurface'], surface_pathway='surface')


def test_aple_manure_p_equation_spring_solid_and_annual_n():
    result = manure_surface_export_kg(applied_n_kg=100, applied_p_kg=30,
             runoff_mm=200, precipitation_mm=1000, available_n_fraction=.2,
             n_exponent=1, p_wep_fraction=.25, solids_pct=30,
             incorporated_fraction=0, season='spring')
    expected_available_p = 30*(.25 + .75*.15)
    assert result['surface_available_p_kg'] == pytest.approx(expected_available_p)
    assert result['TP'] == pytest.approx(expected_available_p*.2**1.225)
    assert result['TN'] == pytest.approx(4.0)
    assert manure_surface_export_kg(applied_n_kg=1, applied_p_kg=1, runoff_mm=0,
            precipitation_mm=1000, available_n_fraction=.5,n_exponent=1,
            p_wep_fraction=.4, solids_pct=30,incorporated_fraction=0,
            season='spring')['TP'] == 0
    with pytest.raises(ValueError,match='cannot exceed'):
        manure_surface_export_kg(applied_n_kg=1, applied_p_kg=1,runoff_mm=1200,
               precipitation_mm=1000,available_n_fraction=.5,n_exponent=1,
               p_wep_fraction=.4,solids_pct=30,incorporated_fraction=0,season='spring')


def test_aple_liquid_fall_and_incorporation_reduce_export():
    kwargs=dict(applied_n_kg=80,applied_p_kg=50,runoff_mm=300,
                precipitation_mm=1000, available_n_fraction=.3,n_exponent=1,
                p_wep_fraction=.3)
    spring=manure_surface_export_kg(**kwargs,solids_pct=20,incorporated_fraction=0,season='spring')
    liquid=manure_surface_export_kg(**kwargs,solids_pct=10,incorporated_fraction=0,season='spring')
    fall=manure_surface_export_kg(**kwargs,solids_pct=20,incorporated_fraction=0,season='fall')
    incorporated=manure_surface_export_kg(**kwargs,solids_pct=20,incorporated_fraction=1,season='spring')
    assert 0 < liquid['TP'] < spring['TP']
    assert 0 < fall['TP'] < spring['TP']
    assert incorporated['TP'] == 0
    assert all(r['TN']==pytest.approx(7.2) for r in (spring,liquid,fall,incorporated))


def test_production_factors_and_same_animal_count_for_feeding(tmp_path):
    factors=reference_excretion_factors()
    assert factors['beef_finishing']['tn_kg_head_day'] > 0
    ctx=_ctx(None)
    row=dict(afo_id='A1',species='beef_steer',count=10,animal_days=180)
    result=animal_production(ctx,[row],'A1')
    assert result['TN']==pytest.approx(10*180*factors['beef_finishing']['tn_kg_head_day'])
    row['tn_kg_head_day']=.5
    assert animal_production(ctx,[row],'A1')['TN']==pytest.approx(900)
    with pytest.raises(ValueError,match='animal_days'):
        animal_production(ctx,[dict(row,animal_days=400)],'A1')


def test_plet_mass_allocation_conserves_and_preserves_subsurface(tmp_path):
    inp=_load(_mass_package(tmp_path,weight=True))
    state=_state()
    ctx=_ctx(inp)
    ctx.current_pathway_load_rates=state.pathway_load_rates
    ctx.parcel_ids = state.parcel_ids
    assignments,ledger=mass_afo_realization(ctx,state=state)
    assert len(assignments)==2 and len(ledger)==2
    assert ledger[0]['tn_applied_kg']/ledger[1]['tn_applied_kg']==pytest.approx(2)
    factors=reference_excretion_factors()
    produced_n=10*365*factors['beef_finishing']['tn_kg_head_day']
    assert sum(r['tn_applied_kg'] for r in ledger)==pytest.approx(produced_n*.8*.9*.7)
    assert sum(r['tp_applied_kg'] for r in ledger)==pytest.approx(10*365*factors['beef_finishing']['tp_kg_head_day']*.8*.9*.9)
    assert sum(r['tn_exported_kg'] for r in ledger) <= sum(r['tn_applied_kg'] for r in ledger)
    assert sum(r['tp_dissolved_exported_kg'] for r in ledger) <= sum(r['tp_applied_kg'] for r in ledger)
    assert state.pathway_load_rates[2,0,1]==pytest.approx(2.)  # no manure subsurface
    assert state.pathway_load_rates[0,0,1]==pytest.approx(1.6)  # feedlot footprint
    assert state.concentrations[2]['TN']>2
    assert ctx.current_afo_feedlot_pathway_rates[0,0,0]>0


def test_statistical_mode_loads_direct_feedlot_and_manure(tmp_path):
    inp=_load(_mass_package(tmp_path),mode='statistical')
    ctx=_ctx(inp)
    ctx.parcel_ids=['1','2','3','4']
    ctx.current_pathway_load_rates=np.array([[[3.,2.],[1.,.5]]]*4)
    before=ctx.current_pathway_load_rates.copy()
    assignments,ledger=mass_afo_realization(ctx)
    assert len(assignments)==2
    assert ctx.current_pathway_load_rates[0,0,1]==pytest.approx(before[0,0,1]*.8)
    assert ctx.current_pathway_load_rates[2,0,0]>before[2,0,0]
    assert ctx.current_afo_feedlot_pathway_rates[0,0,0]>0
    assert ctx.current_afo_feedlot_pathway_rates[0,0,1]==0
    assert ctx.current_afo_manure_pathway_rates[2,0,1]==0
    assert all(row['runoff_mm']==200 for row in ledger)


def test_invalid_missing_management_and_statistical_hydrology(tmp_path):
    path=_mass_package(tmp_path)
    with sqlite3.connect(path) as con:
        con.execute('DELETE FROM afo_manure_management')
    with pytest.raises(ValueError,match='one row'):
        _load(path)


def test_shared_selection_exact_nonnegative_and_reproducible(tmp_path):
    path=_mass_package(tmp_path)
    inp=_load(path,mode='statistical')
    a_ctx=_ctx(inp,seed=10)
    a_ctx.parcel_ids=['1','2','3','4']
    a_ctx.current_pathway_load_rates=np.ones((4,2,2))
    b_ctx=_ctx(inp,seed=10)
    b_ctx.parcel_ids=a_ctx.parcel_ids
    b_ctx.current_pathway_load_rates=np.ones((4,2,2))
    _,a=mass_afo_realization(a_ctx)
    _,b=mass_afo_realization(b_ctx)
    assert a==b


def test_distributed_management_fraction_and_count(tmp_path):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as conn:
        conn.execute('ALTER TABLE afo_manure_management ADD COLUMN f_n_value REAL')
        conn.execute('UPDATE afo_manure_management SET f_n=NULL, f_n_value=0.35')
        conn.execute('ALTER TABLE afo_animals ADD COLUMN count_value REAL')
        conn.execute('UPDATE afo_animals SET count=NULL, count_value=12')
    inputs = _load(package, mode='statistical')
    ctx = _ctx(inputs)
    ctx.parcel_ids = ['1','2','3','4']
    ctx.current_pathway_load_rates = np.ones((4,2,2))
    _, ledger = mass_afo_realization(ctx)
    assert all(row['f_n'] == pytest.approx(.35) for row in ledger)
    assert all(row['animal_n_produced_kg'] == pytest.approx(12*365*reference_excretion_factors()['beef_finishing']['tn_kg_head_day']) for row in ledger)
    assert ctx.current_afo_feedlot_pathway_rates[0,0,0] > 0


def test_statistical_huc_hydrology_inheritance(tmp_path):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as conn:
        conn.execute('DROP TABLE afo_hydrology')
        pd.DataFrame([
            dict(pid=None, huc12=None, annual_runoff_mm=100., annual_precip_mm=800., feedlot_runoff_mm=200.),
            dict(pid=None, huc12='050900000001', annual_runoff_mm=250., annual_precip_mm=1100., feedlot_runoff_mm=350.),
            dict(pid=3, huc12=None, annual_runoff_mm=400., annual_precip_mm=1200., feedlot_runoff_mm=500.),
        ]).to_sql('afo_hydrology', conn, index=False)
    parcel_package = tmp_path / 'parcels.gpkg'
    with sqlite3.connect(parcel_package) as conn:
        pd.DataFrame([dict(pid=1,huc_id='050900000001'),dict(pid=2,huc_id='050900000001'),
                      dict(pid=3,huc_id='050900000001'),dict(pid=4,huc_id='050900000002')]
                    ).to_sql('parcel_huc12',conn,index=False)
    loaded = load_afo_inputs(package,_parcels(),['TN','TP'],method='mass_based',
           mode='statistical',pathways=['surface','subsurface'],parcel_path=parcel_package)
    assert loaded['hydrology']['1']['annual_runoff_mm'] == 250.
    assert loaded['hydrology']['3']['annual_runoff_mm'] == 400.
    assert loaded['hydrology']['4']['annual_runoff_mm'] == 100.
    assert loaded['hydrology']['1'] is loaded['hydrology']['2']


def test_statistical_invalid_missing_feedlot_runoff_is_rejected(tmp_path):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as conn:
        conn.execute('UPDATE afo_hydrology SET feedlot_runoff_mm=NULL')
    with pytest.raises(ValueError, match='feedlot_runoff_mm'):
        _load(package,mode='statistical')
