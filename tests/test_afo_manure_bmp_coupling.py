"""Explicit feedlot BMP manure recovery and source-aware field coupling."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.afo import apply_afo_feedlot_manure_effect, load_afo_inputs, mass_afo_realization
from src.bmp import _apply_pathway_reduction
from src.model import _run_one_scenario
from test_afo import _ctx, _parcels
from test_afo_mass import _mass_package, _load
from test_plet_end_to_end_regression import _config, _single_mass_afo_payload


def _with_recovery(path, *, gain=0.5, collection=None, cps=313):
    with sqlite3.connect(path) as conn:
        pd.DataFrame([dict(afo_id="A1", cps=cps,
                           additional_collection_fraction=gain)]).to_sql(
                               "afo_feedlot_manure_effects", conn, index=False)
        if collection is not None:
            conn.execute("UPDATE afo_manure_management SET collection_fraction=?", (collection,))
    return path


@pytest.mark.parametrize("mode", ["statistical", "plet_rusle"])
@pytest.mark.parametrize("collection", [0.0, 0.8])
def test_collection_gain_increases_field_mass_even_at_zero_baseline(tmp_path, mode, collection):
    path = _with_recovery(_mass_package(tmp_path), collection=collection)
    inp = _load(path, mode=mode, feedlot_bmp_cps=[313])
    ctx = _ctx(inp)
    ctx.parcel_ids = ['1', '2', '3', '4']
    ctx.parcel_area_ha = np.ones(4)
    if mode == 'plet_rusle':
        from test_afo import _state
        state = _state()
        ctx.current_pathway_load_rates = state.pathway_load_rates
        _, ledger = mass_afo_realization(ctx, state=state)
    else:
        ctx.current_pathway_load_rates = np.ones((4, 2, 2))
        _, ledger = mass_afo_realization(ctx)
    before = ctx.current_afo_manure_pathway_rates.copy()
    current = ctx.current_pathway_load_rates.sum(axis=2).copy()
    changes = apply_afo_feedlot_manure_effect(ctx, '1', 313, current)
    expected_collection = collection + (1 - collection) * 0.5
    assert ctx.afo_bmp_operation_state['A1']['current_collection'] == pytest.approx(expected_collection)
    if collection > 0:
        assert ctx.current_afo_manure_pathway_rates[2, 0, 0] == pytest.approx(
            before[2, 0, 0] * expected_collection / collection)
    else:
        assert before.sum() == 0
        assert ctx.current_afo_manure_pathway_rates[2, 0, 0] > 0
    assert len(changes) == 4  # two fields x two nutrients
    assert np.allclose(ctx.current_pathway_load_rates.sum(axis=2), current)
    assert ctx.afo_bmp_effect_records[-1]['added_available_tn_kg'] > 0
    assert sum(r['tn_applied_kg'] for r in ledger) == pytest.approx(
        sum(r['operation_available_tn_kg'] for r in ledger) / len(ledger))


def test_earlier_field_bmp_still_treats_newly_collected_manure(tmp_path):
    path = _with_recovery(_mass_package(tmp_path), collection=0)
    ctx = _ctx(_load(path, mode='statistical', feedlot_bmp_cps=[313]))
    ctx.parcel_ids = ['1', '2', '3', '4']
    ctx.parcel_area_ha = np.ones(4)
    ctx.current_pathway_load_rates = np.ones((4, 2, 2))
    mass_afo_realization(ctx)
    ctx.current_bmp_targets_feedlot = False
    # Full future source also needs the previously installed receiving-field BMP.
    _apply_pathway_reduction(ctx, 2, 0, 1., {'surface': 0.5, 'subsurface': 0.})
    now = ctx.current_pathway_load_rates.sum(axis=2).copy()
    apply_afo_feedlot_manure_effect(ctx, '1', 313, now)
    n_source = next(s for s in ctx.afo_bmp_operation_state['A1']['sources']
                    if s['parcel_idx'] == 2 and s['pol_idx'] == 0)
    # BMP 50% field reduction is retained even though there was no initial manure.
    assert n_source['current_rate'] == pytest.approx(n_source['full_rate'] * 0.5)
    untreated_full = n_source['full_rate'] / 0.5
    assert n_source['current_rate'] == pytest.approx(untreated_full * 0.25)
    assert ctx.current_pathway_load_rates[2, 0, 1] == pytest.approx(1.)


def test_recovery_is_opt_in_and_applies_only_at_configured_host(tmp_path):
    path = _mass_package(tmp_path)
    ctx = _ctx(_load(path, mode='statistical'))
    ctx.parcel_ids = ['1', '2', '3', '4']
    ctx.parcel_area_ha = np.ones(4)
    ctx.current_pathway_load_rates = np.ones((4, 2, 2))
    mass_afo_realization(ctx)
    initial = ctx.current_afo_manure_pathway_rates.copy()
    loads = ctx.current_pathway_load_rates.sum(axis=2).copy()
    assert apply_afo_feedlot_manure_effect(ctx, '1', 313, loads) == []
    assert np.array_equal(initial, ctx.current_afo_manure_pathway_rates)


@pytest.mark.parametrize('bad', [(-0.01, 313), (1.01, 313), (0.2, 999)])
def test_invalid_effect_inputs_rejected(tmp_path, bad):
    gain, cps = bad
    path = _with_recovery(_mass_package(tmp_path), gain=gain, cps=cps)
    with pytest.raises(ValueError, match='additional_collection_fraction|feedlot_bmp_cps'):
        _load(path, feedlot_bmp_cps=[313])


def test_duplicate_effects_and_legacy_mode_rejected(tmp_path):
    path = _with_recovery(_mass_package(tmp_path))
    with sqlite3.connect(path) as conn:
        conn.execute('INSERT INTO afo_feedlot_manure_effects SELECT * FROM afo_feedlot_manure_effects')
    with pytest.raises(ValueError, match='Duplicate'):
        _load(path, feedlot_bmp_cps=[313])
    with pytest.raises(ValueError, match='requires mass_based'):
        load_afo_inputs(path, _parcels(), ['TN', 'TP'], method='concentration_increment')


@pytest.mark.parametrize('statistical', [False, True])
def test_scenario_recovery_offsets_feedlot_removal_and_updates_budgets(
        tmp_path, monkeypatch, statistical):
    written = {}
    monkeypatch.setattr('src.model._write_parquet_atomic',
                        lambda df, path, logger: written.setdefault((path.parent.name, path.name), df.copy()))
    payload = _single_mass_afo_payload(statistical=statistical)
    payload['afo_inputs']['management'][0]['collection_fraction'] = 0.5
    payload['afo_inputs']['feedlot_manure_effects'] = [
        {'afo_id': 'demo', 'cps': 340, 'additional_collection_fraction': 0.5}]
    result = _run_one_scenario(payload, _config(tmp_path), 0, 1234, tmp_path)
    budget = written[('afo_operation_budgets', 's1.parquet')].iloc[0]
    source = written[('afo_load_components', 's1.parquet')]
    tn = source[source.pollutant == 'TN'].iloc[0]
    bmp = written[('bmps', 's1.parquet')].iloc[0]
    effect = written[('afo_manure_bmp_effects', 's1.parquet')].iloc[0]
    assert budget.final_tn_available_kg == pytest.approx(1.5 * budget.tn_available_kg)
    assert budget.final_tp_available_kg == pytest.approx(1.5 * budget.tp_available_kg)
    assert budget.final_tn_available_kg == pytest.approx(budget.final_tn_applied_kg)
    assert tn.final_manure_field_surface_kg_yr == pytest.approx(1.5 * tn.baseline_manure_field_surface_kg_yr)
    assert tn.final_direct_feedlot_surface_kg_yr == pytest.approx(0.5 * tn.baseline_direct_feedlot_surface_kg_yr)
    assert bmp.induced_manure_tn_kg_yr == pytest.approx(
        tn.final_manure_field_surface_kg_yr - tn.baseline_manure_field_surface_kg_yr)
    assert effect.collection_before == pytest.approx(0.5)
    assert effect.collection_after == pytest.approx(0.75)
    assert result[('TN', 'O1', 'count', 'total')][0][2] == pytest.approx(
        tn.baseline_direct_feedlot_surface_kg_yr - tn.final_direct_feedlot_surface_kg_yr
        - bmp.induced_manure_tn_kg_yr)


def test_recovery_field_load_is_delivered_from_receiver_not_host(tmp_path, monkeypatch):
    from shapely.geometry import box
    from src.constants import (
        COL_NDR_F_TO_S, COL_NDR_S_TO_O, COL_SDR_F_TO_S, COL_SDR_S_TO_O,
    )
    written = {}
    monkeypatch.setattr('src.model._write_parquet_atomic',
                        lambda df, path, logger: written.setdefault((path.parent.name, path.name), df.copy()))
    payload = _single_mass_afo_payload(statistical=True)
    payload['afo_inputs']['management'][0]['collection_fraction'] = 0.5
    payload['afo_inputs']['feedlot_manure_effects'] = [
        {'afo_id': 'demo', 'cps': 340, 'additional_collection_fraction': 0.5}]
    payload['afo_inputs']['fields'] = [{'afo_id': 'demo', 'pid': 'P2'}]
    payload['afo_inputs']['geometry']['P2'] = box(100, 0, 200, 100)
    payload['afo_inputs']['area_ha']['P2'] = 1.0
    payload['afo_inputs']['eligible']['P2'] = True
    payload['afo_inputs']['hydrology']['P2'] = {
        'annual_precip_mm': 1000., 'annual_runoff_mm': 200.}
    payload['parcel_ids'] = ['P1', 'P2']
    payload['pid_to_index'] = {'P1': 0, 'P2': 1}
    payload['parcel_area_ha'] = np.array([1., 1.])
    payload['parcel_perim_m'] = np.array([100., 100.])
    payload['parcel_out_oids'] = [['O1'], ['O2']]
    payload['parcel_up_idxs'] = [[], []]
    payload['outlet_oids'] = ['O1', 'O2']
    payload['pollutant_load_rate_stats'].append(payload['pollutant_load_rate_stats'][0])
    payload['delivery_coeffs'][('P2', 'O2')] = {
        COL_NDR_F_TO_S: 1., COL_NDR_S_TO_O: 1.,
        COL_SDR_F_TO_S: 1., COL_SDR_S_TO_O: 1.}
    records = _run_one_scenario(payload, _config(tmp_path), 0, 1234, tmp_path)
    source = written[('afo_load_components', 's1.parquet')]
    field = source[(source.pid == 'P2') & (source.pollutant == 'TN')].iloc[0]
    assert field.final_manure_field_surface_kg_yr > field.baseline_manure_field_surface_kg_yr
    induced = field.final_manure_field_surface_kg_yr - field.baseline_manure_field_surface_kg_yr
    assert records[('TN', 'O2', 'count', 'total')][0][2] == pytest.approx(-induced)
    assert records[('TN', 'O1', 'count', 'total')][0][2] > 0


def test_sequential_capture_is_bounded_and_failure_scales_gain(tmp_path):
    path = _with_recovery(_mass_package(tmp_path), gain=0.5, collection=0.2)
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO afo_feedlot_manure_effects VALUES ('A1', 314, 0.5)")
    ctx = _ctx(_load(path, mode='statistical', feedlot_bmp_cps=[313, 314]))
    ctx.parcel_ids = ['1', '2', '3', '4']
    ctx.parcel_area_ha = np.ones(4)
    ctx.current_pathway_load_rates = np.ones((4, 2, 2))
    mass_afo_realization(ctx)
    current = ctx.current_pathway_load_rates.sum(axis=2).copy()
    apply_afo_feedlot_manure_effect(ctx, '1', 313, current, bmp_effect_scale=0.)
    assert ctx.afo_bmp_operation_state['A1']['current_collection'] == pytest.approx(0.2)
    apply_afo_feedlot_manure_effect(ctx, '1', 313, current)
    assert ctx.afo_bmp_operation_state['A1']['current_collection'] == pytest.approx(0.6)
    apply_afo_feedlot_manure_effect(ctx, '1', 314, current)
    assert ctx.afo_bmp_operation_state['A1']['current_collection'] == pytest.approx(0.8)
