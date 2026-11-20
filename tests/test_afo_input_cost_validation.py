"""Early AFO input validation and operation-specific feedlot cost quantities."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.afo import feedlot_bmp_cost_quantities, load_afo_inputs, animal_production
from src.cost import _estimate_costs_for_probabilities, _get_bmp_cost
from src.input_validation import validate_stats_rows, validate_named_parameter_stats
from test_afo import _package, _parcels
from test_afo_mass import _load, _mass_package
from test_cost_units import DummyModel
from test_input_scopes import Logger, inputs, write_tables
from src.input_config import load_and_validate_all
from src.constants import COL_COST_COMPONENT, COL_CPS, COL_UNIT
from test_input_distribution_standardization import Ctx


def _alter(path, table, column, value, *, field_type='REAL'):
    with sqlite3.connect(path) as conn:
        cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
        if column not in cols:
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {field_type}')
        conn.execute(f'UPDATE "{table}" SET "{column}"=?', (value,))


@pytest.mark.parametrize('parameter,values,pattern', [
    ('count', {'count_mean': 20, 'count_sd': 2}, 'mixes fixed value'),
    ('count', {'count': None, 'count_value': -5}, 'physical domain'),
    ('count', {'count': None, 'count_mean': 15}, 'mean\\+sd'),
    ('count', {'count': None, 'count_min': 20}, 'both min and max'),
    ('count', {'count': None, 'count_min': 30, 'count_max': 20}, 'min > max'),
    ('count', {'count': None, 'count_mean': 10, 'count_sd': -1}, 'sd must'),
    ('animal_days', {'animal_days': 365, 'animal_days_mean': 180, 'animal_days_sd': 30}, 'mixes fixed value'),
    ('animal_days', {'animal_days_value': 400}, 'physical domain'),
    ('animal_days', {'animal_days_min': 0, 'animal_days_max': 367}, 'physical domain'),
    ('animal_days', {'animal_days_mean': 200}, 'mean\\+sd'),
])
def test_animal_parameters_reject_bad_statistics_before_simulation(tmp_path, parameter, values, pattern):
    package = _mass_package(tmp_path)
    for col, value in values.items():
        _alter(package, 'afo_animals', col, value)
    with pytest.raises(ValueError, match=pattern):
        _load(package)


def test_valid_bounded_animal_days_normal_is_sampled_within_366(tmp_path):
    package = _mass_package(tmp_path)
    _alter(package, 'afo_animals', 'animal_days_mean', 360)
    _alter(package, 'afo_animals', 'animal_days_sd', 100)
    data = _load(package)
    ctx = Ctx(seed=4)
    ctx.afo_inputs = {'excretion_factors': data['excretion_factors']}
    for _ in range(30):
        # Does not raise after validation; physical truncation is applied.
        result = animal_production(ctx, data['animals'], 'A1')
        assert np.isfinite(result['TN'])


def test_count_distribution_accepted_without_base_value(tmp_path):
    package = _mass_package(tmp_path)
    _alter(package, 'afo_animals', 'count', None)
    _alter(package, 'afo_animals', 'count_mean', 10)
    _alter(package, 'afo_animals', 'count_sd', 2)
    assert _load(package)['animals'][0]['count_mean'] == 10


@pytest.mark.parametrize('table,field,extra,expected', [
    ('afo_manure_management', 'collection_fraction', {'collection_fraction_mean': .5, 'collection_fraction_sd': .1}, 'mixes fixed value'),
    ('afo_manure_management', 'tn_surface_export_fraction', {'tn_surface_export_fraction_mean': .5, 'tn_surface_export_fraction_sd': .1}, 'mixes fixed value'),
    ('afo_animals', 'tn_kg_1000kg_day', {'tn_kg_1000kg_day': .5, 'tn_kg_1000kg_day_mean': .4, 'tn_kg_1000kg_day_sd': .1}, 'mixes fixed value'),
])
def test_conflict_rejected_in_manure_handling_or_override(tmp_path, table, field, extra, expected):
    package = _mass_package(tmp_path)
    for col, value in extra.items():
        _alter(package, table, col, value)
    with pytest.raises(ValueError, match=expected):
        _load(package)


def test_recovery_fraction_reuses_shared_validation(tmp_path):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as con:
        pd.DataFrame([{'afo_id': 'A1', 'cps': 313,
                       'additional_collection_fraction': .2,
                       'additional_collection_fraction_mean': .3,
                       'additional_collection_fraction_sd': .1}]).to_sql(
                           'afo_feedlot_manure_effects', con, index=False)
    with pytest.raises(ValueError, match='mixes fixed value'):
        _load(package, feedlot_bmp_cps=[313])


def test_sampled_field_counts_reject_conflicts(tmp_path):
    package = _package(tmp_path/'fields.gpkg', allocation='sampled')
    _alter(package, 'afo_operations', 'n_fields_mean', 2.)
    _alter(package, 'afo_operations', 'n_fields_sd', .2)
    with pytest.raises(ValueError, match='mixes fixed value'):
        load_afo_inputs(package, _parcels(), ['TN', 'TP'])


def test_statistical_hydrology_rejects_conflicting_hydrologic_values(tmp_path):
    package = _mass_package(tmp_path)
    _alter(package, 'afo_hydrology', 'annual_precip_mm_mean', 900)
    _alter(package, 'afo_hydrology', 'annual_precip_mm_sd', 100)
    with pytest.raises(ValueError, match='mixes fixed value'):
        _load(package, mode='statistical')


def test_standard_model_input_and_named_parameter_use_same_validation():
    with pytest.raises(ValueError, match='mixes fixed value'):
        validate_stats_rows(pd.DataFrame([{'parameter': 'cn', 'value': 60, 'mean': 70, 'sd': 1}]), 'plet_inputs')
    with pytest.raises(ValueError, match='mixes fixed value'):
        validate_named_parameter_stats({'count': 6, 'count_mean': 7, 'count_sd': 1}, 'count', 'AFO')


def test_feedlot_cost_uses_footprint_by_default_not_parcel_area(tmp_path):
    package = _mass_package(tmp_path)
    data = _load(package, feedlot_bmp_cps=[313])
    assert feedlot_bmp_cost_quantities(data, '1', 313) == (.2, None)
    model = DummyModel(pd.DataFrame([
        {COL_CPS: 313, COL_UNIT: 'usd/ha', 'value': 1000},
    ]))
    assert _get_bmp_cost(model, 313, .2, feedlot=True) == pytest.approx(200.)


def test_feedlot_cost_mixed_components_explicit_area_length_and_project(tmp_path):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as con:
        pd.DataFrame([{'afo_id': 'A1', 'cps': 313,
                       'cost_area_ha': .1, 'cost_length_m': 12}]).to_sql(
                           'afo_feedlot_bmp_cost_dimensions', con, index=False)
    data = _load(package, feedlot_bmp_cps=[313])
    area, length = feedlot_bmp_cost_quantities(data, '1', 313)
    assert area == .1 and length == 12
    rows = pd.DataFrame([
        {COL_CPS: 313, COL_COST_COMPONENT: 'liner', COL_UNIT: 'usd/ha', 'value': 100},
        {COL_CPS: 313, COL_COST_COMPONENT: 'pipe', COL_UNIT: 'usd/m', 'value': 3},
        {COL_CPS: 313, COL_COST_COMPONENT: 'design', COL_UNIT: 'usd/project', 'value': 200},
    ])
    model = DummyModel(rows)
    assert _get_bmp_cost(model, 313, area, feedlot=True, cost_length_m=length) == pytest.approx(246)
    with pytest.raises(ValueError, match='cost_length_m'):
        _get_bmp_cost(model, 313, area, feedlot=True)
    model.data['feedlot_bmp_cps'] = [313]
    model.data['afo_inputs'] = data
    probabilities = _estimate_costs_for_probabilities(model)
    assert probabilities['probability'].iloc[0] == pytest.approx(1.)


def test_feedlot_cost_multiple_operations_sum_areas_and_lengths():
    inputs = {'operations': [
        {'afo_id': 'A1', 'pid': '1', 'feedlot_area_ha': .2},
        {'afo_id': 'A2', 'pid': '1', 'feedlot_area_ha': .15}],
        'feedlot_cost_dimensions': {('A1', 313): {'cost_length_m': 10},
                                    ('A2', 313): {'cost_area_ha': .1, 'cost_length_m': 6}}}
    assert feedlot_bmp_cost_quantities(inputs, '1', 313) == pytest.approx((.3, 16))


def test_invalid_feedlot_cost_dimensions_rejected_at_loading(tmp_path):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as con:
        pd.DataFrame([{'afo_id': 'A1', 'cps': 313, 'cost_area_ha': -1}]).to_sql(
            'afo_feedlot_bmp_cost_dimensions', con, index=False)
    with pytest.raises(ValueError, match='cost_area_ha'):
        _load(package, feedlot_bmp_cps=[313])


def test_missing_length_based_feedlot_cost_dimension_rejected_before_scenarios(inputs, tmp_path):
    cfg, paths, _, _ = inputs
    cfg.pop('huc_parameters')
    cfg['pollutants'] = ['TN', 'TP']
    write_tables(paths['domain'], input_pollutant_load_rate=[
        {'pollutant': p, 'pathway': 'surface', 'value': 1} for p in ('TN', 'TP')])
    pd.DataFrame([{'cps': 340, 'pollutant': p, 'pathway': 'surface', 'value': .5}
                  for p in ('TN', 'TP')]).to_csv(cfg['bmp_efficiency'], index=False)
    cfg['animal_operations'] = str(_mass_package(tmp_path))
    cfg['manure_loading'] = {'method': 'mass_based'}
    cfg['feedlot_bmp_cps'] = [340]
    cost_path = tmp_path/'cost.csv'
    pd.DataFrame([{'cps':340, 'unit':'usd/m', 'value':100}]).to_csv(cost_path, index=False)
    cfg['bmp_cost'] = str(cost_path)
    with pytest.raises(ValueError, match='cost_length_m'):
        load_and_validate_all(cfg, Logger())


def test_scenario_bmp_record_uses_feedlot_cost_area_not_host_parcel(inputs, tmp_path, monkeypatch):
    """End-to-end placement calculation without requiring a Parquet engine."""
    import logging
    from pathlib import Path
    from src.model import Model, _run_one_scenario
    import src.model as model_module

    cfg, paths, parcels, _ = inputs
    cfg.pop('huc_parameters')
    cfg['pollutants'] = ['TN', 'TP']
    write_tables(paths['domain'], input_pollutant_load_rate=[
        {'pollutant': p, 'pathway': 'surface', 'value': 1} for p in ('TN', 'TP')])
    pd.DataFrame([{'cps': 340, 'pollutant': p, 'pathway': 'surface', 'value': .5}
                  for p in ('TN', 'TP')]).to_csv(cfg['bmp_efficiency'], index=False)
    cfg['animal_operations'] = str(_mass_package(tmp_path))
    cfg['manure_loading'] = {'method': 'mass_based'}
    cfg['feedlot_bmp_cps'] = [340]
    path = tmp_path/'costs.csv'
    pd.DataFrame([{'cps': 340, 'unit':'usd/ha', 'value':1000},
                  {'cps': 340, 'cost_component':'mobilization', 'unit':'usd/project', 'value':100}]).to_csv(path,index=False)
    cfg['bmp_cost'] = str(path)
    data = load_and_validate_all(cfg, Logger())
    model = Model(cfg, data, logging.getLogger('feedlot.cost.integration'))
    written = {}

    def record(df, path, *, logger):
        written[path.parent.name] = df.copy()

    monkeypatch.setattr(model_module, '_write_parquet_atomic', record)
    _run_one_scenario(model._shared_payload(), cfg, 0, 7, tmp_path/'output')
    records = written['bmps']
    assert len(records) == 1
    assert int(records.iloc[0]['cps']) == 340
    assert records.iloc[0]['pid'] == '1'
    assert records.iloc[0]['cost_usd'] == pytest.approx(300.)


def test_legacy_concentration_mode_samples_prefixed_animal_count(tmp_path):
    """Legacy PLET load calculation uses the sampled count, not a NULL base."""
    from test_afo import _ctx, _state
    from src.afo import apply_afo_baseline

    package = _package(tmp_path/'legacy.gpkg')
    _alter(package, 'afo_animals', 'count', None)
    _alter(package, 'afo_animals', 'count_value', 10)
    data = load_afo_inputs(package, _parcels(), ['TN', 'TP'])
    ctx = _ctx(data)
    apply_afo_baseline(ctx, _state())
    assert ctx.current_afo_feedlot_pathway_rates[0, 0, 0] > 0


@pytest.mark.parametrize('table,base,extra', [
    ('afo_operations', 'feedlot_area_ha', 'feedlot_area_ha_mean'),
    ('afo_operations', 'paved_pct', 'paved_pct_mean'),
    ('afo_animals', 'factor_tn', 'factor_tn_mean'),
])
def test_fixed_only_afo_fields_reject_ignored_stat_columns(tmp_path, table, base, extra):
    package = _mass_package(tmp_path)
    _alter(package, table, extra, 3)
    with pytest.raises(ValueError, match='mixes fixed value'):
        _load(package)


def test_feedlot_cost_dimension_rejects_distribution_even_with_other_fixed_dimension(tmp_path):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as con:
        pd.DataFrame([{'afo_id': 'A1', 'cps': 313, 'cost_length_m': 5,
                       'cost_area_ha_mean': .2, 'cost_area_ha_sd': .1}]).to_sql(
                           'afo_feedlot_bmp_cost_dimensions', con, index=False)
    with pytest.raises(ValueError, match='cost_area_ha.*fixed scalar'):
        _load(package, feedlot_bmp_cps=[313])
