"""Long-form D384.1 excretion inputs and scenario-level rate sampling."""
from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from src.afo import animal_production, load_afo_inputs, mass_afo_realization, reference_excretion_factors
from src.input_config import load_and_validate_all, resolve_config_paths
from src.model import Model
from test_afo import _package, _parcels, _state
from test_afo_mass import _load, _mass_package
from test_input_distribution_standardization import Ctx
from test_input_scopes import Logger, inputs, write_tables


REFERENCE = Path(__file__).resolve().parents[1] / "src" / "data" / "afo" / "manure_excretion_reference.csv"
PARAMETERS = ("tn_kg_1000kg_day", "tp_kg_1000kg_day")


def _reference_table():
    table = pd.read_csv(REFERENCE)
    table["species"] = table["species"].str.lower()
    return table.sort_values(["species", "parameter"]).reset_index(drop=True)


def _write_reference(tmp_path, table):
    path = tmp_path / "rates.csv"
    table.to_csv(path, index=False)
    return path


def _sampling_ctx(factors, seed=7):
    ctx = Ctx(seed)
    ctx.afo_inputs = {"excretion_factors": factors}
    return ctx


@pytest.mark.parametrize("species,weight,means,sds", [
    ("beef", 360., (.34, .092), (.073, .027)),
    ("dairy", 640., (.45, .094), (.096, .024)),
    ("swine", 61., (.52, .18), (.21, .10)),
    ("broiler", .9, (1.1, .30), (.24, .053)),
    ("sheep", 27., (.42, .087), (.11, .030)),
    ("veal", 91., (.27, .066), (.045, .011)),
    ("goat", 64., (.45, .11), (.12, .016)),
    ("horse", 450., (.30, .071), (.063, .026)),
    ("layer", 1.8, (.84, .30), (.22, .081)),
    ("turkey", 6.8, (.62, .23), (.13, .093)),
    ("duck", 1.4, (1.5, .54), (.54, .21)),
])
def test_default_statistics_match_d384_1_nutrient_rates(species, weight, means, sds):
    table = _reference_table()
    assert len(table) == 33
    assert set(table.parameter) == {"representative_mass_kg", *PARAMETERS}
    assert not table.duplicated(["species", "parameter"]).any()
    rows = table[table.species == species].set_index("parameter")
    loaded = reference_excretion_factors()[species]
    assert loaded["representative_mass_kg"] == {"value": weight}
    for parameter, mean, sd in zip(PARAMETERS, means, sds):
        assert rows.loc[parameter, "units"] == "kg/1000kg/day"
        assert "D384.1" in rows.loc[parameter, "source"]
        assert loaded[parameter] == pytest.approx({"mean": mean, "sd": sd})


def test_default_reference_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(reference_excretion_factors()) == 11


@pytest.mark.parametrize("stats,error", [
    ({}, "no fixed value or distribution"),
    ({"mean": 1.}, "mean\\+sd"),
    ({"mean": 1., "sd": -1.}, "sd must be"),
    ({"mean": np.inf, "sd": 1.}, "finite"),
    ({"mean": -1., "sd": 1.}, "physical domain"),
    ({"value": 1., "mean": 1., "sd": .1}, "mixes fixed value"),
    ({"mean": 1., "sd": .1, "min": 0.}, "both min and max"),
    ({"min": 2., "max": 1.}, "min > max"),
    ({"min": 0., "p25": 2., "p75": 1., "max": 3.}, "not monotonic"),
    ({"mean": 1., "sd": .1, "std": .2}, "Multiple populated columns"),
])
def test_invalid_reference_statistics_fail_before_sampling(tmp_path, stats, error):
    table = _reference_table()
    index = table.index[(table.species == "beef") & (table.parameter == PARAMETERS[0])][0]
    table.loc[index, ["value", "mean", "sd"]] = np.nan
    for key, value in stats.items():
        table.loc[index, key] = value
    with pytest.raises(ValueError, match=error):
        reference_excretion_factors(_write_reference(tmp_path, table))


@pytest.mark.parametrize("problem,error", [
    ("empty", "at least one species"),
    ("missing_schema", "missing required columns"),
    ("missing_parameter", "missing parameters"),
    ("duplicate_alias", "duplicate rows"),
    ("blank_species", "species must not be blank"),
    ("unknown_parameter", "unsupported parameters"),
    ("wet_manure_parameter", "unsupported parameters"),
    ("wrong_units", "dimensionally incompatible"),
    ("unknown_species", "Unknown AFO species"),
    ("missing_mass", "representative_mass_kg"),
])
def test_invalid_reference_structure_is_rejected(tmp_path, problem, error):
    table = _reference_table()
    if problem == "empty":
        table = table.iloc[:0]
    elif problem == "missing_schema":
        table = table.drop(columns="parameter")
    elif problem == "missing_parameter":
        table = table.drop(index=0)
    elif problem == "duplicate_alias":
        duplicate = table.iloc[[0]].copy()
        duplicate["species"] = "beef_steer"
        table = pd.concat([table, duplicate], ignore_index=True)
    elif problem == "blank_species":
        table.loc[0, "species"] = " "
    elif problem == "unknown_parameter":
        table.loc[0, "parameter"] = "bod_kg_head_day"
    elif problem == "wet_manure_parameter":
        table.loc[0, "parameter"] = "manure_kg_1000kg_day"
    elif problem == "wrong_units":
        table.loc[0, "units"] = "mg/L"
    elif problem == "unknown_species":
        table.loc[0, "species"] = "dragon"
    elif problem == "missing_mass":
        table = table[table.parameter != "representative_mass_kg"]
    with pytest.raises(ValueError, match=error):
        reference_excretion_factors(_write_reference(tmp_path, table))


@pytest.mark.parametrize("p_stats", [
    {"min": .01, "max": .03},
    {"min": .01, "p50": .02, "max": .03},
])
def test_custom_reference_accepts_common_numeric_forms_and_units(tmp_path, p_stats):
    table = pd.DataFrame([
        {"species": " Beef ", "parameter": "representative_mass_kg", "value": 1000., "units": "kg"},
        {"species": " Beef_Steer ", "parameter": PARAMETERS[0], "value": 2000., "units": "g/1000kg/day"},
        {"species": "beef_steer", "parameter": PARAMETERS[1], **p_stats},
    ])
    factors = reference_excretion_factors(_write_reference(tmp_path, table))
    assert factors["beef"][PARAMETERS[0]] == {"value": 2.}
    ctx = _sampling_ctx(factors)
    result = animal_production(ctx, [{"afo_id": "A1", "species": "beef_steer", "count": 1, "animal_days": 1}], "A1")
    assert set(result) == {"TN", "TP"}
    assert result["TN"] == 2.
    assert .01 <= result["TP"] <= .03


def test_mean_and_sd_unit_conversion_scales_both_statistics(tmp_path):
    table = _reference_table()
    rates = table.parameter != "representative_mass_kg"
    table.loc[rates, "units"] = "g/1000kg/day"
    table.loc[rates, ["value", "mean", "sd"]] *= 1000
    factors = reference_excretion_factors(_write_reference(tmp_path, table))
    assert factors["beef"][PARAMETERS[0]] == pytest.approx({"mean": .34, "sd": .073})


def test_production_samples_only_animal_mass_and_nutrient_rates(monkeypatch):
    factors = reference_excretion_factors()
    ctx = _sampling_ctx(factors)
    draws = []

    def sample(stats, kind=None):
        draws.append((dict(stats), kind))
        return stats["value"] if "value" in stats else stats["mean"]

    monkeypatch.setattr(ctx, "_sample_from_stats", sample)
    animal = {"afo_id": "A1", "species": "Beef", "count": 10, "animal_days": 100}
    produced = animal_production(ctx, [animal], "A1")
    assert produced == pytest.approx({"TN": 122.4, "TP": 33.12})
    assert draws == [
        ({"value": 360.}, "positive"),
        ({"mean": .34, "sd": .073}, "nonnegative"),
        ({"mean": .092, "sd": .027}, "nonnegative"),
    ]


@pytest.mark.parametrize("value", [0., -1., float("inf")])
def test_invalid_representative_mass_is_rejected(tmp_path, value):
    table = _reference_table()
    table.loc[table.parameter == "representative_mass_kg", "value"] = value
    with pytest.raises(ValueError, match="physical domain|finite"):
        reference_excretion_factors(_write_reference(tmp_path, table))


def test_stochastic_mass_is_drawn_once_and_scales_both_nutrients(tmp_path):
    table = _reference_table()
    beef = table.species == "beef"
    rates = beef & (table.parameter != "representative_mass_kg")
    table.loc[rates, "value"] = [2., 1.]
    table.loc[rates, ["mean", "sd"]] = np.nan
    mass = beef & (table.parameter == "representative_mass_kg")
    table.loc[mass, "value"] = np.nan
    table.loc[mass, ["mean", "sd"]] = [360., 30.]
    factors = reference_excretion_factors(_write_reference(tmp_path, table))
    ctx = _sampling_ctx(factors)
    animal = {"afo_id": "A1", "species": "Beef", "count": 1, "animal_days": 1}
    a = animal_production(ctx, [animal], "A1")
    b = animal_production(ctx, [animal], "A1")
    assert a != b
    for result in (a, b):
        assert set(result) == {"TN", "TP"}
        assert result["TP"] > 0
        assert result["TN"] / result["TP"] == pytest.approx(2.)


def test_animal_mass_override_scales_all_production(tmp_path):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as conn:
        conn.execute("ALTER TABLE afo_animals ADD COLUMN representative_mass_kg_mean REAL")
        conn.execute("ALTER TABLE afo_animals ADD COLUMN representative_mass_kg_sd REAL")
        conn.execute("UPDATE afo_animals SET representative_mass_kg_mean=720, representative_mass_kg_sd=0")
    loaded = _load(package)
    base = dict(loaded["animals"][0])
    base.pop("representative_mass_kg_mean")
    base.pop("representative_mass_kg_sd")
    factors = loaded["excretion_factors"]
    a = animal_production(_sampling_ctx(factors), [base], "A1")
    b = animal_production(_sampling_ctx(factors), loaded["animals"], "A1")
    assert b == pytest.approx({key: value * 2 for key, value in a.items()})


@pytest.mark.parametrize("parameter", [
    "tn_kg_head_day", "manure_kg_1000kg_day", "manure_kg_1000kg_day_mean",
])
def test_unsupported_excretion_override_is_rejected(tmp_path, parameter):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as conn:
        conn.execute(f'ALTER TABLE afo_animals ADD COLUMN "{parameter}" REAL')
        conn.execute(f'UPDATE afo_animals SET "{parameter}"=0.1')
    with pytest.raises(ValueError, match="unsupported excretion"):
        _load(package)


def test_reference_rates_drive_reproducible_independent_truncated_draws():
    factors = reference_excretion_factors()
    row = {"afo_id": "A1", "species": "swine", "count": 1, "animal_days": 1}
    ctx = _sampling_ctx(factors)
    repeated = _sampling_ctx(factors)
    first = animal_production(ctx, [row], "A1")
    assert first == animal_production(repeated, [row], "A1")
    assert first != animal_production(ctx, [row], "A1")
    assert first != animal_production(_sampling_ctx(factors, seed=8), [row], "A1")
    draws = np.array([list(animal_production(ctx, [row], "A1").values()) for _ in range(4000)])
    assert np.all(draws >= 0)
    assert np.max(np.abs(np.corrcoef(draws, rowvar=False) - np.eye(len(PARAMETERS)))) < .06
    for i, parameter in enumerate(PARAMETERS):
        stats = factors["swine"][parameter]
        mean, sd = stats["mean"] * .061, stats["sd"] * .061
        alpha = -mean / sd
        ratio = math.exp(-alpha**2 / 2) / math.sqrt(2 * math.pi) / (1 - NormalDist().cdf(alpha))
        expected_mean = mean + sd * ratio
        expected_sd = sd * math.sqrt(1 + alpha * ratio - ratio**2)
        assert draws[:, i].mean() == pytest.approx(expected_mean, abs=5 * expected_sd / math.sqrt(len(draws)))
        assert draws[:, i].std() == pytest.approx(expected_sd, rel=.05)


def test_reference_rates_are_shared_within_operation_class_but_not_across_operations():
    factors = reference_excretion_factors()
    row = {"afo_id": "A1", "species": "beef_steer", "count": 1, "animal_days": 1}
    one = animal_production(_sampling_ctx(factors), [row], "A1")
    two = animal_production(_sampling_ctx(factors), [row, row], "A1")
    assert two == pytest.approx({key: value * 2 for key, value in one.items()})
    ctx = _sampling_ctx(factors)
    cache = {}
    first = animal_production(ctx, [row], "A1", excretion_cache=cache)
    second = animal_production(ctx, [dict(row, afo_id="A2")], "A2", excretion_cache=cache)
    assert first != second


def test_explicit_sample_group_shares_reference_draws_across_operations(tmp_path):
    table = _reference_table()
    table["sample_group"] = "shared"
    ctx = _sampling_ctx(reference_excretion_factors(_write_reference(tmp_path, table)))
    row = {"afo_id": "A1", "species": "beef_steer", "count": 1, "animal_days": 1}
    cache = {}
    first = animal_production(ctx, [row], "A1", excretion_cache=cache)
    second = animal_production(ctx, [dict(row, afo_id="A2")], "A2", excretion_cache=cache)
    assert first == second


@pytest.mark.parametrize("mode", ["plet_rusle", "statistical"])
def test_mass_realization_uses_loaded_reference_and_preserves_nutrients(tmp_path, monkeypatch, mode):
    loaded = _load(_mass_package(tmp_path, weight=True), mode=mode)
    ctx = _sampling_ctx(loaded["excretion_factors"])
    ctx.afo_inputs = loaded
    ctx.parcel_ids = ["1", "2", "3", "4"]
    ctx.pathway_names = ["surface", "subsurface"]
    state = _state() if mode == "plet_rusle" else None
    ctx.current_pathway_load_rates = state.pathway_load_rates if state is not None else np.ones((4, 2, 2))
    ctx.pollutants = ["TN", "TP"]

    def unexpected_read(*args, **kwargs):
        pytest.fail("Reference CSV must not be read inside a realization")

    monkeypatch.setattr("src.afo.reference_excretion_factors", unexpected_read)
    _, ledger = mass_afo_realization(ctx, state=state)
    for generated, applied, retained in [
        ("animal_n_produced_kg", "tn_applied_kg", .8 * .9 * .7),
        ("animal_p_produced_kg", "tp_applied_kg", .8 * .9 * .9),
    ]:
        assert ledger[0][generated] == ledger[1][generated]
        assert sum(row[applied] for row in ledger) == pytest.approx(ledger[0][generated] * retained)
        assert ledger[0][applied] / ledger[1][applied] == pytest.approx(2)


def test_animal_overrides_replace_reference_distribution(tmp_path):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as conn:
        conn.execute("ALTER TABLE afo_animals ADD COLUMN tn_kg_1000kg_day REAL")
        conn.execute("UPDATE afo_animals SET tn_kg_1000kg_day=0.5")
        conn.execute("ALTER TABLE afo_animals ADD COLUMN tp_kg_1000kg_day_mean REAL")
        conn.execute("ALTER TABLE afo_animals ADD COLUMN tp_kg_1000kg_day_sd REAL")
        conn.execute("UPDATE afo_animals SET tp_kg_1000kg_day_mean=0.2, tp_kg_1000kg_day_sd=0")
    loaded = _load(package)
    ctx = _sampling_ctx(loaded["excretion_factors"])
    result = animal_production(ctx, loaded["animals"], "A1")
    assert result["TN"] == pytest.approx(10 * 365 * .36 * .5)
    assert result["TP"] == pytest.approx(10 * 365 * .36 * .2)


@pytest.mark.parametrize("columns,error", [
    ({"tn_kg_1000kg_day_mean": .5}, "mean\\+sd"),
    ({"tn_kg_1000kg_day_mean": .5, "tn_kg_1000kg_day_sd": -.1}, "sd must be"),
    ({"tn_kg_1000kg_day": -.5}, "physical domain"),
    ({"tn_kg_1000kg_day": .5, "tn_kg_1000kg_day_mean": .5, "tn_kg_1000kg_day_sd": .1}, "mixes fixed value"),
])
def test_invalid_animal_override_is_not_repaired_with_reference_values(tmp_path, columns, error):
    package = _mass_package(tmp_path)
    with sqlite3.connect(package) as conn:
        for column, value in columns.items():
            conn.execute(f'ALTER TABLE afo_animals ADD COLUMN "{column}" REAL')
            conn.execute(f'UPDATE afo_animals SET "{column}"=?', (value,))
    with pytest.raises(ValueError, match=error):
        _load(package)


def test_missing_custom_csv_does_not_fall_back_to_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_afo_inputs(_mass_package(tmp_path), _parcels(), ["TN", "TP"], method="mass_based",
                        excretion_factors_path=tmp_path / "missing.csv")


def test_concentration_increment_does_not_load_excretion_csv(tmp_path, monkeypatch):
    def unexpected_read(*args, **kwargs):
        pytest.fail("Concentration-increment mode does not use excretion rates")

    monkeypatch.setattr("src.afo.reference_excretion_factors", unexpected_read)
    load_afo_inputs(_package(tmp_path / "legacy.gpkg"), _parcels(), ["TN", "TP"])


def test_custom_csv_config_and_parallel_mass_simulations(inputs, tmp_path):
    cfg, paths, _, _ = inputs
    cfg.pop("huc_parameters")
    cfg["pollutants"] = ["TN", "TP"]
    write_tables(paths["domain"], input_pollutant_load_rate=[
        {"pollutant": pollutant, "pathway": "surface", "value": 1.}
        for pollutant in cfg["pollutants"]
    ])
    pd.DataFrame([
        {"cps": 340, "pollutant": pollutant, "pathway": "surface", "value": .5}
        for pollutant in cfg["pollutants"]
    ]).to_csv(cfg["bmp_efficiency"], index=False)
    table = _reference_table()
    table[["mean", "sd"]] *= 2
    _write_reference(tmp_path, table)
    cfg["animal_operations"] = str(_mass_package(tmp_path))
    cfg["manure_loading"] = {"method": "mass_based", "excretion_factors": "../rates.csv"}
    resolve_config_paths(cfg, tmp_path / "config" / "model.yaml")
    data = load_and_validate_all(cfg, Logger())
    assert data["afo_inputs"]["excretion_factors"]["beef"][PARAMETERS[0]] == pytest.approx(
        {"mean": .68, "sd": .146}
    )
    for n_jobs in (1, 2):
        cfg["parallel"] = {"n_jobs": n_jobs}
        cfg["outputs"] = str(tmp_path / f"run_{n_jobs}")
        Model(cfg, data, logging.getLogger(f"afo.excretion.{n_jobs}")).run_all_scenarios()
    for scenario in (1, 2):
        first = pd.read_parquet(tmp_path / "run_1" / "afo_mass_ledger" / f"s{scenario}.parquet")
        second = pd.read_parquet(tmp_path / "run_2" / "afo_mass_ledger" / f"s{scenario}.parquet")
        pd.testing.assert_frame_equal(first, second)
    first_scenario = pd.read_parquet(tmp_path / "run_1" / "afo_mass_ledger" / "s1.parquet")
    second_scenario = pd.read_parquet(tmp_path / "run_1" / "afo_mass_ledger" / "s2.parquet")
    assert first_scenario.iloc[0].animal_n_produced_kg != second_scenario.iloc[0].animal_n_produced_kg
