from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src import input_config, sampling
from src.input_scopes import (
    assign_parcels_to_hucs,
    load_scoped_input_tables,
    normalize_huc_id,
    validate_huc_polygons,
)
from src.input_validation import validate_config
from src.model import Model
from src.plet_rusle import _sample_parameter_table, _sample_plet_hydrology
from utils.create_parcel_huc import write_parcel_huc


class Logger:
    def verbose(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


class Ctx:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)

    _sample_from_stats = sampling._sample_from_stats
    _trunc_normal = sampling._trunc_normal
    _piecewise_quantile_sample = sampling._piecewise_quantile_sample


def write_tables(path, **tables):
    with sqlite3.connect(path) as con:
        for name, rows in tables.items():
            frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
            frame.to_sql(name, con, index=False, if_exists="replace")


@pytest.fixture
def inputs(tmp_path):
    crs = "EPSG:32617"
    parcels = gpd.GeoDataFrame(
        {"pid": [1, 2, 3, 4, 5]},
        geometry=[box(500000 + i * 100, 4300000, 500090 + i * 100, 4300090) for i in range(5)],
        crs=crs,
    )
    hucs = gpd.GeoDataFrame(
        {"huc_id": ["05090202", "05090203"]},
        geometry=[box(499900, 4299900, 500200, 4300200), box(500200, 4299900, 500600, 4300200)],
        crs=crs,
    )
    paths = {name: tmp_path / f"{name}.gpkg" for name in ("parcels", "domain", "huc_parameters", "outlets")}
    parcels.to_file(paths["parcels"], layer="parcels", engine="pyogrio", FID="pid")
    hucs.to_file(paths["huc_parameters"], layer="hucs", engine="pyogrio")
    gpd.GeoDataFrame(geometry=[box(499800, 4299800, 500800, 4300300)], crs=crs).to_file(
        paths["domain"], layer="domain", engine="pyogrio",
    )
    gpd.GeoDataFrame({"oid": ["A"]}, geometry=[Point(500500, 4300100)], crs=crs).to_file(
        paths["outlets"], layer="outlets", engine="pyogrio",
    )
    write_tables(
        paths["parcels"],
        parcel_down=pd.DataFrame(columns=["pid", "pid_down"]),
        parcel_outlets=[{"pid": pid, "oid": "A"} for pid in parcels.pid],
    )
    efficiency = tmp_path / "efficiency.csv"
    pd.DataFrame([{"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.5}]).to_csv(efficiency, index=False)
    cfg = {**{key: str(path) for key, path in paths.items()}, "pollutants": ["TN"], "cps": [340],
           "bmp_efficiency": str(efficiency), "n_scenarios": 2, "bmp_limit_n": 1,
           "random_seed": 7, "outputs": str(tmp_path / "outputs")}
    return cfg, paths, parcels, hucs


def resolve(cfg, parcels):
    return load_scoped_input_tables(cfg, parcels, Logger(), input_config._read_gpkg_input_table)


def test_precedence_is_per_parameter_and_preserves_distributions(inputs):
    cfg, paths, parcels, _ = inputs
    write_tables(paths["parcels"], input_pollutant_load_rate=[
        {"pid": None, "pollutant": "TN", "value": 1.0},
        {"pid": 1, "pollutant": "TN", "value": 4.0},
        {"pid": None, "pollutant": "TP", "value": 5.0},
    ])
    write_tables(paths["domain"], input_pollutant_load_rate=[
        {"pollutant": "TN", "mean": 2.0, "sd": 0.1, "units": "kg/ha/year"},
    ])
    write_tables(paths["huc_parameters"], input_pollutant_load_rate=[
        {"huc_id": "05090202", "pollutant": "TN", "mean": 3.0, "sd": 0.2, "units": "kg/ha/year"},
    ])
    table = resolve(cfg, parcels)["input_pollutant_load_rate"].set_index(["pid", "pollutant"])
    assert table.loc[(1, "TN"), "value"] == 4
    assert pd.isna(table.loc[(1, "TN"), "mean"])
    assert table.loc[(2, "TN"), "mean"] == 3
    assert table.loc[(2, "TN"), "sd"] == 0.2
    assert table.loc[(3, "TN"), "mean"] == 2
    assert table.loc[(4, "TN"), "sample_group"] == table.loc[(3, "TN"), "sample_group"]
    assert table.loc[(2, "TN"), "sample_group"] != table.loc[(3, "TN"), "sample_group"]
    assert table.loc[(5, "TP"), "value"] == 5


@pytest.mark.parametrize("pathways", [False, True])
def test_statistical_domain_only_run_shares_draws_and_is_reproducible(inputs, pathways):
    cfg, paths, _, _ = inputs
    cfg.pop("huc_parameters")
    rows = [{"pollutant": "TN", "mean": 15.0, "sd": 2.0}]
    if pathways:
        rows[0]["pathway"] = "surface"
    write_tables(paths["domain"], input_pollutant_load_rate=rows)
    data = input_config.load_and_validate_all(cfg, Logger())
    rates = data["pollutant_load_rate"]
    assert len(rates) == 5
    assert rates.sample_group.nunique() == 1
    serial = Path(cfg["outputs"])
    Model(cfg, data, logging.getLogger("scope.serial")).run_all_scenarios()
    first = pd.read_parquet(serial / "parcels" / "s1.parquet")
    second = pd.read_parquet(serial / "parcels" / "s2.parquet")
    col = "baseline_load_rate_TN_kg_ha_yr"
    assert first[col].nunique() == 1
    assert first[col].iloc[0] != second[col].iloc[0]
    cfg["outputs"] = str(serial.parent / "parallel")
    cfg["parallel"] = {"n_jobs": 2}
    data["parallel"] = cfg["parallel"]
    Model(cfg, data, logging.getLogger("scope.parallel")).run_all_scenarios()
    for scenario in (1, 2):
        pd.testing.assert_frame_equal(
            pd.read_parquet(serial / "parcels" / f"s{scenario}.parquet"),
            pd.read_parquet(Path(cfg["outputs"]) / "parcels" / f"s{scenario}.parquet"),
        )


def plet_domain_tables():
    tables = {
        "input_land_cover": [{"value": "cropland"}],
        "input_hsg": [{"value": "B"}],
        "input_avg_rain_in": [{"value": 0.6}],
        "input_rain_days": [{"value": 100.0}],
        "input_runoff_day_fraction": [{"value": 0.4}],
        "input_curve_number": [{"land_cover": "cropland", "hsg": "B", "value": 70.0}],
        "input_infiltration_fraction": [{"land_cover": "cropland", "hsg": "B", "value": 0.3}],
        "input_surface_concentration": [{"pollutant": "TN", "value": 2.0}],
        "input_subsurface_concentration": [{"pollutant": "TN", "value": 1.0}],
    }
    for parameter, value in (("r", 100.0), ("k", 0.2), ("ls", 1.5), ("c", 0.1), ("p", 0.5)):
        tables[f"input_rusle_{parameter}"] = [{"value": value}]
    return tables


def test_plet_domain_and_huc_parameters_reach_runtime(inputs):
    cfg, paths, parcels, _ = inputs
    cfg["load_generation"] = {"mode": "plet_rusle"}
    write_tables(paths["domain"], **plet_domain_tables())
    write_tables(paths["huc_parameters"],
        input_avg_rain_in=[{"huc_id": "05090202", "mean": 0.7, "sd": 0.01}],
        input_curve_number=[{"huc_id": "05090202", "land_cover": "cropland", "hsg": "B", "mean": 80, "sd": 1}],
        input_rusle_r=[{"huc_id": "05090202", "land_cover": "cropland", "value": 120.0}],
        input_surface_concentration=[{"huc_id": "05090202", "land_cover": "cropland", "pollutant": "TN", "value": 4.0}],
    )
    write_tables(paths["parcels"],
        input_avg_rain_in=[{"pid": 1, "value": 0.9}],
        input_curve_number=[{"pid": 1, "land_cover": "cropland", "hsg": "B", "value": 90.0}],
    )
    data = input_config.load_and_validate_all(cfg, Logger())
    ctx = Ctx()
    ctx.load_generation = data["load_generation"]
    cache = {}
    assert _sample_plet_hydrology(ctx, "cropland", "B", pid="1", cache=cache)["cn"] == 90
    assert 70 < _sample_plet_hydrology(ctx, "cropland", "B", pid="2", cache=cache)["cn"] < 90
    assert _sample_plet_hydrology(ctx, "cropland", "B", pid="3", cache=cache)["cn"] == 70
    parameters = _sample_parameter_table(ctx, data["plet_inputs"], list(map(str, parcels.pid)), cache_prefix="plet")
    assert parameters[0]["avg_rain_in"] == 0.9
    assert parameters[2]["avg_rain_in"] == 0.6
    surface = data["pollutant_concentrations"].set_index("pid")
    assert surface.loc[1, "value"] == 4.0
    assert surface.loc[3, "value"] == 2.0
    Model(cfg, data, logging.getLogger("scope.plet")).run_all_scenarios()
    realized = pd.read_parquet(Path(cfg["outputs"]) / "load_parameters" / "s1.parquet")
    assert len(realized) == 5


@pytest.mark.parametrize("level", [2, 4, 6, 8, 10, 12, 14, 16])
def test_huc_levels_and_assignment_round_trip(inputs, level):
    cfg, paths, parcels, hucs = inputs
    hucs["huc_id"] = ["0" * (level - 1) + "1", "0" * (level - 1) + "2"]
    assignments = assign_parcels_to_hucs(parcels, hucs)
    assert assignments.huc_id.tolist() == [hucs.huc_id.iloc[0]] * 2 + [hucs.huc_id.iloc[1]] * 3
    assert np.allclose(assignments.area_fraction, 1)
    write_parcel_huc(paths["parcels"], assignments)
    with sqlite3.connect(paths["parcels"]) as con:
        got = pd.read_sql_query("SELECT pid, huc_id, area_fraction FROM parcel_huc ORDER BY pid", con)
    pd.testing.assert_frame_equal(got, assignments)


@pytest.mark.parametrize("value", [5090202, None, "", "*", "050", "a" * 8, "0" * 18])
def test_invalid_huc_ids_are_rejected(value):
    with pytest.raises(ValueError):
        normalize_huc_id(value)


def test_mixed_levels_and_overlapping_hucs_rejected(inputs):
    _, _, _, hucs = inputs
    mixed = hucs.copy()
    mixed.loc[0, "huc_id"] = "050902"
    with pytest.raises(ValueError, match="mixed/nested"):
        validate_huc_polygons(mixed)
    overlap = hucs.copy()
    overlap.loc[1, "geometry"] = overlap.geometry.iloc[0]
    with pytest.raises(ValueError, match="overlap"):
        validate_huc_polygons(overlap)


def test_invalid_scope_rows_are_not_hidden_by_parcel_overrides(inputs):
    cfg, paths, parcels, _ = inputs
    write_tables(paths["parcels"], input_rain_days=[{"pid": pid, "value": 5.0} for pid in parcels.pid])
    write_tables(paths["domain"], input_rain_days=[{"value": -1.0}])
    with pytest.raises(ValueError, match="physical domain"):
        resolve(cfg, parcels)
    write_tables(paths["domain"], input_rain_days=[{"mean": 3.0}])
    with pytest.raises(ValueError, match=r"mean\+sd"):
        resolve(cfg, parcels)


def test_missing_and_unknown_huc_assignments_fail(inputs):
    cfg, paths, parcels, _ = inputs
    write_tables(paths["parcels"], parcel_huc=[{"pid": 1, "huc_id": "05090202"}])
    with pytest.raises(ValueError, match="every modeled parcel"):
        resolve(cfg, parcels)
    write_tables(paths["parcels"], parcel_huc=[{"pid": pid, "huc_id": "99999999"} for pid in parcels.pid])
    with pytest.raises(ValueError, match="unknown HUC"):
        resolve(cfg, parcels)


def test_unknown_table_and_unknown_parameter_huc_fail(inputs):
    cfg, paths, parcels, _ = inputs
    write_tables(paths["domain"], input_typo=[{"value": 1.0}])
    with pytest.raises(ValueError, match="Unknown input_"):
        resolve(cfg, parcels)
    with sqlite3.connect(paths["domain"]) as con:
        con.execute("DROP TABLE input_typo")
    write_tables(paths["huc_parameters"], input_rain_days=[{"huc_id": "99999999", "value": 3}])
    with pytest.raises(ValueError, match="absent from the spatial layer"):
        resolve(cfg, parcels)


def test_config_alias_conflict_and_new_relative_path(inputs):
    cfg, paths, _, _ = inputs
    cfg["plet_forcing"] = cfg["huc_parameters"]
    input_config.normalize_config(cfg)
    with pytest.raises(ValueError, match="not both"):
        validate_config(cfg)
    relative = {"huc_parameters": "../hucs/custom.gpkg"}
    input_config.resolve_config_paths(relative, paths["domain"].parent / "config" / "run.yaml")
    assert Path(relative["huc_parameters"]) == paths["domain"].parent / "hucs" / "custom.gpkg"


def test_selection_and_delivery_tables_obey_scope_precedence(inputs):
    cfg, paths, parcels, _ = inputs
    write_tables(paths["parcels"],
        input_selection_weight=[{"pid": None, "value": 1}, {"pid": 1, "value": 4}],
        input_sdr_f_to_s=[{"pid": None, "oid": None, "value": 0.1}, {"pid": 1, "oid": "A", "value": 0.4}],
    )
    write_tables(paths["domain"],
        input_selection_weight=[{"value": 2}],
        input_sdr_f_to_s=[{"oid": None, "value": 0.2}],
    )
    write_tables(paths["huc_parameters"],
        input_selection_weight=[{"huc_id": "05090202", "value": 3}],
        input_sdr_f_to_s=[{"huc_id": "05090202", "oid": None, "value": 0.3}],
    )
    tables = resolve(cfg, parcels)
    assert tables["input_selection_weight"].set_index("pid").value.to_dict() == {1: 4, 2: 3, 3: 2, 4: 2, 5: 2}
    delivery = input_config._load_delivery_ratios(cfg, Logger(), input_tables=tables)
    assert delivery.set_index("pid").sdr_f_to_s.to_dict() == {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.2, 5: 0.2}


def test_statistical_huc_draws_are_shared_only_within_each_huc(inputs):
    cfg, paths, _, _ = inputs
    write_tables(paths["huc_parameters"], input_pollutant_load_rate=[
        {"huc_id": code, "pollutant": "TN", "mean": 20, "sd": 2}
        for code in ("05090202", "05090203")
    ])
    data = input_config.load_and_validate_all(cfg, Logger())
    Model(cfg, data, logging.getLogger("scope.huc-statistical")).run_all_scenarios()
    first = pd.read_parquet(Path(cfg["outputs"]) / "parcels" / "s1.parquet").set_index("pid")
    values = first["baseline_load_rate_TN_kg_ha_yr"]
    values.index = values.index.astype(int)
    assert values.loc[1] == values.loc[2]
    assert values.loc[3] == values.loc[4] == values.loc[5]
    assert values.loc[1] != values.loc[3]


def test_legacy_huc12_alias_and_mapping_preserve_parameter_values(inputs):
    cfg, paths, parcels, hucs = inputs
    legacy = paths["huc_parameters"].with_name("legacy.gpkg")
    hucs = hucs.rename(columns={"huc_id": "huc12"})
    hucs["huc12"] = ["050902021001", "050902031001"]
    hucs.to_file(legacy, layer="huc12", engine="pyogrio")
    cfg.pop("huc_parameters")
    cfg["plet_forcing"] = str(legacy)
    write_tables(legacy, input_avg_rain_in=[
        {"huc12": "050902021001", "value": 0.6},
        {"huc12": "050902031001", "value": 0.7},
    ])
    write_tables(paths["parcels"],
        parcel_huc12=[{"pid": pid, "huc12": hucs.huc12.iloc[0 if pid <= 2 else 1]} for pid in parcels.pid],
        input_avg_rain_in=[{"pid": None, "value": 0.5}, {"pid": 1, "value": 0.9}],
    )
    old_parcel = input_config._assemble_parcel_parameter_source(cfg, Logger())
    old_hucs = input_config._assemble_huc12_plet_parameter_source(cfg, hucs, Logger())
    mapping = input_config._load_parcel_huc12(cfg, parcels.pid, parcels.pid, Logger())
    previous = input_config._merge_huc12_plet_parameters(old_parcel, mapping, old_hucs, parcels.pid)
    previous = previous[previous.pid.notna()].set_index("pid").value.to_dict()
    current = resolve(cfg, parcels)["input_avg_rain_in"].set_index("pid").value.to_dict()
    assert current == previous
    cfg["huc_parameters"] = cfg.pop("plet_forcing")
    assert resolve(cfg, parcels)["input_avg_rain_in"].set_index("pid").value.to_dict() == current


def test_explicit_sample_group_and_unit_conversion_survive_inheritance(inputs):
    cfg, paths, parcels, _ = inputs
    write_tables(paths["domain"], input_avg_rain_in=[
        {"mean": 25.4, "sd": 0.0, "units": "mm/event", "sample_group": "regional-rain"},
    ])
    tables = resolve(cfg, parcels)
    assert set(tables["input_avg_rain_in"].sample_group) == {"regional-rain"}
    source = input_config._assemble_parcel_parameter_source(cfg, Logger(), input_tables=tables)
    values = _sample_parameter_table(Ctx(), source, list(map(str, parcels.pid)), cache_prefix="plet")
    assert all(value["avg_rain_in"] == pytest.approx(1) for value in values)


@pytest.mark.parametrize("scope", ["domain", "huc_parameters"])
def test_duplicate_rows_fail_before_inheritance(inputs, scope):
    cfg, paths, parcels, _ = inputs
    row = {"value": 1}
    if scope == "huc_parameters":
        row["huc_id"] = "05090202"
    write_tables(paths[scope], input_rain_days=[row, row])
    with pytest.raises(ValueError, match="one row|duplicate"):
        resolve(cfg, parcels)


def test_partial_null_parcel_delivery_key_stays_invalid_with_domain_inputs(inputs):
    cfg, paths, parcels, _ = inputs
    write_tables(paths["parcels"], input_sdr_f_to_s=[{"pid": 1, "oid": None, "value": 0.5}])
    write_tables(paths["domain"], input_sdr_f_to_s=[{"oid": None, "value": 0.7}])
    with pytest.raises(ValueError, match="both pid and oid NULL"):
        resolve(cfg, parcels)


def test_statistical_pathway_shapes_cannot_be_mixed_between_scopes(inputs):
    cfg, paths, parcels, _ = inputs
    write_tables(paths["domain"], input_pollutant_load_rate=[{"pollutant": "TN", "value": 2}])
    write_tables(paths["huc_parameters"], input_pollutant_load_rate=[
        {"huc_id": "05090202", "pollutant": "TN", "pathway": "surface", "value": 3},
    ])
    with pytest.raises(ValueError, match="cannot mix aggregate"):
        resolve(cfg, parcels)


def test_no_huc_intersection_is_an_error(inputs):
    _, _, parcels, hucs = inputs
    with pytest.raises(ValueError, match="missing pid"):
        assign_parcels_to_hucs(parcels, hucs.iloc[:1])


def test_low_overlap_warning_is_file_only(inputs):
    cfg, paths, parcels, _ = inputs
    write_tables(paths["parcels"], parcel_huc=[
        {"pid": pid, "huc_id": "05090202" if pid <= 2 else "05090203", "area_fraction": 0.6}
        for pid in parcels.pid
    ])
    warnings = []

    class RecordingLogger(Logger):
        def warning(self, message, **kwargs):
            warnings.append((message, kwargs))

    load_scoped_input_tables(cfg, parcels, RecordingLogger(), input_config._read_gpkg_input_table)
    assert len(warnings) == 1
    assert "less than 75%" in warnings[0][0]
    assert warnings[0][1] == {"extra": {"file_only": True}}


def test_scoped_inputs_ignore_rows_from_source_parcels_outside_domain(inputs):
    """Clipped source parcels must not reappear through inherited tables."""
    cfg, paths, parcels, _ = inputs
    parcels.attrs["source_pid_universe"] = set(parcels.pid) | {6}
    write_tables(paths["parcels"], input_selection_weight=[
        {"pid": 1, "value": 4.0},
        {"pid": 6, "value": 10.0},
    ])
    write_tables(paths["domain"], input_selection_weight=[{"value": 1.0}])
    scoped = resolve(cfg, parcels)["input_selection_weight"]
    assert set(scoped.pid) == set(parcels.pid)
    assert scoped.set_index("pid").loc[1, "value"] == 4.0
    assert scoped.set_index("pid").loc[2, "value"] == 1.0


def test_huc_assignment_rejects_duplicate_or_invalid_parcel_geometry(inputs):
    """Assignment requires one valid, areal polygon for every parcel ID."""
    _, _, parcels, hucs = inputs
    duplicate = parcels.copy()
    duplicate.loc[1, "pid"] = duplicate.loc[0, "pid"]
    with pytest.raises(ValueError, match="duplicate|one row|unique"):
        assign_parcels_to_hucs(duplicate, hucs)
    invalid = parcels.copy()
    invalid.loc[0, "geometry"] = Point(500000, 4300000)
    with pytest.raises(ValueError, match="polygon geometries"):
        assign_parcels_to_hucs(invalid, hucs)


def write_registered_domain_inputs(path):
    """Create representative scope tables in a temporary GeoPackage.

    Registering the tables as GeoPackage attributes verifies their physical
    format without depending on shipped project data.
    """
    from src.input_schema import INPUT_VARIABLE_SPECS

    rows_by_table = {
        "input_hsg": [{"value": "B", "units": "classification"}],
        "input_selection_weight": [{"value": 1.0, "units": "relative_weight"}],
        "input_avg_rain_in": [{"mean": 0.6, "sd": 0.05, "units": "in/event"}],
        "input_curve_number": [
            {"land_cover": "cropland", "hsg": "B", "value": 78.0, "units": "dimensionless"},
            {"land_cover": "cropland", "hsg": "C", "value": 85.0, "units": "dimensionless"},
        ],
        "input_pollutant_load_rate": [
            {"pollutant": "TN", "mean": 25.0, "sd": 2.0, "units": "kg/ha/year"},
            {"pollutant": "TP", "mean": 1.3, "sd": 0.13, "units": "kg/ha/year"},
            {"pollutant": "TSS", "mean": 250.0, "sd": 25.0, "units": "kg/ha/year"},
        ],
    }
    with sqlite3.connect(path) as con:
        for table_name, rows in rows_by_table.items():
            spec = INPUT_VARIABLE_SPECS[table_name]
            keys = [key for key in spec.key_columns if key not in ("pid", "huc_id")]
            columns = keys + ["value", "mean", "sd", "sample_group", "units"]
            text_columns = {"pollutant", "land_cover", "hsg", "sample_group", "units"}
            definitions = ", ".join(
                f'"{column}" {"TEXT" if column in text_columns else "REAL"}'
                for column in columns
            )
            con.execute(
                f'CREATE TABLE "{table_name}" '
                f'(id INTEGER PRIMARY KEY AUTOINCREMENT, {definitions})'
            )
            names = ", ".join(f'"{column}"' for column in columns)
            placeholders = ", ".join("?" for _ in columns)
            con.executemany(
                f'INSERT INTO "{table_name}" ({names}) VALUES ({placeholders})',
                [[row.get(column) for column in columns] for row in rows],
            )
            con.execute(
                "INSERT INTO gpkg_contents (table_name, data_type, identifier) "
                "VALUES (?, 'attributes', ?)",
                (table_name, table_name),
            )
    return set(rows_by_table)


def test_synthetic_domain_tables_follow_registry_and_geopackage_format(inputs):
    """Generated domain tables use the registered model schema and GPKG format."""
    from src.input_schema import INPUT_VARIABLE_SPECS

    _, paths, _, _ = inputs
    expected_tables = write_registered_domain_inputs(paths["domain"])
    with sqlite3.connect(paths["domain"]) as con:
        tables = {name for (name,) in con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='attributes'"
        )}
        assert expected_tables.issubset(tables)
        for name in expected_tables:
            spec = INPUT_VARIABLE_SPECS[name]
            columns = {col[1]: col for col in con.execute(f'PRAGMA table_info("{name}")')}
            assert columns["id"][5] == 1
            assert "pid" not in columns and "huc_id" not in columns
            assert {key for key in spec.key_columns if key != "pid"}.issubset(columns)
            assert {"value", "mean", "sd", "sample_group", "units"}.issubset(columns)
            assert con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] > 0
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_synthetic_domain_tables_resolve_at_expected_precedence(inputs):
    """Resolve generated defaults with the production reader and parcel overrides."""
    cfg, paths, parcels, _ = inputs
    write_registered_domain_inputs(paths["domain"])
    write_tables(paths["parcels"],
        input_land_cover=[{"pid": None, "value": "cropland"}],
        input_selection_weight=[{"pid": 1, "value": 3.0}],
        input_pollutant_load_rate=[{"pid": 1, "pollutant": "TN", "value": 30.0}],
    )
    tables = resolve(cfg, parcels)
    selection = tables["input_selection_weight"].set_index("pid")
    assert selection.loc[1, "value"] == 3.0
    assert selection.loc[2, "value"] == 1.0
    assert set(tables["input_hsg"].value) == {"B"}
    assert set(tables["input_curve_number"].value) == {78.0, 85.0}
    rainfall = tables["input_avg_rain_in"]
    assert rainfall["sample_group"].nunique() == 1
    assert rainfall["mean"].eq(0.6).all()
    rate = tables["input_pollutant_load_rate"].set_index(["pid", "pollutant"])
    assert rate.loc[(1, "TN"), "value"] == 30.0
    assert rate.loc[(2, "TN"), "mean"] == 25.0
