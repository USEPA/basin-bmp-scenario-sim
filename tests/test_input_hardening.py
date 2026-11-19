from __future__ import annotations

import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src import input_config
from src.input_config import (
    _ensure_projected,
    _load_cps,
    _load_plet_hydrology_lookup,
    _merge_csvs,
    _normalize_integer_identifier_value,
    _plet_parameter_defaults,
    _validate_input_package_schema,
)
from src.io_utils import MissingInputTableError, read_geopackage_table, read_geopackage_table_info


class Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def verbose(self, *args, **kwargs):
        pass


def _write_table(path: Path, name: str, frame: pd.DataFrame) -> None:
    with sqlite3.connect(path) as con:
        frame.to_sql(name, con, if_exists="replace", index=False)


def _write_synthetic_schema_v3_package(path: Path, parcel_count: int = 5) -> None:
    parcels = gpd.GeoDataFrame(
        geometry=[
            box(index * 2, 0, index * 2 + 1, 1)
            for index in range(parcel_count)
        ],
        crs="EPSG:5070",
    )
    parcels.to_file(path, layer="parcels", driver="GPKG")
    with sqlite3.connect(path) as con:
        con.execute("ALTER TABLE parcels RENAME COLUMN fid TO pid")
        con.execute(
            "CREATE TABLE model_input_schema (schema_version INTEGER NOT NULL)"
        )
        con.execute("INSERT INTO model_input_schema(schema_version) VALUES (3)")
        for table in ("input_annual_precip_in", "input_avg_rain_in"):
            con.execute(
                f"CREATE TABLE {table} "
                "(id INTEGER PRIMARY KEY, pid INTEGER, value REAL)"
            )
            con.execute(f"INSERT INTO {table}(pid,value) VALUES (NULL,42)")


def test_required_plet_parameters_are_not_synthesized_by_defaults() -> None:
    defaults = _plet_parameter_defaults(["TN"])
    for required in (
        "annual_precip_in",
        "rain_days",
        "rain_correction_fraction",
        "runoff_day_fraction",
        "land_cover",
        "hsg",
    ):
        assert required not in defaults


def test_cps_rejects_fractional_and_boolean_values() -> None:
    with pytest.raises(ValueError, match="finite integer"):
        _load_cps({"cps": [340.5]})
    with pytest.raises(ValueError, match="boolean"):
        _load_cps({"cps": [True]})


def test_merge_csvs_rejects_conflicting_logical_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "eff.csv"
    pd.DataFrame(
        [
            {"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.1},
            {"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.9},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _merge_csvs(path, ["cps", "pollutant"], "bmp_efficiency", Logger())


def test_merge_csvs_collapses_exact_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "eff.csv"
    row = {"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.1}
    pd.DataFrame([row, row]).to_csv(path, index=False)
    loaded = _merge_csvs(path, ["cps", "pollutant"], "bmp_efficiency", Logger())
    assert len(loaded) == 1


def test_unknown_input_table_is_rejected_with_suggestion(tmp_path: Path) -> None:
    package = tmp_path / "parcels.gpkg"
    _write_table(package, "input_anual_precip_in", pd.DataFrame([{"pid": None, "value": 42.0}]))
    with pytest.raises(ValueError, match="input_anual_precip_in.*did you mean.*input_annual_precip_in"):
        _validate_input_package_schema(
            {"parcels": str(package)}, "plet_rusle", Logger()
        )


def test_missing_geopackage_table_has_specific_exception(tmp_path: Path) -> None:
    package = tmp_path / "parcels.gpkg"
    _write_table(package, "present", pd.DataFrame([{"x": 1}]))
    with pytest.raises(MissingInputTableError):
        read_geopackage_table(package, "missing")


def test_projected_feet_are_reprojected_to_metric_analysis_crs() -> None:
    geographic = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(-82.95, 39.98)], crs="EPSG:4326"
    )
    # Ohio South State Plane uses US survey feet.
    feet = geographic.to_crs("EPSG:3735")
    assert feet.crs.is_projected
    assert any("foot" in axis.unit_name.lower() for axis in feet.crs.axis_info)

    metric = _ensure_projected(feet, Logger())
    assert metric.crs.is_projected
    assert all(abs(float(axis.unit_conversion_factor) - 1.0) < 1e-12 for axis in metric.crs.axis_info[:2])


def test_hydrology_lookup_can_be_reusable_but_only_used_pairs_must_be_complete() -> None:
    table = pd.DataFrame(
        [
            {"land_cover": "cropland", "hsg": "B", "parameter": "cn", "value": 78.0},
            {"land_cover": "cropland", "hsg": "B", "parameter": "infiltration_fraction", "value": 0.30},
            # Extra incomplete unused pairing should not invalidate this run.
            {"land_cover": "forest", "hsg": "A", "parameter": "cn", "value": 39.0},
        ]
    )
    loaded = _load_plet_hydrology_lookup(
        table, Logger(), required_pairs=[("cropland", "B")]
    )
    assert len(loaded) == 3



def test_clipped_structural_relationships_are_filtered_but_true_unknown_ids_error() -> None:
    rows = pd.DataFrame(
        [
            {"pid": "P2", "pid_down": "P1"},
            {"pid": "P3", "pid_down": "P2"},  # P3 exists in source but is outside domain.
        ]
    )
    result = input_config._build_parcel_down_map(
        rows,
        ["P1", "P2"],
        source_parcel_ids=["P1", "P2", "P3"],
        logger=Logger(),
    )
    assert result["P2"] == "P1"

    bad = pd.DataFrame([{"pid": "P2", "pid_down": "TYPO"}])
    with pytest.raises(ValueError, match="source parcels"):
        input_config._build_parcel_down_map(
            bad,
            ["P1", "P2"],
            source_parcel_ids=["P1", "P2", "P3"],
            logger=Logger(),
        )


def test_validate_config_rejects_unknown_keys() -> None:
    from src.input_validation import validate_config

    cfg = input_config.normalize_config({"n_scenarios": 1, "mdoe": "plet_rusle"})
    with pytest.raises(ValueError, match="Unknown configuration key.*mdoe"):
        validate_config(cfg)


def test_validate_config_rejects_unknown_load_generation_keys() -> None:
    from src.input_validation import validate_config

    cfg = input_config.normalize_config(
        {"n_scenarios": 1, "load_generation": {"mode": "plet_rusle", "mdoe": "x"}}
    )
    with pytest.raises(ValueError, match="Unknown load_generation key.*mdoe"):
        validate_config(cfg)


def test_schema_v3_rejects_legacy_star_pid_and_accepts_null_default() -> None:
    with pytest.raises(ValueError, match=r"legacy '\*' parcel default"):
        _normalize_integer_identifier_value("*", "pid", allow_null=True)
    assert _normalize_integer_identifier_value(None, "pid", allow_null=True) is None
    assert _normalize_integer_identifier_value("42", "pid") == 42


def test_synthetic_geopackage_uses_integer_pid_primary_key_and_editable_attribute_ids(
    tmp_path: Path,
) -> None:
    package = tmp_path / "synthetic_parcels.gpkg"
    _write_synthetic_schema_v3_package(package)

    parcels = {row["name"].lower(): row for row in read_geopackage_table_info(package, "parcels")}
    assert "fid" not in parcels
    assert parcels["pid"]["pk"] == 1
    assert "INT" in parcels["pid"]["type"].upper()

    annual = {row["name"].lower(): row for row in read_geopackage_table_info(package, "input_annual_precip_in")}
    assert annual["id"]["pk"] == 1
    assert "INT" in annual["pid"]["type"].upper()

    avg_rain = {row["name"].lower(): row for row in read_geopackage_table_info(package, "input_avg_rain_in")}
    assert avg_rain["id"]["pk"] == 1
    assert "INT" in avg_rain["pid"]["type"].upper()

    with sqlite3.connect(package) as con:
        assert con.execute("SELECT schema_version FROM model_input_schema").fetchone()[0] == 3
        default_pid = con.execute("SELECT pid FROM input_annual_precip_in LIMIT 1").fetchone()[0]
        assert default_pid is None
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []


def test_load_parcels_recovers_pid_when_reader_hides_feature_id(
    monkeypatch, tmp_path: Path,
) -> None:
    """Schema-v3 pid must survive GeoPandas/Fiona versions that hide the FID."""
    package = tmp_path / "synthetic_parcels.gpkg"
    _write_synthetic_schema_v3_package(package)
    domain = gpd.GeoDataFrame(
        geometry=[box(-1, -1, 20, 2)], crs="EPSG:5070"
    )
    domain = input_config._ensure_projected(domain, Logger())

    def read_without_feature_id(path, *, layer=None, fid_as_index=False):
        # Mimic readers that accept/ignore fid_as_index but expose only geometry
        # when the GeoPackage INTEGER PRIMARY KEY is used as the feature ID.
        return gpd.read_file(path, layer=layer)

    monkeypatch.setattr(input_config, "read_geodataframe", read_without_feature_id)
    parcels = input_config._load_parcels({"parcels": str(package)}, domain, Logger())

    assert parcels["pid"].dtype.kind in "iu"
    assert parcels["pid"].is_unique
    assert parcels["pid"].iloc[:5].tolist() == [1, 2, 3, 4, 5]


def test_gpkg_reader_accepts_legacy_registry_wildcard_field(monkeypatch, tmp_path) -> None:
    """A partially updated tree must not crash on the old registry field name."""
    from types import SimpleNamespace

    fake_spec = SimpleNamespace(wildcard_allowed=True, key_columns=("pid",))
    monkeypatch.setitem(input_config.INPUT_VARIABLE_SPECS, "input_fake", fake_spec)
    monkeypatch.setattr(
        input_config,
        "read_geopackage_table",
        lambda package, table: pd.DataFrame([{"pid": None, "value": 1.0}]),
    )
    package = tmp_path / "inputs.gpkg"
    package.touch()

    result = input_config._read_gpkg_input_table(
        package, "input_fake", ["pid"], "input_fake", Logger()
    )
    assert pd.isna(result.loc[0, "pid"])


def test_plet_validation_applies_null_classification_defaults_to_integer_pids() -> None:
    """Schema-v3 NULL defaults must satisfy classifications for every integer parcel."""
    from src.input_validation import validate_plet_input_table

    table = pd.DataFrame(
        [
            {"pid": None, "parameter": "land_cover", "value": "cropland"},
            {"pid": None, "parameter": "hsg", "value": "B"},
            {"pid": None, "parameter": "annual_precip_in", "value": 42.0},
        ]
    )
    validated = validate_plet_input_table(table, [1, 2])
    classes = validated[validated["parameter"].isin(["land_cover", "hsg"])]
    assert classes["pid"].isna().all()
    assert set(classes["parameter"]) == {"land_cover", "hsg"}


def test_plet_concentration_sampling_applies_null_defaults_to_integer_pids() -> None:
    """Runtime concentration sampling must honor schema-v3 NULL parcel defaults."""
    from src.plet_rusle import _sample_concentrations

    class FixedContext:
        @staticmethod
        def _sample_from_stats(stats, kind=None):
            return float(stats["value"])

    table = pd.DataFrame(
        [
            {"pid": None, "pollutant": "TN", "value": 4.4},
            {"pid": None, "pollutant": "TP", "value": 0.35},
            {"pid": 2, "pollutant": "TN", "value": 5.0},
        ]
    )
    sampled = _sample_concentrations(FixedContext(), table, [1, 2])
    assert sampled[0] == {"TN": pytest.approx(4.4), "TP": pytest.approx(0.35)}
    assert sampled[1] == {"TN": pytest.approx(5.0), "TP": pytest.approx(0.35)}
