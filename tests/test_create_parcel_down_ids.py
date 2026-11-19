"""GeoPackage primary-key / layer selection regression tests."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

pyogrio = pytest.importorskip("pyogrio")

from utils import create_parcel_down


def _write_parcels(path: Path, *, fid_primary=True, layer="parcels"):
    parcels = gpd.GeoDataFrame(
        {"pid": [11, 33], "name": ["first", "second"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    kwargs = {"layer_options": {"FID": "pid"}} if fid_primary else {}
    pyogrio.write_dataframe(parcels, path, driver="GPKG", layer=layer, **kwargs)


def test_gpkg_pid_stored_as_fid_is_restored(tmp_path):
    path = tmp_path / "input.gpkg"
    _write_parcels(path)
    # Pyogrio hides the pid FID column when it returns attributes normally.
    raw = gpd.read_file(path, layer="parcels")
    assert "pid" not in raw.columns
    parcels = create_parcel_down.read_parcels(str(path), "pid")
    assert parcels["pid"].tolist() == [11, 33]
    assert parcels.geometry.is_valid.all()


def test_multi_layer_gpkg_picks_parcels_geometry_layer(tmp_path):
    path = tmp_path / "mixed.gpkg"
    _write_parcels(path)
    other = gpd.GeoDataFrame({"id": [99]}, geometry=[box(3, 0, 4, 1)], crs="EPSG:4326")
    pyogrio.write_dataframe(other, path, layer="outlets", driver="GPKG", append=False)
    parcels = create_parcel_down.read_parcels(str(path), "pid")
    assert parcels["pid"].tolist() == [11, 33]


def test_explicit_layer_for_other_named_geometry_layer(tmp_path):
    path = tmp_path / "other.gpkg"
    _write_parcels(path, layer="fields")
    parcels = create_parcel_down.read_parcels(str(path), "pid", layer="fields")
    assert parcels["pid"].tolist() == [11, 33]
    with pytest.raises(ValueError, match="not found"):
        create_parcel_down.read_parcels(str(path), "pid", layer="absent")


def test_normal_pid_attribute_still_works(tmp_path):
    path = tmp_path / "regular.gpkg"
    _write_parcels(path, fid_primary=False)
    parcels = create_parcel_down.read_parcels(str(path), "pid")
    assert parcels["pid"].tolist() == [11, 33]


def test_missing_id_does_not_use_row_number(tmp_path):
    path = tmp_path / "wrong.gpkg"
    _write_parcels(path)
    with pytest.raises(ValueError, match="not found"):
        create_parcel_down.read_parcels(str(path), "different_id")


def test_geojson_pid_attribute_still_works(tmp_path):
    path = tmp_path / "input.geojson"
    source = gpd.GeoDataFrame(
        {"pid": [11, 33]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs="EPSG:4326"
    )
    source.to_file(path, driver="GeoJSON")
    assert create_parcel_down.read_parcels(str(path), "pid")["pid"].tolist() == [11, 33]
