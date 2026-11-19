"""USGS 3DEP discovery/download tests; all network operations are mocked."""

import argparse
from pathlib import Path
import sys
import zipfile

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from utils import usgs_dem, create_parcel_up
from utils.rasterio_compat import read_band


def study():
    return gpd.GeoDataFrame(
        {"pid": [1, 2]},
        geometry=[box(500000, 4400000, 500040, 4400040),
                  box(500040, 4400000, 500080, 4400040)],
        crs="EPSG:32617",
    )


def tif(path, bounds=(499950, 4399950, 500150, 4400150), *, nodata_patch=False):
    minx, miny, maxx, maxy = bounds
    data = np.arange(400, dtype=np.float32).reshape(20, 20)
    if nodata_patch:
        data[5:15, 5:15] = -9999
    with rasterio.open(
        path, "w", driver="GTiff", height=20, width=20, count=1,
        dtype="float32", crs="EPSG:32617",
        transform=from_origin(minx, maxy, (maxx-minx)/20, (maxy-miny)/20),
        nodata=-9999,
    ) as ds:
        ds.write(data, 1)


def test_availability_index_requires_full_aoi_not_any_intersection(monkeypatch):
    full = box(-83, 39, -82, 40)
    half = box(-83, 39, -82.5, 40)

    def fake_json(url, params):
        assert params["f"] == "geojson"
        assert params["inSR"] == params["outSR"] == 4326
        index = int(url.split("/")[-2])
        geometries = {1: [half], 4: [half, box(-82.5, 39, -82, 40)], 5: [full]}
        return {"features": [{"type": "Feature", "geometry": mapping(g), "properties": {}}
                             for g in geometries.get(index, [])]}
    monkeypatch.setattr(usgs_dem, "_json_get", fake_json)
    assert usgs_dem._index_coverage(1, full) == pytest.approx(.5)
    assert usgs_dem._index_coverage(4, full) == pytest.approx(1.0)
    assert usgs_dem._index_coverage(2, full) == 0


def test_index_pages_and_does_not_silently_ignore_api_failure(monkeypatch):
    g = box(-83, 39, -82, 40)
    calls = []

    def pages(url, params):
        calls.append(params["resultOffset"])
        if len(calls) == 1:
            return {"features": [{"geometry": mapping(g)}] * 200, "exceededTransferLimit": True}
        return {"features": []}
    monkeypatch.setattr(usgs_dem, "_json_get", pages)
    assert usgs_dem._index_coverage(4, g) == pytest.approx(1)
    assert calls == [0, 200]

    def failure(url, params):
        raise usgs_dem.USGSDownloadError("HTTP 503")
    monkeypatch.setattr(usgs_dem, "_json_get", failure)
    with pytest.raises(usgs_dem.USGSDownloadError, match="HTTP 503"):
        usgs_dem.check_resolutions(study())


def test_selection_accepts_only_full_coverage_and_requires_explicit_choice():
    coverage = {r.key: (1.0 if r.key == "10m" else 0.5) for r in usgs_dem.RESOLUTIONS}
    assert usgs_dem.choose_resolution(coverage, None, input_fn=lambda _: "5").key == "10m"
    assert usgs_dem.choose_resolution(coverage, "10m").key == "10m"
    with pytest.raises(usgs_dem.USGSDownloadError, match="not available"):
        usgs_dem.choose_resolution(coverage, "1m")
    with pytest.raises(usgs_dem.USGSDownloadError, match="Noninteractive"):
        def eof(_):
            raise EOFError
        usgs_dem.choose_resolution(coverage, None, input_fn=eof)


def product(i, extent, date="2024-01-01"):
    return {"sourceId": str(i), "downloadURL": f"https://prd-tnm.s3.amazonaws.com/data/{i}.tif",
            "boundingBox": {"minX": extent[0], "minY": extent[1],
                            "maxX": extent[2], "maxY": extent[3]},
            "publicationDate": date, "sizeInBytes": 100}


def test_product_query_selects_tiling_and_not_historical_duplicates(monkeypatch):
    aoi = box(-83, 39, -82, 40)
    all_products = [
        product("old", (-83,39,-82.5,40), "2015-01-01"),
        product("new", (-83,39,-82.5,40), "2024-01-01"),
        product("east", (-82.5,39,-82,40), "2023-01-01"),
    ]
    monkeypatch.setattr(usgs_dem, "_json_get", lambda url, params: {"items": all_products})
    selected = usgs_dem.query_products(next(r for r in usgs_dem.RESOLUTIONS if r.key == "10m"), aoi)
    assert {r["sourceId"] for r in selected} == {"new", "east"}

    monkeypatch.setattr(usgs_dem, "_json_get", lambda url, params: {"items": all_products[:2]})
    with pytest.raises(usgs_dem.USGSDownloadError, match="full"):
        usgs_dem.query_products(next(r for r in usgs_dem.RESOLUTIONS if r.key == "10m"), aoi)


def fake_usgs(monkeypatch, *, nodata=False):
    local, geo, crs = usgs_dem._metric_aoi(study(), 10)
    cov = {r.key: (1.0 if r.key == "10m" else 0.0) for r in usgs_dem.RESOLUTIONS}
    monkeypatch.setattr(usgs_dem, "check_resolutions", lambda parcels, buffer_m, **kwargs: (local, geo, crs, cov))
    monkeypatch.setattr(usgs_dem, "query_products", lambda res, geo: [product("1", geo.bounds)])

    def download(url, path, timeout=90):
        path.parent.mkdir(parents=True, exist_ok=True)
        tif(path, nodata_patch=nodata)
    monkeypatch.setattr(usgs_dem, "_download_tile", download)


def test_usgs_creates_projected_meter_dem_and_verifies_coverage(monkeypatch, tmp_path):
    fake_usgs(monkeypatch)
    out = usgs_dem.get_dem_from_usgs(
        study(), resolution="10m", buffer_m=10,
        cache_dir=tmp_path, max_cells=100_000,
    )
    assert out.exists()
    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 32617
        assert abs(ds.res[0] - 10) < 1e-7
        assert abs(ds.res[1] - 10) < 1e-7
        assert np.isfinite(read_band(ds)).all()


def test_usgs_does_not_accept_index_only_if_actual_raster_has_gaps(monkeypatch, tmp_path):
    fake_usgs(monkeypatch, nodata=True)
    with pytest.raises(usgs_dem.USGSDownloadError, match="nodata"):
        usgs_dem.get_dem_from_usgs(
            study(), resolution="10m", buffer_m=10,
            cache_dir=tmp_path, max_cells=100_000,
        )


def test_list_only_does_not_download(monkeypatch, tmp_path):
    fake_usgs(monkeypatch)
    monkeypatch.setattr(usgs_dem, "_download_tile", lambda *a, **k: pytest.fail("download not allowed"))
    assert usgs_dem.get_dem_from_usgs(
        study(), resolution=None, buffer_m=10, cache_dir=tmp_path,
        max_cells=100_000, list_only=True,
    ) is None
    assert not list(tmp_path.iterdir())


def test_large_dem_refused_before_query_or_download(monkeypatch, tmp_path):
    fake_usgs(monkeypatch)
    monkeypatch.setattr(usgs_dem, "query_products", lambda *a: pytest.fail("no query yet"))
    with pytest.raises(usgs_dem.USGSDownloadError, match="max-dem-cells"):
        usgs_dem.get_dem_from_usgs(
            study(), resolution="10m", buffer_m=10,
            cache_dir=tmp_path, max_cells=1,
        )


def test_zip_path_traversal_protection(tmp_path):
    archive = tmp_path/"bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.tif", "danger")
    with pytest.raises(usgs_dem.USGSDownloadError, match="Unsafe path"):
        usgs_dem._extract_raster(archive, tmp_path/"extract")


def test_local_dem_cli_path_skips_online_api(tmp_path, monkeypatch):
    from whitebox_test_support import fake_whitebox_file_pipeline
    fake_whitebox_file_pipeline(monkeypatch)
    study().to_file(tmp_path / "parcels.gpkg", driver="GPKG")
    raster = tmp_path / "local.tif"
    tif(raster)
    target = tmp_path / "out.csv"
    monkeypatch.setattr(sys, "argv", ["create_parcel_up.py", str(tmp_path / "parcels.gpkg"),
                                      str(raster), "--out", str(target)])
    monkeypatch.setattr(usgs_dem, "get_dem_from_usgs", lambda *a, **k: pytest.fail("network not allowed"))
    create_parcel_up.main()
    assert target.read_text().startswith("pid,pid_up")
    # Normalized model schema: zero or more direct edges; no placeholder rows.
    assert target.read_text().count("\n") >= 1


def test_missing_dem_cli_routes_to_usgs(monkeypatch, tmp_path):
    from whitebox_test_support import fake_whitebox_file_pipeline
    fake_whitebox_file_pipeline(monkeypatch)
    study().to_file(tmp_path / "parcels.gpkg", driver="GPKG")
    raster = tmp_path / "usgs_fake.tif"
    tif(raster)
    called = {}

    def get_dem(parcels, **kwargs):
        called.update(kwargs)
        assert len(parcels) == 2
        return raster
    monkeypatch.setattr(usgs_dem, "get_dem_from_usgs", get_dem)
    monkeypatch.setattr(sys, "argv", ["create_parcel_up.py", str(tmp_path / "parcels.gpkg"),
                                      "--usgs-resolution", "10m", "--out", str(tmp_path / "out.csv")])
    create_parcel_up.main()
    assert called["resolution"] == "10m"
    assert (tmp_path / "out.csv").exists()


def test_s1m_uses_separate_official_availability_layer(monkeypatch):
    observed = []
    def report(layer, geo, *, index_url):
        observed.append((layer, index_url))
        return 1.0
    monkeypatch.setattr(usgs_dem, "_index_coverage", report)
    _, _, _, coverage = usgs_dem.check_resolutions(study(), buffer_m=0)
    s1m = next(r for r in usgs_dem.RESOLUTIONS if r.key == "s1m")
    assert (s1m.index_layer, usgs_dem.S1M_INDEX_URL) in observed
    assert all(v == 1 for v in coverage.values())


def test_product_pagination(monkeypatch):
    calls = []
    def paged(url, params):
        calls.append(params["offset"])
        if len(calls) == 1:
            return {"items": [product(f"unused_{i}", (-85, 35, -84, 36)) for i in range(500)]}
        return {"items": [product("target", (-83, 39, -82, 40))]}
    monkeypatch.setattr(usgs_dem, "_json_get", paged)
    output = usgs_dem.query_products(
        next(r for r in usgs_dem.RESOLUTIONS if r.key == "10m"), box(-83,39,-82,40)
    )
    assert calls == [0,500]
    assert len(output) == 1 and output[0]["sourceId"] == "target"


def test_zip_dem_is_readable_and_safe(tmp_path):
    tif_file = tmp_path/"tile.tif"
    tif(tif_file)
    zipped = tmp_path/"tile.zip"
    with zipfile.ZipFile(zipped, "w") as z:
        z.write(tif_file, arcname="data/tile.tif")
    extracted = usgs_dem._extract_raster(zipped, tmp_path/"downloads")
    with rasterio.open(extracted) as ds:
        assert ds.count == 1 and ds.crs.to_epsg() == 32617


def test_usgs_requires_user_download_confirmation(monkeypatch, tmp_path):
    fake_usgs(monkeypatch)
    called = []
    monkeypatch.setattr(usgs_dem, "_download_tile", lambda *a, **kw: called.append(a))
    responses = iter(["10m", "n"])
    with pytest.raises(usgs_dem.USGSDownloadError, match="cancelled"):
        usgs_dem.get_dem_from_usgs(
            study(), resolution=None, buffer_m=10,
            cache_dir=tmp_path, max_cells=100_000,
            input_fn=lambda prompt: next(responses),
        )
    assert not called


def test_missing_dem_list_only_does_not_create_csv(monkeypatch, tmp_path):
    parcels = tmp_path / "parcels.gpkg"
    study().to_file(parcels, driver="GPKG")
    out = tmp_path / "unused.csv"
    observed = []
    def check(gdf, **kwargs):
        observed.append(kwargs)
        assert len(gdf) == 2
        return None
    monkeypatch.setattr(usgs_dem, "get_dem_from_usgs", check)
    monkeypatch.setattr(sys, "argv", ["create_parcel_up.py", str(parcels),
                                      "--list-usgs-resolutions", "--out", str(out)])
    create_parcel_up.main()
    assert observed[0]["list_only"] is True
    assert not out.exists()


def test_download_manifest_records_3dep_source(monkeypatch, tmp_path):
    import json
    fake_usgs(monkeypatch)
    dem = usgs_dem.get_dem_from_usgs(
        study(), resolution="10m", buffer_m=10, cache_dir=tmp_path,
        max_cells=100_000,
    )
    data = json.loads((tmp_path / "usgs_3dep_10m_sources.json").read_text())
    assert data["source"] == "USGS 3DEP / TNMAccess"
    assert data["availability_source"] == "index"
    assert data["availability_provisional"] is False
    assert data["nominal_cell_size_m"] == 10
    assert len(data["products"]) == 1
    assert data["dem"] == str(dem)
