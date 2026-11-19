"""USGS HTTP 500 and TNMAccess fallback regressions (no live requests)."""

import io
import json
import urllib.error

import pytest
from shapely.geometry import box

from utils import usgs_dem
from test_create_parcel_down_usgs import product, study


def test_index_http_500_falls_back_to_independent_catalog(monkeypatch, capsys):
    """The exact reported 10-meter index failure must not abort all choices."""
    layers = []

    def index(layer, geo, *, index_url):
        layers.append(layer)
        if layer == 4:  # 1/3 arc-second / 10m
            raise usgs_dem.USGSDownloadError("HTTP 500 Internal Server Error")
        return 0.0

    queries = []

    def api(url, params):
        queries.append((url, dict(params)))
        assert url == usgs_dem.PRODUCTS_URL
        assert params["datasets"] == "National Elevation Dataset (NED) 1/3 arc-second"
        minx, miny, maxx, maxy = (float(n) for n in params["bbox"].split(","))
        return {"items": [product("10m", (minx - .1, miny - .1, maxx + .1, maxy + .1))]}

    monkeypatch.setattr(usgs_dem, "_index_coverage", index)
    monkeypatch.setattr(usgs_dem, "_json_get", api)
    local, geo, crs, coverage = usgs_dem.check_resolutions(study())
    assert 4 in layers
    assert len(queries) == 1
    assert coverage["10m"] == pytest.approx(1.0)
    assert coverage.sources["10m"] == "catalog"
    assert usgs_dem.choose_resolution(coverage, "10m").key == "10m"
    usgs_dem.show_resolutions(coverage)
    out = capsys.readouterr().out
    assert "CATALOG FULL*" in out
    assert "provisional" in out
    assert "10m" in out


def test_fallback_partial_is_not_selectable(monkeypatch):
    monkeypatch.setattr(usgs_dem, "_index_coverage", lambda *a, **kw: (_ for _ in ()).throw(
        usgs_dem.USGSDownloadError("500")))

    def partial(res, geo):
        # The bounding box covers only half, not the full parcel area.
        x0, y0, x1, y1 = geo.bounds
        return [product("west", (x0, y0, (x0 + x1) / 2, y1))]

    monkeypatch.setattr(usgs_dem, "_fetch_catalog_products", partial)
    _, _, _, coverage = usgs_dem.check_resolutions(study())
    assert all(value is not None and 0 < value < 1 for value in coverage.values())
    with pytest.raises(usgs_dem.USGSDownloadError, match="no DEM resolution"):
        usgs_dem.choose_resolution(coverage, "10m")


def test_one_unknown_resolution_does_not_block_others(monkeypatch, capsys):
    def index(layer, geo, *, index_url):
        if layer == 4:
            raise usgs_dem.USGSDownloadError("HTTP 500")
        return 1.0 if layer == 5 else 0.0

    def catalog(*args):
        raise usgs_dem.USGSDownloadError("TNMAccess HTTP 503")

    monkeypatch.setattr(usgs_dem, "_index_coverage", index)
    monkeypatch.setattr(usgs_dem, "_catalog_coverage", catalog)
    _, _, _, coverage = usgs_dem.check_resolutions(study())
    assert coverage["10m"] is None
    assert coverage["30m"] == pytest.approx(1.0)
    assert usgs_dem.choose_resolution(coverage, "30m").key == "30m"
    with pytest.raises(usgs_dem.USGSDownloadError, match="not available"):
        usgs_dem.choose_resolution(coverage, "10m")
    usgs_dem.show_resolutions(coverage)
    out = capsys.readouterr().out
    assert "UNKNOWN" in out
    assert "FULL" in out


def test_all_index_and_catalog_queries_fail_raise_not_no_coverage(monkeypatch):
    def down(*args, **kwargs):
        raise usgs_dem.USGSDownloadError("HTTP 500")
    monkeypatch.setattr(usgs_dem, "_index_coverage", down)
    monkeypatch.setattr(usgs_dem, "_catalog_coverage", down)
    with pytest.raises(usgs_dem.USGSDownloadError, match="All USGS index queries and catalog fallbacks failed"):
        usgs_dem.check_resolutions(study())


def test_product_catalog_coverage_distinguishes_partial_full_and_none(monkeypatch):
    res = next(r for r in usgs_dem.RESOLUTIONS if r.key == "10m")
    aoi = box(-84, 39, -83, 40)
    west = product("west", (-84, 39, -83.5, 40))
    east = product("east", (-83.5, 39, -83, 40))
    monkeypatch.setattr(usgs_dem, "_fetch_catalog_products", lambda *a: [west])
    assert usgs_dem._catalog_coverage(res, aoi) == pytest.approx(.5)
    monkeypatch.setattr(usgs_dem, "_fetch_catalog_products", lambda *a: [west, east])
    assert usgs_dem._catalog_coverage(res, aoi) == pytest.approx(1.)
    monkeypatch.setattr(usgs_dem, "_fetch_catalog_products", lambda *a: [])
    assert usgs_dem._catalog_coverage(res, aoi) == 0.


def test_http_500_is_retried_and_then_successful(monkeypatch):
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self, *a): return b'{"items": []}'

    def fetch(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 500, "Internal Server Error", {}, None)
        return Response()

    monkeypatch.setattr(usgs_dem.urllib.request, "urlopen", fetch)
    monkeypatch.setattr(usgs_dem.time, "sleep", lambda _: None)
    assert usgs_dem._json_get("https://usgs.example/query", {"f": "json"}) == {"items": []}
    assert len(calls) == 2


def test_http_400_not_retried(monkeypatch):
    calls = []
    def fetch(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, None)
    monkeypatch.setattr(usgs_dem.urllib.request, "urlopen", fetch)
    monkeypatch.setattr(usgs_dem.time, "sleep", lambda _: pytest.fail("no retry on 400"))
    with pytest.raises(usgs_dem.USGSDownloadError, match="HTTP 400"):
        usgs_dem._json_get("https://usgs.example/query", {"f": "json"})
    assert len(calls) == 1
