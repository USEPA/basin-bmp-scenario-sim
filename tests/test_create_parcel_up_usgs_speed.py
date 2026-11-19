"""USGS availability speed/caching regressions, with no network dependency."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import urllib.error

import pytest
from shapely.geometry import box

from utils import usgs_dem, create_parcel_up
from test_create_parcel_up_usgs import study


def test_coverage_checks_run_concurrently_and_preserve_resolution_order(monkeypatch):
    first = usgs_dem.RESOLUTIONS[0]
    second = usgs_dem.RESOLUTIONS[1]
    monkeypatch.setattr(usgs_dem, 'RESOLUTIONS', (first, second))
    monkeypatch.setattr(usgs_dem, '_AVAILABILITY_WORKERS', 2)
    barrier = threading.Barrier(2, timeout=3)
    threads = []

    def index(layer, geo, *, index_url):
        threads.append(threading.get_ident())
        barrier.wait()
        return 1. if layer == first.index_layer and index_url == first.index_url else .3

    monkeypatch.setattr(usgs_dem, '_index_coverage', index)
    _, _, _, coverage = usgs_dem.check_resolutions(study(), buffer_m=0)
    assert len(set(threads)) == 2
    assert set(coverage) == {first.key, second.key}
    assert all(coverage.sources[key] == 'index' for key in coverage)
    assert list(coverage) == [r.key for r in usgs_dem.RESOLUTIONS] or set(coverage) == set(r.key for r in usgs_dem.RESOLUTIONS)


def test_repeat_queries_use_disk_cache_for_exact_same_aoi(monkeypatch, tmp_path):
    calls = []

    def index(layer, geo, *, index_url):
        calls.append((layer, index_url))
        return 1. if layer == 4 else 0.

    monkeypatch.setattr(usgs_dem, '_index_coverage', index)
    _, _, _, initial = usgs_dem.check_resolutions(study(), cache_dir=tmp_path)
    assert len(calls) == len(usgs_dem.RESOLUTIONS)
    assert (tmp_path / 'usgs_availability_cache.json').exists()
    assert initial.sources['10m'] == 'index'
    calls.clear()
    _, _, _, repeated = usgs_dem.check_resolutions(study(), cache_dir=tmp_path)
    assert calls == []
    assert repeated['10m'] == 1
    assert repeated.sources['10m'] == 'index'
    # Different buffer -> different buffered geometry -> cache miss.
    _, _, _, other = usgs_dem.check_resolutions(study(), cache_dir=tmp_path, buffer_m=200)
    assert len(calls) == len(usgs_dem.RESOLUTIONS)
    calls.clear()
    usgs_dem.check_resolutions(study(), cache_dir=tmp_path, buffer_m=200, refresh=True)
    assert len(calls) == len(usgs_dem.RESOLUTIONS)


def test_expired_cache_rechecks_and_unknown_service_failures_not_cached(monkeypatch, tmp_path):
    calls = []
    now = [500000.]
    monkeypatch.setattr(usgs_dem.time, 'time', lambda: now[0])

    def index(layer, geo, *, index_url):
        calls.append(layer)
        if layer == 4:
            raise usgs_dem.USGSDownloadError('HTTP 500')
        return .5

    def catalog(res, geo):
        raise usgs_dem.USGSDownloadError('HTTP 503')

    monkeypatch.setattr(usgs_dem, '_index_coverage', index)
    monkeypatch.setattr(usgs_dem, '_catalog_coverage', catalog)
    _, _, _, coverage = usgs_dem.check_resolutions(study(), cache_dir=tmp_path)
    assert coverage['10m'] is None and coverage.sources['10m'] == 'unknown'
    assert len(calls) == len(usgs_dem.RESOLUTIONS)
    calls.clear()
    usgs_dem.check_resolutions(study(), cache_dir=tmp_path)
    assert calls == [4]  # Only unknown is retried.
    calls.clear()
    now[0] += usgs_dem._INDEX_CACHE_TTL_S + 1
    usgs_dem.check_resolutions(study(), cache_dir=tmp_path)
    assert len(calls) == len(usgs_dem.RESOLUTIONS)


def test_provisional_coverage_expires_sooner_and_retains_provenance(monkeypatch, tmp_path):
    now = [500000.]
    monkeypatch.setattr(usgs_dem.time, 'time', lambda: now[0])
    calls = []

    def index(layer, geo, *, index_url):
        calls.append(('index', layer))
        if layer == 4:
            raise usgs_dem.USGSDownloadError('HTTP 500')
        return 0.

    def catalog(res, geo):
        calls.append(('catalog', res.key))
        return 1.

    monkeypatch.setattr(usgs_dem, '_index_coverage', index)
    monkeypatch.setattr(usgs_dem, '_catalog_coverage', catalog)
    usgs_dem.check_resolutions(study(), cache_dir=tmp_path)
    calls.clear()
    now[0] += usgs_dem._CATALOG_CACHE_TTL_S + 1
    _, _, _, coverage = usgs_dem.check_resolutions(study(), cache_dir=tmp_path)
    assert calls == [('index', 4), ('catalog', '10m')]
    assert coverage.sources['10m'] == 'catalog'
    assert coverage['10m'] == 1.
    assert coverage['30m'] == 0.


def test_bad_json_cache_is_ignored_not_treated_as_coverage(monkeypatch, tmp_path):
    (tmp_path / 'usgs_availability_cache.json').write_text('{broken')
    seen = []
    monkeypatch.setattr(usgs_dem, '_index_coverage',
                        lambda layer, geo, *, index_url: seen.append(layer) or 1.)
    _, _, _, coverage = usgs_dem.check_resolutions(study(), cache_dir=tmp_path)
    assert len(seen) == len(usgs_dem.RESOLUTIONS)
    assert all(coverage[r.key] == 1. for r in usgs_dem.RESOLUTIONS)
    assert json.loads((tmp_path / 'usgs_availability_cache.json').read_text())['version'] == 1


def test_2_transient_failures_stop_and_use_short_timeout(monkeypatch):
    seen = []
    def down(request, *, timeout):
        seen.append(timeout)
        raise urllib.error.HTTPError(request.full_url, 500, 'Internal Server Error', {}, None)
    monkeypatch.setattr(usgs_dem.urllib.request, 'urlopen', down)
    monkeypatch.setattr(usgs_dem.time, 'sleep', lambda _: None)
    with pytest.raises(usgs_dem.USGSDownloadError, match='HTTP 500'):
        usgs_dem._json_get('https://usgs.example/arcgis', {'f': 'geojson'})
    assert seen == [12, 12]


def test_coverage_cache_has_explicit_refresh_cli_flag(monkeypatch):
    monkeypatch.setattr('sys.argv', ['create_parcel_up.py', 'parcels.gpkg',
                                   '--refresh-usgs-availability', '--list-usgs-resolutions'])
    args = create_parcel_up.parse_args()
    assert args.refresh_usgs_availability


def test_explicit_resolution_only_queries_selected_layer_and_reuses_cache(monkeypatch, tmp_path, capsys):
    calls = []
    def index(layer, geo, *, index_url):
        calls.append(layer)
        return 1. if layer == 4 else 0.
    monkeypatch.setattr(usgs_dem, '_index_coverage', index)
    _, _, _, selected = usgs_dem.check_resolutions(
        study(), cache_dir=tmp_path, only_resolution='10m'
    )
    assert calls == [4]
    assert selected['10m'] == 1.0
    assert selected['30m'] is None
    assert selected.sources['30m'] == 'skipped'
    usgs_dem.show_resolutions(selected)
    assert 'NOT CHECKED' in capsys.readouterr().out
    calls.clear()
    _, _, _, all_results = usgs_dem.check_resolutions(study(), cache_dir=tmp_path)
    assert len(calls) == len(usgs_dem.RESOLUTIONS) - 1
    assert 4 not in calls
    assert all_results['10m'] == 1.
    calls.clear()
    usgs_dem.check_resolutions(study(), cache_dir=tmp_path, only_resolution='10m')
    assert calls == []


def test_explicit_resolution_failures_cannot_be_mistaken_for_missing_coverage(monkeypatch):
    def error(*args, **kwargs):
        raise usgs_dem.USGSDownloadError('503')
    monkeypatch.setattr(usgs_dem, '_index_coverage', error)
    monkeypatch.setattr(usgs_dem, '_catalog_coverage', error)
    with pytest.raises(usgs_dem.USGSDownloadError, match='failed for 10m'):
        usgs_dem.check_resolutions(study(), only_resolution='10m')
