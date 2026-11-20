"""Regression tests for direct D8 graph edges and atomic GeoPackage updates."""
from __future__ import annotations

import csv
from pathlib import Path
import sqlite3
import sys

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from utils import create_parcel_down, parcel_down_graph


def eastward_flow(labels: np.ndarray):
    """D8 pointers to the right, with the final column draining off-grid."""
    nr = np.broadcast_to(np.arange(labels.shape[0])[:, None], labels.shape).copy().astype('int32')
    nc = np.broadcast_to(np.arange(labels.shape[1])[None, :]+1, labels.shape).copy().astype('int32')
    nc[:, -1] = -1
    return nr, nc


def test_three_parcel_chain_stores_only_immediate_connections():
    labels = np.tile(np.array([1, 1, 2, 2, 3, 3]), (3, 1))
    nr, nc = eastward_flow(labels)
    edges = parcel_down_graph.direct_edges_from_d8(labels, nr, nc, {1: '1', 2: '2', 3: '3'})
    assert edges == [(1, 2), (2, 3)]
    assert (1, 3) not in edges


def test_background_gap_connects_first_next_parcel_only():
    labels = np.array([[1, 1, 0, 0, 2, 2, 0, 3, 3]])
    nr, nc = eastward_flow(labels)
    assert parcel_down_graph.direct_edges_from_d8(labels, nr, nc, {1: 1, 2: 2, 3: 3}) == [(1, 2), (2, 3)]


def test_multiple_immediate_upstream_parcels_and_one_duplicate_only():
    labels = np.array([[1, 0, 0, 3], [2, 0, 0, 3]], dtype='int32')
    nr = np.full(labels.shape, -1, dtype='int32')
    nc = np.full(labels.shape, -1, dtype='int32')
    # Parcel 1 and parcel 2 drain through separate background corridors to 3.
    for r in range(2):
        for c in range(3):
            nr[r, c] = r
            nc[r, c] = c+1
    edges = parcel_down_graph.direct_edges_from_d8(labels, nr, nc, {1: 1, 2: 2, 3: 3})
    assert edges == [(1, 3), (2, 3)]


def test_same_parcel_reentry_is_not_an_edge():
    labels = np.array([[1, 0, 1, 0, 2, 2]])
    nr, nc = eastward_flow(labels)
    assert parcel_down_graph.direct_edges_from_d8(labels, nr, nc, {1: 1, 2: 2}) == [(1, 2)]


def test_d8_cycle_in_background_does_not_loop_forever():
    labels = np.array([[1, 0, 0, 2]])
    nr, nc = eastward_flow(labels)
    nc[0, 2] = 1  # background cycle at indices 1 and 2
    assert parcel_down_graph.direct_edges_from_d8(labels, nr, nc, {1: 1, 2: 2}) == []


def test_validation_rejects_cycles_and_unknown_ids():
    assert parcel_down_graph.validate_direct_edges([(2, 3), (1, 2), (2, 3)], [1, 2, 3]) == [(1, 2), (2, 3)]
    with pytest.raises(ValueError, match='self-link'):
        parcel_down_graph.validate_direct_edges([(2, 2)], [1, 2, 3])
    with pytest.raises(ValueError, match='directed cycle'):
        parcel_down_graph.validate_direct_edges([(1, 2), (2, 3), (3, 1)], [1, 2, 3])
    with pytest.raises(ValueError, match='unknown'):
        parcel_down_graph.validate_direct_edges([(1, 888)], [1, 2, 3])
    with pytest.raises(ValueError, match='integer'):
        parcel_down_graph.validate_direct_edges([], [1, 'not-an-integer'])


def test_long_chain_avoids_recursion_limit():
    edges = [(i+1, i) for i in range(1, 2500)]
    assert len(parcel_down_graph.validate_direct_edges(edges, range(1, 2501))) == len(edges)


def write_gpkg(path: Path):
    gdf = gpd.GeoDataFrame(
        {'pid': [11, 22, 33]},
        geometry=[box(0,0,2,3), box(2,0,4,3), box(4,0,6,3)],
        crs='EPSG:3857',
    )
    pyogrio.write_dataframe(gdf, path, driver='GPKG', layer='parcels', layer_options={'FID': 'pid'})
    # A separate model table must remain unchanged when parcel_up is replaced.
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE "parcel_outlets" (id INTEGER PRIMARY KEY, pid INTEGER, oid TEXT)')
        conn.execute("INSERT INTO parcel_outlets(pid,oid) VALUES (11,'OutletA')")
        conn.execute('CREATE TABLE "parcel_down" (id INTEGER PRIMARY KEY, pid INTEGER, pid_down INTEGER)')
        conn.execute('INSERT INTO parcel_down(pid,pid_down) VALUES (11,33)')
        conn.execute(
            "INSERT INTO gpkg_contents(table_name,data_type,identifier,description,last_change) "
            "VALUES ('parcel_down','attributes','parcel_down','original',strftime('%Y-%m-%dT%H:%M:%fZ','now'))"
        )
    return path


def test_schema_and_atomic_replacement_preserves_other_tables(tmp_path):
    gpkg = write_gpkg(tmp_path/'parcels.gpkg')
    n = parcel_down_graph.write_gpkg_parcel_down(gpkg, [(11,22),(22,33),(11,22)], [11,22,33])
    assert n == 2
    with sqlite3.connect(gpkg) as conn:
        rows = conn.execute('SELECT pid,pid_down FROM parcel_down ORDER BY pid').fetchall()
        assert rows == [(11,22),(22,33)]
        cols = conn.execute('PRAGMA table_info("parcel_down")').fetchall()
        assert [(r[1],r[2]) for r in cols] == [('id','INTEGER'),('pid','INTEGER'),('pid_down','INTEGER')]
        assert cols[0][5] == 1
        assert conn.execute('SELECT pid,oid FROM parcel_outlets').fetchall() == [(11,'OutletA')]
        assert conn.execute("SELECT data_type,description FROM gpkg_contents WHERE table_name='parcel_down'").fetchone() == ('attributes','original')
        fk = conn.execute('PRAGMA foreign_key_list("parcel_down")').fetchall()
        assert len(fk) == 2
        assert conn.execute('PRAGMA foreign_key_check("parcel_down")').fetchall() == []
    with pytest.raises(sqlite3.IntegrityError):
        # Simulate a stale input graph referring to a parcel no longer in GPKG.
        parcel_down_graph.write_gpkg_parcel_down(gpkg, [(22,33), (11,999)], [11,22,33,999])
    with sqlite3.connect(gpkg) as conn:
        assert conn.execute('SELECT pid,pid_down FROM parcel_down ORDER BY pid').fetchall() == [(11,22),(22,33)]
        assert conn.execute('SELECT pid,oid FROM parcel_outlets').fetchall() == [(11,'OutletA')]


def test_replacement_refuses_dependency_that_could_cascade_into_other_tables(tmp_path):
    gpkg = write_gpkg(tmp_path/'parcels.gpkg')
    with sqlite3.connect(gpkg) as conn:
        conn.execute('CREATE TABLE other_table (edge_id INTEGER REFERENCES parcel_down(id) ON DELETE CASCADE)')
        conn.execute('INSERT INTO other_table(edge_id) VALUES (1)')
    with pytest.raises(ValueError, match='foreign key referencing parcel_down'):
        parcel_down_graph.write_gpkg_parcel_down(gpkg, [(11,22)], [11,22,33])
    with sqlite3.connect(gpkg) as conn:
        assert conn.execute('SELECT pid,pid_down FROM parcel_down').fetchall() == [(11,33)]
        assert conn.execute('SELECT edge_id FROM other_table').fetchall() == [(1,)]


def test_sqlite_rejects_duplicate_row_and_foreign_id(tmp_path):
    gpkg = write_gpkg(tmp_path/'parcels.gpkg')
    parcel_down_graph.write_gpkg_parcel_down(gpkg, [(11,22)], [11,22,33])
    with sqlite3.connect(gpkg) as conn:
        conn.execute('PRAGMA foreign_keys=ON')
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('INSERT INTO parcel_down(pid,pid_down) VALUES (11,22)')
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('INSERT INTO parcel_down(pid,pid_down) VALUES (99,11)')


def test_model_reference_geopackage_schema_and_qgis_registration(tmp_path):
    """Synthetic fixture verifies the model's nonspatial link-table contract."""
    gpkg = write_gpkg(tmp_path / 'schema.gpkg')
    with sqlite3.connect(gpkg) as conn:
        before = conn.execute('SELECT COUNT(*) FROM parcels').fetchone()[0]
        assert before == 3
        assert conn.execute('SELECT pid,pid_down FROM parcel_down').fetchall() == [(11,33)]
        assert conn.execute('SELECT pid,oid FROM parcel_outlets').fetchall() == [(11,'OutletA')]
    parcel_down_graph.write_gpkg_parcel_down(gpkg, [(11,22),(22,33)], [11,22,33])
    with sqlite3.connect(gpkg) as conn:
        assert conn.execute('SELECT COUNT(*) FROM parcels').fetchone()[0] == before
        assert conn.execute('SELECT pid,pid_down FROM parcel_down ORDER BY pid').fetchall() == [(11,22),(22,33)]
        assert conn.execute("SELECT data_type FROM gpkg_contents WHERE table_name='parcel_down'").fetchone()[0] == 'attributes'
        assert len(conn.execute('PRAGMA foreign_key_list("parcel_down")').fetchall()) == 2
        assert conn.execute('PRAGMA foreign_key_check("parcel_down")').fetchall() == []
        assert conn.execute('SELECT pid,oid FROM parcel_outlets').fetchall() == [(11,'OutletA')]


def test_end_to_end_cli_updates_gpkg_only_with_immediate_edges(tmp_path, monkeypatch):
    from whitebox_test_support import fake_whitebox_file_pipeline
    fake_whitebox_file_pipeline(monkeypatch)
    gpkg = write_gpkg(tmp_path/'parcels.gpkg')
    dem = tmp_path/'dem.tif'
    arr = np.tile(np.arange(6,0,-1,dtype='float32'), (3,1))
    with rasterio.open(dem,'w',driver='GTiff',width=6,height=3,count=1,dtype='float32',
                       transform=from_origin(0,3,1,1),crs='EPSG:3857',nodata=-9999) as ds:
        ds.write(arr,1)
    monkeypatch.setattr(sys, 'argv', ['create_parcel_down.py', str(gpkg), str(dem)])
    create_parcel_down.main()
    with sqlite3.connect(gpkg) as conn:
        assert conn.execute('SELECT pid,pid_down FROM parcel_down ORDER BY pid').fetchall() == [(11,22),(22,33)]
    assert not (tmp_path/'parcels_downgradient.csv').exists()
    monkeypatch.setattr(sys, 'argv', ['create_parcel_down.py', str(gpkg), str(dem),
                                      '--out', str(tmp_path/'connections.csv')])
    create_parcel_down.main()
    with (tmp_path/'connections.csv').open(newline='') as f:
        assert list(csv.DictReader(f)) == [{'pid':'11','pid_down':'22'}, {'pid':'22','pid_down':'33'}]


def test_list_usgs_resolutions_does_not_touch_gpkg(tmp_path, monkeypatch):
    from utils import usgs_dem
    gpkg = write_gpkg(tmp_path/'parcels.gpkg')
    old = gpkg.read_bytes()
    monkeypatch.setattr(usgs_dem, 'get_dem_from_usgs', lambda *a, **k: None)
    monkeypatch.setattr(sys, 'argv', ['create_parcel_down.py', str(gpkg), '--list-usgs-resolutions'])
    create_parcel_down.main()
    assert gpkg.read_bytes() == old


def test_csv_only_opt_out_preserves_gpkg(tmp_path, monkeypatch):
    from whitebox_test_support import fake_whitebox_file_pipeline
    fake_whitebox_file_pipeline(monkeypatch)
    gpkg = write_gpkg(tmp_path/'parcels.gpkg')
    old = gpkg.read_bytes()
    dem = tmp_path/'dem.tif'
    arr = np.tile(np.arange(6,0,-1,dtype='float32'), (3,1))
    with rasterio.open(dem,'w',driver='GTiff',width=6,height=3,count=1,dtype='float32',
                       transform=from_origin(0,3,1,1),crs='EPSG:3857') as ds:
        ds.write(arr,1)
    monkeypatch.setattr(sys, 'argv', ['create_parcel_down.py', str(gpkg), str(dem),
                                    '--no-gpkg-update', '--out', str(tmp_path/'only.csv')])
    create_parcel_down.main()
    assert gpkg.read_bytes() == old
    assert (tmp_path/'only.csv').read_text().startswith('pid,pid_down')
