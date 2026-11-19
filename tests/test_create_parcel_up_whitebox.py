"""Whitebox D8 cell routing and deterministic parcel-graph regression tests.

Whitebox executable integration is mocked here so the checks run without a
network request or a separate WhiteboxTools installation. The real Whitebox
binary should also be used in a smoke test before production adoption.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from utils import whitebox_parcel_routing as routing
from utils import create_parcel_up
from utils.parcel_up_graph import validate_direct_edges
from whitebox_test_support import fake_whitebox_file_pipeline


def candidates_for(labels, downstream, *, cell_area=900.0):
    routing.validate_cell_dag(np.asarray(downstream).reshape(-1))
    destinations = routing.compute_parcel_origin_destinations(
        np.asarray(labels), np.asarray(downstream).reshape(-1)
    )
    pids = {int(i): str(int(i)) for i in np.unique(labels) if i != 0}
    return routing.score_candidate_connections(np.asarray(labels), destinations,
                                                pids, cell_area_m2=cell_area)


def test_whitebox_default_pointer_encoding_all_directions():
    p = np.asarray([[64, 128, 1], [32, 0, 2], [16, 8, 4]], dtype=np.uint8)
    nxt = routing.whitebox_d8_to_next(p, np.ones(p.shape, bool))
    # Directions leaving the 3x3 extent are outlets, not wrapped indexes.
    assert nxt.reshape(3,3)[1,2] == -1  # E from right edge
    assert nxt.reshape(3,3)[1,0] == -1  # W from left edge
    assert nxt.reshape(3,3)[2,1] == -1  # S from bottom
    assert nxt.reshape(3,3)[0,1] == -1  # N from top
    # Interior cell eastward: Whitebox 2, not Esri 1.
    p[1,1] = 2
    assert routing.whitebox_d8_to_next(p, np.ones(p.shape, bool))[4] == 5


def test_invalid_esri_direction_and_nodata_target_rejected():
    with pytest.raises(ValueError, match='Invalid Whitebox'):
        routing.whitebox_d8_to_next(np.array([[3]]), np.array([[True]]))
    with pytest.raises(ValueError, match='into NoData'):
        routing.whitebox_d8_to_next(np.array([[2, 0]]), np.array([[True, False]]))


def test_raster_cell_cycle_is_rejected_before_parcel_edges():
    with pytest.raises(ValueError, match='cell-level cycle'):
        routing.validate_cell_dag(np.array([1, 0], dtype=np.int32))
    routing.validate_cell_dag(np.array([1, 2, -1], dtype=np.int32))


def test_chain_has_only_immediate_links_and_no_transitive_edges():
    labels = np.array([[1, 1, 2, 2, 3, 3]], dtype=np.int32)
    nxt = np.array([1, 2, 3, 4, 5, -1], dtype=np.int32)
    rows = candidates_for(labels, nxt)
    edges, result, summary = routing.select_acyclic_connections(rows)
    assert edges == [(2,1), (3,2)]
    assert (3,1) not in edges
    assert validate_direct_edges(edges,[1,2,3]) == edges
    assert summary['ambiguous_parcels'] == 0


def test_gap_tracing_finds_next_parcel_and_does_not_skip_it():
    labels = np.array([[1, 1, 0, 0, 2, 3]])
    nxt = np.array([1, 2, 3, 4, 5, -1])
    rows = candidates_for(labels, nxt)
    edges, _, _ = routing.select_acyclic_connections(rows)
    assert edges == [(2,1), (3,2)]


def test_return_to_same_parcel_after_gap_continues_to_real_destination():
    labels = np.array([[1, 0, 1, 2]])
    nxt = np.array([1, 2, 3, -1])
    rows = candidates_for(labels, nxt)
    edges, _, _ = routing.select_acyclic_connections(rows)
    assert edges == [(2, 1)]
    assert sum(r['origin_cells'] for r in rows if r['upstream_pid']==1) == 2


def test_aggregation_uses_own_cells_not_upstream_throughflow():
    # One source parcel's three cells trace to two different destinations:
    # 2/3 go to 2 and 1/3 goes to 3. A separate 40-cell upstream parcel that
    # flows through the source must NOT be included in its voting weight.
    labels = np.asarray([[4]*40 + [1,1,1,2,3]], dtype=np.int32)
    nxt = np.full(labels.size, -1, dtype=np.int32)
    nxt[:40] = 40
    nxt[40] = 43  # 1 -> 2
    nxt[41] = 43  # 1 -> 2
    nxt[42] = 44  # 1 -> 3
    rows = candidates_for(labels, nxt)
    source1 = [r for r in rows if r['upstream_pid'] == 1]
    assert sorted((r['downstream_pid'], r['origin_cells']) for r in source1) == [(2,2),(3,1)]
    assert sum(r['parcel_cells'] for r in source1)==6
    assert next(r['share'] for r in source1 if r['downstream_pid']==2) == pytest.approx(2/3)
    edges, *_ = routing.select_acyclic_connections(rows)
    assert (2,1) in edges and (3,1) not in edges


def test_acyclic_cell_paths_can_aggregate_to_cycle_but_selection_cannot():
    # Disjoint cell paths: 1 -> 2, 2 -> 1 (never a raster D8 cycle).
    labels = np.array([[1,2,0,2,1,0]], dtype=np.int32)
    nxt = np.array([1,-1,-1,4,-1,-1], dtype=np.int32)
    rows = candidates_for(labels, nxt)
    assert (2,1) in {(r['downstream_pid'],r['upstream_pid']) for r in rows
                       if r['downstream_pid'] is not None}
    assert (1,2) in {(r['downstream_pid'],r['upstream_pid']) for r in rows
                       if r['downstream_pid'] is not None}
    # Both split their area 50/50 between the other parcel and outside.
    edges, result, summary = routing.select_acyclic_connections(rows)
    assert not edges
    assert summary['ambiguous_parcels']==2


def test_greedy_cycle_rejection_retains_stronger_connection_and_reports_weak():
    evidence = [
      {'upstream_pid':6, 'downstream_pid':14,'origin_cells':85,'parcel_cells':100,'share':.85,'origin_area_m2':85.},
      {'upstream_pid':6, 'downstream_pid':20,'origin_cells':15,'parcel_cells':100,'share':.15,'origin_area_m2':15.},
      {'upstream_pid':14, 'downstream_pid':6,'origin_cells':65,'parcel_cells':100,'share':.65,'origin_area_m2':65.},
      {'upstream_pid':14, 'downstream_pid':20,'origin_cells':35,'parcel_cells':100,'share':.35,'origin_area_m2':35.},
    ]
    edges, diag, summary = routing.select_acyclic_connections(evidence)
    assert edges == [(14,6)]
    assert summary['rejected_cycle_candidates']==1
    assert next(r for r in diag if r['upstream_pid']==14 and r['downstream_pid']==6)['decision']=='rejected_cycle'
    assert validate_direct_edges(edges,[6,14,20])==edges


def test_candidate_with_dominant_outside_route_gets_no_link():
    rows = [
      {'upstream_pid':6, 'downstream_pid':None,'origin_cells':90,'parcel_cells':100,'share':.9,'origin_area_m2':90.},
      {'upstream_pid':6, 'downstream_pid':14,'origin_cells':10,'parcel_cells':100,'share':.1,'origin_area_m2':10.},
    ]
    edges, diag, stats = routing.select_acyclic_connections(rows)
    assert edges==[] and stats['outside_parcels']==1
    assert next(r for r in diag if r['downstream_pid'] is None)['decision']=='outside_parcel_domain'


def test_substantial_multioutlet_split_is_unresolved_not_arbitrarily_connected():
    rows = [
      {'upstream_pid':6, 'downstream_pid':14,'origin_cells':45,'parcel_cells':100,'share':.45,'origin_area_m2':45.},
      {'upstream_pid':6, 'downstream_pid':20,'origin_cells':40,'parcel_cells':100,'share':.4,'origin_area_m2':40.},
      {'upstream_pid':6, 'downstream_pid':None,'origin_cells':15,'parcel_cells':100,'share':.15,'origin_area_m2':15.},
    ]
    edges, diag, summary = routing.select_acyclic_connections(rows)
    assert edges == [] and summary['ambiguous_parcels']==1
    assert all(r['decision']=='unresolved_ambiguous' for r in diag)


def test_selection_is_independent_of_candidate_row_order():
    import random
    candidates = [
      {'upstream_pid':1, 'downstream_pid':2,'origin_cells':70,'parcel_cells':100,'share':.7,'origin_area_m2':70.},
      {'upstream_pid':2, 'downstream_pid':3,'origin_cells':70,'parcel_cells':100,'share':.7,'origin_area_m2':70.},
      {'upstream_pid':3, 'downstream_pid':1,'origin_cells':65,'parcel_cells':100,'share':.65,'origin_area_m2':65.},
    ]
    expected,_,_=routing.select_acyclic_connections([dict(x) for x in candidates])
    random.Random(7).shuffle(candidates)
    assert expected==routing.select_acyclic_connections(candidates)[0]
    assert len(expected)==2


def test_diagnostic_csv_retains_excluded_connections_and_outside_rows(tmp_path):
    rows = [
        {'upstream_pid':6, 'downstream_pid':None,'origin_cells':10,'origin_area_m2':9000,
         'parcel_cells':20,'share':.5,'decision':'unresolved_ambiguous','reason':'tied_majority'},
    ]
    path=tmp_path/'diag.csv'
    routing.write_candidate_diagnostics(path, rows)
    with path.open(newline='') as handle:
        data=list(csv.DictReader(handle))
    assert data[0]['downstream_pid']=='OUTSIDE'
    assert data[0]['reason']=='tied_majority'


def test_whitebox_frontend_pipeline_invokes_breach_then_d8(tmp_path, monkeypatch):
    dem=tmp_path/'dem.tif'; dem.write_bytes(b'input')
    events=[]
    class FakeWhitebox:
        def set_verbose_mode(self, verbose):
            assert verbose is False
        def breach_depressions_least_cost(self, dem, out, **kwargs):
            events.append(('breach',kwargs))
            Path(out).write_bytes(b'conditioned')
            return 0
        def d8_pointer(self, dem, out, **kwargs):
            events.append(('d8',kwargs))
            Path(out).write_bytes(b'pointer')
            return 0
    monkeypatch.setitem(sys.modules,'whitebox',SimpleNamespace(WhiteboxTools=FakeWhitebox))
    path=routing.run_whitebox_d8(dem,tmp_path/'out',breach_dist=150)
    assert path.is_file()
    assert events==[('breach',{'dist':150,'min_dist':True,'fill':True}),
                    ('d8',{'esri_pntr':False})]


def test_whitebox_run_propagates_tool_failure_and_preserves_gpkg(tmp_path, monkeypatch):
    dem=tmp_path/'dem.tif'; dem.write_bytes(b'dem')
    class BadWhitebox:
        def set_verbose_mode(self, verbose): pass
        def breach_depressions_least_cost(self, *args, **kwargs): return 1
    monkeypatch.setitem(sys.modules,'whitebox',SimpleNamespace(WhiteboxTools=BadWhitebox))
    with pytest.raises(RuntimeError,match='conditioning failed'):
        routing.run_whitebox_d8(dem,tmp_path/'work')


def test_cli_writes_diagnostics_and_normalized_gpkg_after_fake_whitebox(tmp_path, monkeypatch):
    import sqlite3
    from test_create_parcel_up_table import write_gpkg
    gpkg=write_gpkg(tmp_path/'parcels.gpkg')
    dem=tmp_path/'dem.tif'
    with rasterio.open(dem,'w',driver='GTiff',width=6,height=3,count=1,
                       dtype='float32',transform=from_origin(0,3,1,1),
                       crs='EPSG:3857') as ds:
        ds.write(np.tile(np.arange(6,0,-1,dtype='float32'),(3,1)),1)
    fake_whitebox_file_pipeline(monkeypatch)
    monkeypatch.setattr(sys,'argv',['create_parcel_up.py',str(gpkg),str(dem)])
    create_parcel_up.main()
    with sqlite3.connect(gpkg) as conn:
        edges=conn.execute('SELECT pid,pid_up FROM parcel_up ORDER BY pid').fetchall()
    assert edges==[(22,11),(33,22)]
    diagnostics=tmp_path/'parcels_parcel_up_candidates.csv'
    assert diagnostics.is_file()
    with diagnostics.open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert any(r['decision']=='selected' for r in rows)


def test_abort_on_ambiguity_leaves_existing_gpkg_unchanged(tmp_path, monkeypatch):
    from test_create_parcel_up_table import write_gpkg
    gpkg = write_gpkg(tmp_path/'parcels.gpkg')
    original = gpkg.read_bytes()
    dem = tmp_path/'dem.tif'
    with rasterio.open(dem,'w',driver='GTiff',width=6,height=3,count=1,
                       dtype='float32',transform=from_origin(0,3,1,1),
                       crs='EPSG:3857') as ds:
        ds.write(np.ones((3,6),dtype='float32'),1)
    fake_whitebox_file_pipeline(monkeypatch)
    original_select = routing.select_acyclic_connections
    def forced_ambiguity(candidates, **kwargs):
        edges, rows, summary = original_select(candidates, **kwargs)
        summary['ambiguous_parcels'] += 1
        return edges, rows, summary
    monkeypatch.setattr(routing,'select_acyclic_connections',forced_ambiguity)
    monkeypatch.setattr(sys,'argv',['create_parcel_up.py',str(gpkg),str(dem),
                                   '--abort-on-ambiguity'])
    with pytest.raises(ValueError,match='GeoPackage was not modified'):
        create_parcel_up.main()
    assert gpkg.read_bytes() == original
    assert (tmp_path/'parcels_parcel_up_candidates.csv').exists()


def test_whitebox_pointer_reader_rejects_misaligned_georeferencing(tmp_path):
    def make(path, x):
        with rasterio.open(path,'w',driver='GTiff',width=2,height=2,count=1,
                           dtype='uint8',transform=from_origin(x,2,1,1),
                           crs='EPSG:3857') as ds:
            ds.write(np.zeros((2,2),dtype='uint8'),1)
    dem=tmp_path/'dem.tif';ptr=tmp_path/'pointer.tif'
    make(dem,0); make(ptr,1)
    with pytest.raises(ValueError,match='not aligned'):
        routing.read_whitebox_pointers(ptr,dem)
