"""Strict parcel_down schema, safe migration, and spatial validation tests."""
from __future__ import annotations

import json
import sqlite3

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.input_config import _build_parcel_down_map, _load_parcel_graph
from utils.parcel_down_graph import validate_direct_edges, write_gpkg_parcel_down
from utils.drainage_validation import validate_dem_coverage, validate_parcel_overlaps
from utils.drainage_manifest import write_drainage_manifest, parcel_geometry_sha256
from utils.migrate_parcel_down import migrate
from utils.whitebox_parcel_routing import select_acyclic_connections
from utils.rasterio_compat import read_mask
from test_create_parcel_down_table import write_gpkg


class Logger:
    def verbose(self,*a): pass


def test_strict_loader_rejects_legacy_only(tmp_path):
    gpkg=write_gpkg(tmp_path/'input.gpkg')
    with sqlite3.connect(gpkg) as conn:
        conn.execute('ALTER TABLE parcel_down RENAME TO parcel_up')
        conn.execute("UPDATE gpkg_contents SET table_name='parcel_up',identifier='parcel_up' "
                     "WHERE table_name='parcel_down'")
    with pytest.raises(ValueError, match='Required.*parcel_down.*Legacy parcel_up is not supported'):
        _load_parcel_graph({'parcels': str(gpkg)},Logger())


def test_migration_removes_legacy_and_enforces_single_destination(tmp_path):
    gpkg=write_gpkg(tmp_path/'input.gpkg')
    with sqlite3.connect(gpkg) as conn:
        conn.execute('ALTER TABLE parcel_down RENAME TO parcel_up')
        conn.execute("UPDATE gpkg_contents SET table_name='parcel_up',identifier='parcel_up' "
                     "WHERE table_name='parcel_down'")
    assert migrate(gpkg, [(11,22),(22,33)], backup=False)==2
    with sqlite3.connect(gpkg) as conn:
        assert conn.execute('SELECT pid,pid_down FROM parcel_down ORDER BY pid').fetchall()==[(11,22),(22,33)]
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='parcel_up'").fetchone() is None
        assert conn.execute('PRAGMA foreign_key_check').fetchall()==[]
        assert conn.execute("SELECT table_name FROM gpkg_contents WHERE table_name='parcel_down'").fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('INSERT INTO parcel_down(pid,pid_down) VALUES(11,33)')


def test_loader_one_downstream_many_sources_and_cycle_rejection():
    d=pd.DataFrame([{'pid':1,'pid_down':3},{'pid':2,'pid_down':3},{'pid':3,'pid_down':4}])
    assert _build_parcel_down_map(d,[1,2,3,4]) == {'1':'3','2':'3','3':'4','4':None}
    with pytest.raises(ValueError,match='directed cycle'):
        _build_parcel_down_map(pd.DataFrame([{'pid':1,'pid_down':2},
                                             {'pid':2,'pid_down':1}]),[1,2])
    with pytest.raises(ValueError,match='more than one'):
        _build_parcel_down_map(pd.DataFrame([{'pid':1,'pid_down':2},
                                             {'pid':1,'pid_down':3}]),[1,2,3])


def test_select_largest_even_when_below_half_and_ties_by_numeric_pid():
    evidence=[
        {'upstream_pid':11,'downstream_pid':15,'share':.4,'origin_cells':40,'parcel_cells':100,'origin_area_m2':40},
        {'upstream_pid':11,'downstream_pid':14,'share':.4,'origin_cells':40,'parcel_cells':100,'origin_area_m2':40},
        {'upstream_pid':11,'downstream_pid':None,'share':.2,'origin_cells':20,'parcel_cells':100,'origin_area_m2':20},
    ]
    edges,diags,stats=select_acyclic_connections(evidence)
    assert edges == [(11,14)]
    assert stats['selected_edges'] == 1
    assert next(x for x in diags if x['downstream_pid']==15)['decision']=='rejected_secondary'


def _parcels(*polygons):
    return gpd.GeoDataFrame({'pid':list(range(1,len(polygons)+1))}, geometry=list(polygons), crs='EPSG:3857')


def _dem(path,*,size=(10,10),transform=None,nodata=None):
    with rasterio.open(path,'w',driver='GTiff',height=size[0],width=size[1],count=1,
                       crs='EPSG:3857',transform=transform or from_origin(0,10,1,1),
                       dtype='float32',nodata=nodata) as dst:
        arr=np.ones(size,dtype='float32')
        if nodata is not None: arr[0,0]=nodata
        dst.write(arr,1)


def test_dem_coverage_rejects_parcel_99_percent_outside(tmp_path):
    path=tmp_path/'dem.tif';_dem(path)
    parcels=_parcels(box(0,0,100,100))
    with rasterio.open(path) as ds:
        with pytest.raises(ValueError,match='does not cover full parcel area'):
            validate_dem_coverage(parcels,ds,read_mask(ds)>0)


def test_dem_coverage_rejects_nodata_inside(tmp_path):
    path=tmp_path/'dem.tif';_dem(path,nodata=-9999)
    parcels=_parcels(box(0,0,10,10))
    with rasterio.open(path) as ds:
        with pytest.raises(ValueError,match='NoData'):
            validate_dem_coverage(parcels,ds,read_mask(ds)>0)


def test_dem_coverage_rejects_partial_nodata_pixel_without_center(tmp_path):
    """A narrow parcel can cross a NoData cell without containing its center."""
    path = tmp_path / 'partial_nodata.tif'
    _dem(path, size=(1, 2), transform=from_origin(0, 1, 1, 1), nodata=-9999)
    parcels = _parcels(box(0.8, 0.2, 1.2, 0.8))
    # The left pixel is NoData. Only 0.2 m * 0.6 m of the parcel
    # intersects it; its center (0.5, 0.5) is outside the parcel.
    with rasterio.open(path) as ds:
        mask = read_mask(ds) > 0
        with pytest.raises(ValueError, match='NoData or nonfinite elevations in parcel area'):
            validate_dem_coverage(parcels, ds, mask)
        with pytest.raises(ValueError, match='NoData or nonfinite elevations in parcel area'):
            validate_dem_coverage(parcels, ds, mask, tolerance_m2=0.10)
        validate_dem_coverage(parcels, ds, mask, tolerance_m2=0.13)


def test_dem_coverage_accepts_boundary_only_contact_with_nodata(tmp_path):
    """all_touched candidate cells must not cause false-positive NoData."""
    path = tmp_path / 'boundary_nodata.tif'
    _dem(path, size=(1, 2), transform=from_origin(0, 1, 1, 1), nodata=-9999)
    parcels = _parcels(box(1.0, 0.2, 1.2, 0.8))
    with rasterio.open(path) as ds:
        validate_dem_coverage(parcels, ds, read_mask(ds) > 0)


def test_overlap_detection_is_independent_of_feature_order():
    first=box(0,0,2,2);second=box(1,0,3,2)
    for polygons in [(first,second),(second,first)]:
        with pytest.raises(ValueError,match='Overlapping parcel geometries'):
            validate_parcel_overlaps(_parcels(*polygons),'pid')


def test_manifest_geometry_hash_changes_only_with_geometry_or_ids(tmp_path):
    p=_parcels(box(0,0,3,3))
    dem=tmp_path/'dem.tif';_dem(dem)
    first=parcel_geometry_sha256(p,'pid')
    q=p.copy();q['unrelated']=99
    assert parcel_geometry_sha256(q,'pid') == first
    q.geometry=[box(0,0,4,3)]
    assert parcel_geometry_sha256(q,'pid') != first
    manifest=tmp_path/'manifest.json'
    write_drainage_manifest(manifest, parcels=p, id_field='pid',
                            dem_path=dem,pointer_path=dem,edges=[(1,2)],
                            summary={'selected_edges':1},parameters={'conditioning':'breach'},
                            diagnostics_path=tmp_path/'candidates.csv')
    document=json.loads(manifest.read_text())
    assert document['schema'] == 'parcel_down(pid,pid_down)'
    assert document['outputs']['edge_count']==1
    assert document['source']['parcel_geometry_sha256']==first


def test_model_reverses_downstream_links_for_wetland_traversal(tmp_path):
    """A/B both drain into C; C drains into D; wetland traverses the reverse."""
    import logging
    from src.model import Model, _ScenarioContext
    from model_test_cases import make_statistical_case
    cfg, data = make_statistical_case(tmp_path, n_scenarios=1)
    # P1 and P2 converge to P3. The model has exactly one downstream per PID.
    data['parcel_down_map'] = {'P1':'P3','P2':'P3','P3':None}
    model = Model(cfg,data,logging.getLogger('parcel.down.reverse.test'))
    assert model._get_parcel_up_list('P3') == ['P1','P2']
    assert model.parcel_up_idxs[model.pid_to_index['P3']] == [0,1]
    scenario = _ScenarioContext(cfg,model._shared_payload(),logging.getLogger('parcel.down.ctx'),123)
    assert scenario._get_parcel_up_list('P3') == ['P1','P2']
