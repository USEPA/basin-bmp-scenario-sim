"""Reproducible fingerprints for parcel-to-downstream drainage graph generation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from datetime import datetime, timezone

import geopandas as gpd
import rasterio


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parcel_geometry_sha256(parcels: gpd.GeoDataFrame, id_field: str) -> str:
    """Hash parcel identifiers, CRS, and geometry, independent of GPKG metadata."""
    h = hashlib.sha256()
    h.update(str(parcels.crs).encode())
    for _, row in parcels.sort_values(id_field, key=lambda x: x.astype(int)).iterrows():
        h.update(str(row[id_field]).encode() + b'\0')
        h.update(bytes(row.geometry.wkb) + b'\0')
    return h.hexdigest()


def write_drainage_manifest(path: str | Path, *, parcels, id_field: str,
                            dem_path: str | Path, pointer_path: str | Path,
                            edges: list[tuple[int,int]], parameters: dict,
                            summary: dict, diagnostics_path: str | Path) -> None:
    """Write after a successful run; path is atomically replaced on success."""
    with rasterio.open(dem_path) as src:
        raster = {'crs': str(src.crs), 'transform': list(src.transform)[:6],
                  'width': src.width, 'height': src.height,
                  'bounds': list(src.bounds), 'nodata': src.nodata}
    try:
        import whitebox
        wbt_version = str(getattr(whitebox, '__version__', 'unknown'))
    except ImportError:
        wbt_version = 'external_executable_or_unavailable'
    import numpy, geopandas
    try:
        import pyogrio
        pyogrio_version = str(getattr(pyogrio, '__version__', 'unknown'))
    except ImportError:
        pyogrio_version = 'unavailable'
    edges_hash = hashlib.sha256(''.join(f'{a},{b}\n' for a,b in sorted(edges)).encode()).hexdigest()
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'schema': 'parcel_down(pid,pid_down)', 'routing': 'whole-parcel-dominant',
        'source': {'parcel_count': len(parcels),
                   'parcel_geometry_sha256': parcel_geometry_sha256(parcels, id_field),
                   'dem_sha256': _file_sha256(dem_path),
                   'd8_pointer_sha256': _file_sha256(pointer_path)},
        'dem': raster,
        'parameters': parameters,
        'quality': summary,
        'outputs': {'edge_count': len(edges), 'edges_sha256': edges_hash,
                    'diagnostics_path': str(diagnostics_path)},
        'software': {'python': platform.python_version(),
                     'rasterio': rasterio.__version__, 'geopandas': geopandas.__version__,
                     'numpy': numpy.__version__, 'pyogrio': pyogrio_version,
                     'whitebox': wbt_version},
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, default=str)
            stream.write('\n')
        os.replace(tmp,path)
    finally:
        Path(tmp).unlink(missing_ok=True)
