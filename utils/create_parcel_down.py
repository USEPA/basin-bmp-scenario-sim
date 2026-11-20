#!/usr/bin/env python3
"""Create parcel drainage connections using a local or USGS 3DEP DEM.

Run ``python utils/create_parcel_down.py --help`` for examples. If the DEM
positional argument is omitted, query USGS for full-area 3DEP availability,
allow the user to choose the resolution, download tiles and build a metric DEM.
The existing local-DEM path never makes network requests.
"""
import argparse
import logging
import sys
from typing import Dict, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive a representative, acyclic parcel_down network using "
                    "WhiteboxTools D8 hydrology and parcel-origin drainage shares."
    )
    parser.add_argument("parcels", help="Path to parcel boundaries (GeoPackage/GeoJSON/Shapefile). Must have an ID field.")
    parser.add_argument(
        "dem", nargs="?", default=None,
        help="Local DEM raster (GeoTIFF). Omit to discover and download a USGS 3DEP DEM.",
    )
    parser.add_argument(
        "--usgs-resolution", choices=("1m", "s1m", "3m", "5m", "10m", "30m", "60m"),
        help="USGS 3DEP resolution; still verifies full parcel-area availability. "
             "Skips interactive selection and download confirmation.",
    )
    parser.add_argument(
        "--list-usgs-resolutions", action="store_true",
        help="Query and display USGS 3DEP full-area coverage, then exit without downloading.",
    )
    parser.add_argument(
        "--refresh-usgs-availability", action="store_true",
        help="Ignore saved USGS availability checks and query all resolution indexes again.",
    )
    parser.add_argument(
        "--usgs-cache-dir", default=None,
        help="Directory for USGS source tiles and the projected DEM. "
             "Default: <parcels_basename>_usgs_dem beside the parcel file.",
    )
    parser.add_argument(
        "--dem-buffer-m", type=float, default=100.,
        help="Buffer in meters outside all parcel geometries for the USGS DEM (default: 100).",
    )
    parser.add_argument(
        "--max-dem-cells", type=int, default=20_000_000,
        help="Safety limit on output DEM cells before downloading (default: 20 million).",
    )
    parser.add_argument("--id-field", default="pid", help="Name of the parcel ID field (default: pid).")
    parser.add_argument(
        "--layer", default=None,
        help="Parcel geometry layer in a GeoPackage. Defaults to 'parcels' when present, "
             "otherwise the only geometry layer if there is just one.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Optional CSV export of immediate (pid,pid_down) edges, one per row. "
             "With a GeoPackage, the parcel_down table is written by default."
    )
    parser.add_argument(
        "--no-gpkg-update", action="store_true",
        help="Do not modify the input GeoPackage; requires --out for CSV-only output."
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use-filled-dem", action="store_true",
                        help="Deprecated compatibility alias for --whitebox-conditioning fill; "
                             "does not use richdem.")
    parser.add_argument(
        "--whitebox-conditioning", choices=("breach", "fill"), default="breach",
        help="WhiteboxTools depression conditioning. Default: least-cost breach "
             "with filling of unbreached depressions."
    )
    parser.add_argument("--whitebox-breach-dist", type=int, default=100,
                        help="Maximum least-cost breaching path search distance in "
                             "cells (default: 100).")
    parser.add_argument("--whitebox-exe", default=None,
                        help="Path to whitebox_tools executable; otherwise requires "
                             "the Python 'whitebox' package.")
    parser.add_argument("--whitebox-output-dir", default=None,
                        help="Directory for conditioned DEM and Whitebox D8 pointer "
                             "rasters. Default: <DEM_directory>/<DEM_stem>_whitebox_flow.")
    parser.add_argument("--max-overlap-m2", type=float, default=0.01,
                        help="Allowed polygon overlap in square meters (default 0.01).")
    parser.add_argument("--manifest-out", default=None,
                        help="Run manifest JSON; default <parcels_stem>_parcel_down_manifest.json.")
    parser.add_argument("--diagnostics-out", default=None,
                        help="CSV recording all parcel-local downstream destinations "
                             "and selected/excluded edges. Default: <parcels_stem>_parcel_down_candidates.csv.")
    return parser.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def _reject_missing_parcel_geometries(gdf: gpd.GeoDataFrame, id_field: str) -> None:
    """Never omit a parcel from drainage routing because its geometry is absent."""
    missing = gdf.geometry.isna() | gdf.geometry.is_empty
    if missing.any():
        pids = gdf.loc[missing, id_field].astype(str).tolist()
        sample = ', '.join(pids[:12]) + (' ...' if len(pids) > 12 else '')
        raise ValueError(
            f'Missing or empty parcel geometries for {len(pids)} parcel(s): '
            f'PID(s) {sample}. Repair the parcel layer before generating parcel_down.'
        )


def read_parcels(parcels_path: str, id_field: str, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read parcel geometries, including GeoPackage feature-ID primary keys.

    GDAL's GeoPackage driver treats the primary key as a special feature ID;
    Pyogrio does not expose it as a normal attribute. Recover the real FIDs,
    never a synthetic zero-based row index, when the requested field is the
    GeoPackage feature-ID column (e.g., ``pid``).
    """
    from pathlib import Path

    if Path(parcels_path).suffix.lower() == ".gpkg":
        import pyogrio

        layers = [str(name) for name, geometry_type in pyogrio.list_layers(parcels_path)
                  if geometry_type is not None]
        if layer is None:
            if "parcels" in layers:
                layer = "parcels"
            elif len(layers) == 1:
                layer = layers[0]
            else:
                raise ValueError(
                    f"GeoPackage {parcels_path} has multiple or no geometry layers "
                    f"({layers}); specify the parcel layer with --layer."
                )
        if layer not in layers:
            raise ValueError(
                f"Parcel geometry layer {layer!r} not found in {parcels_path}. "
                f"Available geometry layers: {layers}"
            )
        info = pyogrio.read_info(parcels_path, layer=layer)
        fid_column = info.get("fid_column")
        # fid_as_index preserves the real GeoPackage primary key values.
        gdf = gpd.read_file(
            parcels_path, layer=layer, engine="pyogrio", fid_as_index=True,
        )
        if id_field not in gdf.columns and id_field == fid_column:
            gdf[id_field] = gdf.index.to_numpy()
            logging.info(
                "Recovered parcel ID %r from GeoPackage feature IDs in "
                "layer %r (%d parcels).",
                id_field, layer, len(gdf),
            )
    else:
        # Preserve existing Shapefile / GeoJSON attribute handling.
        kwargs = {"layer": layer} if layer is not None else {}
        gdf = gpd.read_file(parcels_path, **kwargs)

    if id_field not in gdf.columns:
        raise ValueError(
            f"Parcel ID field {id_field!r} not found in {parcels_path}. "
            f"Columns: {list(gdf.columns)}. For a GeoPackage primary key, "
            "specify its name with --id-field."
        )
    # Input IDs must be unique to preserve one raster code per parcel.
    if gdf[id_field].isna().any():
        raise ValueError(f"Parcel ID field {id_field!r} contains null values")
    if gdf[id_field].astype(str).duplicated().any():
        raise ValueError(f"Parcel ID field {id_field!r} contains duplicate values")
    if gdf.empty:
        raise ValueError("No parcel records found.")
    _reject_missing_parcel_geometries(gdf, id_field)
    # Attempt to fix invalid geometries
    if not gdf.geometry.is_valid.all():
        logging.info("Fixing invalid parcel geometries with buffer(0)")
        gdf["geometry"] = gdf.geometry.buffer(0)
        _reject_missing_parcel_geometries(gdf, id_field)
    return gdf


def reproject_to_match(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Parcels have no CRS. Please define a CRS before running.")
    if target_crs is None:
        raise ValueError("DEM has no CRS. DEM must have a valid CRS.")
    if gdf.crs != target_crs:
        logging.info(f"Reprojecting parcels from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf


def rasterize_parcels(
    gdf: gpd.GeoDataFrame,
    id_field: str,
    out_shape: Tuple[int, int],
    transform
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, int]]:
    """
    Rasterize parcels to match DEM grid.
    Returns:
      - pid_raster: int array where each cell is code for parcel id, 0 for background
      - code_to_pid: dict mapping int code -> original pid (string repr)
      - pid_to_code: dict mapping original pid (string repr) -> int code
    """
    pids = gdf[id_field].astype(str).tolist()
    unique_pids = list(dict.fromkeys(pids))  # preserve order as in gdf
    pid_to_code = {pid: (i + 1) for i, pid in enumerate(unique_pids)}  # 0 is background
    code_to_pid = {v: k for k, v in pid_to_code.items()}

    shapes = ((geom, pid_to_code[str(pid)]) for geom, pid in zip(gdf.geometry, gdf[id_field]))
    pid_raster = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.int32
    )
    return pid_raster, code_to_pid, pid_to_code


def main():
    args = parse_args()
    setup_logging(args.log_level)

    from pathlib import Path
    is_gpkg = Path(args.parcels).suffix.lower() == ".gpkg"
    if args.no_gpkg_update and not args.out:
        raise ValueError("--no-gpkg-update requires --out for a CSV-only result")
    if not is_gpkg and not args.out:
        raise ValueError("A non-GeoPackage parcel input requires --out CSV")
    if is_gpkg and not args.no_gpkg_update:
        if args.id_field != "pid" or (args.layer is not None and args.layer != "parcels"):
            raise ValueError(
                "The model's GeoPackage schema requires layer 'parcels' and ID 'pid'; "
                "use --no-gpkg-update --out FILE for a different schema"
            )

    if args.dem is not None and (args.usgs_resolution or args.list_usgs_resolutions
                                 or args.refresh_usgs_availability):
        raise ValueError("USGS availability options cannot be used with a local DEM.")
    if args.dem_buffer_m < 0 or not np.isfinite(args.dem_buffer_m):
        raise ValueError("--dem-buffer-m must be a finite nonnegative number.")
    if args.max_dem_cells < 1:
        raise ValueError("--max-dem-cells must be positive.")

    logging.info("Reading parcels: %s", args.parcels)
    gdf = read_parcels(args.parcels, args.id_field, layer=args.layer)
    if args.dem is None:
        # Support both direct execution (`python utils/create_parcel_down.py`) and
        # package import (`from utils import create_parcel_down`).
        if __package__:
            from .usgs_dem import get_dem_from_usgs
        else:
            from usgs_dem import get_dem_from_usgs
        from pathlib import Path
        parcels_path = Path(args.parcels)
        cache_dir = (Path(args.usgs_cache_dir) if args.usgs_cache_dir else
                     parcels_path.with_name(parcels_path.stem + "_usgs_dem"))
        dem_path = get_dem_from_usgs(
            gdf, resolution=args.usgs_resolution, buffer_m=args.dem_buffer_m,
            cache_dir=cache_dir, max_cells=args.max_dem_cells,
            list_only=args.list_usgs_resolutions,
            refresh_availability=args.refresh_usgs_availability,
        )
        if dem_path is None:  # --list-usgs-resolutions
            return
        args.dem = str(dem_path)

    # WhiteboxTools, not a custom steepest-descent / uphill fallback,
    # supplies a depression-conditioned D8 pointer raster. Keep its outputs so
    # apparent cycles or drainage choices can be inspected in GIS afterward.
    from pathlib import Path
    if __package__:
        from .whitebox_parcel_routing import (
            run_whitebox_d8, read_whitebox_pointers, validate_cell_dag,
            compute_parcel_origin_destinations, score_candidate_connections,
            select_acyclic_connections, write_candidate_diagnostics,
        )
        from .parcel_down_graph import (validate_direct_edges,
                                      write_gpkg_parcel_down, write_normalized_csv)
    else:
        from whitebox_parcel_routing import (
            run_whitebox_d8, read_whitebox_pointers, validate_cell_dag,
            compute_parcel_origin_destinations, score_candidate_connections,
            select_acyclic_connections, write_candidate_diagnostics,
        )
        from parcel_down_graph import (validate_direct_edges,
                                     write_gpkg_parcel_down, write_normalized_csv)

    if __package__:
        from .drainage_validation import validate_dem_coverage, validate_parcel_overlaps
        from .drainage_manifest import write_drainage_manifest
    else:
        from drainage_validation import validate_dem_coverage, validate_parcel_overlaps
        from drainage_manifest import write_drainage_manifest

    conditioning = args.whitebox_conditioning
    if args.use_filled_dem:
        conditioning = 'fill'
        logging.warning('--use-filled-dem is deprecated; using WhiteboxTools FillDepressions')

    dem_path = Path(args.dem).resolve()
    wb_dir = (Path(args.whitebox_output_dir) if args.whitebox_output_dir else
              dem_path.parent / (dem_path.stem + '_whitebox_flow'))
    with rasterio.open(dem_path) as dem_ds:
        if dem_ds.crs is None or not dem_ds.crs.is_projected:
            raise ValueError('DEM must have a projected coordinate system')
        projected_parcels = reproject_to_match(gdf, dem_ds.crs)
        unit_to_m = float(dem_ds.crs.linear_units_factor[1])
        validate_parcel_overlaps(projected_parcels, args.id_field,
                                 tolerance_m2=args.max_overlap_m2,
                                 linear_units_to_m=unit_to_m)
        try:
            from .rasterio_compat import read_mask as _dem_read_mask
        except ImportError:
            from rasterio_compat import read_mask as _dem_read_mask
        validate_dem_coverage(projected_parcels, dem_ds,
                              _dem_read_mask(dem_ds) > 0,
                              linear_units_to_m=unit_to_m)
    logging.info('Parcel overlap and DEM extent/NoData checks passed.')
    logging.info('Conditioning DEM and calculating flow with WhiteboxTools...')
    pointer_path = run_whitebox_d8(
        dem_path, wb_dir, method=conditioning,
        breach_dist=args.whitebox_breach_dist, exe=args.whitebox_exe,
    )
    with rasterio.open(args.dem) as src:
        dem_crs = src.crs
        transform = src.transform
        dem_shape = src.shape
        if dem_crs is None or not dem_crs.is_projected:
            raise ValueError('DEM must have a projected CRS for D8 and parcel area calculations')
        unit_factor = float(dem_crs.linear_units_factor[1])
        if unit_factor <= 0 or not np.isfinite(unit_factor):
            raise ValueError('Could not determine projected DEM linear units')
        cell_area_m2 = abs(transform.a * transform.e - transform.b * transform.d) * unit_factor**2
    next_flat, valid_mask = read_whitebox_pointers(pointer_path, dem_path)
    logging.info('Validating all Whitebox D8 flow directions for cycles...')
    validate_cell_dag(next_flat)

    gdf = reproject_to_match(gdf, dem_crs)
    logging.info('Rasterizing parcels to Whitebox D8 grid...')
    pid_raster, code_to_pid, pid_to_code = rasterize_parcels(
        gdf, args.id_field, out_shape=dem_shape, transform=transform
    )
    if np.any((pid_raster != 0) & ~valid_mask):
        raise ValueError('The DEM has NoData in parcel raster cells; cannot infer drainage')
    logging.info('Tracing drainage originating inside every parcel cell...')
    destinations = compute_parcel_origin_destinations(pid_raster, next_flat)
    candidates = score_candidate_connections(
        pid_raster, destinations, code_to_pid, cell_area_m2=cell_area_m2,
    )
    edges, diagnostics, summary = select_acyclic_connections(candidates)
    # Sub-cell parcels cannot be evaluated at this raster resolution. Report
    # them instead of inventing flow directions or silently pretending they
    # have been represented.
    represented = set(np.unique(pid_raster[pid_raster > 0]))
    missing = sorted(set(code_to_pid) - represented)
    for code in missing:
        diagnostics.append({
            'upstream_pid': int(code_to_pid[code]), 'downstream_pid': None,
            'origin_cells': 0, 'origin_area_m2': 0., 'parcel_cells': 0,
            'share': 0., 'decision': 'unresolved_not_rasterized',
            'reason': 'parcel_smaller_than_dem_cell_or_not_rasterized',
        })
    logging.info('Whitebox parcel routing: %d selected links; %d cycle-causing '
                 'candidates excluded; %d ambiguous parcels; %d unrasterized '
                 'parcels; %d outlets to unmodeled terrain',
                 len(edges), summary['rejected_cycle_candidates'],
                 summary['ambiguous_parcels'], len(missing), summary['outside_parcels'])
    if summary['ambiguous_parcels'] or missing or summary['rejected_cycle_candidates']:
        logging.warning('Some parcel connections were not inferred or were rejected. '
                        'Inspect the candidate diagnostic CSV before using parcel_down.')
    diagnostic_path = (Path(args.diagnostics_out) if args.diagnostics_out else
                       Path(args.parcels).with_name(Path(args.parcels).stem +
                       '_parcel_down_candidates.csv'))
    write_candidate_diagnostics(diagnostic_path, diagnostics)
    logging.info('Candidate drainage evidence written to %s', diagnostic_path)
    edges = validate_direct_edges(edges, gdf[args.id_field].tolist())
    logging.info('Computed %d acyclic, representative downstream connections', len(edges))

    if is_gpkg and not args.no_gpkg_update:
        count = write_gpkg_parcel_down(args.parcels, edges, gdf[args.id_field])
        logging.info("Atomically replaced parcel_down in %s (%d rows)", args.parcels, count)
    if args.out:
        write_normalized_csv(args.out, edges)
        logging.info("Wrote normalized immediate-edge CSV to %s", args.out)

    manifest_path = (Path(args.manifest_out) if args.manifest_out else
                     Path(args.parcels).with_name(Path(args.parcels).stem +
                     '_parcel_down_manifest.json'))
    write_drainage_manifest(
        manifest_path, parcels=gdf, id_field=args.id_field,
        dem_path=dem_path, pointer_path=pointer_path,
        edges=edges,
        parameters={'whitebox_conditioning': conditioning,
                    'whitebox_breach_dist': args.whitebox_breach_dist,
                    'max_overlap_m2': args.max_overlap_m2,
                    'dem_resolution': args.usgs_resolution,
                    'whole_parcel_drainage': True},
        summary={**summary, 'unrasterized_parcels': len(missing),
                 'unrasterized_pids': [int(code_to_pid[c]) for c in missing]},
        diagnostics_path=diagnostic_path,
    )
    logging.info('Drainage provenance manifest written to %s', manifest_path)
    logging.info("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)