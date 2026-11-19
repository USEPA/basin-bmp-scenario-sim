"""Discover, download and mosaic USGS 3DEP DEMs for a parcel study area.

Availability preferentially uses the USGS 3DEP elevation *index* (polygons).
When an index layer is unavailable, TNMAccess catalog product bounding boxes
provide clearly labeled *provisional* availability, not definitive raster
coverage. Downloadable products come from the official TNMAccess API.  Final
raster validity is checked after mosaicking; published tile footprints are not
a guarantee that every cell contains elevation data.
"""

from __future__ import annotations

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import logging
import hashlib
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import geopandas as gpd
import numpy as np
import rasterio

try:
    from .rasterio_compat import read_band, read_mask
except ImportError:  # Direct execution of utils/create_parcel_up.py
    from rasterio_compat import read_band, read_mask
from rasterio import features
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely.geometry import box, shape, mapping
from shapely.ops import unary_union

INDEX_URL = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer"
PRODUCTS_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
S1M_INDEX_URL = "https://index.nationalmap.gov/arcgis/rest/services/USGS_Seamless1m_Index/MapServer"
# Dataset tags are published by https://tnmaccess.nationalmap.gov/api/v1/datasets
# Layer IDs are published by the USGS 3DEP Elevation Index MapServer.


@dataclass(frozen=True)
class Resolution:
    key: str
    label: str
    meters: int
    index_layer: int
    dataset: str
    index_url: str = INDEX_URL


RESOLUTIONS = (
    Resolution("1m", "1 meter (standard tiles)", 1, 1, "Digital Elevation Model (DEM) 1 meter"),
    Resolution("s1m", "1 meter seamless (S1M; limited availability)", 1, 0,
               "Seamless 1-m DEM (S1M)", S1M_INDEX_URL),
    Resolution("3m", "1/9 arc-second (~3 meters)", 3, 2,
               "National Elevation Dataset (NED) 1/9 arc-second"),
    Resolution("5m", "5 meters (Alaska)", 5, 3, "Alaska IFSAR 5 meter DEM"),
    Resolution("10m", "1/3 arc-second (~10 meters)", 10, 4,
               "National Elevation Dataset (NED) 1/3 arc-second"),
    Resolution("30m", "1 arc-second (~30 meters)", 30, 5,
               "National Elevation Dataset (NED) 1 arc-second"),
    Resolution("60m", "2 arc-second (~60 meters; Alaska)", 60, 6,
               "National Elevation Dataset (NED) Alaska 2 arc-second"),
)


class USGSDownloadError(RuntimeError):
    """A USGS data discovery, download, or coverage check failed."""


def _json_get(url: str, params: dict, *, timeout: float = 12) -> dict:
    """Fetch API JSON with a short discovery timeout and one transient retry.

    4xx errors other than HTTP 429 are not retried because they typically
    indicate an invalid query. A repeated 500 is a service error, not proof
    that the queried resolution has no coverage.
    """
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url + "?" + query,
        headers={"User-Agent": "basin-bmp-scenario-sim/create_parcel_up (USGS 3DEP)",
                 "Accept": "application/json"},
    )
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                obj = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < 1:
                logging.warning("USGS HTTP %s at %s; retrying (%d/2)",
                                exc.code, url, attempt + 1)
                time.sleep(0.35 * 2**attempt)
                continue
            raise USGSDownloadError(
                f"USGS service request failed ({url}): HTTP {exc.code} {exc.reason}"
            ) from exc
        except (OSError, ValueError, urllib.error.URLError) as exc:
            # Connection resets and timeouts are transient, but bounded.
            if attempt < 1:
                logging.warning("USGS connection problem at %s; retrying (%d/2): %s",
                                url, attempt + 1, exc)
                time.sleep(0.35 * 2**attempt)
                continue
            raise USGSDownloadError(f"USGS service request failed ({url}): {exc}") from exc
        if not isinstance(obj, dict) or obj.get("error") or obj.get("errors"):
            # ArcGIS can return a JSON error with HTTP 200. Preserve its error.
            raise USGSDownloadError(f"USGS service returned an error ({url}): {str(obj)[:500]}")
        return obj
    raise AssertionError("unreachable: retry loop must return or raise")


class CoverageMap(dict):
    """Coverage fractions with provenance; None means service status unknown.

    Source is ``index`` (polygon query), ``catalog`` (downloadable product
    bounding boxes), or ``unknown``. Catalog coverage is provisional until
    downloaded DEM pixels are verified for NoData.
    """

    def __init__(self):
        super().__init__()
        self.sources: dict[str, str] = {}
        self.errors: dict[str, str] = {}


def _metric_aoi(parcels: gpd.GeoDataFrame, buffer_m: float):
    if parcels.crs is None:
        raise ValueError("Parcel file lacks a CRS; define its CRS before querying USGS.")
    utm = parcels.estimate_utm_crs()
    if utm is None:
        raise ValueError("Could not determine a suitable projected CRS for this study area.")
    projected = parcels.to_crs(utm)
    aoi = unary_union([g for g in projected.geometry if g is not None and not g.is_empty])
    if aoi.is_empty or aoi.area <= 0:
        raise ValueError("No nonempty parcel polygons were found.")
    aoi = aoi.buffer(buffer_m)
    geo = gpd.GeoSeries([aoi], crs=utm).to_crs("EPSG:4326").iloc[0]
    if geo.bounds[0] >= geo.bounds[2] or geo.bounds[2] - geo.bounds[0] > 180:
        raise USGSDownloadError("Antimeridian-spanning study areas are not supported.")
    return aoi, geo, utm


def _index_coverage(layer: int, aoi_geographic, *, index_url: str = INDEX_URL) -> float:
    """Compute fraction of the *entire* buffered AOI covered by indexed polygons."""
    bbox = aoi_geographic.bounds
    params = {
        "f": "geojson", "where": "1=1", "geometry": ",".join(map(str, bbox)),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects", "outFields": "OBJECTID",
        "returnGeometry": "true", "resultRecordCount": 200,
    }
    polygons = []
    offset = 0
    while True:
        data = _json_get(f"{index_url}/{layer}/query", {**params, "resultOffset": offset})
        records = data.get("features")
        if not isinstance(records, list):
            raise USGSDownloadError("USGS 3DEP availability index returned no feature list.")
        for item in records:
            geom = item.get("geometry")
            if geom:
                polygon = shape(geom)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if not polygon.is_empty:
                    polygons.append(polygon)
        offset += len(records)
        if offset > 5000:
            raise USGSDownloadError("USGS index returned over 5,000 features; narrow the study area.")
        if not records or (len(records) < 200 and not data.get("exceededTransferLimit")):
            break
    if not polygons:
        return 0.0
    union = unary_union(polygons)
    # A fractional area is diagnostic; accept only effectively complete coverage.
    missing = aoi_geographic.difference(union)
    return max(0.0, min(1.0, 1.0 - missing.area / aoi_geographic.area))


# Cached *coverage* only, never downloaded elevation products. USGS metadata
# can be updated, so verified index results expire after one day and provisional
# catalog footprints expire after three hours. Unknown errors are never cached.
_AVAILABILITY_CACHE_VERSION = 1
_INDEX_CACHE_TTL_S = 24 * 60 * 60
_CATALOG_CACHE_TTL_S = 3 * 60 * 60
_AVAILABILITY_WORKERS = 4


def _coverage_cache_key(aoi_geo) -> str:
    """Key on the exact buffered AOI and configured resolution sources."""
    specs = [(r.key, r.index_url, r.index_layer, r.dataset) for r in RESOLUTIONS]
    material = aoi_geo.wkb + json.dumps(specs, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _read_coverage_cache(path: Path | None, key: str) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if (obj.get("version") != _AVAILABILITY_CACHE_VERSION
                or obj.get("aoi_sha256") != key):
            return {}
        return obj.get("results", {}) if isinstance(obj.get("results"), dict) else {}
    except (OSError, ValueError, TypeError, AttributeError):
        logging.warning("Ignoring unreadable USGS availability cache: %s", path)
        return {}


def _write_coverage_cache(path: Path | None, key: str, results: dict) -> None:
    if path is None or not results:
        return
    # Atomic replacement avoids leaving truncated JSON after interruption.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps({
            "version": _AVAILABILITY_CACHE_VERSION, "aoi_sha256": key,
            "results": results,
        }, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        logging.warning("Could not save USGS availability cache (%s): %s", path, exc)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _query_one_resolution(res: Resolution, geo):
    """Independent index + catalog fallback; service errors are never 'no data'."""
    try:
        return (_index_coverage(res.index_layer, geo, index_url=res.index_url),
                "index", None)
    except USGSDownloadError as index_exc:
        logging.warning("USGS %s index unavailable: %s", res.key, index_exc)
        try:
            amount = _catalog_coverage(res, geo)
            logging.info("Used TNMAccess provisional coverage for %s", res.key)
            return amount, "catalog", None
        except USGSDownloadError as catalog_exc:
            logging.warning("Cannot verify %s through either USGS service: %s",
                            res.key, catalog_exc)
            return None, "unknown", (
                f"index: {index_exc}; product catalog: {catalog_exc}")


def check_resolutions(parcels: gpd.GeoDataFrame, *, buffer_m: float = 100,
                      cache_dir: Path | None = None, refresh: bool = False,
                      only_resolution: str | None = None):
    """Check full-area 3DEP coverage concurrently; reuse fresh cached evidence.

    Results remain exact index polygon coverage or provisional TNMAccess tile
    bounding-box coverage. Unknown service failures cannot become a false NONE.
    Each resolution is independent and the printed order remains stable.
    """
    if only_resolution is not None and only_resolution not in {r.key for r in RESOLUTIONS}:
        raise ValueError(f"Unsupported USGS DEM resolution: {only_resolution}")
    local, geo, crs = _metric_aoi(parcels, buffer_m)
    logging.info("Checking USGS 3DEP availability for the full buffered parcel area...")
    coverage = CoverageMap()
    path = Path(cache_dir) / "usgs_availability_cache.json" if cache_dir is not None else None
    key = _coverage_cache_key(geo)
    stored = {} if refresh else _read_coverage_cache(path, key)
    now = time.time()
    records = {}
    pending = []
    cache_hits = 0
    for res in RESOLUTIONS:
        record = stored.get(res.key)
        valid_cached = False
        if isinstance(record, dict):
            source = record.get("source")
            ttl = _INDEX_CACHE_TTL_S if source == "index" else _CATALOG_CACHE_TTL_S
            timestamp = record.get("checked_at")
            value = record.get("coverage")
            if (source in {"index", "catalog"} and isinstance(timestamp, (int, float))
                    and isinstance(value, (int, float)) and math.isfinite(value)
                    and 0 <= value <= 1 and 0 <= now - timestamp < ttl):
                records[res.key] = record
                valid_cached = True
        if only_resolution is not None and res.key != only_resolution:
            coverage[res.key] = None
            coverage.sources[res.key] = "skipped"
            continue
        if valid_cached:
            coverage[res.key] = float(records[res.key]["coverage"])
            coverage.sources[res.key] = records[res.key]["source"]
            cache_hits += 1
        else:
            pending.append(res)
    if cache_hits:
        logging.info("Reusing cached USGS availability for %d requested resolution(s)",
                     cache_hits)
    if pending:
        logging.info("Querying %d resolution(s) using up to %d simultaneous USGS checks",
                     len(pending), min(len(pending), _AVAILABILITY_WORKERS))
        with ThreadPoolExecutor(max_workers=min(len(pending), _AVAILABILITY_WORKERS)) as executor:
            futures = {executor.submit(_query_one_resolution, res, geo): res
                       for res in pending}
            for count, future in enumerate(as_completed(futures), 1):
                res = futures[future]
                try:
                    fraction, source, error = future.result()
                except USGSDownloadError as exc:
                    # A service failure is UNKNOWN, not evidence of zero coverage.
                    logging.warning("USGS availability unavailable for %s: %s", res.key, exc)
                    fraction, source, error = None, "unknown", str(exc)
                coverage[res.key] = fraction
                coverage.sources[res.key] = source
                if error:
                    coverage.errors[res.key] = error
                if source in {"index", "catalog"} and fraction is not None:
                    records[res.key] = {
                        "coverage": fraction, "source": source, "checked_at": time.time()
                    }
                logging.info("USGS availability: %d/%d checks complete (%s)",
                             count, len(pending), res.key)
        _write_coverage_cache(path, key, records)
    if all(value is None for value in coverage.values()):
        summary = (f"USGS availability check failed for {only_resolution}. "
                   if only_resolution is not None else
                   "All USGS index queries and catalog fallbacks failed. ")
        raise USGSDownloadError(
            summary +
            "The services may be temporarily unavailable. Retry later or "
            "supply a local DEM as the second positional argument. "
            f"Example: {next(iter(coverage.errors.values()), 'no details')}"
        )
    return local, geo, crs, coverage


def show_resolutions(coverage: dict[str, float | None]):
    print("\nUSGS 3DEP DEM coverage for the entire parcel study area (including buffer):")
    catalog_used = False
    for i, res in enumerate(RESOLUTIONS, 1):
        fraction = coverage.get(res.key)
        source = getattr(coverage, "sources", {}).get(res.key, "index")
        if source == "skipped":
            state = "NOT CHECKED (use --list-usgs-resolutions)"
        elif fraction is None:
            state = "UNKNOWN (USGS service unavailable)"
        elif fraction >= 0.999999:
            state = "FULL" if source == "index" else "CATALOG FULL*"
        elif fraction:
            state = f"PARTIAL ({fraction:.1%})" if source == "index" else f"CATALOG PARTIAL* ({fraction:.1%})"
        else:
            state = "NONE" if source == "index" else "NO DOWNLOADABLE PRODUCTS*"
        catalog_used |= source == "catalog"
        print(f"  {i}. {res.key:>3}  {res.label:<39} {state}")
    print("Only full-coverage resolutions can be selected. Raster NoData is "
          "verified after downloading.")
    if catalog_used:
        print("* The USGS coverage-index service failed for these resolutions. "
              "TNMAccess tile bounding boxes provide provisional coverage, "
              "not proof that all raster cells contain valid elevations.")
    print()


def choose_resolution(coverage: dict[str, float], choice: str | None = None, *, input_fn=input) -> Resolution:
    options = [r for r in RESOLUTIONS
               if coverage.get(r.key) is not None and coverage[r.key] >= 0.999999]
    if not options:
        raise USGSDownloadError("USGS reports no DEM resolution covering the full buffered parcel study area.")
    if choice:
        selected = next((r for r in RESOLUTIONS if r.key == choice.lower()), None)
        if selected is None or selected not in options:
            raise USGSDownloadError(f"USGS resolution {choice!r} is not available over the full area.")
        return selected
    try:
        answer = input_fn("Select an available resolution by number or key (e.g., 10m): ").strip().lower()
    except EOFError as exc:
        raise USGSDownloadError("Noninteractive session: provide --usgs-resolution, e.g. 10m.") from exc
    selected = next((r for r in RESOLUTIONS if answer in {str(RESOLUTIONS.index(r)+1), r.key}), None)
    if selected is None or selected not in options:
        raise USGSDownloadError(f"Invalid or unavailable DEM resolution: {answer!r}.")
    return selected


def _product_box(product: dict):
    bounds = product.get("boundingBox") or {}
    try:
        return box(*(float(bounds[k]) for k in ("minX", "minY", "maxX", "maxY")))
    except (ValueError, TypeError, KeyError):
        return None


def _fetch_catalog_products(res: Resolution, aoi_geo) -> list[dict]:
    """Get all matching TNMAccess products, including paginated results."""
    params = {
        "datasets": res.dataset,
        "bbox": ",".join(map(str, aoi_geo.bounds)),
        "max": 500,
    }
    products = []
    offset = 0
    while True:
        data = _json_get(PRODUCTS_URL, {**params, "offset": offset})
        items = data.get("items")
        if not isinstance(items, list):
            raise USGSDownloadError("TNMAccess returned an invalid product list.")
        products.extend(items)
        offset += len(items)
        if offset > 5000:
            raise USGSDownloadError(
                "Over 5,000 USGS DEM products match the AOI; narrow the study area."
            )
        if not items or len(items) < 500:
            break
    return products


def _usable_product_boxes(products: list[dict], aoi_geo):
    """Products that have a recognized downloadable raster and usable bounds."""
    allowed = (".tif", ".tiff", ".img", ".zip")
    usable = []
    for item in products:
        url = item.get("downloadURL", "")
        if not isinstance(url, str) or urllib.parse.urlsplit(url).scheme != "https":
            continue
        if Path(urllib.parse.urlsplit(url).path).suffix.lower() not in allowed:
            continue
        geom = _product_box(item)
        if geom is None or not geom.is_valid or geom.is_empty or not geom.intersects(aoi_geo):
            continue
        usable.append((item, geom))
    return usable


def _catalog_coverage(res: Resolution, aoi_geo) -> float:
    """Approximate full-AOI availability from downloadable product footprints.

    This is an independent fallback for failed polygon-index queries. Product
    bounding boxes may include raster NoData; full raster coverage is still
    required and checked after the mosaic has been downloaded.
    """
    products = _fetch_catalog_products(res, aoi_geo)
    boxes = [geom for _, geom in _usable_product_boxes(products, aoi_geo)]
    if not boxes:
        return 0.0
    covered = unary_union(boxes)
    missing = aoi_geo.difference(covered)
    return max(0.0, min(1.0, 1.0 - missing.area / aoi_geo.area))


def query_products(res: Resolution, aoi_geo):
    """Find downloadable tiles, paginate, and choose a covering subset.

    Prefer most recent tiles when footprints overlap, rather than downloading
    many superseded/historical versions of the same geographic tile.
    """
    products = _fetch_catalog_products(res, aoi_geo)
    usable = _usable_product_boxes(products, aoi_geo)
    # Greedy set coverage, newest on equal increments.
    uncovered = aoi_geo
    selected = []
    while uncovered.area / aoi_geo.area > 1e-6:
        ranked = sorted(
            usable,
            key=lambda pair: (uncovered.intersection(pair[1]).area,
                              str(pair[0].get("publicationDate", ""))),
            reverse=True,
        )
        if not ranked or uncovered.intersection(ranked[0][1]).area <= aoi_geo.area * 1e-10:
            raise USGSDownloadError(
                f"USGS product catalog does not provide tiles covering the full "
                f"{res.key} study area, even though the availability index does."
            )
        product, geom = ranked[0]
        selected.append(product)
        uncovered = uncovered.difference(geom)
        usable.remove(ranked[0])
        if len(selected) > 300:
            raise USGSDownloadError("Over 300 DEM tiles needed; use a smaller AOI or coarser DEM.")
    return selected


def _download_tile(url: str, destination: Path, *, timeout=90):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        logging.info("Using cached USGS tile %s", destination.name)
        return
    tmp = destination.with_name(destination.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "basin-bmp-scenario-sim/create_parcel_up"})
        with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp, "wb") as fd:
            shutil.copyfileobj(response, fd, 1024 * 1024)
        if tmp.stat().st_size == 0:
            raise USGSDownloadError(f"Downloaded an empty tile from {url}")
        tmp.replace(destination)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise USGSDownloadError(f"Failed to download USGS tile {url}: {exc}") from exc


def _extract_raster(file_path: Path, cache: Path) -> Path:
    if file_path.suffix.lower() != ".zip":
        return file_path
    extracted = cache / (file_path.stem + "_unpacked")
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(file_path) as archive:
        for item in archive.infolist():
            target = (extracted / item.filename).resolve()
            if not target.is_relative_to(extracted.resolve()):
                raise USGSDownloadError(f"Unsafe path in USGS archive: {item.filename}")
            if item.file_size > 3_000_000_000:
                raise USGSDownloadError(f"Oversized raster in USGS archive: {item.filename}")
        archive.extractall(extracted)
    rasters = [p for p in extracted.rglob("*") if p.is_file()
               and p.suffix.lower() in {".tif", ".tiff", ".img"}]
    if rasters:
        # In USGS bundles the DEM is normally the largest raster.  Avoid
        # picking a small browse/thumbnail GeoTIFF if one is also included.
        return max(rasters, key=lambda p: p.stat().st_size)
    hdr_files = [p for p in extracted.rglob("*") if p.name.lower() == "hdr.adf"]
    if hdr_files:
        # GDAL's AIG (ArcInfo Grid) driver opens the *directory* containing
        # hdr.adf and w001001.adf; hdr.adf is not itself a raster dataset.
        return hdr_files[0].parent
    raise USGSDownloadError(f"No supported DEM raster found in USGS archive {file_path.name}")


def _verify_full_raster_coverage(dem_path: Path, target_geometry):
    """Reject mosaics containing nodata on any cell intersecting the buffered AOI."""
    with rasterio.open(dem_path) as dataset:
        # Never report full coverage if the raster itself is spatially smaller
        # than the parcel AOI (a common one-cell rounding edge case).
        if not box(*dataset.bounds).covers(target_geometry):
            raise USGSDownloadError("Mosaicked USGS DEM does not cover the buffered parcel extent.")
        for _, window in dataset.block_windows(1):
            chunk_bounds = rasterio.windows.bounds(window, dataset.transform)
            if not target_geometry.intersects(box(*chunk_bounds)):
                continue
            target = features.geometry_mask(
                [mapping(target_geometry)], out_shape=(window.height, window.width),
                transform=dataset.window_transform(window), invert=True, all_touched=False,
            )
            if target.any():
                # The GDAL validity mask is equivalent to masked=True but
                # avoids allocating a NumPy MaskedArray for each raster block.
                data = read_band(dataset, window=window)
                invalid = (read_mask(dataset, window=window) == 0) | ~np.isfinite(data)
                if np.any(target & invalid):
                    raise USGSDownloadError(
                        "Downloaded USGS DEM has nodata within the buffered parcel area; "
                        "try a coarser resolution or supply a complete local DEM."
                    )


def get_dem_from_usgs(parcels: gpd.GeoDataFrame, *, resolution: str | None,
                      buffer_m: float, cache_dir: Path, max_cells: int,
                      list_only: bool = False, input_fn=input,
                      refresh_availability: bool = False) -> Path | None:
    """Perform discovery, interactive selection, tile downloads, and mosaic."""
    local, geo, crs, coverage = check_resolutions(
        parcels, buffer_m=buffer_m, cache_dir=cache_dir,
        refresh=refresh_availability,
        only_resolution=resolution if not list_only else None,
    )
    show_resolutions(coverage)
    if list_only:
        return None
    chosen = choose_resolution(coverage, resolution, input_fn=input_fn)
    left, bottom, right, top = local.bounds
    width = math.ceil((right - left) / chosen.meters)
    height = math.ceil((top - bottom) / chosen.meters)
    pixels = width * height
    # Rasterio merge rounds requested bounds to cell counts. Expand to a full
    # pixel grid so the eastern and southern edges are not truncated.
    raster_bounds = (left, top - height * chosen.meters,
                     left + width * chosen.meters, top)
    print(f"Selected USGS {chosen.label}; approximate mosaic size: {width:,} x "
          f"{height:,} = {pixels:,} pixels (projected {crs}).")
    if pixels > max_cells:
        raise USGSDownloadError(
            f"The requested DEM exceeds --max-dem-cells ({pixels:,} > {max_cells:,}). "
            "Choose a coarser resolution or raise the limit if RAM permits."
        )
    products = query_products(chosen, geo)
    estimated_bytes = sum(int(p.get("sizeInBytes") or 0) for p in products)
    print(f"Download: {len(products)} USGS product(s), approximately "
          f"{estimated_bytes / (1024**2):,.1f} MiB if catalog sizes are known.")
    if resolution is None:
        try:
            confirmation = input_fn("Download and build this DEM? [y/N]: ").strip().lower()
        except EOFError as exc:
            raise USGSDownloadError("Noninteractive session: set --usgs-resolution.") from exc
        if confirmation not in {"y", "yes"}:
            raise USGSDownloadError("USGS DEM download cancelled; no input files changed.")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, prod in enumerate(products, 1):
        url = prod["downloadURL"]
        basename = Path(urllib.parse.urlsplit(url).path).name
        identifier = str(prod.get("sourceId") or i).replace("/", "_")
        dest = cache_dir / f"{identifier}_{basename}"
        logging.info("Downloading USGS tile %d/%d: %s", i, len(products), basename)
        _download_tile(url, dest)
        paths.append(_extract_raster(dest, cache_dir))
    # Merge as a projected, meter-spaced raster so D8 slope distance is valid.
    output = cache_dir / f"usgs_3dep_{chosen.key}_study_dem.tif"
    temp_output = cache_dir / (output.name + ".partial.tif")
    with ExitStack() as stack:
        sources = []
        for path in paths:
            source = stack.enter_context(rasterio.open(path))
            if source.crs is None:
                raise USGSDownloadError(f"USGS raster has no CRS: {path}")
            vrt = stack.enter_context(WarpedVRT(
                source, crs=crs, resampling=Resampling.bilinear,
                nodata=-9999.0,
            ))
            sources.append(vrt)
        try:
            merge(
                sources, bounds=raster_bounds, res=chosen.meters,
                nodata=-9999.0, dtype="float32", dst_path=str(temp_output),
                dst_kwds={"driver": "GTiff", "tiled": True, "compress": "deflate",
                           "BIGTIFF": "IF_SAFER", "nodata": -9999.0,
                           "blockxsize": 256, "blockysize": 256},
                mem_limit=64,
            )
            _verify_full_raster_coverage(temp_output, local)
            temp_output.replace(output)
        except Exception:
            temp_output.unlink(missing_ok=True)
            raise
    # Audit trail for reproducible analyses and future offline reuse.
    import json
    manifest = cache_dir / f"usgs_3dep_{chosen.key}_sources.json"
    manifest.write_text(json.dumps({
        "source": "USGS 3DEP / TNMAccess",
        "resolution": chosen.key,
        "availability_source": getattr(coverage, "sources", {}).get(chosen.key, "index"),
        "availability_provisional": (
            getattr(coverage, "sources", {}).get(chosen.key) == "catalog"
        ),
        "dataset": chosen.dataset,
        "nominal_cell_size_m": chosen.meters,
        "projected_crs": str(crs),
        "buffer_m": buffer_m,
        "dem": str(output),
        "products": [{"sourceId": item.get("sourceId"),
                      "publicationDate": item.get("publicationDate"),
                      "downloadURL": item["downloadURL"]} for item in products],
    }, indent=2) + "\n", encoding="utf-8")
    logging.info("USGS DEM created and coverage-checked: %s", output)
    logging.info("USGS source manifest: %s", manifest)
    return output
