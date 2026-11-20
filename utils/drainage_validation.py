"""Pre-routing validations preventing spatially incomplete or ambiguous graphs."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box
from rasterio.features import rasterize


def validate_parcel_overlaps(parcels, id_field: str, *, tolerance_m2: float = 0.01,
                             linear_units_to_m: float = 1.) -> None:
    """Reject duplicate or materially overlapping polygons before rasterization.

    Sindex candidate discovery is subquadratic on non-overlapping data. Polygon
    overlaps are tested on projected DEM coordinates (converted to m²).
    """
    if tolerance_m2 < 0 or not np.isfinite(tolerance_m2):
        raise ValueError('Overlap tolerance must be finite and nonnegative')
    if parcels.crs is None or not parcels.crs.is_projected:
        raise ValueError('Parcel overlap detection requires a projected CRS')
    geom = parcels.geometry
    for i, g in enumerate(geom):
        for j in parcels.sindex.query(g, predicate='intersects'):
            if j <= i:
                continue
            overlap = float(g.intersection(geom.iloc[j]).area) * linear_units_to_m**2
            if overlap > tolerance_m2:
                a, b = parcels.iloc[i][id_field], parcels.iloc[j][id_field]
                raise ValueError(f'Overlapping parcel geometries: pid {a} and {b}, '
                                 f'overlap {overlap:.3f} m² exceeds {tolerance_m2:.3f} m²')


def validate_dem_coverage(parcels, dem_dataset, valid_mask: np.ndarray,
                          *, tolerance_m2: float = 0., linear_units_to_m: float = 1.) -> None:
    """Check full parcel polygon extent AND NoData inside DEM footprint.

    Complete parcel coverage is required, but tiny/subcell parcels can still
    be unresolved by rasterization; that is a separate resolution diagnostic.
    """
    if valid_mask.shape != dem_dataset.shape:
        raise ValueError('DEM validity mask does not match raster shape')
    bounds = box(*dem_dataset.bounds)
    for geom in parcels.geometry:
        outside_m2 = geom.difference(bounds).area * linear_units_to_m**2
        if outside_m2 > tolerance_m2 + 1e-8:
            raise ValueError(f'DEM does not cover full parcel area: '
                             f'{outside_m2:.3f} m² outside raster bounds')
    if tolerance_m2 < 0 or not np.isfinite(tolerance_m2):
        raise ValueError('DEM coverage tolerance must be finite and nonnegative')
    if linear_units_to_m <= 0 or not np.isfinite(linear_units_to_m):
        raise ValueError('DEM linear unit conversion must be finite and positive')

    # A default (cell-center) rasterization misses invalid cells that overlap
    # only a narrow or irregular portion of a parcel. Find candidate cells
    # using all_touched=True, then check their *actual intersection area* to
    # avoid rejecting parcels that merely share a pixel boundary.
    covered = rasterize(((g, 1) for g in parcels.geometry),
                        out_shape=dem_dataset.shape,
                        transform=dem_dataset.transform, fill=0,
                        all_touched=True, dtype='uint8')
    potentially_missing = (covered > 0) & ~np.asarray(valid_mask, dtype=bool)
    if not np.any(potentially_missing):
        return

    transform = dem_dataset.transform
    spatial_index = parcels.sindex
    missing_area_m2 = 0.0
    missing_pixels = 0
    for row, col in np.argwhere(potentially_missing):
        # This also works with rotated or sheared pixel grids.
        pixel = Polygon((transform * (col, row),
                         transform * (col + 1, row),
                         transform * (col + 1, row + 1),
                         transform * (col, row + 1)))
        cell_overlap = sum(
            parcels.geometry.iloc[int(i)].intersection(pixel).area
            for i in spatial_index.query(pixel, predicate='intersects')
        ) * linear_units_to_m**2
        if cell_overlap <= 0:
            continue  # shared boundary/corner, not missing parcel area
        missing_area_m2 += cell_overlap
        missing_pixels += 1
        if missing_area_m2 > tolerance_m2 + 1e-8:
            raise ValueError(
                'DEM has NoData in parcel area: at least '
                f'{missing_area_m2:.6f} m² across {missing_pixels} '
                'partly or fully intersected invalid pixel(s)'
            )
