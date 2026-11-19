"""Pre-routing validations preventing spatially incomplete or ambiguous graphs."""
from __future__ import annotations

import numpy as np
from shapely.geometry import box
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
    # Raster cells occupied by any parcel must be valid DEM data. Deliberately
    # use a shared mask rather than reallocating arrays for each parcel.
    covered = rasterize(((g,1) for g in parcels.geometry),
                        out_shape=dem_dataset.shape,
                        transform=dem_dataset.transform, fill=0, dtype='uint8')
    missing = (covered > 0) & ~valid_mask
    if np.any(missing):
        cell_area_m2 = abs(dem_dataset.transform.a * dem_dataset.transform.e -
                           dem_dataset.transform.b * dem_dataset.transform.d) * linear_units_to_m**2
        raise ValueError(f'DEM has NoData in parcel area: '
                         f'{int(missing.sum())} invalid pixel(s), '
                         f'approximately {missing.sum()*cell_area_m2:.3f} m²')
