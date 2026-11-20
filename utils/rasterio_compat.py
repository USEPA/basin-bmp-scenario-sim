"""Compatibility wrappers for NumPy 2.5 / rasterio band reads.

Some rasterio versions assign to ``ndarray.shape`` internally while reading
raster bands. NumPy 2.5 warns about that implementation detail even though the
read succeeds. Confine suppression of *that one warning* to rasterio reads;
other DeprecationWarnings and all data/IO exceptions remain visible.

These functions do not change the arrays or their masks.
"""
from __future__ import annotations

import warnings
from typing import Any


_SHAPE_WARNING = r"^Setting the shape on a NumPy array has been deprecated in NumPy 2\.5\."


def read_band(dataset: Any, band: int = 1, **kwargs: Any) -> Any:
    """Read an unchanged raster band, hiding only the NumPy 2.5 shape warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_SHAPE_WARNING, category=DeprecationWarning
        )
        return dataset.read(band, **kwargs)


def read_mask(dataset: Any, band: int = 1, **kwargs: Any) -> Any:
    """Read an unchanged GDAL validity mask, hiding only the same warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_SHAPE_WARNING, category=DeprecationWarning
        )
        return dataset.read_masks(band, **kwargs)
