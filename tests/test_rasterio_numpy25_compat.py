"""Suppress only the NumPy 2.5 raster-read shape deprecation."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from utils.rasterio_compat import read_band, read_mask


_SHAPE_MESSAGE = "Setting the shape on a NumPy array has been deprecated in NumPy 2.5."


class WarningDataset:
    def read(self, band, **kwargs):
        warnings.warn(_SHAPE_MESSAGE, DeprecationWarning, stacklevel=2)
        return np.array([[12., 13.]], dtype=np.float32)

    def read_masks(self, band, **kwargs):
        warnings.warn(_SHAPE_MESSAGE, DeprecationWarning, stacklevel=2)
        return np.array([[255, 0]], dtype=np.uint8)


def test_numpy25_shape_deprecation_is_filtered_only_during_raster_reads():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert read_band(WarningDataset()).tolist() == [[12., 13.]]
        assert read_mask(WarningDataset()).tolist() == [[255, 0]]
        with pytest.raises(DeprecationWarning, match="NumPy 2.5"):
            WarningDataset().read(1)


def test_other_deprecations_are_not_hidden():
    class OtherWarningDataset:
        def read(self, band, **kwargs):
            warnings.warn("A distinct deprecation", DeprecationWarning, stacklevel=2)
        def read_masks(self, band, **kwargs):
            warnings.warn("A distinct deprecation", DeprecationWarning, stacklevel=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="distinct"):
            read_band(OtherWarningDataset())
        with pytest.raises(DeprecationWarning, match="distinct"):
            read_mask(OtherWarningDataset())


def test_read_exceptions_are_not_suppressed():
    class BadDataset:
        def read(self, band, **kwargs):
            raise OSError("DEM read failed")
    with pytest.raises(OSError, match="DEM read failed"):
        read_band(BadDataset())
