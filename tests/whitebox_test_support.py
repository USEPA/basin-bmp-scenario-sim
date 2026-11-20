"""Offline WhiteboxTools stub for CLI orchestration tests only."""
from pathlib import Path

import numpy as np
import rasterio


def fake_whitebox_file_pipeline(monkeypatch):
    """Simulate a valid Whitebox D8 pointer raster for an eastward slope.

    This stub checks CLI I/O and aggregation, *not* the WhiteboxTools binary.
    Actual Whitebox method calls are covered by separate mock-interaction tests.
    """
    from utils import whitebox_parcel_routing

    def run(dem_file, output_dir, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pointer = output_dir / 'pointer.tif'
        with rasterio.open(dem_file) as src:
            meta = src.profile.copy()
            arr = np.zeros(src.shape, dtype=np.uint16)
            arr[:, :-1] = 2  # Whitebox convention: E = 2
            meta.update(dtype='uint16', nodata=65535)
            with rasterio.open(pointer, 'w', **meta) as dst:
                dst.write(arr, 1)
        return pointer

    monkeypatch.setattr(whitebox_parcel_routing, 'run_whitebox_d8', run)
