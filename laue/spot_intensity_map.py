"""
spot_intensity_map.py — Integrated spot intensity map for a Laue raster scan.

For each scan point, integrates the detector counts within a ROI centred on a
given diffraction spot and returns a 2D map of the integrated intensity.  This
reveals how the intensity of a specific reflection varies across the sample —
directly related to local strain, mosaicity, and dislocation density.

Usage
-----
>>> fig = spot_intensity_map(
...     h5_path="scan_001.h5",
...     img_source="path/to/tifs",
...     roi_center=(1255, 1390),   # (x, y) detector pixel
...     roi_boxsize=(50, 60),      # (width, height) in pixels
... )
>>> fig.show()
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

from lauexplore.image import read as read_image
from lauexplore.plots import heatmap, scan_hovermenu
from lauexplore.plots.base import _as_grid
from lauexplore.scan import Scan


def spot_intensity_map(
    h5_path: str | Path,
    img_source: str | Path,
    roi_center: tuple[int, int],
    roi_boxsize: tuple[int, int],
    *,
    scan_number: int = 1,
    img_prefix: str = "",
    img_suffix: str = ".tif",
    img_index_pad: int = 4,
    h5_img_key: str | None = None,
    workers: int = 8,
    width: int = 600,
    height: int = 600,
    colorscale: str = "inferno",
    title: str | None = None,
) -> "go.Figure":
    """Compute and plot the integrated intensity of a diffraction spot across a scan.

    For each scan point, loads the detector image, crops to the ROI centred on
    the spot, and sums all pixel counts.  The result is a 2D heatmap with the
    same spatial coordinates as a fluorescence map.

    Parameters
    ----------
    h5_path : str or Path
        Path to the scan HDF5 file (used to read scan geometry).
    img_source : str or Path
        TIF folder path, or HDF5 file containing detector images.
    roi_center : (x, y)
        Centre of the ROI in detector pixel coordinates (column, row).
    roi_boxsize : (width, height)
        Size of the ROI in pixels (columns, rows).
    scan_number : int
        Scan entry number inside the HDF5 (default 1).
    img_prefix, img_suffix, img_index_pad :
        TIF filename format: ``{prefix}{index:0>{pad}d}{suffix}``.
    h5_img_key : str, optional
        Dataset key inside ``img_source`` when it is an HDF5 file.
        Indexed as ``h5f[h5_img_key][file_index]``.
    workers : int
        Number of threads for parallel image loading (default 8).
    width, height : int
        Figure dimensions in pixels.
    colorscale : str
        Plotly colorscale name.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    h5_path    = Path(h5_path)
    img_source = Path(img_source)

    scan = Scan.from_h5(h5_path, scan_number)

    cx, cy   = roi_center
    hw, hh   = roi_boxsize[0] // 2, roi_boxsize[1] // 2
    col_slice = slice(cx - hw, cx + hw)
    row_slice = slice(cy - hh, cy + hh)

    def _integrate(file_index: int) -> tuple[int, float]:
        if img_source.suffix in ('.h5', '.hdf5'):
            if h5_img_key is None:
                raise ValueError("h5_img_key must be set when img_source is an HDF5 file.")
            with h5py.File(img_source, "r") as h5f:
                crop = h5f[h5_img_key][file_index][row_slice, col_slice]
        else:
            fname = img_source / f"{img_prefix}{file_index:0>{img_index_pad}d}{img_suffix}"
            crop = read_image(fname)[row_slice, col_slice]
        return file_index, float(crop.sum())

    intensity = np.empty(scan.length)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_integrate, i): i for i in range(scan.length)}
        for future in as_completed(futures):
            idx, val = future.result()
            intensity[idx] = val

    intensity_grid = _as_grid(intensity, scan)   # (nbypoints, nbxpoints)
    motor_x        = scan.xpoints * 1e3           # µm
    motor_y        = scan.ypoints * 1e3           # µm

    customdata, hovertemplate = scan_hovermenu(scan)

    return heatmap(
        intensity_grid, motor_x, motor_y,
        customdata=customdata,
        hovertemplate=hovertemplate,
        width=width,
        height=height,
        title=title or f"Spot intensity — center {roi_center}, boxsize {roi_boxsize}",
        xlabel="Position [μm]",
        ylabel="Position [μm]",
        colorscale=colorscale,
        cbartitle="Integrated counts",
    )
