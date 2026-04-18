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
import matplotlib.pyplot as plt
import numpy as np

from lauexplore.image import read as read_image
from lauexplore.plots.base import _as_grid
from lauexplore.scan import Scan


def spot_intensity_map(
    h5_path: str | Path,
    img_source: str | Path,
    roi_center: tuple[int, int],
    roi_boxsize: tuple[int, int],
    *,
    scan_number: int = 1,
    img_prefix: str = "img_",
    img_suffix: str = ".tif",
    img_index_pad: int = 4,
    h5_img_key: str | None = None,
    normalize_to_monitor: bool = True,
    workers: int = 8,
    figsize: tuple[float, float] = (7, 6),
    cmap: str = "inferno",
    title: str | None = None,
) -> plt.Figure:
    """Compute and plot the integrated intensity of a diffraction spot across a scan.

    For each scan point, loads the detector image, crops to the ROI centred on
    the spot, and sums all pixel counts.  The result is a 2D heatmap with the
    same spatial coordinates as a fluorescence map.

    For HDF5 image sources the entire ROI stack is read in a single operation,
    which is significantly faster than opening the file once per image.
    For TIF sources, images are loaded in parallel with a thread pool.

    Parameters
    ----------
    h5_path : str or Path
        Path to the scan HDF5 file (used to read scan geometry and monitor).
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
        Dataset shape must be ``(n_images, height, width)``.
    normalize_to_monitor : bool
        If True (default), divide integrated intensity by the monitor counts at
        each scan point, removing synchrotron ring refill artefacts.
    workers : int
        Number of threads for parallel TIF loading (ignored for HDF5).
    figsize : (width, height)
        Figure size in inches.
    cmap : str
        Matplotlib colormap (default ``"inferno"``).
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    h5_path    = Path(h5_path)
    img_source = Path(img_source)
    is_h5      = img_source.suffix in ('.h5', '.hdf5')

    scan = Scan.from_h5(h5_path, scan_number)

    cx, cy    = roi_center
    hw, hh    = roi_boxsize[0] // 2, roi_boxsize[1] // 2
    col_slice = slice(cx - hw, cx + hw)
    row_slice = slice(cy - hh, cy + hh)

    # ── Integrate counts per scan point ───────────────────────────────────────
    if is_h5:
        if h5_img_key is None:
            raise ValueError("h5_img_key must be set when img_source is an HDF5 file.")
        # Read the full ROI stack in one shot — much faster than per-image access
        print("Reading ROI from HDF5...")
        with h5py.File(img_source, "r") as h5f:
            crops = h5f[h5_img_key][:scan.length, row_slice, col_slice]
        intensity = crops.sum(axis=(1, 2)).astype(float)

    else:
        def _integrate(file_index: int) -> tuple[int, float]:
            fname = img_source / f"{img_prefix}{file_index:0>{img_index_pad}d}{img_suffix}"
            crop  = read_image(fname)[row_slice, col_slice]
            return file_index, float(crop.sum())

        intensity = np.empty(scan.length)
        print(f"Loading {scan.length} TIF images ({workers} threads)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_integrate, i): i for i in range(scan.length)}
            done = 0
            for future in as_completed(futures):
                idx, val = future.result()
                intensity[idx] = val
                done += 1
                if done % max(1, scan.length // 10) == 0:
                    print(f"  {done}/{scan.length}")

    # ── Monitor normalisation ─────────────────────────────────────────────────
    if normalize_to_monitor:
        monitor  = scan.monitor_data.astype(float)
        monitor  = np.where(monitor == 0, np.nan, monitor)
        intensity = intensity / monitor

    # ── Reshape and plot ──────────────────────────────────────────────────────
    intensity_grid = _as_grid(intensity, scan)   # (nbypoints, nbxpoints)
    motor_x        = scan.xpoints * 1e3           # mm → µm
    motor_y        = scan.ypoints * 1e3           # mm → µm

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(motor_x, motor_y, intensity_grid, cmap=cmap, shading="auto")
    plt.colorbar(mesh, ax=ax, label="Integrated counts" + (" / monitor" if normalize_to_monitor else ""))
    ax.set_aspect("equal")
    ax.set_xlabel("Position [μm]")
    ax.set_ylabel("Position [μm]")
    ax.set_title(title or f"Spot intensity  —  center {roi_center},  boxsize {roi_boxsize}")
    fig.tight_layout()

    return fig
