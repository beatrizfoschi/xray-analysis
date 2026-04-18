"""
spot_intensity_map.py — Integrated spot intensity map for a Laue raster scan.

For each scan point, integrates the detector counts within a ROI centred on a
given diffraction spot and returns a 2D map of the integrated intensity.  This
reveals how the intensity of a specific reflection varies across the sample —
directly related to local strain, mosaicity, and dislocation density.

Normalisation options
---------------------
normalize_to_monitor : bool
    Divides by the monitor counts (incident beam intensity).  Removes beam
    current fluctuations measured upstream of the sample.
bg_center / bg_boxsize : tuple
    Defines a background region on the detector with no diffraction spots.
    The integrated counts in that region are used to normalise each frame,
    correcting for detector-level variations (diffuse scattering, gain drift,
    residual ring-refill effects not captured by the monitor).
    Applied after monitor normalisation when both are active.

Usage
-----
>>> fig = spot_intensity_map(
...     h5_path="scan_001.h5",
...     img_source="path/to/tifs",
...     roi_center=(1255, 1390),
...     roi_boxsize=(50, 60),
...     bg_center=(800, 400),      # empty detector region
...     bg_boxsize=(60, 60),
... )
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


def _make_slices(center: tuple[int, int], boxsize: tuple[int, int]):
    cx, cy = center
    hw, hh = boxsize[0] // 2, boxsize[1] // 2
    return slice(cy - hh, cy + hh), slice(cx - hw, cx + hw)


def _read_h5_roi(img_source: Path, h5_img_key: str, n: int,
                 row_slice: slice, col_slice: slice) -> np.ndarray:
    with h5py.File(img_source, "r") as h5f:
        return h5f[h5_img_key][:n, row_slice, col_slice].astype(float)


def _integrate_tifs(img_source: Path, img_prefix: str, img_suffix: str,
                    img_index_pad: int, n: int, row_slice: slice,
                    col_slice: slice, workers: int) -> np.ndarray:
    def _worker(i):
        fname = img_source / f"{img_prefix}{i:0>{img_index_pad}d}{img_suffix}"
        return i, float(read_image(fname)[row_slice, col_slice].sum())

    result = np.empty(n)
    print(f"Loading {n} TIF images ({workers} threads)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, i): i for i in range(n)}
        done = 0
        for future in as_completed(futures):
            idx, val = future.result()
            result[idx] = val
            done += 1
            if done % max(1, n // 10) == 0:
                print(f"  {done}/{n}")
    return result


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
    bg_center: tuple[int, int] | None = None,
    bg_boxsize: tuple[int, int] | None = None,
    workers: int = 8,
    figsize: tuple[float, float] = (7, 6),
    cmap: str = "inferno",
    title: str | None = None,
) -> plt.Figure:
    """Compute and plot the integrated intensity of a diffraction spot across a scan.

    Parameters
    ----------
    h5_path : str or Path
        Path to the scan HDF5 file (scan geometry and monitor data).
    img_source : str or Path
        TIF folder path, or HDF5 file containing detector images.
    roi_center : (x, y)
        Centre of the spot ROI in detector pixel coordinates (column, row).
    roi_boxsize : (width, height)
        Size of the spot ROI in pixels.
    scan_number : int
        Scan entry number inside the HDF5 (default 1).
    img_prefix, img_suffix, img_index_pad :
        TIF filename format: ``{prefix}{index:0>{pad}d}{suffix}``.
    h5_img_key : str, optional
        Dataset key when ``img_source`` is an HDF5 file.
        Shape must be ``(n_images, height, width)``.
    normalize_to_monitor : bool
        Divide by monitor counts to correct for incident beam fluctuations
        (default True).
    bg_center : (x, y), optional
        Centre of a background ROI — a detector region with no diffraction
        spots.  The integrated counts there normalise each frame, correcting
        for detector-level variations not captured by the monitor (diffuse
        scattering, gain drift, residual ring-refill artefacts).
        Applied after monitor normalisation.
    bg_boxsize : (width, height), optional
        Size of the background ROI.  Required when ``bg_center`` is set.
    workers : int
        Threads for parallel TIF loading (ignored for HDF5).
    figsize, cmap, title :
        Matplotlib figure parameters.

    Returns
    -------
    matplotlib.figure.Figure
    """
    h5_path    = Path(h5_path)
    img_source = Path(img_source)
    is_h5      = img_source.suffix in ('.h5', '.hdf5')

    if bg_center is not None and bg_boxsize is None:
        raise ValueError("bg_boxsize must be set when bg_center is provided.")

    scan = Scan.from_h5(h5_path, scan_number)
    n    = scan.length

    spot_row, spot_col = _make_slices(roi_center, roi_boxsize)

    # ── Integrate spot ROI ────────────────────────────────────────────────────
    if is_h5:
        if h5_img_key is None:
            raise ValueError("h5_img_key must be set when img_source is an HDF5 file.")
        print("Reading spot ROI from HDF5...")
        crops     = _read_h5_roi(img_source, h5_img_key, n, spot_row, spot_col)
        intensity = crops.sum(axis=(1, 2))
    else:
        intensity = _integrate_tifs(img_source, img_prefix, img_suffix,
                                    img_index_pad, n, spot_row, spot_col, workers)

    # ── Monitor normalisation ─────────────────────────────────────────────────
    if normalize_to_monitor:
        monitor   = np.where(scan.monitor_data == 0, np.nan,
                             scan.monitor_data.astype(float))
        intensity = intensity / monitor

    # ── Background normalisation ──────────────────────────────────────────────
    if bg_center is not None:
        bg_row, bg_col = _make_slices(bg_center, bg_boxsize)

        if is_h5:
            print("Reading background ROI from HDF5...")
            bg_crops = _read_h5_roi(img_source, h5_img_key, n, bg_row, bg_col)
            bg       = bg_crops.sum(axis=(1, 2)).astype(float)
        else:
            bg = _integrate_tifs(img_source, img_prefix, img_suffix,
                                 img_index_pad, n, bg_row, bg_col, workers)

        if normalize_to_monitor:
            bg = bg / monitor

        bg        = np.where(bg == 0, np.nan, bg)
        intensity = intensity / bg

    # ── Colorbar label ────────────────────────────────────────────────────────
    label = "Integrated counts"
    if normalize_to_monitor:
        label += " / monitor"
    if bg_center is not None:
        label += " / background"

    # ── Reshape and plot ──────────────────────────────────────────────────────
    intensity_grid = _as_grid(intensity, scan)
    motor_x        = scan.xpoints * 1e3
    motor_y        = scan.ypoints * 1e3

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(motor_x, motor_y, intensity_grid, cmap=cmap, shading="auto")
    plt.colorbar(mesh, ax=ax, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("Position [μm]")
    ax.set_ylabel("Position [μm]")
    ax.set_title(title or f"Spot intensity  —  center {roi_center},  boxsize {roi_boxsize}")
    fig.tight_layout()

    return fig
