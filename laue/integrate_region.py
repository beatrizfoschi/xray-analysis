"""
integrate_region.py — Sum diffraction frames from a rectangular scan map region.

For a raster scan stored as a 4-D HDF5 dataset (scan_rows, scan_cols, det_y, det_x),
selects a rectangular sub-region of the scan grid and sums (or averages / max-projects)
all detector frames within it, producing a single 2-D image.

Typical use: accumulate signal from a specific sample area — a cluster of V-pits,
a grain boundary, or a uniform reference zone — without point-by-point analysis.

Usage (scan indices)
--------------------
>>> from laue.integrate_region import integrate_map_region, plot_integrated_frame
>>> img = integrate_map_region(
...     h5_path="scan_001.h5",
...     h5_key="1.1/measurement/eiger4m",
...     scan_number=1,
...     col_range=(10, 20),   # x-direction columns (i), inclusive
...     row_range=(5, 15),    # y-direction rows (j), inclusive
... )
>>> fig = plot_integrated_frame(img)

Usage (physical coordinates in µm)
-----------------------------------
>>> img = integrate_map_region(
...     h5_path="scan_001.h5",
...     h5_key="1.1/measurement/eiger4m",
...     scan_number=1,
...     x_um=(-50.0, 50.0),
...     y_um=(-30.0, 30.0),
... )
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lauexplore.scan import Scan


# ── helpers ───────────────────────────────────────────────────────────────────

def _um_to_col_range(scan: Scan, x_um: tuple[float, float]) -> tuple[int, int]:
    """Convert physical x coordinates (µm) to inclusive column indices."""
    xpts = scan.xpoints * 1e3  # mm → µm
    x0, x1 = min(x_um), max(x_um)
    i0 = int(np.searchsorted(xpts, x0))
    i1 = int(np.searchsorted(xpts, x1, side="right")) - 1
    i0 = int(np.clip(i0, 0, scan.nbxpoints - 1))
    i1 = int(np.clip(i1, 0, scan.nbxpoints - 1))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1


def _um_to_row_range(scan: Scan, y_um: tuple[float, float]) -> tuple[int, int]:
    """Convert physical y coordinates (µm) to inclusive row indices."""
    ypts = scan.ypoints * 1e3  # mm → µm
    y0, y1 = min(y_um), max(y_um)
    j0 = int(np.searchsorted(ypts, y0))
    j1 = int(np.searchsorted(ypts, y1, side="right")) - 1
    j0 = int(np.clip(j0, 0, scan.nbypoints - 1))
    j1 = int(np.clip(j1, 0, scan.nbypoints - 1))
    if j1 < j0:
        j0, j1 = j1, j0
    return j0, j1


# ── main function ─────────────────────────────────────────────────────────────

def integrate_map_region(
    h5_path: str | Path,
    h5_key: str,
    *,
    scan_number: int = 1,
    col_range: tuple[int, int] | None = None,
    row_range: tuple[int, int] | None = None,
    x_um: tuple[float, float] | None = None,
    y_um: tuple[float, float] | None = None,
    method: Literal["sum", "mean", "max"] = "sum",
) -> np.ndarray:
    """Integrate detector frames from a rectangular sub-region of the scan map.

    The HDF5 dataset is expected to be either:

    * **4-D** ``(scan_rows, scan_cols, det_y, det_x)`` — raster scan stored on a grid.
    * **3-D** ``(n_frames, det_y, det_x)`` — flat list; frame order follows the scan
      sequence returned by ``scan.ij_to_index(col, row)``.

    Region can be specified as scan indices **or** physical coordinates (µm).
    If both are given, the µm coordinates take precedence.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file containing the diffraction dataset.
    h5_key : str
        Dataset key inside the HDF5 file, e.g. ``"1.1/measurement/eiger4m"``.
    scan_number : int
        Scan entry in the scan HDF5 file (for reading geometry, default 1).
        If ``h5_path`` is the same file that holds the scan geometry, set this
        to the appropriate scan number.
    col_range : (i0, i1), optional
        Inclusive column indices (x-direction) of the scan region.
        Uses 0-based indexing; ``i1`` is included.
    row_range : (j0, j1), optional
        Inclusive row indices (y-direction) of the scan region.
        Uses 0-based indexing; ``j1`` is included.
    x_um : (x_min, x_max), optional
        x-axis coordinate range in µm. Overrides ``col_range`` when provided.
    y_um : (y_min, y_max), optional
        y-axis coordinate range in µm. Overrides ``row_range`` when provided.
    method : {"sum", "mean", "max"}
        Aggregation method (default ``"sum"``).

    Returns
    -------
    np.ndarray, shape (det_y, det_x)
        Integrated detector image.

    Examples
    --------
    Select columns 10–20 and rows 5–15 (inclusive scan indices):

    >>> img = integrate_map_region("scan.h5", "1.1/measurement/eiger4m",
    ...                            col_range=(10, 20), row_range=(5, 15))

    Select by physical coordinates:

    >>> img = integrate_map_region("scan.h5", "1.1/measurement/eiger4m",
    ...                            x_um=(-50.0, 50.0), y_um=(-30.0, 30.0))

    Use all scan points (no region restriction):

    >>> img = integrate_map_region("scan.h5", "1.1/measurement/eiger4m")
    """
    h5_path = Path(h5_path)
    scan    = Scan.from_h5(h5_path, scan_number)

    # ── resolve coordinate ranges ─────────────────────────────────────────────
    if x_um is not None:
        col_range = _um_to_col_range(scan, x_um)
    if y_um is not None:
        row_range = _um_to_row_range(scan, y_um)

    if col_range is None:
        col_range = (0, scan.nbxpoints - 1)
    if row_range is None:
        row_range = (0, scan.nbypoints - 1)

    i0, i1 = col_range
    j0, j1 = row_range

    cols = list(range(i0, i1 + 1))
    rows = list(range(j0, j1 + 1))
    n_frames = len(cols) * len(rows)

    print(
        f"Region: cols {i0}–{i1} ({len(cols)} pts), "
        f"rows {j0}–{j1} ({len(rows)} pts)  →  {n_frames} frames"
    )

    # ── accumulate ────────────────────────────────────────────────────────────
    with h5py.File(h5_path, "r") as h5f:
        ds    = h5f[h5_key]
        ndim  = len(ds.shape)

        if ndim == 4:
            # (scan_rows, scan_cols, det_y, det_x)
            det_y, det_x = ds.shape[2], ds.shape[3]
        elif ndim == 3:
            # (n_frames, det_y, det_x)
            det_y, det_x = ds.shape[1], ds.shape[2]
        else:
            raise ValueError(
                f"Dataset has {ndim} dimensions; expected 3 or 4."
            )

        if method == "max":
            result = np.zeros((det_y, det_x), dtype=np.float64)
        else:
            result = np.zeros((det_y, det_x), dtype=np.float64)

        pairs = [(j, i) for j in rows for i in cols]
        for j, i in tqdm(pairs, desc=f"Integrating ({method})", unit="frame"):
            if ndim == 4:
                frame = ds[j, i, :, :].astype(np.float64)
            else:
                idx   = scan.ij_to_index(i, j)
                frame = ds[idx, :, :].astype(np.float64)

            if method == "max":
                np.maximum(result, frame, out=result)
            else:
                result += frame

    if method == "mean":
        result /= n_frames

    print(
        f"Done. Image shape: {result.shape}  |  "
        f"total counts: {result.sum():.3e}  |  "
        f"max pixel: {result.max():.3e}"
    )
    return result


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_integrated_frame(
    img: np.ndarray,
    *,
    percentile_clip: tuple[float, float] = (1, 99),
    cmap: str = "seismic",
    figsize: tuple[float, float] = (8, 7),
    title: str | None = None,
    log_scale: bool = False,
) -> plt.Figure:
    """Display the integrated diffraction frame.

    Parameters
    ----------
    img : np.ndarray
        2-D array returned by :func:`integrate_map_region`.
    percentile_clip : (lo, hi)
        Percentiles used to set the colour scale (ignores zero and negative).
    cmap : str
        Matplotlib colormap.
    figsize : (w, h)
        Figure size in inches.
    title : str, optional
        Figure title.
    log_scale : bool
        Apply ``log1p`` before displaying (useful when dynamic range is large).
    """
    data = np.log1p(img) if log_scale else img.copy()

    positive = data[data > 0]
    if positive.size == 0:
        lo, hi = data.min(), data.max()
    else:
        lo = np.percentile(positive, percentile_clip[0])
        hi = np.percentile(positive, percentile_clip[1])

    ny, nx = data.shape
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data,
        origin="upper",
        aspect="equal",
        extent=[0, nx, ny, 0],
        cmap=cmap,
        vmin=lo,
        vmax=hi,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="log(1 + counts)" if log_scale else "counts")
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_title(title or "Integrated diffraction frame")
    fig.tight_layout()
    return fig
