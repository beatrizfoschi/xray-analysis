"""
spot_intensity_map.py — Integrated spot intensity map for a Laue raster scan.

For each scan point, integrates the detector counts within a ROI centred on a
given diffraction spot and returns a 2D map of the integrated intensity.  This
reveals how the intensity of a specific reflection varies across the sample —
directly related to local strain, mosaicity, and dislocation density.

Pedestal subtraction
--------------------
pedestal_ring : int
    Width in pixels of a frame drawn around each ROI.  The median of that
    frame is taken as the per-pixel pedestal (detector offset + diffuse
    background) and removed from every ROI pixel before summing.

    This is what makes the monitor normalisation work.  The raw ROI sum is
    ``S + B``: a signal ``S`` that scales with the beam and a pedestal ``B``
    (``pixel offset × box area``) that does not.  Dividing ``S + B`` by the
    monitor leaves ``B / monitor``, which *grows* when the beam drops — the
    normalisation then over-corrects, and a refill or a beam dip reappears in
    the map with the opposite sign.  On a weak spot in a 20×20 box, ``B`` can
    be comparable to ``S``.  Removing the pedestal first leaves only the part
    that is proportional to the beam.

Normalisation options
---------------------
normalize_to_monitor : bool
    Divides by the monitor counts (incident beam intensity).  Removes beam
    current fluctuations measured upstream of the sample.

bg_center / bg_boxsize : tuple
    Defines a reference ROI used to normalise each frame.  Applied after
    monitor normalisation when both are active — but the monitor then cancels
    exactly in the ratio, so with a reference ROI ``normalize_to_monitor`` has
    no effect on the map.

    Two strategies, in increasing effectiveness:

    1. Empty detector region (no diffraction spots)
       Corrects for detector-level variations: diffuse scattering, gain
       drift, residual ring-refill artefacts not captured by the monitor.
       Use ``pedestal_ring=0`` here: on an empty region the ring median *is*
       the content of the box, and subtracting it would leave ~0.

    2. Substrate spot of the same reflection family  ← recommended
       The substrate and epitaxial layer are illuminated by the same beam
       at the same sample point simultaneously.  Any intensity variation
       (ring refill, beam fluctuation, monitor drift) affects both spots
       equally and cancels in the ratio.  The result is a map of
       ``I_layer / I_substrate``, which reflects only real structural
       variations in the epitaxial layer: local mosaicity, strain, and
       dislocation density — free of instrumental artefacts.

Usage
-----
>>> fig = spot_intensity_map(
...     h5_path="scan_001.h5",
...     img_source="path/to/tifs",
...     roi_center=(1255, 1390),   # epitaxial layer spot
...     roi_boxsize=(20, 20),
...     bg_center=(1255, 1500),    # same reflection in substrate
...     bg_boxsize=(20, 20),
... )
"""

from __future__ import annotations

from pathlib import Path
from warnings import warn

import h5py
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from lauexplore.image import read as read_image
from lauexplore.plots.base import _as_grid
from lauexplore.scan import Scan


def _make_window(center: tuple[int, int], boxsize: tuple[int, int], ring: int,
                 detector_shape: tuple[int, int], name: str):
    """Slices of the ROI grown by ``ring`` pixels on every side.

    The box is exactly ``boxsize`` pixels, odd sizes included: it spans
    ``c - size//2 .. c - size//2 + size - 1``, so an odd box is centred on
    ``c`` and an even one has ``c`` just right/below of its middle.
    """
    cx, cy = center
    w, h   = boxsize
    r0 = cy - h // 2 - ring
    c0 = cx - w // 2 - ring
    r1 = r0 + h + 2 * ring
    c1 = c0 + w + 2 * ring
    H, W = detector_shape
    # A negative start does not raise in numpy — it wraps around, and the sum
    # silently comes from the wrong place or from an empty slice.
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        raise ValueError(
            f"{name} (center {center}, boxsize {boxsize}, pedestal ring {ring}) "
            f"covers rows {r0}:{r1}, cols {c0}:{c1}, outside the {H}×{W} detector."
        )
    return slice(r0, r1), slice(c0, c1)


def _net_counts(window: np.ndarray, ring: int) -> np.ndarray:
    """Sum of the inner box minus the ring median times the box area.

    ``window`` is (..., h + 2·ring, w + 2·ring); a leading frame axis is kept.
    """
    window = np.asarray(window, dtype=float)
    if ring == 0:
        return window.sum(axis=(-2, -1))
    inner = window[..., ring:-ring, ring:-ring]
    edge  = np.ones(window.shape[-2:], dtype=bool)
    edge[ring:-ring, ring:-ring] = False
    pedestal = np.median(window[..., edge], axis=-1)
    return inner.sum(axis=(-2, -1)) - pedestal * inner.shape[-2] * inner.shape[-1]


def _read_h5_frame(index, file_path, key, squeeze, windows):
    """Top-level so loky can ship it to worker processes."""
    with h5py.File(file_path, "r") as f:
        ds = f[key]
        vals = [_net_counts(ds[0, rs, cs] if squeeze else ds[rs, cs], ring)
                for rs, cs, ring in windows]
    return index, vals


def _read_tif_frame(index, fname, windows):
    img = read_image(fname)
    return index, [_net_counts(img[rs, cs], ring) for rs, cs, ring in windows]


def _vds_sources(img_source: Path, h5_img_key: str) -> list[tuple[str, str, bool]]:
    """(file, key, squeeze) for each frame, in the order the VDS maps them.

    Taken from the VDS mapping itself rather than from re-globbing the source
    folder: a lexical sort of the file names only matches the frame order when
    the indices are zero-padded, and the mapping is the order the stack was
    actually built with.
    """
    with h5py.File(img_source, "r") as f:
        ds       = f[h5_img_key]
        n_frames = ds.shape[0]
        fallback = Path(f.attrs["source_folder"]) if "source_folder" in f.attrs else None
        sources  = [None] * n_frames
        for vs in ds.virtual_sources():
            frame = vs.vspace.get_select_bounds()[0][0]
            path  = Path(vs.file_name)
            if not path.is_absolute():
                path = img_source.parent / path
            if not path.exists() and fallback is not None:
                path = fallback / path.name
            squeeze = len(vs.src_space.shape) == 3   # (1, H, W) stored per file
            sources[frame] = (str(path), vs.dset_name, squeeze)
    return sources


def _integrate(
    img_source: Path,
    is_h5: bool,
    h5_img_key: str | None,
    tif_name,
    n: int,
    windows: list[tuple[slice, slice, int]],
    workers: int,
) -> np.ndarray:
    """Net counts of every window in the first ``n`` frames → (len(windows), n).

    Each frame is read once, whatever the number of windows.
    """
    result = np.empty((len(windows), n), dtype=float)

    if is_h5:
        with h5py.File(img_source, "r") as f:
            is_virtual = f[h5_img_key].is_virtual

        if not is_virtual:
            # Real dataset: single-threaded batch read is optimal
            batch = 500
            with h5py.File(img_source, "r") as f:
                ds = f[h5_img_key]
                with tqdm(total=n, desc="Loading H5 frames", unit="img") as pbar:
                    for start in range(0, n, batch):
                        end = min(start + batch, n)
                        for k, (rs, cs, ring) in enumerate(windows):
                            result[k, start:end] = _net_counts(ds[start:end, rs, cs], ring)
                        pbar.update(end - start)
            return result

        # Virtual dataset (1 file per frame): bypass virtual links and read the
        # source files directly so worker processes achieve true parallel I/O.
        # joblib/loky rather than ProcessPoolExecutor, which hangs at 0/N when
        # driven from a Jupyter kernel (see scan_pipeline._run_parallel).
        sources = _vds_sources(img_source, h5_img_key)
        jobs    = (delayed(_read_h5_frame)(i, *sources[i], windows) for i in range(n))
        backend, desc = "loky", "Loading H5 frames"
    else:
        jobs    = (delayed(_read_tif_frame)(i, tif_name(i), windows) for i in range(n))
        backend, desc = "threading", "Loading TIFs"

    results = Parallel(n_jobs=workers, backend=backend,
                       return_as="generator_unordered")(jobs)
    with tqdm(total=n, desc=desc, unit="img") as pbar:
        for i, vals in results:
            result[:, i] = vals
            pbar.update(1)
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
    pedestal_ring: int = 3,
    normalize_to_monitor: bool = True,
    bg_center: tuple[int, int] | None = None,
    bg_boxsize: tuple[int, int] | None = None,
    workers: int = 8,
    figsize: tuple[float, float] = (7, 6),
    cmap: str = "inferno",
    title: str | None = None,
    return_data: bool = False,
) -> plt.Figure | tuple[plt.Figure, np.ndarray]:
    """Compute and plot the integrated intensity of a diffraction spot across a scan.

    Parameters
    ----------
    h5_path : str or Path
        Path to the scan HDF5 file (scan geometry and monitor data).
    img_source : str or Path
        TIF folder path, or HDF5 file containing detector images.
    roi_center : (x, y)
        Centre of the spot ROI in 0-based detector pixel coordinates
        (column, row).
    roi_boxsize : (width, height)
        Size of the spot ROI in pixels.
    scan_number : int
        Scan entry number inside the HDF5 (default 1).
    img_prefix, img_suffix, img_index_pad :
        TIF filename format: ``{prefix}{index:0>{pad}d}{suffix}``.
    h5_img_key : str, optional
        Dataset key when ``img_source`` is an HDF5 file.
        Shape must be ``(n_images, height, width)``.
    pedestal_ring : int
        Width of the frame around each ROI whose median is subtracted, per
        pixel, before summing (default 3; 0 disables).  Needed for the monitor
        normalisation to be correct — see module docstring.  Applied to the
        reference ROI too; set 0 when the reference is an empty region.
    normalize_to_monitor : bool
        Divide by monitor counts to correct for incident beam fluctuations
        (default True).
    bg_center : (x, y), optional
        Centre of the reference ROI used to normalise each frame.
        Two recommended choices (see module docstring):
        (a) an empty detector region with no diffraction spots, or
        (b) a substrate spot of the same reflection family as ``roi_center``
            — the preferred option, as it cancels all instrumental artefacts
            (ring refill, beam fluctuations) and yields a pure
            ``I_layer / I_substrate`` ratio map.
        The monitor cancels in this ratio.
    bg_boxsize : (width, height), optional
        Size of the background ROI.  Required when ``bg_center`` is set.
    workers : int
        Parallel readers: threads for TIFs, processes for a virtual HDF5
        stack.  A real HDF5 dataset is read in batches on one thread.
    figsize, cmap, title :
        Matplotlib figure parameters.
    return_data : bool
        Also return the (ny, nx) intensity grid (default False).

    Returns
    -------
    matplotlib.figure.Figure, or (Figure, ndarray) when ``return_data``.
    """
    h5_path    = Path(h5_path)
    img_source = Path(img_source)
    is_h5      = img_source.suffix in ('.h5', '.hdf5')

    if bg_center is not None and bg_boxsize is None:
        raise ValueError("bg_boxsize must be set when bg_center is provided.")
    if is_h5 and h5_img_key is None:
        raise ValueError("h5_img_key must be set when img_source is an HDF5 file.")

    scan    = Scan.from_h5(h5_path, scan_number)
    monitor = np.asarray(scan.monitor_data, dtype=float)

    def tif_name(i):
        return img_source / f"{img_prefix}{i:0>{img_index_pad}d}{img_suffix}"

    if is_h5:
        with h5py.File(img_source, "r") as f:
            n_frames       = f[h5_img_key].shape[0]
            detector_shape = f[h5_img_key].shape[-2:]
    else:
        n_frames       = scan.length
        detector_shape = read_image(tif_name(0)).shape[-2:]

    # An interrupted scan has fewer monitor points and frames than its title
    # promises; integrate what was acquired and leave the rest NaN.
    n = min(scan.length, monitor.size, n_frames)
    if n < scan.length:
        warn(f"Only {n} of {scan.length} scan points available "
             f"(monitor: {monitor.size}, frames: {n_frames}); the rest is NaN.",
             RuntimeWarning)

    windows = [(*_make_window(roi_center, roi_boxsize, pedestal_ring,
                              detector_shape, "Spot ROI"), pedestal_ring)]
    if bg_center is not None:
        windows.append((*_make_window(bg_center, bg_boxsize, pedestal_ring,
                                      detector_shape, "Reference ROI"), pedestal_ring))

    counts    = _integrate(img_source, is_h5, h5_img_key, tif_name, n, windows, workers)
    intensity = counts[0]

    # ── Monitor normalisation ─────────────────────────────────────────────────
    if normalize_to_monitor:
        mon       = np.where(monitor[:n] > 0, monitor[:n], np.nan)
        intensity = intensity / mon

    # ── Reference normalisation ───────────────────────────────────────────────
    # The monitor would divide the reference too and cancel, so it is left out.
    if bg_center is not None:
        bg        = np.where(counts[1] == 0, np.nan, counts[1])
        intensity = counts[0] / bg

    # ── Colorbar label ────────────────────────────────────────────────────────
    label = "Net counts" if pedestal_ring else "Integrated counts"
    if bg_center is not None:
        label += " / reference"
    elif normalize_to_monitor:
        label += " / monitor"

    # ── Reshape and plot ──────────────────────────────────────────────────────
    full           = np.full(scan.length, np.nan)
    full[:n]       = intensity
    intensity_grid = _as_grid(full, scan)
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

    if return_data:
        return fig, intensity_grid
    return fig
