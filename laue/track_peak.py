"""
track_peak.py
=============

Track a diffraction peak across a sequence of Laue peaksearch files.

Returns a single :class:`PeakTrack` object whose attributes are
``np.ndarray`` of length ``len(img_range)``.  Missing images or unmatched
peaks are stored as ``np.nan`` so every array is directly plottable.

Usage
-----
>>> track = track_peak("/data/dat", target_x=1357.0, target_y=1693.0,
...                    img_range=range(0, 20001))
>>> track.peak_Itot          # full array, shape (20001,)
>>> track.fwhm_eff[42]       # scalar for image index 42
>>> track.peak_X[100:200]    # slice

Derived property
----------------
``fwhm_eff``
    Effective (isotropic) FWHM — geometric mean of the two ellipse axes::

        fwhm_eff = sqrt(peak_fwaxmaj * peak_fwaxmin)

    Equals the FWHM of a circular Gaussian with the same integrated area
    as the fitted elliptical spot.

``dist``
    Euclidean distance (pixels) between the matched peak and the target
    position.  Useful to monitor peak drift over time.

``found``
    Boolean mask: ``True`` where a peak was matched, ``False`` otherwise.
"""

import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

COLUMNS = [
    "peak_X", "peak_Y", "peak_Itot", "peak_Isub",
    "peak_fwaxmaj", "peak_fwaxmin", "peak_inclination",
    "Xdev", "Ydev", "peak_bkg", "Ipixmax",
]
COL_INDEX = {name: i for i, name in enumerate(COLUMNS)}


# ---------------------------------------------------------------------------
# PeakTrack — struct-of-arrays result object
# ---------------------------------------------------------------------------

@dataclass
class PeakTrack:
    """
    Tracked peak data across an image sequence.

    Every attribute is a ``np.ndarray`` of shape ``(n_images,)``.
    Entries are ``np.nan`` (or ``False`` for ``found``) where the file was
    missing or no peak was matched within the tolerance.

    Attributes
    ----------
    img_indices : np.ndarray of int
        Image indices corresponding to each position in the arrays.
    found : np.ndarray of bool
        ``True`` where a peak was successfully matched.
    peak_X, peak_Y : np.ndarray
        Fitted peak centroid (pixels).
    peak_Itot : np.ndarray
        Total integrated intensity.
    peak_Isub : np.ndarray
        Background-subtracted integrated intensity.
    peak_fwaxmaj : np.ndarray
        FWHM along the major axis of the fitted ellipse (pixels).
    peak_fwaxmin : np.ndarray
        FWHM along the minor axis of the fitted ellipse (pixels).
    peak_inclination : np.ndarray
        Inclination angle of the ellipse major axis (degrees).
    Xdev, Ydev : np.ndarray
        Centroid deviation from the predicted position (pixels).
    peak_bkg : np.ndarray
        Local background level.
    Ipixmax : np.ndarray
        Maximum pixel intensity in the peak region.
    fwhm_eff : np.ndarray
        Effective isotropic FWHM: ``sqrt(peak_fwaxmaj * peak_fwaxmin)``.
    dist : np.ndarray
        Euclidean distance between matched peak and the target position.
    """
    img_indices: np.ndarray
    found: np.ndarray
    peak_X: np.ndarray
    peak_Y: np.ndarray
    peak_Itot: np.ndarray
    peak_Isub: np.ndarray
    peak_fwaxmaj: np.ndarray
    peak_fwaxmin: np.ndarray
    peak_inclination: np.ndarray
    Xdev: np.ndarray
    Ydev: np.ndarray
    peak_bkg: np.ndarray
    Ipixmax: np.ndarray
    fwhm_eff: np.ndarray
    dist: np.ndarray

    def __len__(self) -> int:
        return len(self.img_indices)

    def __repr__(self) -> str:
        n = len(self)
        found = int(self.found.sum())
        return (
            f"PeakTrack(n_images={n}, found={found}, missing={n - found})"
        )


# ---------------------------------------------------------------------------
# Row-level result (internal use only)
# ---------------------------------------------------------------------------

# One entry per image: either a row vector + dist, or None
_RowResult = Optional[tuple]   # (np.ndarray of shape (N_COLS,), float dist)


def _process_image(
    img_index: int,
    folder: str,
    target_x: float,
    target_y: float,
    tolerance: float,
    filename_pattern: str,
) -> _RowResult:
    """
    Load one peaksearch file and return the raw data row of the nearest peak.

    Returns ``None`` if the file is missing, empty, malformed, or no peak
    falls within *tolerance* pixels of the target.
    """
    filepath = os.path.join(folder, filename_pattern.format(img_index))

    if not os.path.isfile(filepath):
        return None

    try:
        data = np.loadtxt(filepath, skiprows=1)
    except Exception:
        return None

    if data.size == 0:
        return None

    if data.ndim == 1:
        data = data[np.newaxis, :]

    dx = data[:, COL_INDEX["peak_X"]] - target_x
    dy = data[:, COL_INDEX["peak_Y"]] - target_y
    dist = np.sqrt(dx**2 + dy**2)

    nearest = int(np.argmin(dist))
    if dist[nearest] > tolerance:
        return None

    return (data[nearest], float(dist[nearest]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def track_peak(
    folder: str,
    target_x: float,
    target_y: float,
    img_range: range,
    tolerance: float = 5.0,
    filename_pattern: str = "img_{:0>4d}.dat",
    n_workers: int = None,
) -> PeakTrack:
    """
    Track a diffraction peak across a sequence of Laue peaksearch files.

    Parameters
    ----------
    folder : str
        Path to the directory containing the .dat files.
    target_x : float
        Expected X position of the peak (pixels).
    target_y : float
        Expected Y position of the peak (pixels).
    img_range : range
        Sequence of image indices, e.g. ``range(0, 20001)``.
    tolerance : float, optional
        Maximum Euclidean distance (pixels) to accept a peak match.
        Default: ``5.0``.
    filename_pattern : str, optional
        Format string to build filenames from an index.
        Default: ``"img_{:0>4d}.dat"``.
    n_workers : int, optional
        Number of parallel worker processes.
        Defaults to ``multiprocessing.cpu_count()``.

    Returns
    -------
    PeakTrack
        Object whose attributes are arrays of length ``len(img_range)``.
        Access any property directly::

            track = track_peak(...)
            track.peak_Itot        # full array
            track.fwhm_eff[42]     # single image
            track.peak_X[10:20]    # slice
    """
    if n_workers is None:
        n_workers = cpu_count()

    worker = partial(
        _process_image,
        folder=folder,
        target_x=target_x,
        target_y=target_y,
        tolerance=tolerance,
        filename_pattern=filename_pattern,
    )

    indices = list(img_range)
    with Pool(processes=n_workers) as pool:
        raw = pool.map(worker, indices)

    # Assemble struct-of-arrays
    n = len(indices)
    nan = np.full(n, np.nan)

    arrays = {col: nan.copy() for col in COLUMNS}
    arrays["fwhm_eff"] = nan.copy()
    arrays["dist"] = nan.copy()
    found = np.zeros(n, dtype=bool)

    for k, result in enumerate(raw):
        if result is None:
            continue
        row, dist = result
        found[k] = True
        for col, idx in COL_INDEX.items():
            arrays[col][k] = row[idx]
        maj = row[COL_INDEX["peak_fwaxmaj"]]
        minn = row[COL_INDEX["peak_fwaxmin"]]
        arrays["fwhm_eff"][k] = float(np.sqrt(maj * minn))
        arrays["dist"][k] = dist

    return PeakTrack(
        img_indices=np.array(indices),
        found=found,
        **arrays,
    )


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    FOLDER = "/path/to/dat/files"
    TARGET_X = 1357.0
    TARGET_Y = 1693.0
    IMG_RANGE = range(0, 20001)
    TOLERANCE = 5.0   # pixels
    N_WORKERS = 8     # None → use all available CPUs

    track = track_peak(
        folder=FOLDER,
        target_x=TARGET_X,
        target_y=TARGET_Y,
        img_range=IMG_RANGE,
        tolerance=TOLERANCE,
        n_workers=N_WORKERS,
    )

    print(track)
    # PeakTrack(n_images=20001, found=19834, missing=167)

    # Access any property as a full array
    print(track.peak_Itot)          # shape (20001,)
    print(track.fwhm_eff[42])       # scalar for image 42
    print(track.peak_X[100:110])    # slice

    # Plot three properties
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, prop in zip(axes, ["peak_Itot", "fwhm_eff", "dist"]):
        ax.plot(track.img_indices, getattr(track, prop), lw=0.8)
        ax.set_ylabel(prop)
    axes[-1].set_xlabel("Image index")
    fig.suptitle(f"Peak at ({TARGET_X}, {TARGET_Y}) — tolerance {TOLERANCE} px")
    fig.tight_layout()
    fig.savefig("peak_track.png", dpi=150)
    plt.show()
