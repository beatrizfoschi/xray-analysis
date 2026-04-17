#!/usr/bin/env python3
"""
peak_search_2g.py — Laue diffraction peak search with 1- or 2-Gaussian fitting.

Processes .tif diffraction images and writes per-image .dat peak files that are
compatible with the LaueTools format (11 standard columns) plus two extra columns
(n_gaussians, chi2_reduced).

For overlapping spots (e.g. GaN substrate + porous GaN layer), the fitter
automatically upgrades from 1G to 2G when:
  1. Two distinct local maxima are detected within the fit ROI  (primary trigger)
  2. The 1G Poisson-normalised reduced chi² exceeds --chi2-threshold  (secondary)

When a 2-Gaussian fit succeeds and improves chi², two rows are written to the
.dat file — one per component — each in the standard column format.

Usage (single image):
    python peak_search_2g.py --exp-folder /path/to/tifs --out-folder all_peaks \\
        --first-img 0 --last-img 0

Usage (batch via SLURM — see run_peaksearch.sh):
    python peak_search_2g.py --exp-folder /path/to/tifs --out-folder all_peaks \\
        --first-img 0 --last-img 9802 --n-cpus 32
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fabio
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.stats

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

try:
    from LaueTools.CrystalParameters import Prepare_Grain
    from LaueTools.lauecore import SimulateLaue_full_np
    from LaueTools.dict_LaueTools import dict_Materials, dict_CCD
    import LaueTools.IOLaueTools as _IOLT
    _HAS_LAUETOOLS_SIM = True
except ImportError:
    _HAS_LAUETOOLS_SIM = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FWHM_FACTOR = 2.3548200450309493   # 2 * sqrt(2 * ln 2)
MAX_NFEV    = 1550                  # reject fit if optimiser exceeds this


# ---------------------------------------------------------------------------
# 1. Laue diffraction simulation
# ---------------------------------------------------------------------------

def simulate_peaks(
    material: str,
    ub_matrix: np.ndarray,
    calibration_parameters,
    emin: float = 5.0,
    emax: float = 27.0,
    camera_label: str = "sCMOS",
    detector_diameter: float = 148.1212,
    material_dictionary: Optional[dict] = None,
) -> np.ndarray:
    """Simulate Laue diffraction spot positions on the detector.

    Parameters
    ----------
    material              : material key in dict_Materials (e.g. 'GaN', 'Al2O3')
    ub_matrix             : (3, 3) orientation / UB matrix
    calibration_parameters: detector calibration list from readCalib_det_file
                            (CCDCalibParameters field)
    emin, emax            : energy range [keV]
    camera_label          : CCD label in dict_CCD (default 'sCMOS')
    detector_diameter     : detector diameter [mm] (default 148.1212, multiplied by 1.75
                            internally to match LaueTools convention)
    material_dictionary   : override for dict_Materials (None → use LaueTools default)

    Returns
    -------
    (N, 2) float array of simulated (X, Y) positions in XMAS pixel coordinates
    (X = col + 1, Y = row + 1), clipped to detector area.

    Raises
    ------
    ImportError if LaueTools simulation modules are not available.
    """
    if not _HAS_LAUETOOLS_SIM:
        raise ImportError(
            "LaueTools simulation modules not found. "
            "Install LaueTools or omit --material / --ub arguments."
        )

    mat_dict = material_dictionary if material_dictionary is not None else dict_Materials
    sim_params = Prepare_Grain(material, ub_matrix, dictmaterials=mat_dict)

    pixel_size  = dict_CCD[camera_label][1]
    frame_shape = dict_CCD[camera_label][0]   # (nrows, ncols)

    result = SimulateLaue_full_np(
        sim_params,
        emin,
        emax,
        calibration_parameters,
        detectordiameter = detector_diameter * 1.75,
        pixelsize        = pixel_size,
        dim              = frame_shape,
        dictmaterials    = mat_dict,
        kf_direction     = "Z>0",
        removeharmonics  = 0,
    )

    x_pos = result[3]   # col direction (XMAS, 1-based)
    y_pos = result[4]   # row direction (XMAS, 1-based)

    on_detector = (
        (x_pos > 0) & (x_pos < frame_shape[1]) &
        (y_pos > 0) & (y_pos < frame_shape[0])
    )
    return np.column_stack([x_pos[on_detector], y_pos[on_detector]])


def load_calibration(det_file: str):
    """Load detector calibration parameters from a LaueTools .det file.

    Returns the CCDCalibParameters list needed by SimulateLaue_full_np.
    """
    if not _HAS_LAUETOOLS_SIM:
        raise ImportError("LaueTools not available. Please import lauetools")
    calib_dict = _IOLT.readCalib_det_file(det_file)
    return calib_dict["CCDCalibParameters"]


# ---------------------------------------------------------------------------
# 2. Simulation-guided local maxima search
# ---------------------------------------------------------------------------

def find_local_maxima_sim_guided(
    bg_sub_image: np.ndarray,
    sim_xy: np.ndarray,
    tolerance: float,
    threshold: float = 0.0,
) -> np.ndarray:
    """Find local maxima near simulated spot positions.

    For each simulated spot, searches within a circle of radius `tolerance`
    pixels for the brightest pixel above `threshold` in the background-subtracted
    image.  Multiple simulated spots that resolve to the same pixel are
    deduplicated (brightest kept).

    Parameters
    ----------
    bg_sub_image : background-subtracted image (float array)
    sim_xy       : (N, 2) simulated positions in XMAS coords (X=col+1, Y=row+1)
    tolerance    : search radius [pixels]
    threshold    : minimum bg-subtracted intensity to accept (default 0)

    Returns
    -------
    (M, 2) int array of (row, col) candidates, sorted by intensity descending.
    """
    nrows, ncols = bg_sub_image.shape
    candidates: List[Tuple[float, int, int]] = []

    for x_sim, y_sim in sim_xy:
        # XMAS → 0-based array indices
        col_sim = x_sim - 1.0
        row_sim = y_sim - 1.0

        # Bounding box (clamped)
        r0 = max(0,      int(np.floor(row_sim - tolerance)))
        r1 = min(nrows,  int(np.ceil (row_sim + tolerance)) + 1)
        c0 = max(0,      int(np.floor(col_sim - tolerance)))
        c1 = min(ncols,  int(np.ceil (col_sim + tolerance)) + 1)

        if r1 <= r0 or c1 <= c0:
            continue

        rr, cc = np.ogrid[r0:r1, c0:c1]
        dist   = np.sqrt((rr - row_sim) ** 2 + (cc - col_sim) ** 2)
        patch  = np.where(dist <= tolerance, bg_sub_image[r0:r1, c0:c1], -np.inf)

        if patch.max() < threshold:
            continue

        local_pos = np.unravel_index(patch.argmax(), patch.shape)
        abs_row   = r0 + local_pos[0]
        abs_col   = c0 + local_pos[1]
        candidates.append((float(bg_sub_image[abs_row, abs_col]), abs_row, abs_col))

    if not candidates:
        return np.empty((0, 2), dtype=int)

    # Sort by intensity, deduplicate by position
    candidates.sort(reverse=True)
    kept: List[Tuple[float, int, int]] = []
    seen_positions: set = set()
    for intens, row, col in candidates:
        if (row, col) not in seen_positions:
            seen_positions.add((row, col))
            kept.append((intens, row, col))

    return np.array([[r, c] for _, r, c in kept], dtype=int)


# ---------------------------------------------------------------------------
# 3. Background estimation
# ---------------------------------------------------------------------------

def auto_background(image: np.ndarray, boxsize: int = 10) -> np.ndarray:
    """Background via minimum filter — replicates LaueTools auto_background.

    Equivalent to LaueTools imageprocessing.compute_autobackground_image with
    filter_minimum(..., boxsize=boxsize).
    """
    return scipy.ndimage.minimum_filter(image.astype(float), size=2 * boxsize + 1)


# ---------------------------------------------------------------------------
# 4. Threshold-based local maxima search (full image)
# ---------------------------------------------------------------------------

def find_local_maxima(
    bg_sub_image: np.ndarray,
    threshold: float,
    pixel_near_radius: int = 2,
    max_candidates: int = 5000,
) -> np.ndarray:
    """Return (N, 2) int array of (row, col) candidate positions.

    Steps:
      1. Label connected components above threshold.
      2. Keep the brightest pixel in each component.
      3. Remove duplicates within pixel_near_radius of a brighter candidate.
      4. Return sorted by intensity descending (capped at max_candidates).
    """
    above = bg_sub_image >= threshold
    if not np.any(above):
        return np.empty((0, 2), dtype=int)

    labelled, n_labels = scipy.ndimage.label(above)
    if n_labels == 0:
        return np.empty((0, 2), dtype=int)

    candidates: List[Tuple[float, int, int]] = []
    for lab in range(1, n_labels + 1):
        local_vals = np.where(labelled == lab, bg_sub_image, -np.inf)
        pos = np.unravel_index(local_vals.argmax(), bg_sub_image.shape)
        candidates.append((float(bg_sub_image[pos]), int(pos[0]), int(pos[1])))

    # Sort by intensity descending
    candidates.sort(key=lambda c: c[0], reverse=True)
    candidates = candidates[:max_candidates]

    # Remove duplicates within pixel_near_radius
    kept: List[Tuple[float, int, int]] = []
    for intens, row, col in candidates:
        too_close = any(
            abs(row - kr) <= pixel_near_radius and abs(col - kc) <= pixel_near_radius
            for _, kr, kc in kept
        )
        if not too_close:
            kept.append((intens, row, col))

    if not kept:
        return np.empty((0, 2), dtype=int)

    return np.array([[r, c] for _, r, c in kept], dtype=int)


# ---------------------------------------------------------------------------
# 5. Blacklist
# ---------------------------------------------------------------------------

def load_blacklist(filepath: str) -> Optional[np.ndarray]:
    """Load (X, Y) positions to ignore from a text file.

    Accepts space/tab-delimited files; lines starting with '#' are skipped.
    Reads the first two numeric columns as X (col direction) and Y (row direction),
    compatible with LaueTools .dat and .fit file formats.
    Returns an (N, 2) float array or None.
    """
    if not filepath or not os.path.isfile(filepath):
        return None
    rows = []
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return np.array(rows, dtype=float) if rows else None


def filter_blacklisted(
    candidates: np.ndarray,
    blacklist_xy: Optional[np.ndarray],
    radius: float = 5.0,
) -> np.ndarray:
    """Remove candidates within radius pixels of any blacklisted (X, Y) position.

    Candidates are (row, col) pairs; blacklist is in XMAS (X=col+1, Y=row+1).
    """
    if blacklist_xy is None or len(blacklist_xy) == 0 or len(candidates) == 0:
        return candidates

    # Convert candidates from 0-based (row, col) to XMAS (X=col+1, Y=row+1)
    cand_xy = np.stack([candidates[:, 1] + 1.0, candidates[:, 0] + 1.0], axis=1)

    keep_mask = np.ones(len(candidates), dtype=bool)
    for i, cxy in enumerate(cand_xy):
        dists = np.sqrt(np.sum((blacklist_xy - cxy) ** 2, axis=1))
        if dists.min() <= radius:
            keep_mask[i] = False

    return candidates[keep_mask]


# ---------------------------------------------------------------------------
# 6. ROI extraction
# ---------------------------------------------------------------------------

def extract_roi(
    image: np.ndarray,
    seed_row: int,
    seed_col: int,
    boxsize: int,
    saturation_value: float = 65000.0,
) -> Tuple[np.ndarray, bool, int, int]:
    """Extract a (2*boxsize+1) square ROI, zero-padded at image boundaries.

    Returns
    -------
    roi            : float array, shape (2*boxsize+1, 2*boxsize+1)
    is_saturated   : True if any pixel in the original crop >= saturation_value
    row_origin     : image row corresponding to roi[0, :]
    col_origin     : image col corresponding to roi[:, 0]
    """
    nrows, ncols = image.shape
    roi_size = 2 * boxsize + 1
    r0 = seed_row - boxsize
    c0 = seed_col - boxsize

    # Source window clamped to image bounds
    ir0 = max(r0, 0);      ir1 = min(r0 + roi_size, nrows)
    ic0 = max(c0, 0);      ic1 = min(c0 + roi_size, ncols)

    roi = np.zeros((roi_size, roi_size), dtype=float)
    dr0 = ir0 - r0;  dc0 = ic0 - c0
    roi[dr0: dr0 + (ir1 - ir0), dc0: dc0 + (ic1 - ic0)] = image[ir0:ir1, ic0:ic1]

    is_saturated = bool(np.any(image[ir0:ir1, ic0:ic1] >= saturation_value))
    return roi, is_saturated, r0, c0


# ---------------------------------------------------------------------------
# 7. Gaussian models
# ---------------------------------------------------------------------------

def _gauss2d(params: np.ndarray, row_arr: np.ndarray, col_arr: np.ndarray) -> np.ndarray:
    """Evaluate a rotated elliptical Gaussian on a grid.

    params = (height, amplitude, cen_row, cen_col, width_row, width_col, rota_deg)

    Convention identical to LaueTools fit2Dintensity.twodgaussian (rotate=1,
    vheight=1, circle=0).  Note: numpy array axis-0 = rows, axis-1 = cols.
    """
    height, amplitude, cen_row, cen_col, width_row, width_col, rota_deg = (
        float(params[0]), float(params[1]),
        float(params[2]), float(params[3]),
        abs(float(params[4])) + 1e-6,
        abs(float(params[5])) + 1e-6,
        float(params[6]),
    )
    rota = np.deg2rad(rota_deg)
    cos_r, sin_r = np.cos(rota), np.sin(rota)

    # Rotate both the reference centre and the evaluation points
    rcen_r = cen_row * cos_r - cen_col * sin_r
    rcen_c = cen_row * sin_r + cen_col * cos_r
    rp = row_arr * cos_r - col_arr * sin_r
    cp = row_arr * sin_r + col_arr * cos_r

    return height + amplitude * np.exp(
        -(((rcen_r - rp) / width_row) ** 2 + ((rcen_c - cp) / width_col) ** 2) / 2.0
    )


def _gauss2d_2peaks(params: np.ndarray, row_arr: np.ndarray, col_arr: np.ndarray) -> np.ndarray:
    """Shared baseline + two independent rotated elliptical Gaussians.

    params = (height,
              amp1, cen_row1, cen_col1, width_row1, width_col1, rota1_deg,
              amp2, cen_row2, cen_col2, width_row2, width_col2, rota2_deg)
    Total: 13 parameters.
    """
    height = float(params[0])
    p1 = np.array([height,      params[1],  params[2],  params[3],
                   params[4],   params[5],  params[6]])
    p2 = np.array([0.0,         params[7],  params[8],  params[9],
                   params[10],  params[11], params[12]])
    return _gauss2d(p1, row_arr, col_arr) + _gauss2d(p2, row_arr, col_arr)


# ---------------------------------------------------------------------------
# 8. Moment-based initial parameter guess
# ---------------------------------------------------------------------------

def _moments_guess(roi: np.ndarray) -> np.ndarray:
    """Estimate 1G starting parameters from image moments (no rotation).

    Returns array [height, amplitude, cen_row, cen_col, width_row, width_col, 0.0].
    """
    data = roi.astype(float)
    total = data.sum() or 1.0
    rows_idx, cols_idx = np.indices(data.shape)

    cen_row = (rows_idx * data).sum() / total
    cen_col = (cols_idx * data).sum() / total

    cr = int(np.clip(round(cen_row), 0, data.shape[0] - 1))
    cc = int(np.clip(round(cen_col), 0, data.shape[1] - 1))

    col_slice = data[:, cc]
    cs = col_slice.sum() or 1.0
    width_row = np.sqrt(
        max(((np.arange(len(col_slice)) - cen_row) ** 2 * col_slice).sum() / cs, 1e-4)
    )
    row_slice = data[cr, :]
    rs = row_slice.sum() or 1.0
    width_col = np.sqrt(
        max(((np.arange(len(row_slice)) - cen_col) ** 2 * row_slice).sum() / rs, 1e-4)
    )

    # scipy.stats.mode API changed in 1.9 (keepdims) and again in 1.11 (deprecated)
    try:
        height = float(scipy.stats.mode(data.ravel(), keepdims=True)[0][0])
    except TypeError:
        height = float(scipy.stats.mode(data.ravel())[0][0])

    amplitude = max(float(data.max()) - height, 1.0)
    return np.array([height, amplitude, cen_row, cen_col,
                     max(width_row, 0.1), max(width_col, 0.1), 0.0])


# ---------------------------------------------------------------------------
# 9. Single-Gaussian fit
# ---------------------------------------------------------------------------

def _chi2_reduced(roi: np.ndarray, model: np.ndarray, n_params: int) -> float:
    """Poisson-normalised reduced chi² for a 2-D fit."""
    residuals = roi - model
    dof = max(roi.size - n_params, 1)
    return float(np.sum(residuals ** 2 / np.maximum(np.abs(roi), 1.0)) / dof)


def fit_1gaussian(
    roi: np.ndarray,
    seed_row_roi: float,
    seed_col_roi: float,
    xtol: float = 0.5,
    peak_size_range: Tuple[float, float] = (0.1, 6.0),
    fit_pixel_dev: float = 3.0,
) -> Optional[Tuple[np.ndarray, float, int]]:
    """Fit a single rotated elliptical Gaussian to roi.

    Returns (params_opt, chi2_reduced, nfev) or None if the fit is rejected.
    params_opt = [height, amplitude, cen_row, cen_col, width_row, width_col, rota_deg]
    """
    grid = np.indices(roi.shape)   # (2, H, W): grid[0]=rows, grid[1]=cols

    def errfunc(p):
        return np.ravel(_gauss2d(p, grid[0], grid[1]) - roi)

    p0 = _moments_guess(roi)
    try:
        result = scipy.optimize.leastsq(errfunc, p0, full_output=True, xtol=xtol)
    except Exception:
        return None

    p_opt, _, infodict, _, _ = result
    nfev = int(infodict['nfev'])

    height, amplitude, cen_row, cen_col, width_row, width_col, _ = p_opt

    # --- Rejection checks (matching LaueTools) ---
    if nfev >= MAX_NFEV:
        return None
    if amplitude <= 0 or height < 0:
        return None

    dev = np.sqrt((cen_row - seed_row_roi) ** 2 + (cen_col - seed_col_roi) ** 2)
    if dev > fit_pixel_dev:
        return None

    fwhm_row = abs(width_row) * FWHM_FACTOR
    fwhm_col = abs(width_col) * FWHM_FACTOR
    fwhm_avg = (fwhm_row + fwhm_col) / 2.0
    if not (peak_size_range[0] <= fwhm_avg <= peak_size_range[1]):
        return None

    model = _gauss2d(p_opt, grid[0], grid[1])
    chi2_r = _chi2_reduced(roi, model, n_params=7)

    return p_opt, chi2_r, nfev


# ---------------------------------------------------------------------------
# 10. Two-peak detection in ROI
# ---------------------------------------------------------------------------

def find_two_peaks_in_roi(
    roi: np.ndarray,
    threshold_fraction: float = 0.1,
    min_separation: float = 2.0,
    max_separation: float = np.inf,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Detect two distinct local maxima in a ROI.

    Returns ((row1, col1), (row2, col2)) — the two brightest components whose
    separation is within [min_separation, max_separation] — or None if no such
    pair exists.
    """
    bg = float(roi.min())
    span = float(roi.max()) - bg
    if span <= 0:
        return None

    above = (roi - bg) >= threshold_fraction * span
    labelled, n_labels = scipy.ndimage.label(above)
    if n_labels < 2:
        return None

    components: List[Tuple[float, float, float]] = []
    for lab in range(1, n_labels + 1):
        local_vals = np.where(labelled == lab, roi, -np.inf)
        pos = np.unravel_index(local_vals.argmax(), roi.shape)
        components.append((float(roi[pos]), float(pos[0]), float(pos[1])))

    components.sort(reverse=True)   # brightest first
    r1, c1 = components[0][1], components[0][2]

    # Look for the brightest second component within [min_separation, max_separation]
    for _, r2, c2 in components[1:]:
        sep = np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
        if min_separation <= sep <= max_separation:
            return (r1, c1), (r2, c2)

    return None


# ---------------------------------------------------------------------------
# 11. Two-Gaussian fit
# ---------------------------------------------------------------------------

def fit_2gaussian(
    roi: np.ndarray,
    seed1: Tuple[float, float],
    seed2: Tuple[float, float],
    xtol: float = 0.5,          # NOTE: ignored — least_squares uses xtol=1e-4 internally.
    peak_size_range: Tuple[float, float] = (0.1, 6.0),
    fit_pixel_dev: float = 3.0,
    p1g_opt: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, int, str]]:
    """Fit a two-Gaussian model to roi.

    Returns (params_c1, params_c2, chi2_reduced, nfev, reject_reason) on success,
    where reject_reason is an empty string.
    Returns (None, None, None, None, reject_reason) on failure, where reject_reason
    is one of:
      'exception'   — optimizer raised an exception
      'nfev'        — max function evaluations reached without convergence
      'amp'         — one or both fitted amplitudes are non-positive
      'height'      — fitted baseline is negative
      'drift_c1'    — primary component centre drifted too far from its seed
      'drift_c2'    — secondary component centre drifted too far from its seed

    Each params_cX = [height_shared, amplitude, cen_row, cen_col,
                      width_row, width_col, rota_deg]
    where height_shared is the common baseline.
    """
    grid = np.indices(roi.shape)
    r1, c1 = seed1
    r2, c2 = seed2

    # Initial parameter guess
    if p1g_opt is not None:
        height0 = float(p1g_opt[0])
        wr0 = max(abs(float(p1g_opt[4])), 0.1)
        wc0 = max(abs(float(p1g_opt[5])), 0.1)
        # leastsq (used in fit_1gaussian) has no bounds, so the angle can be
        # outside [-180, 180].  Wrap it into the valid range before passing to
        # least_squares, which will raise ValueError if p0 is infeasible.
        th0 = float(p1g_opt[6]) % 360.0
        if th0 > 180.0:
            th0 -= 360.0
    else:
        guess = _moments_guess(roi)
        height0 = float(guess[0])
        wr0 = max(abs(float(guess[4])), 0.1)
        wc0 = max(abs(float(guess[5])), 0.1)
        th0 = 0.0

    # Each component should start narrower than the 1G fit: two peaks of width σ
    # that overlap produce a combined blob of width ~√2·σ, so individual widths
    # should initialise at σ/√2.  This prevents the optimizer from widening one
    # component to absorb signal that belongs to the other.
    wr0 = max(wr0 / np.sqrt(2.0), peak_size_range[0] / FWHM_FACTOR)
    wc0 = max(wc0 / np.sqrt(2.0), peak_size_range[0] / FWHM_FACTOR)

    # Scale amplitudes by 1/(1+overlap) so the combined initial model does not
    # start at ~2x the data level when seeds are close.
    sep_init  = np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
    mean_w    = (wr0 + wc0) / 2.0
    overlap   = np.exp(-sep_init ** 2 / (4.0 * max(mean_w ** 2, 0.01)))
    amp_scale = 1.0 / (1.0 + overlap)   # each component gets this fraction of signal

    def _roi_val(r, c):
        ri = int(np.clip(round(r), 0, roi.shape[0] - 1))
        ci = int(np.clip(round(c), 0, roi.shape[1] - 1))
        return max(float(roi[ri, ci]) - height0, 1.0) * amp_scale

    p0 = np.array([
        height0,
        _roi_val(r1, c1), r1, c1, wr0, wc0, th0,
        _roi_val(r2, c2), r2, c2, wr0, wc0, th0,
    ], dtype=float)

    # Use least_squares with bounds to prevent width blow-up.
    # Widths are constrained to peak_size_range (in sigma units = FWHM / FWHM_FACTOR).
    w_lo   = peak_size_range[0] / FWHM_FACTOR
    w_hi   = peak_size_range[1] / FWHM_FACTOR
    roi_h, roi_w = roi.shape
    lb = [-np.inf,  0, 0,     0,     w_lo, w_lo, -180,
                    0, 0,     0,     w_lo, w_lo, -180]
    ub = [ np.inf,  np.inf, roi_h, roi_w, w_hi, w_hi,  180,
                    np.inf, roi_h, roi_w, w_hi, w_hi,  180]

    _fail = lambda reason: (None, None, None, None, reason)

    try:
        # xtol here is the relative change in parameters (not pixels as in leastsq).
        # Use 1e-4 regardless of cfg.fit_xtol to ensure proper convergence.
        result = scipy.optimize.least_squares(
            lambda p: np.ravel(_gauss2d_2peaks(p, grid[0], grid[1]) - roi),
            p0, method="trf", bounds=(lb, ub),
            xtol=1e-4, ftol=1e-4, gtol=1e-4, max_nfev=MAX_NFEV,
        )
    except Exception:
        return _fail("exception")

    p_opt = result.x
    nfev  = int(result.nfev)

    height = p_opt[0]
    amp1, cr1, cc1, wr1, wc1, th1 = p_opt[1:7]
    amp2, cr2, cc2, wr2, wc2, th2 = p_opt[7:13]

    if nfev >= MAX_NFEV:
        return _fail("nfev")
    if amp1 <= 0 or amp2 <= 0:
        return _fail("amp")
    if height < 0:
        return _fail("height")

    # Validate centre displacement from each seed
    for cr, cc, seed_r, seed_c, dev_mult, label in [
        (cr1, cc1, r1, c1, 1.0, "drift_c1"),
        (cr2, cc2, r2, c2, 1.5, "drift_c2"),
    ]:
        dev = np.sqrt((cr - seed_r) ** 2 + (cc - seed_c) ** 2)
        if dev > fit_pixel_dev * dev_mult:
            return _fail(label)

    model = _gauss2d_2peaks(p_opt, grid[0], grid[1])
    chi2_r = _chi2_reduced(roi, model, n_params=13)

    params_c1 = np.array([height, amp1, cr1, cc1, wr1, wc1, th1])
    params_c2 = np.array([height, amp2, cr2, cc2, wr2, wc2, th2])
    return params_c1, params_c2, chi2_r, nfev, ""


# ---------------------------------------------------------------------------
# 12. PeakResult and conversion helper
# ---------------------------------------------------------------------------

@dataclass
class PeakResult:
    peak_X:           float   # col position, XMAS convention (1-based)
    peak_Y:           float   # row position, XMAS convention (1-based)
    peak_Itot:        float   # height + amplitude
    peak_Isub:        float   # amplitude (signal above baseline)
    peak_fwaxmaj:     float   # FWHM along major axis [pixels]
    peak_fwaxmin:     float   # FWHM along minor axis [pixels]
    peak_inclination: float   # rotation angle [degrees, 0-360]
    Xdev:             float   # fitted_col - seed_col
    Ydev:             float   # fitted_row - seed_row
    peak_bkg:         float   # baseline (height)
    Ipixmax:          float   # max pixel value in ROI
    n_gaussians:      int     # 1 or 2
    chi2_reduced:     float   # Poisson-normalised reduced chi²


def _params_to_result(
    params: np.ndarray,
    roi_row0: int,
    roi_col0: int,
    seed_row: int,
    seed_col: int,
    roi_orig: np.ndarray,
    n_gaussians: int,
    chi2_r: float,
) -> PeakResult:
    """Convert Gaussian fit parameters to a PeakResult in image coordinates."""
    height, amplitude, cen_row, cen_col, width_row, width_col, rota_deg = params

    # ROI → image coordinates (0-based), then +1 for XMAS convention
    img_row = roi_row0 + float(cen_row)
    img_col = roi_col0 + float(cen_col)

    fwhm_row = abs(float(width_row)) * FWHM_FACTOR
    fwhm_col = abs(float(width_col)) * FWHM_FACTOR

    return PeakResult(
        peak_X           = img_col + 1.0,
        peak_Y           = img_row + 1.0,
        peak_Itot        = float(height) + float(amplitude),
        peak_Isub        = float(amplitude),
        peak_fwaxmaj     = max(fwhm_row, fwhm_col),
        peak_fwaxmin     = min(fwhm_row, fwhm_col),
        peak_inclination = float(rota_deg) % 360.0,
        Xdev             = img_col - float(seed_col),
        Ydev             = img_row - float(seed_row),
        peak_bkg         = float(height),
        Ipixmax          = float(roi_orig.max()),
        n_gaussians      = n_gaussians,
        chi2_reduced     = chi2_r,
    )


# ---------------------------------------------------------------------------
# 13. Peak fitting driver
# ---------------------------------------------------------------------------

def _find_2g_seeds(
    roi: np.ndarray,
    p1g: np.ndarray,
    max_separation: float = np.inf,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return seed positions for a 2G fit, or None if a second seed cannot be found.

    Tries two strategies in order:

    A) Two distinct local maxima in the ROI (find_two_peaks_in_roi).
       The ROI is thresholded and labelled; if two well-separated connected
       components exist, their brightest pixels become the two seeds.

    B) Residual-based second seed.
       The 1G fitted centre is seed1.  The largest-residual pixel (at least 1 px
       away from seed1) is seed2.  If the raw separation is smaller than MIN_SEP,
       seed2 is extended along the same direction to reach MIN_SEP, giving the
       optimizer enough room to distinguish the two components.
    """
    two_peaks = find_two_peaks_in_roi(roi, max_separation=max_separation)
    if two_peaks is not None:
        return two_peaks

    # Strategy B: residual-based second seed.
    # The residual maximum gives the correct *direction* from the 1G centre toward
    # the second peak, even when the two peaks are only ~1 px apart.  If the raw
    # separation is too small for the optimizer to distinguish the two components,
    # we rescale along the same direction to a guaranteed minimum of MIN_SEP pixels.
    MIN_SEP = 2.0
    grid     = np.indices(roi.shape)
    model_1g = _gauss2d(p1g, grid[0], grid[1])
    resid    = np.abs(roi - model_1g)
    dist     = np.sqrt((grid[0] - float(p1g[2])) ** 2 + (grid[1] - float(p1g[3])) ** 2)
    resid[dist < 1.0] = 0.0
    if resid.max() > 0:
        pos2    = np.unravel_index(resid.argmax(), roi.shape)
        seed1_b = (float(p1g[2]), float(p1g[3]))
        seed2_b = (float(pos2[0]), float(pos2[1]))
        sep = np.sqrt((seed1_b[0] - seed2_b[0]) ** 2 + (seed1_b[1] - seed2_b[1]) ** 2)
        if sep > 0 and sep <= max_separation:
            if sep < MIN_SEP:
                # Extend seed2 along the same direction to reach MIN_SEP.
                scale   = MIN_SEP / sep
                dir_r   = (seed2_b[0] - seed1_b[0]) * scale
                dir_c   = (seed2_b[1] - seed1_b[1]) * scale
                seed2_b = (seed1_b[0] + dir_r, seed1_b[1] + dir_c)
            h, w = roi.shape
            if 0 <= seed2_b[0] < h and 0 <= seed2_b[1] < w:
                return seed1_b, seed2_b

    return None


def fit_peak(
    image: np.ndarray,
    seed_row: int,
    seed_col: int,
    cfg,
) -> List[PeakResult]:
    """Fit 1 or 2 Gaussians to a candidate peak.

    The fitting strategy is controlled by cfg.fitting_mode:
      '1G'   — always fit a single Gaussian; never attempt 2G.
      '2G'   — always attempt a 2G fit; reject the spot entirely if it fails
               or does not pass quality criteria.  No 1G fallback.
      'auto' — fit 1G first; upgrade to 2G when two local maxima are found in the
               ROI or when the 1G reduced chi² exceeds cfg.chi2_threshold.
               Falls back to the 1G result if 2G does not improve chi².

    Returns a list of PeakResult (1 item for 1G, 2 items for 2G, empty on failure).
    """
    roi, _, r0, c0 = extract_roi(
        image, seed_row, seed_col, cfg.boxsize, cfg.saturation
    )

    seed_r_roi = float(seed_row - r0)
    seed_c_roi = float(seed_col - c0)

    fitting_mode = getattr(cfg, "fitting_mode", "auto")

    # Always start with 1G to get a baseline fit and initial parameters
    fit1g = fit_1gaussian(
        roi, seed_r_roi, seed_c_roi,
        xtol            = cfg.fit_xtol,
        peak_size_range = (cfg.peak_size_min, cfg.peak_size_max),
        fit_pixel_dev   = cfg.fit_pixel_dev,
    )
    if fit1g is None:
        return []

    p1g, chi2_1g, _ = fit1g

    # ── '1G' mode: done ──────────────────────────────────────────────────────
    if fitting_mode == "1G":
        return [_params_to_result(p1g, r0, c0, seed_row, seed_col, roi, 1, chi2_1g)]

    # ── '2G' mode: always fit 2G, reject if quality too poor, never fallback ──
    if fitting_mode == "2G":
        seeds = _find_2g_seeds(roi, p1g, max_separation=getattr(cfg, "max_sep_2g", np.inf))
        if seeds is None:
            return []   # cannot find two seeds → reject spot entirely
        fit2g = fit_2gaussian(
            roi, seeds[0], seeds[1],
            xtol            = cfg.fit_xtol,
            peak_size_range = (cfg.peak_size_min, cfg.peak_size_max),
            fit_pixel_dev   = cfg.fit_pixel_dev,
            p1g_opt         = p1g,
        )
        pc1, pc2, chi2_2g, _, reject_reason = fit2g
        if pc1 is None:
            if getattr(cfg, "verbose", 0) >= 2:
                print(f"    [2G] seed ({seed_row},{seed_col}): fit2g rejected — {reject_reason}")
            return []   # fit failed → reject, no 1G fallback
        # Sort so that pc1 = higher amplitude (substrate), pc2 = lower (porous layer)
        if pc2[1] > pc1[1]:
            pc1, pc2 = pc2, pc1
        chi2_max = getattr(cfg, "chi2_max_2g", np.inf)
        if chi2_2g > chi2_max:
            return []   # quality too poor → reject
        # Reject if the two fitted centres are too close: this catches single
        # symmetric peaks that the optimizer splits into two coincident Gaussians.
        min_sep_2g = getattr(cfg, "min_sep_2g", 1.0)
        sep_fitted = np.sqrt((pc1[2] - pc2[2]) ** 2 + (pc1[3] - pc2[3]) ** 2)
        if sep_fitted < min_sep_2g:
            return []   # components too close → single peak, reject in 2G mode
        # Reject if the secondary component is too weak relative to the primary.
        # A genuine porous-layer spot should carry a physically significant fraction
        # of the substrate intensity.  When the second Gaussian merely absorbs
        # background noise its amplitude is negligible, producing a spurious pair.
        amp_ratio_min = getattr(cfg, "amp_ratio_min", 0.15)
        if pc2[1] / pc1[1] < amp_ratio_min:
            return []   # secondary component too weak → spurious fit, reject
        # Reject if the secondary component collapsed to a sub-pixel spike along
        # any axis.  This happens when the optimizer reaches the lower width bound
        # while fitting a slight asymmetry of a single-peak spot.  Both axes of a
        # physically real porous-layer spot must be at least fwhm_min_2g pixels wide.
        fwhm_min_2g = getattr(cfg, "fwhm_min_2g", 0.5)
        if min(pc2[4], pc2[5]) * FWHM_FACTOR < fwhm_min_2g:
            return []   # secondary component degenerate spike → reject
        r1 = _params_to_result(pc1, r0, c0, seed_row, seed_col, roi, 2, chi2_2g)
        r2 = _params_to_result(pc2, r0, c0, seed_row, seed_col, roi, 2, chi2_2g)
        max_drift = float(cfg.boxsize)
        if (np.sqrt(r1.Xdev ** 2 + r1.Ydev ** 2) <= max_drift and
                np.sqrt(r2.Xdev ** 2 + r2.Ydev ** 2) <= max_drift):
            return [r1, r2]
        return []   # components drifted outside ROI → reject

    # ── 'auto' mode ──────────────────────────────────────────────────────────
    _max_sep = getattr(cfg, "max_sep_2g", np.inf)
    # Trigger A: two distinct local maxima
    seeds  = find_two_peaks_in_roi(roi, threshold_fraction=0.15, max_separation=_max_sep)
    try_2g = seeds is not None
    # Trigger B: poor 1G chi² (only if A didn't fire)
    if not try_2g and chi2_1g > cfg.chi2_threshold:
        seeds  = _find_2g_seeds(roi, p1g, max_separation=_max_sep)
        try_2g = seeds is not None

    if try_2g and seeds is not None:
        fit2g = fit_2gaussian(
            roi, seeds[0], seeds[1],
            xtol            = cfg.fit_xtol,
            peak_size_range = (cfg.peak_size_min, cfg.peak_size_max),
            fit_pixel_dev   = cfg.fit_pixel_dev,
            p1g_opt         = p1g,
        )
        pc1, pc2, chi2_2g, _, reject_reason = fit2g
        if pc1 is not None:
            # Sort so that pc1 = higher amplitude (substrate), pc2 = lower (porous layer)
            if pc2[1] > pc1[1]:
                pc1, pc2 = pc2, pc1
            amp_ratio_min = getattr(cfg, "amp_ratio_min", 0.15)
            fwhm_min_2g   = getattr(cfg, "fwhm_min_2g", 0.5)
            if (chi2_2g < chi2_1g and
                    pc2[1] / pc1[1] >= amp_ratio_min and
                    min(pc2[4], pc2[5]) * FWHM_FACTOR >= fwhm_min_2g):
                r1 = _params_to_result(pc1, r0, c0, seed_row, seed_col, roi, 2, chi2_2g)
                r2 = _params_to_result(pc2, r0, c0, seed_row, seed_col, roi, 2, chi2_2g)
                max_drift = float(cfg.boxsize)
                if (np.sqrt(r1.Xdev ** 2 + r1.Ydev ** 2) <= max_drift and
                        np.sqrt(r2.Xdev ** 2 + r2.Ydev ** 2) <= max_drift):
                    return [r1, r2]

    # Fall back to single-Gaussian result
    return [_params_to_result(p1g, r0, c0, seed_row, seed_col, roi, 1, chi2_1g)]


# ---------------------------------------------------------------------------
# 14. Post-fit proximity rejection
# ---------------------------------------------------------------------------

def _reject_close_peaks(
    peaks: List[PeakResult],
    min_dist: float,
) -> List[PeakResult]:
    """Remove peaks within min_dist pixels of a brighter peak.

    Peaks are sorted by amplitude (peak_Isub) descending.  For each peak,
    if it falls within min_dist of an already-kept brighter peak, it is
    discarded.  Exception: two 2G peaks are never rejected against each other,
    because the two components of an overlapping spot are intentionally close
    and must both survive.
    """
    if len(peaks) <= 1 or min_dist <= 0:
        return peaks

    sorted_peaks = sorted(peaks, key=lambda p: p.peak_Isub, reverse=True)
    kept: List[PeakResult] = []
    for pk in sorted_peaks:
        too_close = False
        for kp in kept:
            # Do not reject two components of the same 2G fit against each other
            if pk.n_gaussians == 2 and kp.n_gaussians == 2:
                continue
            d = np.sqrt((pk.peak_X - kp.peak_X) ** 2 + (pk.peak_Y - kp.peak_Y) ** 2)
            if d < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append(pk)
    return kept


# ---------------------------------------------------------------------------
# 15. Output
# ---------------------------------------------------------------------------

_DAT_HEADER = (
    "peak_X  peak_Y  peak_Itot  peak_Isub  peak_fwaxmaj  peak_fwaxmin  "
    "peak_inclination  Xdev  Ydev  peak_bkg  Ipixmax  n_gaussians  chi2_reduced"
)


def write_dat(peaks: List[PeakResult], filepath: str) -> None:
    """Write peak list to a space-delimited .dat file.

    The header line contains column names without a leading '#', matching the
    format written by LaueTools writefile_Peaklist() so that getcolumnsname_dat()
    and getspotsproperties_dat() parse it correctly.
    The first 11 columns are identical to LaueTools output.
    Two extra columns (n_gaussians, chi2_reduced) are appended.
    For 2-Gaussian fits, two rows are written — one per Gaussian component.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w') as fh:
        fh.write(f"{_DAT_HEADER}\n")
        for pk in peaks:
            fh.write(
                f"{pk.peak_X:.2f}   {pk.peak_Y:.2f}   {pk.peak_Itot:.2f}   "
                f"{pk.peak_Isub:.2f}   {pk.peak_fwaxmaj:.2f}   {pk.peak_fwaxmin:.2f}   "
                f"{pk.peak_inclination:.3f}   {pk.Xdev:.2f}   {pk.Ydev:.2f}   "
                f"{pk.peak_bkg:.2f}   {pk.Ipixmax:.0f}   {pk.n_gaussians:d}   "
                f"{pk.chi2_reduced:.4f}\n"
            )


# ---------------------------------------------------------------------------
# 16. Per-image worker (top-level so it is picklable)
# ---------------------------------------------------------------------------

# Module-level config set by pool initializer (avoids per-task pickle overhead)
_worker_cfg = None


def _worker_init(cfg) -> None:
    global _worker_cfg
    _worker_cfg = cfg


def process_image(img_idx: int) -> Tuple[int, int, float]:
    """Process one image: load → background → maxima → fit → write .dat.

    Returns (img_idx, n_peaks_written, elapsed_seconds).
    """
    cfg = _worker_cfg
    t0 = time.time()

    img_path = os.path.join(cfg.exp_folder, f"img_{img_idx:0>4}.tif")
    if not os.path.isfile(img_path):
        return img_idx, 0, 0.0

    try:
        frame = fabio.open(img_path)
        image = frame.data.astype(float)
    except Exception as exc:
        print(f"  [WARNING] Could not read {img_path}: {exc}")
        return img_idx, 0, 0.0

    # Background and subtracted image for maxima search
    bg     = auto_background(image, boxsize=cfg.bg_boxsize)
    bg_sub = image - bg

    # Candidate positions
    sim_xy = getattr(cfg, "sim_xy", None)
    if sim_xy is not None:
        # Simulation-guided search: look only near simulated spot positions
        candidates = find_local_maxima_sim_guided(
            bg_sub,
            sim_xy    = sim_xy,
            tolerance = cfg.sim_tolerance,
            threshold = cfg.intensity_thresh,
        )
    else:
        # Full-image threshold search
        candidates = find_local_maxima(
            bg_sub,
            threshold         = cfg.intensity_thresh,
            pixel_near_radius = cfg.pixel_near_radius,
            max_candidates    = cfg.max_peaks_per_image,
        )
    candidates = filter_blacklisted(candidates, cfg.blacklist_xy, radius=5.0)

    # Fit each candidate.
    # pairs_primary / pairs_secondary track 2G components explicitly so that the
    # substrate→primary and porous→secondary assignment is never broken by the
    # post-fit reordering performed by _reject_close_peaks.
    all_peaks:       List[PeakResult] = []
    pairs_primary:   List[PeakResult] = []
    pairs_secondary: List[PeakResult] = []

    for row, col in candidates:
        results = fit_peak(image, int(row), int(col), cfg)
        all_peaks.extend(results)
        if len(results) == 2:
            # fit_peak returns [r1, r2] where r1=primary (substrate, higher amplitude)
            # and r2=secondary (porous layer, lower amplitude) — guaranteed by the
            # amplitude sort inside fit_peak.
            pairs_primary.append(results[0])
            pairs_secondary.append(results[1])

    # Post-fit proximity rejection on the full flat list (used for the combined .dat)
    all_peaks = _reject_close_peaks(all_peaks, cfg.max_pixel_dist_rejection)

    if getattr(cfg, "verbose", 0) >= 1:
        n_2g = len(pairs_primary)
        n_1g = sum(1 for p in all_peaks if p.n_gaussians == 1)
        print(f"  img_{img_idx:04d}: {len(candidates)} candidates → "
              f"{n_1g} 1G + {n_2g} 2G pairs = {len(all_peaks)} peaks")

    # Write output .dat
    # In 2G mode with a secondary output folder defined, write:
    #   out_folder   → primary component (substrate, higher amplitude)
    #   out_folder_2 → secondary component (porous layer, lower amplitude)
    # Pairs are sourced from pairs_primary/pairs_secondary collected above, which
    # preserves the correct per-fit assignment regardless of _reject_close_peaks
    # reordering.  1G peaks go only to out_folder.
    out_folder_2 = getattr(cfg, "out_folder_2", None)
    if out_folder_2 and getattr(cfg, "fitting_mode", "auto") == "2G":
        peaks_1g   = [p for p in all_peaks if p.n_gaussians == 1]
        out_path   = os.path.join(cfg.out_folder, f"img_{img_idx:0>4}.dat")
        out_path_2 = os.path.join(out_folder_2,   f"img_{img_idx:0>4}.dat")
        write_dat(peaks_1g + pairs_primary, out_path)
        write_dat(pairs_secondary,          out_path_2)
    else:
        out_path = os.path.join(cfg.out_folder, f"img_{img_idx:0>4}.dat")
        write_dat(all_peaks, out_path)

    return img_idx, len(all_peaks), time.time() - t0


# ---------------------------------------------------------------------------
# 17. Batch runner
# ---------------------------------------------------------------------------

def run_batch(cfg) -> None:
    """Run peak search over [first_img, last_img] using multiprocessing."""
    os.makedirs(cfg.out_folder, exist_ok=True)
    out_folder_2 = getattr(cfg, "out_folder_2", None)
    if out_folder_2:
        os.makedirs(out_folder_2, exist_ok=True)

    img_indices = list(range(cfg.first_img, cfg.last_img + 1))
    n_images = len(img_indices)

    print(f"Peak search: {n_images} images | {cfg.n_cpus} workers")
    print(f"  Input : {cfg.exp_folder}")
    print(f"  Output: {cfg.out_folder}")
    fitting_mode = getattr(cfg, "fitting_mode", "auto")
    sim_xy       = getattr(cfg, "sim_xy", None)
    search_mode  = (f"sim-guided (tol={cfg.sim_tolerance} px)" if sim_xy is not None
                    else "full-image threshold")
    print(f"  Search : {search_mode}  |  fitting={fitting_mode}")
    print(f"  Params : thresh={cfg.intensity_thresh}  boxsize={cfg.boxsize}  "
          f"xtol={cfg.fit_xtol}  chi2_thr={cfg.chi2_threshold}")

    t_start = time.time()
    results: List[Tuple[int, int, float]] = []

    pool = multiprocessing.Pool(
        processes        = cfg.n_cpus,
        initializer      = _worker_init,
        initargs         = (cfg,),
        maxtasksperchild = 10,
    )
    try:
        imap = pool.imap_unordered(process_image, img_indices, chunksize=1)
        if _HAS_TQDM:
            for res in tqdm(imap, total=n_images, desc="Peak search"):
                results.append(res)
        else:
            for i, res in enumerate(imap):
                results.append(res)
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{n_images} images …")
        pool.close()
        pool.join()
    except Exception:
        pool.terminate()
        raise

    elapsed = time.time() - t_start
    total_peaks = sum(r[1] for r in results)
    n_ok        = sum(1 for r in results if r[1] >= 0)
    print(f"\nDone: {n_ok}/{n_images} images, {total_peaks} peaks total in {elapsed:.1f} s  "
          f"({elapsed / max(n_ok, 1):.2f} s/image)")


# ---------------------------------------------------------------------------
# 18. CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # I/O
    p.add_argument("--exp-folder",     required=True,
                   help="Directory containing img_XXXX.tif files")
    p.add_argument("--out-folder",     default="all_peaks",
                   help="Output directory for .dat files (default: all_peaks/). "
                        "In 2G mode, receives the primary component (substrate, higher amplitude).")
    p.add_argument("--out-folder-2",   default=None,
                   help="(2G mode only) output directory for the secondary component .dat files "
                        "(porous layer, lower amplitude). If omitted, all peaks are written to "
                        "--out-folder. When set, creates two parallel folders with identical "
                        "filenames so that indexing tools can process each layer independently.")
    p.add_argument("--first-img",      type=int, default=0,
                   help="First image index (default: 0)")
    p.add_argument("--last-img",       type=int, required=True,
                   help="Last image index (inclusive)")
    p.add_argument("--n-cpus",         type=int, default=os.cpu_count() or 1,
                   help="Number of parallel workers (default: all CPUs)")
    # Peak search
    p.add_argument("--intensity-thresh",  type=float, default=200.0,
                   help="Background-subtracted intensity threshold (default: 200)")
    p.add_argument("--boxsize",           type=int,   default=5,
                   help="Half-box size for ROI extraction and fitting (default: 5)")
    p.add_argument("--bg-boxsize",        type=int,   default=10,
                   help="Half-box size for minimum-filter background (default: 10)")
    p.add_argument("--pixel-near-radius", type=int,   default=2,
                   help="Merge maxima within this radius during search (default: 2)")
    p.add_argument("--max-peaks-per-image", type=int, default=5000,
                   help="Cap on candidates per image (default: 5000)")
    # Fit quality
    p.add_argument("--fit-xtol",          type=float, default=0.5,
                   help="Convergence tolerance for leastsq (default: 0.5)")
    p.add_argument("--peak-size-min",     type=float, default=0.1,
                   help="Minimum peak FWHM in pixels (default: 0.1)")
    p.add_argument("--peak-size-max",     type=float, default=6.0,
                   help="Maximum peak FWHM in pixels (default: 6.0)")
    p.add_argument("--fit-pixel-dev",     type=float, default=3.0,
                   help="Max allowed fitted-centre to seed displacement (default: 3.0)")
    p.add_argument("--max-pixel-dist-rejection", type=float, default=3.0,
                   help="Reject peaks closer than this to a brighter one (default: 3.0)")
    p.add_argument("--saturation",        type=float, default=65000.0,
                   help="Pixel saturation value in counts (default: 65000)")
    # Fitting mode
    p.add_argument("--fitting-mode",      choices=["1G", "2G", "auto"], default="auto",
                   help="Gaussian fitting strategy: '1G' always fits one Gaussian, "
                        "'2G' always fits two, 'auto' decides per peak (default: auto)")
    p.add_argument("--chi2-threshold",    type=float, default=2.0,
                   help="(auto mode) reduced chi² above which 2G fit is attempted (default: 2.0)")
    p.add_argument("--chi2-max-2g",      type=float, default=float("inf"),
                   help="(2G mode) reject spot if 2G reduced chi² exceeds this value (default: no limit)")
    p.add_argument("--min-sep-2g",       type=float, default=1.0,
                   help="(2G mode) reject pair if fitted centre separation < this value in pixels (default: 1.0)")
    p.add_argument("--max-sep-2g",       type=float, default=float("inf"),
                   help="(2G mode) reject pair if seed separation > this value in pixels during seed search "
                        "(default: no limit). Use to prevent the second seed from landing on a distant unrelated spot.")
    p.add_argument("--amp-ratio-min",    type=float, default=0.15,
                   help="(2G mode) reject pair if amplitude of secondary component (porous layer) is less than "
                        "this fraction of primary component amplitude (substrate). Prevents accepting spurious "
                        "fits where the second Gaussian merely absorbs background noise. (default: 0.15)")
    p.add_argument("--fwhm-min-2g",      type=float, default=0.5,
                   help="(2G mode) reject pair if the smallest FWHM axis of the secondary component is below "
                        "this value in pixels. Catches degenerate fits where the optimizer collapses one "
                        "Gaussian to a sub-pixel spike while absorbing a slight asymmetry of a single peak. "
                        "(default: 0.5)")
    # Simulation-guided search
    p.add_argument("--material",          default=None,
                   help="Material name in LaueTools dict_Materials (e.g. 'GaN', 'Al2O3'). "
                        "If given together with --ub and --det-file, enables simulation-guided search.")
    p.add_argument("--ub",                nargs="+", default=None,
                   help="UB / orientation matrix: either a path to a .npy file (3×3) "
                        "or 9 space-separated floats in row-major order.")
    p.add_argument("--det-file",          default=None,
                   help="Path to LaueTools .det calibration file (required for simulation).")
    p.add_argument("--emin",              type=float, default=5.0,
                   help="Minimum X-ray energy for simulation [keV] (default: 5.0)")
    p.add_argument("--emax",              type=float, default=27.0,
                   help="Maximum X-ray energy for simulation [keV] (default: 27.0)")
    p.add_argument("--camera-label",      default="sCMOS",
                   help="Camera label in LaueTools dict_CCD (default: 'sCMOS')")
    p.add_argument("--sim-tolerance",     type=float, default=15.0,
                   help="Search radius around each simulated spot [pixels] (default: 15)")
    # Blacklist
    p.add_argument("--blacklist",         default=None,
                   help="Path to blacklist file with (X Y) positions to ignore")
    # Verbosity
    p.add_argument("--verbose",           type=int, default=0,
                   help="Verbosity level: 0 = summary only (default), "
                        "1 = per-image rejection counts, "
                        "2 = per-spot rejection reasons (slow, use on single images)")
    return p


def main() -> None:
    cfg = _build_parser().parse_args()

    # ── Blacklist ─────────────────────────────────────────────────────────────
    cfg.blacklist_xy = load_blacklist(cfg.blacklist)
    if cfg.blacklist_xy is not None:
        print(f"Blacklist: {len(cfg.blacklist_xy)} positions loaded from {cfg.blacklist}")

    # ── Simulation-guided search setup ────────────────────────────────────────
    cfg.sim_xy = None
    if cfg.material is not None:
        missing = [arg for arg, val in [("--ub", cfg.ub), ("--det-file", cfg.det_file)] if val is None]
        if missing:
            raise SystemExit(
                f"Simulation requires {', '.join(missing)} when --material is given."
            )
        # Parse UB matrix
        ub_arg = cfg.ub
        if len(ub_arg) == 1 and os.path.isfile(ub_arg[0]):
            ub_matrix = np.load(ub_arg[0])
        else:
            try:
                vals = [float(v) for v in ub_arg]
            except ValueError:
                raise SystemExit("--ub must be 9 floats or a path to a .npy file.")
            if len(vals) != 9:
                raise SystemExit(f"--ub expects 9 values for a 3×3 matrix, got {len(vals)}.")
            ub_matrix = np.array(vals).reshape(3, 3)

        print(f"Simulation: material={cfg.material}  E={cfg.emin}–{cfg.emax} keV  "
              f"tolerance={cfg.sim_tolerance} px")
        print(f"  UB matrix:\n{ub_matrix}")

        calib_params = load_calibration(cfg.det_file)
        sim_xy = simulate_peaks(
            material               = cfg.material,
            ub_matrix              = ub_matrix,
            calibration_parameters = calib_params,
            emin                   = cfg.emin,
            emax                   = cfg.emax,
            camera_label           = cfg.camera_label,
        )
        cfg.sim_xy = sim_xy
        print(f"  Simulated spots on detector: {len(sim_xy)}")

    # ── Run ───────────────────────────────────────────────────────────────────
    if cfg.first_img == cfg.last_img:
        _worker_init(cfg)
        idx, n_peaks, elapsed = process_image(cfg.first_img)
        print(f"img_{idx:0>4}: {n_peaks} peaks in {elapsed:.2f} s")
    else:
        run_batch(cfg)


if __name__ == "__main__":
    main()
