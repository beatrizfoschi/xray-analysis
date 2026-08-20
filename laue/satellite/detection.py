"""Satellite peak detection in Laue spot images from MQW structures.

A periodic MQW stack with period Λ = t_QW + t_barrier adds extra reciprocal
lattice points spaced by 2π/Λ along the growth direction, flanking the main
Bragg peak. In a Laue spot image these appear as a discrete series of satellite
spots (SL₋₃, …, SL₀, …, SL₊₃) aligned along the growth axis projection on
the detector.

Public API
----------
detect_satellites(image, ...)      Main entry point — returns all peaks + metadata
subtract_background(image, sigma)  FFT-Gaussian background removal
find_sl0_centroid(image)           Locate the zero-order peak
find_satellite_axis(image, center) PCA-based axis angle estimation
extract_1d_profile(image, ...)     Project image onto satellite axis
fit_gaussian_1d(...)               Refine a single peak position / width
locate_sl0_by_local_max(...)       Find SL0 near its predicted position
make_synthetic_image(...)          Generate a test image with known satellites
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from typing import Any, Dict, List, Optional, Tuple

from utils.fitting import gaussian


# ── Hot pixel removal ─────────────────────────────────────────────────────────

def clip_hot_pixels(
    image: np.ndarray,
    filter_size: int = 3,
    n_sigma: float = 10.0,
) -> np.ndarray:
    """Replace isolated bright pixels (hot pixels, cosmic rays) with local median.

    A pixel is classified as "hot" if it exceeds the local median by more than
    n_sigma times the global noise estimate (MAD-based).  Operates on a copy;
    does not modify the input.
    """
    img = image.astype(np.float32)
    local_med = median_filter(img, size=filter_size).astype(np.float32)
    diff = img - local_med
    noise = float(np.median(np.abs(diff))) * 1.4826   # MAD → σ
    if noise == 0.0:
        return img
    hot = diff > n_sigma * noise
    out = img.copy()
    out[hot] = local_med[hot]
    return out


# ── Background subtraction ────────────────────────────────────────────────────

def subtract_background(image: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """Remove smooth background via Gaussian blurring.

    sigma <= 0 → no subtraction (returns the image as-is).  Use this when the
    raw image already has a flat background or when bg_sigma would be so large
    it has no effect.
    """
    img = image.astype(np.float32)
    if sigma <= 0.0:
        return img
    bg = gaussian_filter(img, sigma=sigma)
    return np.maximum(img - bg, 0.0)


# ── SL₀ localisation ─────────────────────────────────────────────────────────

def find_sl0_centroid(
    image: np.ndarray,
    top_fraction: float = 0.02,
) -> Tuple[float, float]:
    """Return (row, col) centroid of the brightest `top_fraction` of pixels.

    The zero-order peak (SL₀) is the brightest and most compact feature in a
    background-subtracted spot image, so its centroid is a robust seed for the
    satellite axis search.
    """
    threshold = float(np.quantile(image, 1.0 - top_fraction))
    rows, cols = np.where(image >= threshold)
    weights = image[rows, cols].astype(np.float64) if len(rows) > 0 else np.array([])
    total = float(weights.sum())
    if total == 0.0 or len(rows) == 0:
        return float(image.shape[0] / 2), float(image.shape[1] / 2)
    cy = float((weights * rows).sum() / total)
    cx = float((weights * cols).sum() / total)
    return cy, cx


# ── Axis detection ────────────────────────────────────────────────────────────

def _canonical_axis_angle(angle_deg: float) -> float:
    """Fold an axis-line angle into (-90, 90].

    A satellite axis is a line, not a direction: angle and angle+180 describe
    the same line, but np.linalg.eigh returns an eigenvector with arbitrary
    sign, so the raw angle flips between the two equivalent representations
    from one call to the next (e.g. frame to frame in a scan). That flip
    silently swaps which side of SL0 gets positive vs negative order labels.
    Folding to a single canonical half-plane makes "positive along-axis
    distance" a fixed, reproducible convention.
    """
    angle = angle_deg % 180.0
    if angle > 90.0:
        angle -= 180.0
    return angle


def find_satellite_axis(
    image: np.ndarray,
    center: Optional[Tuple[float, float]] = None,
) -> float:
    """Return satellite axis angle (degrees from +x) via intensity-weighted PCA.

    The principal axis of the intensity distribution corresponds to the [0001]
    projection on the detector, which is the direction along which satellites
    are lined up.  If `center` is supplied the moments are computed about it;
    otherwise the intensity-weighted centroid is used.
    """
    img = np.maximum(image, 0.0).astype(np.float64)
    rows, cols = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    w = img.ravel()
    total = w.sum()
    if total == 0.0:
        return 0.0

    if center is None:
        cy = float((w * rows.ravel()).sum() / total)
        cx = float((w * cols.ravel()).sum() / total)
    else:
        cy, cx = center

    dy = (rows - cy).ravel()
    dx = (cols - cx).ravel()
    cyy = float((w * dy * dy).sum() / total)
    cyx = float((w * dy * dx).sum() / total)
    cxx = float((w * dx * dx).sum() / total)

    # eigh returns eigenvalues in ascending order; index 1 = largest eigenvalue
    _, eigvecs = np.linalg.eigh(np.array([[cyy, cyx], [cyx, cxx]]))
    principal = eigvecs[:, 1]   # [dy-component, dx-component]
    angle = float(np.degrees(np.arctan2(principal[0], principal[1])))
    return _canonical_axis_angle(angle)


# ── 1-D profile extraction ────────────────────────────────────────────────────

def extract_1d_profile(
    image: np.ndarray,
    center: Tuple[float, float],
    axis_angle: float,
    half_length: Optional[float] = None,
    strip_width: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the 1-D intensity profile along the satellite axis.

    Pixel intensities within a strip of `strip_width` centred on the axis are
    summed into bins of 1 px width, using np.histogram for speed.

    Parameters
    ----------
    image       : 2-D array
    center      : (row, col) of SL₀
    axis_angle  : degrees from +x axis
    half_length : half-length of the profile (px). Defaults to 60 % of min(shape).
    strip_width : strip width (px) perpendicular to axis

    Returns
    -------
    distances   : signed pixel distances from SL₀ along the axis
    intensities : summed counts per bin
    """
    if half_length is None:
        # Use the full diagonal so the profile always reaches every pixel,
        # regardless of where SL₀ sits within the crop.
        half_length = float(np.hypot(image.shape[0], image.shape[1]))

    rows, cols = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    dy = (rows - center[0]).ravel()
    dx = (cols - center[1]).ravel()
    rad = float(np.radians(axis_angle))

    s = dx * np.cos(rad) + dy * np.sin(rad)   # along-axis coordinate
    t = -dx * np.sin(rad) + dy * np.cos(rad)  # cross-axis coordinate

    img_flat = image.ravel().astype(np.float64)
    in_strip = (np.abs(t) <= strip_width / 2.0) & (np.abs(s) <= half_length)

    n_bins = max(int(2 * half_length), 50)
    bin_edges = np.linspace(-half_length, half_length, n_bins + 1)
    distances = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    intensities, _ = np.histogram(s[in_strip], bins=bin_edges, weights=img_flat[in_strip])

    return distances, intensities.astype(np.float64)


# ── Gaussian fitting ──────────────────────────────────────────────────────────

def fit_gaussian_1d(
    distances: np.ndarray,
    intensities: np.ndarray,
    peak_pos: float,
    search_window: float = 12.0,
) -> Optional[Dict[str, Any]]:
    """Fit a 1-D Gaussian to the profile near `peak_pos`.

    Returns a dict with keys pos, amplitude, sigma, fwhm, background, success;
    or None if fewer than 5 points are available or the fit fails.
    """
    mask = np.abs(distances - peak_pos) <= search_window
    if mask.sum() < 5:
        return None
    x, y = distances[mask], intensities[mask]
    try:
        popt, _ = curve_fit(
            gaussian, x, y,
            p0=[float(y.max()), float(peak_pos), 3.0, float(y.min())],
            bounds=(
                [0.0,  peak_pos - search_window, 0.5, 0.0],
                [np.inf, peak_pos + search_window, search_window, np.inf],
            ),
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return None
    A, mu, sigma, bkg = popt
    return {
        'pos': float(mu),
        'amplitude': float(A),
        'sigma': float(abs(sigma)),
        'fwhm': float(2.3548 * abs(sigma)),
        'background': float(bkg),
        'success': True,
    }


def locate_sl0_by_local_max(
    image: np.ndarray,
    sl0_pos_along_axis: float,
    axis_angle: float,
    sl0_center: Tuple[float, float],
    boxsize: float = 3.0,
) -> Dict[str, Any]:
    """Locate SL0 as a local maximum in a small box around its predicted position.

    SL0 is buried in the flank of the much brighter bulk peak, so it is a
    handful of pixels standing out from a steep gradient rather than a
    resolvable peak with a width of its own.  The test is correspondingly
    weak: exactly one pixel in the box strictly brighter than all 8 of its
    neighbours.  No amplitude, width, or line shape is assumed.

    Parameters
    ----------
    image               : the raw crop.  NOT the background-subtracted image
                          and NOT a hot-pixel-clipped one: `bg_sigma`'s wide
                          Gaussian lets the bulk's tail inflate the background
                          estimate here, and `clip_hot_pixels` mistakes a
                          narrow SL0 for a cosmic ray and replaces it with its
                          own 3x3 median.  Either one erases the feature.
    sl0_pos_along_axis  : predicted position, from `locate_sl0_from_ladder`.
    axis_angle          : satellite axis in degrees, as used to place every
                          other peak's `position_2d` (see `_make_peak`).
    sl0_center          : origin for that same projection.
    boxsize             : half-width (px) of the search box.  Keep it well
                          inside the gap to the nearest satellite and to the
                          bulk peak, or either supplies a second local maximum
                          and the box is rejected as ambiguous.

    Returns
    -------
    dict with `sl0_confirmed` (bool).  When True: `sl0_measured_position_2d`
    (row, col), `sl0_measured_pos` (along the axis, same units as every other
    peak's `pos_along_axis`) and `sl0_measured_amplitude`.  When False,
    `reason`, plus `candidates` when the box was ambiguous.  Finding nothing
    is a valid outcome, not an error: SL0 is not resolvable everywhere.
    """
    rad = np.radians(axis_angle)
    row_f = sl0_center[0] + sl0_pos_along_axis * np.sin(rad)
    col_f = sl0_center[1] + sl0_pos_along_axis * np.cos(rad)

    # round(), not floor()/ceil(): the latter pair rounds outward on both
    # sides and by different amounts (a fractional centre made boxsize=4 search
    # 10x10 px instead of 9x9, pulling in noise maxima the user never asked to
    # include).  round() keeps the box symmetric about the prediction.
    row_lo = max(int(round(row_f - boxsize)), 1)
    row_hi = min(int(round(row_f + boxsize)) + 1, image.shape[0] - 1)
    col_lo = max(int(round(col_f - boxsize)), 1)
    col_hi = min(int(round(col_f + boxsize)) + 1, image.shape[1] - 1)
    if row_hi <= row_lo or col_hi <= col_lo:
        return {'sl0_confirmed': False, 'reason': 'search box falls outside the image'}

    candidate = image[row_lo:row_hi, col_lo:col_hi]
    is_max = np.ones_like(candidate, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbour = image[row_lo + dr:row_hi + dr, col_lo + dc:col_hi + dc]
            is_max &= candidate > neighbour

    rows, cols = np.nonzero(is_max)
    if len(rows) != 1:
        candidates = [(row_lo + int(r), col_lo + int(c),
                      float(image[row_lo + int(r), col_lo + int(c)]))
                     for r, c in zip(rows, cols)]
        return {'sl0_confirmed': False,
                'reason': f'{len(rows)} local maxima in the box (need exactly 1)',
                'candidates': candidates}

    r, c = row_lo + int(rows[0]), col_lo + int(cols[0])
    pos_along_axis = (r - sl0_center[0]) * np.sin(rad) + (c - sl0_center[1]) * np.cos(rad)
    return {
        'sl0_confirmed': True,
        'sl0_measured_position_2d': (r, c),
        'sl0_measured_pos': float(pos_along_axis),
        'sl0_measured_amplitude': float(image[r, c]),
    }


# ── Order assignment ──────────────────────────────────────────────────────────

def _assign_orders(
    peak_positions: np.ndarray,
    sl0_pos: float,
    min_spacing_px: float = 4.0,
) -> List[int]:
    """Assign integer satellite order indices relative to SL₀.

    Orders are assigned by rank: the closest non-SL0 peak on each side gets
    order ±1, the next ±2, etc.  This is robust to non-uniform inter-satellite
    spacing, which can occur when the projected axis is slightly misaligned or
    the true superlattice spacing differs from the SL0→SL1 distance.
    """
    diffs = peak_positions - sl0_pos
    orders = np.zeros(len(peak_positions), dtype=int)

    neg_mask = diffs < -min_spacing_px
    if neg_mask.any():
        neg_idx = np.where(neg_mask)[0]
        for rank, idx in enumerate(neg_idx[np.argsort(np.abs(diffs[neg_mask]))]):
            orders[idx] = -(rank + 1)

    pos_mask = diffs > min_spacing_px
    if pos_mask.any():
        pos_idx = np.where(pos_mask)[0]
        for rank, idx in enumerate(pos_idx[np.argsort(np.abs(diffs[pos_mask]))]):
            orders[idx] = rank + 1

    return orders.tolist()


# ── Main detection helpers ────────────────────────────────────────────────────

def _make_peak(
    order: int,
    fit: Dict[str, Any],
    sl0_center: Tuple[float, float],
    rad: float,
) -> Dict[str, Any]:
    return {
        'order': int(order),
        'pos_along_axis': fit['pos'],
        'position_2d': (
            float(sl0_center[0] + fit['pos'] * np.sin(rad)),
            float(sl0_center[1] + fit['pos'] * np.cos(rad)),
        ),
        'amplitude': fit['amplitude'],
        'fwhm': fit['fwhm'],
        'sigma': fit['sigma'],
        'fit_success': fit['success'],
    }


def _fill_missing_orders(
    detected: List[Dict[str, Any]],
    distances: np.ndarray,
    intensities: np.ndarray,
    sl0_center: Tuple[float, float],
    rad: float,
    n_min_eff: int,
    n_max_eff: int,
    sl0_pos: float,
    spacing_est: float,
    search_win: float,
    adaptive_fill_win: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Targeted Gaussian fit for satellite orders missing from the first pass.

    Strategy: find ALL peaks in the log profile with a very low prominence
    threshold, remove those already assigned to a detected order, then for
    each missing order try the remaining candidates ordered by log-intensity
    (strongest first) on the correct side of SL0.  A linear-fit extrapolation
    is used as a fallback when no candidate passes the amplitude floor.

    Accepted only if fitted amplitude ≥ 5 % of the weakest confirmed non-SL0
    satellite (or 1 % of SL0), preventing curve_fit from inventing noise peaks.
    """
    detected_orders = {pk['order'] for pk in detected}
    result = list(detected)

    missing = [n for n in range(n_min_eff, n_max_eff + 1)
               if n not in detected_orders]
    if not missing:
        if verbose:
            print('[FILL] No missing orders — skip.')
        return result

    non_sl0 = [pk for pk in detected if pk['order'] != 0]
    sl0_list = [pk for pk in detected if pk['order'] == 0]
    if non_sl0:
        min_amp = 0.05 * min(pk['amplitude'] for pk in non_sl0)
    elif sl0_list:
        min_amp = 0.01 * sl0_list[0]['amplitude']
    else:
        min_amp = 1.0

    # Linear-fit spacing (used as fallback / rough expected position).
    ref_sl0_pos = sl0_list[0]['pos_along_axis'] if sl0_list else sl0_pos
    if len(non_sl0) >= 2:
        det_orders = np.array([pk['order'] for pk in non_sl0], dtype=float)
        det_pos = np.array([pk['pos_along_axis'] for pk in non_sl0])
        fill_spacing = float(np.polyfit(det_orders, det_pos, 1)[0])
    elif len(non_sl0) == 1:
        pk0 = non_sl0[0]
        fill_spacing = pk0['pos_along_axis'] / float(pk0['order'])
    else:
        fill_spacing = spacing_est

    if adaptive_fill_win and abs(fill_spacing) > 0.5:
        win = max(abs(fill_spacing) * 0.45, 2.0)
    else:
        win = search_win

    if verbose:
        print(f'[FILL] Missing orders: {missing}')
        print(f'[FILL] ref_sl0_pos={ref_sl0_pos:.2f}  fill_spacing={fill_spacing:.2f}  '
              f'search_win={win:.2f}  min_amp={min_amp:.2f}')

    # All log-profile peaks with very low prominence — catches weak satellites
    # that the main find_peaks pass misses.
    log_prof = np.log1p(intensities)
    log_max = float(log_prof.max()) or 1.0
    cand_idx, _ = find_peaks(log_prof / log_max, prominence=0.005, width=1.5)

    # Remove candidates already claimed by a detected peak.
    assigned_pos = [pk['pos_along_axis'] for pk in detected]
    free_cands: List[tuple] = []
    for idx in cand_idx:
        pos = float(distances[idx])
        if all(abs(pos - ap) > win * 0.5 for ap in assigned_pos):
            free_cands.append((float(log_prof[idx]), pos))
    if verbose:
        print(f'[FILL] Free candidates (log_val, pos): '
              + ', '.join(f'({lv:.3f}, {p:.2f})' for lv, p in free_cands))

    used_pos: List[float] = []

    for n in sorted(missing, key=abs):   # fill closest orders first
        expected_pos = ref_sl0_pos + n * fill_spacing

        # Minimum distance from SL0: candidate for order n must be farther
        # than the adjacent inner satellite (order n+sign(n)), if detected.
        # Falls back to (|n|-0.5)*|fill_spacing| when inner order is absent.
        inner_order = n + int(np.sign(n))
        inner_peak = next(
            (pk for pk in detected if pk['order'] == inner_order), None
        )
        if inner_peak is not None:
            min_dist_sl0 = abs(inner_peak['pos_along_axis'] - ref_sl0_pos)
        else:
            min_dist_sl0 = (abs(n) - 0.5) * abs(fill_spacing)

        # Candidates on the correct side of SL0, not yet consumed,
        # and farther from SL0 than the inner satellite.
        candidates = [
            (lv, pos) for lv, pos in free_cands
            if (n == 0 or np.sign(pos - ref_sl0_pos) == np.sign(n))
            and all(abs(pos - up) > win * 0.5 for up in used_pos)
            and abs(pos - ref_sl0_pos) > min_dist_sl0
        ]
        # Sort: closest to linearly extrapolated position first, then by strength.
        candidates.sort(key=lambda x: (abs(x[1] - expected_pos), -x[0]))

        if verbose:
            print(f'[FILL] n={n}  expected_pos={expected_pos:.2f}  '
                  f'candidates (sorted): '
                  + ', '.join(f'pos={p:.2f}(lv={lv:.3f})' for lv, p in candidates))

        accepted = False
        for _, guess_pos in candidates:
            fit = fit_gaussian_1d(distances, intensities, guess_pos,
                                  search_window=win)
            if verbose:
                if fit is None:
                    print(f'  guess={guess_pos:.2f} -> fit FAILED')
                else:
                    print(f'  guess={guess_pos:.2f} -> amp={fit["amplitude"]:.1f} '
                          f'(floor={min_amp:.1f}) fwhm={fit["fwhm"]:.2f} '
                          f'-> {"ACCEPT" if fit["amplitude"] >= min_amp else "REJECT (below floor)"}')
            if fit is None:
                continue
            if fit['amplitude'] < min_amp:
                continue
            result.append(_make_peak(n, fit, sl0_center, rad))
            used_pos.append(fit['pos'])
            accepted = True
            break

        if not accepted:
            if verbose:
                print(f'[FILL] n={n}: no candidate accepted — trying linear fallback at {expected_pos:.2f}')
            # Fallback: linear-fit extrapolation + log-argmax refinement.
            if expected_pos < float(distances[0]) or expected_pos > float(distances[-1]):
                if verbose:
                    print(f'[FILL] n={n}: fallback position out of range — skip')
                continue
            local_mask = np.abs(distances - expected_pos) <= win
            if not local_mask.any():
                continue
            refined_pos = float(
                distances[local_mask][np.argmax(np.log1p(intensities[local_mask]))]
            )
            fit = fit_gaussian_1d(distances, intensities, refined_pos,
                                  search_window=win)
            if verbose:
                if fit is None:
                    print(f'[FILL] n={n}: fallback fit FAILED')
                else:
                    print(f'[FILL] n={n}: fallback amp={fit["amplitude"]:.1f} '
                          f'(floor={min_amp:.1f}) '
                          f'-> {"ACCEPT" if fit["amplitude"] >= min_amp else "REJECT"}')
            if fit is None:
                continue
            if fit['amplitude'] < min_amp:
                continue
            if abs(fit['pos'] - ref_sl0_pos) <= min_dist_sl0:
                if verbose:
                    print(f'[FILL] n={n}: fallback pos={fit["pos"]:.2f} too close '
                          f'to SL0 (min_dist={min_dist_sl0:.2f}) — REJECT')
                continue
            result.append(_make_peak(n, fit, sl0_center, rad))
            used_pos.append(fit['pos'])

    return result


def _correct_by_spacing(
    detected: List[Dict[str, Any]],
    peak_positions: np.ndarray,
    orders: List[int],
    distances: np.ndarray,
    intensities: np.ndarray,
    sl0_center: Tuple[float, float],
    rad: float,
    search_win: float,
    n_min_eff: int,
    n_max_eff: int,
    forced_spacing: Optional[float] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Fix misassigned |n|≥3 satellites using spacing inferred from n=±1,2.

    When rank-based assignment places an extra spurious peak between two real
    satellites (e.g. shoulder at -41 px between SL-2 at -34 px and the real
    SL-3 at -46 px), the real satellite is pushed to rank 4 and filtered by
    n_range.  This function detects that case and swaps in the closer peak.

    Swap conditions (both must hold):
      • out-of-range peak is within 0.3 × spacing of the extrapolated position
      • currently-assigned peak deviates > 0.3 × spacing from that position

    If `forced_spacing` is supplied it overrides the SL1→SL2 estimate.
    """
    det_by_order = {pk['order']: pk for pk in detected}
    result = list(detected)

    for side in [-1, 1]:
        n1, n2 = side, 2 * side

        if forced_spacing is not None:
            if n1 not in det_by_order:
                continue
            p1 = det_by_order[n1]['pos_along_axis']
            spacing = forced_spacing
        else:
            if n1 not in det_by_order or n2 not in det_by_order:
                continue
            p1 = det_by_order[n1]['pos_along_axis']
            p2 = det_by_order[n2]['pos_along_axis']
            spacing = abs(p2 - p1)
            if spacing < 1.0:
                continue

        max_k = max(abs(n_min_eff), abs(n_max_eff))
        oor_mask = np.array([not (n_min_eff <= o <= n_max_eff) for o in orders])
        for k in range(3, max_k + 1):
            order = side * k
            if not (n_min_eff <= order <= n_max_eff):
                continue
            if order not in det_by_order:
                continue

            expected = p1 + side * (k - 1) * spacing
            current_dist = abs(det_by_order[order]['pos_along_axis'] - expected)

            # Find the closest out-of-range peak to expected.
            best_pos, best_dist = None, current_dist
            for pos, is_oor in zip(peak_positions, oor_mask):
                if not is_oor:
                    continue
                d = abs(pos - expected)
                if d < best_dist:
                    best_dist, best_pos = d, pos

            if best_pos is None:
                continue
            if best_dist >= 0.3 * spacing or current_dist <= 0.3 * spacing:
                continue

            fit = fit_gaussian_1d(distances, intensities, best_pos,
                                  search_window=search_win)
            if verbose:
                if fit is None:
                    print(f'[CORR] order={order}: swap to {best_pos:.2f} '
                          f'(expected={expected:.2f}) -> fit FAILED, keep original')
                else:
                    print(f'[CORR] order={order}: {det_by_order[order]["pos_along_axis"]:.2f} '
                          f'(dist={current_dist:.2f}) -> {fit["pos"]:.2f} '
                          f'(dist={best_dist:.2f})  expected={expected:.2f}  '
                          f'spacing={spacing:.2f}')
            if fit is None:
                continue
            result = [pk for pk in result if pk['order'] != order]
            result.append(_make_peak(order, fit, sl0_center, rad))
            det_by_order[order] = result[-1]

    return result


def _detect_in_profile(
    image_sub: np.ndarray,
    sl0_center: Tuple[float, float],
    axis_angle: float,
    strip_width: float,
    min_prominence: float,
    peak_min_distance: int,
    peak_min_width: Optional[float],
    n_min_eff: int,
    n_max_eff: int,
    spacing_px: Optional[float] = None,
    adaptive_fill_win: bool = False,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Extract 1-D profile and find + fit satellite peaks.

    Returns (detected_peaks, distances, intensities).  detected_peaks is an
    empty list when no peaks survive the filters.
    """
    distances, intensities = extract_1d_profile(
        image_sub, sl0_center, axis_angle, strip_width=strip_width,
    )

    prof_max = float(intensities.max())
    if prof_max == 0.0:
        return [], distances, intensities

    rad = float(np.radians(axis_angle))

    # Peak detection on the log-scale profile so that weak high-order satellites
    # have prominence comparable to SL0, even though their absolute intensity is
    # orders of magnitude smaller.  The floor (1 count) prevents log(0).
    log_prof = np.log1p(intensities)
    log_max = float(log_prof.max())
    if log_max == 0.0:
        return [], distances, intensities

    fp_kwargs: Dict[str, Any] = {
        'prominence': min_prominence,
        'distance': peak_min_distance,
    }
    if peak_min_width is not None:
        fp_kwargs['width'] = float(peak_min_width)

    peaks_idx, props = find_peaks(log_prof / log_max, **fp_kwargs)
    if len(peaks_idx) == 0:
        return [], distances, intensities

    # Identify SL₀: highest-prominence peak within the central 25 % of the image
    peak_positions = distances[peaks_idx]
    central_radius = float(min(image_sub.shape)) * 0.25
    central_mask = np.abs(peak_positions) < central_radius
    if central_mask.any():
        idx_in_central = np.where(central_mask)[0]
        best = idx_in_central[np.argmax(props['prominences'][central_mask])]
    else:
        best = int(np.argmax(props['prominences']))
    sl0_pos = float(peak_positions[best])

    orders = _assign_orders(peak_positions, sl0_pos,
                            min_spacing_px=float(peak_min_distance) * 0.8)

    # Two separate fit windows:
    #   main_search_win — tight, based on minimum CONSECUTIVE inter-peak spacing
    #     among in-range peaks only.  Prevents the Gaussian from drifting into
    #     an adjacent peak when two satellites are close together.
    #   fill_search_win — wider, based on SL0-to-nearest spacing.  Used by the
    #     fill pass where peaks are not yet located precisely.
    abs_diffs = np.abs(peak_positions - sl0_pos)
    nonzero_diffs = abs_diffs[abs_diffs > float(peak_min_distance) * 0.8]
    if len(nonzero_diffs) >= 1:
        spacing_est = float(nonzero_diffs.min())
        fill_search_win = min(float(peak_min_distance) * 2.5, spacing_est * 0.45)
        in_range_pos = peak_positions[
            np.array([n_min_eff <= o <= n_max_eff for o in orders])
        ]
        if len(in_range_pos) >= 2:
            min_inter = float(np.diff(np.sort(in_range_pos)).min())
            main_search_win = min(fill_search_win, min_inter * 0.45)
        else:
            main_search_win = fill_search_win
    else:
        spacing_est = float(peak_min_distance) * 2.0
        main_search_win = fill_search_win = float(peak_min_distance) * 2.5

    # User-supplied spacing overrides the auto-detected estimate for fill and
    # correction passes (main_search_win stays inter-peak-based for fit safety).
    if spacing_px is not None:
        spacing_est = float(spacing_px)
        fill_search_win = min(float(peak_min_distance) * 2.5, spacing_est * 0.45)

    if verbose:
        print(f'[MAIN] sl0_pos={sl0_pos:.2f}  spacing_est={spacing_est:.2f}  '
              f'main_search_win={main_search_win:.2f}  fill_search_win={fill_search_win:.2f}')
        print(f'[MAIN] find_peaks found {len(peak_positions)} peaks:')
        for pos, ord_ in zip(peak_positions, orders):
            print(f'  pos={pos:+.2f}  assigned_order={ord_}  in_range={n_min_eff<=ord_<=n_max_eff}')

    detected: List[Dict[str, Any]] = []
    for pos, order in zip(peak_positions, orders):
        if not (n_min_eff <= order <= n_max_eff):
            continue
        fit = fit_gaussian_1d(distances, intensities, pos, search_window=main_search_win)
        if verbose:
            if fit is None:
                print(f'[MAIN] order={order}  pos={pos:+.2f} -> Gaussian fit FAILED')
            else:
                print(f'[MAIN] order={order}  pos={pos:+.2f} -> '
                      f'fitted={fit["pos"]:+.2f}  amp={fit["amplitude"]:.1f}  '
                      f'fwhm={fit["fwhm"]:.2f}  ACCEPTED')
        if fit is None:
            continue
        detected.append(_make_peak(order, fit, sl0_center, rad))

    # Spacing consistency correction: if a spurious peak between two real
    # satellites pushed the true SL-k to rank k+1 (filtered by n_range),
    # this swaps the out-of-range peak back to order k when it is much closer
    # to the position extrapolated from the SL-1 / SL-2 spacing.
    if len(detected) >= 2:
        detected = _correct_by_spacing(
            detected, peak_positions, orders, distances, intensities,
            sl0_center, rad, main_search_win, n_min_eff, n_max_eff,
            forced_spacing=spacing_px, verbose=verbose,
        )

    # Supplemental pass: for any order within n_range still missing after the
    # prominence-based search, fit a Gaussian directly at the expected position
    # (sl0_pos + n × spacing).  This recovers satellites that are too weak or
    # too close to SL0 for find_peaks to pick up, without lowering the global
    # prominence threshold.
    if len(nonzero_diffs) >= 1:
        detected = _fill_missing_orders(
            detected, distances, intensities, sl0_center, rad,
            n_min_eff, n_max_eff, sl0_pos, spacing_est, fill_search_win,
            adaptive_fill_win=adaptive_fill_win, verbose=verbose,
        )

    # Deduplicate same-order: keep highest-amplitude when two peaks share an order.
    order_to_peak: Dict[int, Dict[str, Any]] = {}
    for pk in detected:
        order = pk['order']
        if order not in order_to_peak or pk['amplitude'] > order_to_peak[order]['amplitude']:
            order_to_peak[order] = pk

    if verbose:
        dupes = {o for o in order_to_peak
                 if sum(1 for p in detected if p['order'] == o) > 1}
        if dupes:
            print(f'[MAIN] Deduplication removed lower-amplitude duplicates for orders: {sorted(dupes)}')

    detected = sorted(order_to_peak.values(), key=lambda p: p['order'])

    # Deduplicate cross-order: if two peaks from different orders are within
    # half the minimum inter-peak spacing in position, keep the one with lower
    # |order| (closer to SL0 wins — more physically reliable assignment).
    pos_dedup_win = max(main_search_win, 2.0)
    kept: List[Dict[str, Any]] = []
    for pk in sorted(detected, key=lambda p: abs(p['order'])):
        pos = pk['pos_along_axis']
        if any(abs(pos - k['pos_along_axis']) < pos_dedup_win for k in kept):
            if verbose:
                print(f'[MAIN] Cross-order dedup: order={pk["order"]} at {pos:.2f} '
                      f'too close to an already-kept peak — removed')
            continue
        kept.append(pk)
    detected = sorted(kept, key=lambda p: p['order'])
    return detected, distances, intensities


def _refine_axis_from_peaks(
    peaks: List[Dict[str, Any]],
    sl0_center: Tuple[float, float],
    image_sub: np.ndarray,
) -> Optional[float]:
    """Refine satellite axis angle from true 2D intensity centroids.

    For each detected satellite, the intensity-weighted 2D centroid is computed
    inside a circular neighbourhood of radius 2*FWHM in the background-subtracted
    image.  PCA on these real 2D positions (equal weight per satellite) gives a
    better axis estimate than the full-image PCA, which is dominated by the bright
    SL0 and diffuse background streaks.

    Note: using the projected position_2d values from the 1D profile would be
    wrong — those points are collinear by construction and carry no new direction
    information.
    """
    if len(peaks) < 2:
        return None

    H, W = image_sub.shape
    rows_grid, cols_grid = np.mgrid[0:H, 0:W]

    centroids_r: List[float] = []
    centroids_c: List[float] = []
    for pk in peaks:
        r0, c0 = pk['position_2d']
        radius = max(pk['fwhm'] * 2.0, 5.0)
        mask = (rows_grid - r0) ** 2 + (cols_grid - c0) ** 2 <= radius ** 2
        w = np.maximum(image_sub[mask], 0.0)
        total = float(w.sum())
        if total == 0.0:
            centroids_r.append(r0)
            centroids_c.append(c0)
        else:
            centroids_r.append(float((w * rows_grid[mask]).sum() / total))
            centroids_c.append(float((w * cols_grid[mask]).sum() / total))

    rows = np.array(centroids_r)
    cols = np.array(centroids_c)
    cy, cx = sl0_center
    dy = rows - cy
    dx = cols - cx

    cov = np.array([
        [float(np.dot(dy, dy)), float(np.dot(dy, dx))],
        [float(np.dot(dy, dx)), float(np.dot(dx, dx))],
    ]) / len(peaks)

    _, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, 1]   # direction of maximum spread
    angle = float(np.degrees(np.arctan2(principal[0], principal[1])))
    return _canonical_axis_angle(angle)


def detect_satellites(
    image: np.ndarray,
    axis_angle: Optional[float] = None,
    n_max: int = 3,
    min_prominence: float = 0.05,
    strip_width: float = 5.0,
    bg_sigma: float = 20.0,
    sl0_top_fraction: float = 0.02,
    peak_min_distance: int = 5,
    peak_min_width: Optional[float] = 2.0,
    hot_pixel_sigma: Optional[float] = 10.0,
    n_range: Optional[Tuple[int, int]] = None,
    spacing_px: Optional[float] = None,
    adaptive_fill_win: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Detect satellite peaks in a Laue spot image from a MQW structure.

    Parameters
    ----------
    image             : 2-D array, spot image (raw or pre-processed)
    axis_angle        : float or None — satellite axis angle in degrees from +x.
                        Auto-detected via PCA when None.
    n_max             : maximum satellite order to accept (e.g. 3 for ±3).
    min_prominence    : minimum peak prominence as fraction of log-profile maximum.
                        Peak detection runs on log1p(intensities) so that weak
                        high-order satellites have comparable prominence to SL0.
    strip_width       : strip width (px) for the 1-D profile projection.
    bg_sigma          : Gaussian sigma (px) for background subtraction.
                        Set to 0 to skip subtraction entirely.
    sl0_top_fraction  : fraction of brightest pixels used to seed SL₀ centroid.
    peak_min_distance : minimum pixel spacing between candidate peaks.
    peak_min_width    : minimum 1-D peak width in profile bins to reject
                        single-bin noise spikes (default 2.0).  Set to None to
                        disable.  Real satellite peaks typically have width ≥ 3
                        bins; single-pixel noise spikes have width < 2.
    hot_pixel_sigma   : n_sigma for hot-pixel removal before background subtraction.
                        None = skip.
    n_range           : (n_min, n_max) tuple to restrict which satellite orders are
                        accepted.  E.g. (-3, 0) accepts only SL-3…SL0 and discards
                        any peak assigned to a positive order (useful when a nearby
                        Bragg reflection falls on the positive side).
                        Overrides n_max when provided.
    spacing_px        : float or None — if provided, overrides the automatic satellite
                        spacing estimate.  Use when the auto-detected spacing is
                        unreliable (e.g. only one satellite visible, or strongly
                        non-uniform spacing due to axis misalignment).
    verbose           : if True, print step-by-step diagnostic messages tracing
                        every detection, assignment, correction, and fill decision.

    Returns
    -------
    dict with keys
        'peaks'      : list of peak dicts (order, pos_along_axis, position_2d,
                       amplitude, fwhm, sigma, fit_success)
        'axis_angle' : float — refined axis angle used (degrees)
        'sl0_center' : (row, col)
        'profile'    : (distances, intensities) 1-D profile tuple
        'image_sub'  : background-subtracted image
    """
    image = np.asarray(image, dtype=np.float32)

    if hot_pixel_sigma is not None:
        image = clip_hot_pixels(image, n_sigma=float(hot_pixel_sigma))

    image_sub = subtract_background(image, sigma=bg_sigma)
    sl0_center = find_sl0_centroid(image_sub, top_fraction=sl0_top_fraction)

    if axis_angle is None:
        axis_angle = find_satellite_axis(image_sub, center=sl0_center)

    n_min_eff = n_range[0] if n_range is not None else -n_max
    n_max_eff = n_range[1] if n_range is not None else n_max

    # Pass 1 — detect with initial axis (PCA or user-supplied)
    if verbose:
        print('[DETECT] ── Pass 1 ──────────────────────────────────────')
    detected, distances, intensities = _detect_in_profile(
        image_sub, sl0_center, axis_angle, strip_width,
        min_prominence, peak_min_distance, peak_min_width,
        n_min_eff, n_max_eff, spacing_px=spacing_px,
        adaptive_fill_win=adaptive_fill_win, verbose=verbose,
    )

    if intensities.max() == 0.0:
        return _empty_result(axis_angle, sl0_center, distances, intensities, image_sub)

    # Pass 2 — refine axis from detected satellite positions, then re-detect.
    # This gives equal weight to every satellite instead of being dominated by
    # the bright SL0 (as full-image PCA would be).
    if len(detected) >= 2:
        refined = _refine_axis_from_peaks(detected, sl0_center, image_sub)
        if refined is not None:
            axis_angle = refined
            if verbose:
                print(f'[DETECT] ── Pass 2 (refined axis={axis_angle:.2f}°) ──────────')
            detected, distances, intensities = _detect_in_profile(
                image_sub, sl0_center, axis_angle, strip_width,
                min_prominence, peak_min_distance, peak_min_width,
                n_min_eff, n_max_eff, spacing_px=spacing_px,
                adaptive_fill_win=adaptive_fill_win, verbose=verbose,
            )

    return {
        'peaks': detected,
        'axis_angle': float(axis_angle),
        'sl0_center': sl0_center,
        'profile': (distances, intensities),
        'image_sub': image_sub,
    }


def _empty_result(axis_angle, sl0_center, distances, intensities, image_sub):
    return {
        'peaks': [],
        'axis_angle': float(axis_angle),
        'sl0_center': sl0_center,
        'profile': (distances, intensities),
        'image_sub': image_sub,
    }


# ── Synthetic test image ──────────────────────────────────────────────────────

def make_synthetic_image(
    shape: Tuple[int, int] = (150, 150),
    n_satellites: int = 3,
    spacing: float = 22.0,
    axis_angle: float = 35.0,
    sl0_amplitude: float = 5000.0,
    envelope_decay: float = 0.5,
    fwhm: float = 4.0,
    background_level: float = 200.0,
    noise_level: float = 50.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic MQW Laue spot image with known satellite positions.

    Useful for testing detection and metrics without real data.  The true
    parameters are printed to stdout so detection results can be compared.

    Parameters
    ----------
    shape          : image size (rows, cols)
    n_satellites   : number of orders on each side (total = 2·n + 1)
    spacing        : pixel distance between consecutive orders
    axis_angle     : satellite axis in degrees from +x
    sl0_amplitude  : peak counts of SL₀
    envelope_decay : α in I_n = I₀·exp(−α·|n|)
    fwhm           : peak FWHM in pixels
    background_level : flat background level
    noise_level    : Gaussian noise σ
    seed           : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    cy, cx = shape[0] / 2.0, shape[1] / 2.0
    sigma = fwhm / 2.3548

    image = np.full(shape, background_level, dtype=np.float64)
    # Smooth diffuse background (dislocation-induced streaking proxy)
    image += 0.15 * sl0_amplitude * np.exp(-((rows - cy) ** 2 + (cols - cx) ** 2) / (2 * 55.0**2))

    rad = np.radians(axis_angle)
    for order in range(-n_satellites, n_satellites + 1):
        amp = sl0_amplitude * np.exp(-envelope_decay * abs(order))
        s = order * spacing
        r0 = cy + s * np.sin(rad)
        c0 = cx + s * np.cos(rad)
        image += amp * np.exp(-((rows - r0) ** 2 + (cols - c0) ** 2) / (2 * sigma**2))

    image += rng.normal(0.0, noise_level, shape)
    return np.maximum(image, 0.0).astype(np.float32)
