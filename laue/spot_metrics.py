"""
spot_metrics.py — Per-spot morphology metrics for Laue diffraction images.

Each function operates on a single 2D intensity array (one ROI crop).
All coordinates are in pixels; angles in degrees measured from the
positive x-axis (horizontal right), with y increasing downward.

Functions
---------
preprocess          Background subtraction + optional smoothing/masking
center_of_mass      Intensity-weighted centroid
inertia_tensor      Second moments → eigenvalues, streak direction, aspect ratio
streak_length       D50 / D95 along the principal streak axis
core_tail_ratio     Fraction of intensity within a fixed-radius core
analyze_spot        Convenience wrapper: full pipeline, returns a dict
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(
    img: np.ndarray,
    *,
    bg_method: str = "corners",
    bg_percentile: float = 5.0,
    corner_size: int = 5,
    smooth_sigma: float = 0.0,
    noise_nsigma: float = 0.0,
) -> np.ndarray:
    """Subtract background and optionally smooth and threshold the ROI.

    Parameters
    ----------
    img : (H, W) array
        Raw intensity values.
    bg_method : {"corners", "percentile"}
        "corners" estimates background from the four corner patches (robust
        for centred spots); "percentile" uses a global percentile.
    bg_percentile : float
        Percentile used when bg_method="percentile" (default 5).
    corner_size : int
        Side of each corner patch in pixels (default 5).
    smooth_sigma : float
        Sigma for Gaussian smoothing (0 = disabled).
    noise_nsigma : float
        Pixels below bg + noise_nsigma * bg_std are set to zero (0 = disabled).

    Returns
    -------
    out : (H, W) float64 array, non-negative
    """
    img = np.asarray(img, dtype=np.float64)

    if bg_method == "corners":
        cs = max(1, corner_size)
        patches = [
            img[:cs, :cs], img[:cs, -cs:],
            img[-cs:, :cs], img[-cs:, -cs:],
        ]
        flat = np.concatenate([p.ravel() for p in patches])
        bg    = np.median(flat)
        bg_std = flat.std()
    else:
        bg     = np.percentile(img, bg_percentile)
        bg_std = 0.0

    out = img - bg

    if smooth_sigma > 0:
        out = gaussian_filter(out, sigma=smooth_sigma)

    if noise_nsigma > 0:
        out[out < noise_nsigma * bg_std] = 0.0

    return np.clip(out, 0, None)


# ── Center of mass ─────────────────────────────────────────────────────────────

def center_of_mass(img: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted centroid.

    Returns
    -------
    (x_com, y_com) : float
        Column (x) and row (y) coordinates in pixels, 0-based.
        Returns image centre if total intensity is zero.
    """
    total = img.sum()
    if total == 0:
        return img.shape[1] / 2.0, img.shape[0] / 2.0
    gy, gx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    x_com = (gx * img).sum() / total
    y_com = (gy * img).sum() / total
    return float(x_com), float(y_com)


# ── Inertia tensor ────────────────────────────────────────────────────────────

def inertia_tensor(
    img: np.ndarray,
    x_com: float,
    y_com: float,
) -> tuple[float, float, float, float]:
    """Second intensity moments → streak geometry.

    Computes the 2×2 inertia matrix M = [[σ_xx, σ_xy], [σ_xy, σ_yy]] and
    diagonalises it to find the principal streak axis.

    Returns
    -------
    lambda1 : float
        Larger eigenvalue (variance along streak axis).
    lambda2 : float
        Smaller eigenvalue (variance perpendicular to streak).
    aspect_ratio : float
        lambda1 / lambda2  (1 = round, >> 1 = strongly streaked).
    theta : float
        Streak angle in degrees, measured from the positive x-axis,
        restricted to (-90°, 90°].  y increases downward (image convention).
    """
    total = img.sum()
    if total == 0:
        return np.nan, np.nan, np.nan, np.nan

    gy, gx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    dx = (gx - x_com).ravel()
    dy = (gy - y_com).ravel()
    w  = img.ravel() / total

    sxx = (dx * dx * w).sum()
    syy = (dy * dy * w).sum()
    sxy = (dx * dy * w).sum()

    M = np.array([[sxx, sxy], [sxy, syy]])
    eigenvalues, eigenvectors = np.linalg.eigh(M)   # ascending order

    lambda2, lambda1 = float(eigenvalues[0]), float(eigenvalues[1])
    v1 = eigenvectors[:, 1]                          # eigenvector of lambda1

    theta = float(np.degrees(np.arctan2(v1[1], v1[0])))
    # Normalise to (-90°, 90°] — a streak and its 180° mirror are equivalent
    if theta > 90.0:
        theta -= 180.0
    elif theta <= -90.0:
        theta += 180.0

    aspect_ratio = (lambda1 / lambda2) if lambda2 > 0 else np.inf

    return lambda1, lambda2, aspect_ratio, theta


# ── Streak length ─────────────────────────────────────────────────────────────

def streak_length(
    img: np.ndarray,
    x_com: float,
    y_com: float,
    theta_deg: float,
) -> tuple[float, float]:
    """D50 and D95 along the principal streak axis.

    Projects all pixels onto the streak direction and computes the distance
    from the COM that encloses 50% (core width) and 95% (streak extent) of
    the total intensity.

    Returns
    -------
    (D50, D95) : float
        Distances in pixels.
    """
    total = img.sum()
    if total == 0 or np.isnan(theta_deg):
        return np.nan, np.nan

    theta_rad = np.radians(theta_deg)
    sx, sy = np.cos(theta_rad), np.sin(theta_rad)   # streak unit vector (x, y)

    gy, gx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    proj = (gx - x_com) * sx + (gy - y_com) * sy    # signed projection

    abs_proj = np.abs(proj).ravel()
    weights  = img.ravel()

    # Sort by absolute projection distance from COM
    order     = np.argsort(abs_proj)
    sorted_d  = abs_proj[order]
    cumsum    = np.cumsum(weights[order])
    cumsum   /= cumsum[-1]                           # normalise to [0, 1]

    d50 = float(sorted_d[np.searchsorted(cumsum, 0.50)])
    d95 = float(sorted_d[np.searchsorted(cumsum, 0.95)])
    return d50, d95


# ── Core-to-tail ratio ────────────────────────────────────────────────────────

def core_tail_ratio(
    img: np.ndarray,
    x_com: float,
    y_com: float,
    r_core: float = 3.0,
) -> float:
    """Fraction of total intensity within a circular core of radius r_core.

    R close to 1 → compact spot; R << 1 → significant diffuse tail / V-pit.
    """
    total = img.sum()
    if total == 0:
        return np.nan
    gy, gx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    dist    = np.sqrt((gx - x_com) ** 2 + (gy - y_com) ** 2)
    return float(img[dist <= r_core].sum() / total)


# ── Full pipeline for one spot ────────────────────────────────────────────────

def analyze_spot(
    img: np.ndarray,
    *,
    r_core: float = 3.0,
    smooth_sigma: float = 0.0,
    bg_method: str = "corners",
    bg_percentile: float = 5.0,
    corner_size: int = 5,
    noise_nsigma: float = 0.0,
    min_counts: float = 10.0,
) -> dict:
    """Run the full morphology pipeline on one ROI image.

    Returns a dict with keys:
        x_com, y_com          — centroid (pixels, relative to ROI origin)
        x_com_rel, y_com_rel  — centroid relative to ROI centre
        lambda1, lambda2      — inertia eigenvalues
        aspect_ratio          — lambda1 / lambda2
        theta                 — streak angle (degrees, image convention)
        streak_D50            — core half-width (pixels)
        streak_D95            — streak extent (pixels)
        core_tail_ratio       — I_core / I_total

    All values are NaN if the spot has insufficient counts.
    """
    nan_result: dict = {k: np.nan for k in (
        "x_com", "y_com", "x_com_rel", "y_com_rel",
        "lambda1", "lambda2", "aspect_ratio", "theta",
        "streak_D50", "streak_D95", "core_tail_ratio",
    )}

    img = np.asarray(img, dtype=np.float64)
    proc = preprocess(
        img,
        bg_method=bg_method,
        bg_percentile=bg_percentile,
        corner_size=corner_size,
        smooth_sigma=smooth_sigma,
        noise_nsigma=noise_nsigma,
    )

    if proc.sum() < min_counts:
        return nan_result

    x_com, y_com = center_of_mass(proc)
    cx = (proc.shape[1] - 1) / 2.0
    cy = (proc.shape[0] - 1) / 2.0

    lam1, lam2, ar, theta = inertia_tensor(proc, x_com, y_com)
    d50, d95              = streak_length(proc, x_com, y_com, theta)
    ctr                   = core_tail_ratio(proc, x_com, y_com, r_core)

    return {
        "x_com":          x_com,
        "y_com":          y_com,
        "x_com_rel":      x_com - cx,
        "y_com_rel":      y_com - cy,
        "lambda1":        lam1,
        "lambda2":        lam2,
        "aspect_ratio":   ar,
        "theta":          theta,
        "streak_D50":     d50,
        "streak_D95":     d95,
        "core_tail_ratio": ctr,
    }
