"""Shared peak-fitting primitives.

The Gaussian model, the FWHM conversion and the R² helper used to exist in three
places (``emission/xeol_peak_map.py``, ``emission/stats_utils.py`` and
``utils/plot_histograms.py``), each with a different baseline. They are the same
model; only the baseline degree differed, so the richest form subsumes the rest.
"""

from __future__ import annotations

import numpy as np

# FWHM = 2 sqrt(2 ln 2) sigma ≈ 2.3548 sigma
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def gaussian(x, amplitude, center, sigma, background=0.0, slope=0.0):
    """Gaussian peak over a linear baseline.

    ``A exp(-(x - x0)² / 2σ²) + slope·x + background``

    Parameter order is fixed by ``scipy.optimize.curve_fit``, which passes them
    positionally: amplitude, center, sigma, then background before slope. Callers
    fitting a flat baseline simply stop at ``background``.
    """
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + slope * x + background


def r_squared(y, y_fit):
    """Coefficient of determination; NaN when y has no variance."""
    y = np.asarray(y, dtype=float)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def fwhm_from_sigma(sigma):
    """Convert a Gaussian sigma to full width at half maximum."""
    return FWHM_FACTOR * abs(sigma)
