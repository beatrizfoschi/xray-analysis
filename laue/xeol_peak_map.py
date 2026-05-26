"""
xeol_peak_map.py — Spatial map of XEOL emission peak position.

For each scan point, fits a Gaussian to the emission peak within a
user-defined wavelength window and maps the peak centre wavelength
across the scan to evaluate emission homogeneity.

Usage
-----
>>> from laue.xeol_peak_map import fit_xeol_peak_map, plot_xeol_peak_map
>>> df = fit_xeol_peak_map(
...     h5_path="scan_001.h5",
...     wl_window=(360.0, 380.0),   # nm
... )
>>> fig = plot_xeol_peak_map(df)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

from lauexplore.emission import XEOL
from lauexplore.plots.base import _as_grid


_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2√(2 ln 2) ≈ 2.355


def _gaussian(wl, amplitude, center, sigma, background):
    return amplitude * np.exp(-0.5 * ((wl - center) / sigma) ** 2) + background


def _r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def _fit_one(wl_win, spec_win):
    """Fit a Gaussian to one spectrum window. Returns (center, amplitude, fwhm, bg, r2, converged)."""
    bg0    = spec_win.min()
    amp0   = spec_win.max() - bg0
    cen0   = wl_win[spec_win.argmax()]
    sig0   = (wl_win[-1] - wl_win[0]) / 6.0   # rough guess: window/6

    if amp0 <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, False

    try:
        popt, _ = curve_fit(
            _gaussian, wl_win, spec_win,
            p0=[amp0, cen0, sig0, bg0],
            bounds=(
                [0,          wl_win[0],  1e-3,        -np.inf],
                [np.inf,     wl_win[-1], wl_win.ptp(), np.inf],
            ),
            maxfev=800,
        )
        amplitude, center, sigma, background = popt
        y_fit   = _gaussian(wl_win, *popt)
        r2      = _r_squared(spec_win, y_fit)
        fwhm    = _FWHM_FACTOR * abs(sigma)
        return center, amplitude, fwhm, background, r2, True
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, False


def fit_xeol_peak_map(
    h5_path: str | Path,
    wl_window: tuple[float, float],
    *,
    scan_number: int = 1,
    normalize_to_monitor: bool = True,
    norm_zone: tuple[float, float] | None = None,
    min_amplitude: float = 0.0,
) -> pd.DataFrame:
    """Fit a Gaussian peak to the XEOL spectrum in ``wl_window`` at each scan point.

    Parameters
    ----------
    h5_path : Path
        Scan HDF5 file.
    wl_window : (wl_min, wl_max)
        Wavelength window in nm to fit.
    scan_number : int
        Scan entry in the HDF5 (default 1).
    normalize_to_monitor : bool
        Normalise spectra by monitor counts before fitting (default True).
    norm_zone : (wl_min, wl_max) or None
        Additional normalisation zone passed to XEOL.from_h5.
    min_amplitude : float
        Points where the fitted amplitude is below this threshold are marked
        as not converged (useful to reject noise fits).

    Returns
    -------
    pd.DataFrame with columns:
        i, j, x_um, y_um        — scan grid position
        peak_wl   (nm)          — Gaussian centre wavelength
        amplitude                — peak amplitude
        fwhm      (nm)          — full width at half maximum
        background               — fitted constant background
        r_squared                — goodness of fit [0, 1]
        converged  (bool)        — True if the fit succeeded
    """
    xeol = XEOL.from_h5(
        h5_path, scan_number,
        roi=wl_window,
        normalize_to_monitor=normalize_to_monitor,
        norm_zone=norm_zone,
    )

    wl      = xeol.wl_array
    spectra = xeol.spectra
    scan    = xeol.scan

    # Mask to wavelength window
    mask    = (wl >= wl_window[0]) & (wl <= wl_window[1])
    wl_win  = wl[mask]
    if mask.sum() < 4:
        raise ValueError(
            f"Wavelength window {wl_window} nm contains only {mask.sum()} "
            f"channels — too narrow to fit."
        )

    positions = [
        (i, j)
        for i in range(scan.nbxpoints)
        for j in range(scan.nbypoints)
    ]

    rows = []
    for i, j in tqdm(positions, desc="Fitting peaks"):
        pt_idx   = scan.ij_to_index(i, j)
        spec_win = spectra[pt_idx, mask].astype(float)

        if normalize_to_monitor:
            spec_win = spec_win * 1e5 / scan.monitor_data[pt_idx]

        center, amplitude, fwhm, background, r2, ok = _fit_one(wl_win, spec_win)

        if ok and amplitude < min_amplitude:
            ok = False
            center = amplitude = fwhm = background = r2 = np.nan

        x_um, y_um = scan.ij_to_xy(i, j)

        rows.append({
            "i":          i,
            "j":          j,
            "x_um":       float(x_um) * 1e3,
            "y_um":       float(y_um) * 1e3,
            "peak_wl":    center,
            "amplitude":  amplitude,
            "fwhm":       fwhm,
            "background": background,
            "r_squared":  r2,
            "converged":  ok,
        })

    df = pd.DataFrame(rows)
    df.sort_values(["i", "j"], inplace=True, ignore_index=True)

    n_ok  = df["converged"].sum()
    n_tot = len(df)
    print(f"Converged: {n_ok}/{n_tot} points  "
          f"({100*n_ok/n_tot:.1f}%)  |  "
          f"peak_wl = {df['peak_wl'].mean():.2f} ± {df['peak_wl'].std():.2f} nm")
    return df


def plot_xeol_peak_map(
    df: pd.DataFrame,
    *,
    percentile_clip: tuple[float, float] = (2, 98),
    cmap_wl: str = "RdBu",
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Plot 2D spatial maps of the Gaussian fit parameters.

    Four panels:
      - Peak wavelength λ₀  (emission homogeneity — main result)
      - Amplitude
      - FWHM
      - R² goodness of fit  (quality control)

    Parameters
    ----------
    df : DataFrame
        Output of ``fit_xeol_peak_map``.
    percentile_clip : (lo, hi)
        Colour scale percentiles for each panel.
    cmap_wl : str
        Colormap for the peak wavelength map (default "RdBu" to highlight shifts).
    figsize : (w, h)
        Figure size in inches.
    """
    panels = [
        ("peak_wl",    "Peak wavelength λ₀ (nm)",   cmap_wl),
        ("amplitude",  "Amplitude (counts)",          "inferno"),
        ("fwhm",       "FWHM (nm)",                  "plasma"),
        ("r_squared",  "R² goodness of fit",          "viridis"),
    ]

    x_um = np.sort(df["x_um"].unique())
    y_um = np.sort(df["y_um"].unique())
    extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]

    i_min, j_min = df["i"].min(), df["j"].min()
    nbx = df["i"].nunique()
    nby = df["j"].nunique()

    # Use NaN for non-converged points
    df_plot = df.copy()
    df_plot.loc[~df_plot["converged"], ["peak_wl", "amplitude", "fwhm", "r_squared"]] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, (col, title, cmap) in zip(axes.flat, panels):
        grid = np.full((nbx, nby), np.nan)
        for _, row in df_plot.iterrows():
            grid[int(row["i"] - i_min), int(row["j"] - j_min)] = row[col]

        data = grid.T
        lo = np.nanpercentile(data, percentile_clip[0])
        hi = np.nanpercentile(data, percentile_clip[1])

        im = ax.imshow(data, origin="lower", aspect="equal",
                       extent=extent, cmap=cmap, vmin=lo, vmax=hi)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

    plt.tight_layout()
    return fig
