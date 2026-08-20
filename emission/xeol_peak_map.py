"""
xeol_peak_map.py — Spatial map of XEOL emission peak position and InGaN composition.

For each scan point, fits a Gaussian to the emission peak within a
user-defined wavelength window and maps the peak centre wavelength
across the scan.  An additional post-processing step converts the peak
wavelength to InGaN In fraction (x) via Vegard's law + bowing, allowing
spatial mapping of In concentration variation.

Usage
-----
>>> from emission.xeol_peak_map import fit_xeol_peak_map, plot_xeol_peak_map
>>> from emission.xeol_peak_map import add_In_fraction, plot_In_content_map
>>> df = fit_xeol_peak_map(
...     h5_path="scan_001.h5",
...     wl_window=(360.0, 380.0),   # nm
... )
>>> fig = plot_xeol_peak_map(df)
>>> add_In_fraction(df)
>>> fig_in = plot_In_content_map(df)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm

from utils.fitting import fwhm_from_sigma, gaussian, r_squared

# lauexplore is imported inside fit_xeol_peak_map: it is the only function that
# needs it, and it requires Python >= 3.9, so a module-level import would make the
# Gaussian helpers and the In-fraction conversion unusable on older interpreters.


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
            gaussian, wl_win, spec_win,
            p0=[amp0, cen0, sig0, bg0],
            bounds=(
                [0,          wl_win[0],  1e-3,        -np.inf],
                [np.inf,     wl_win[-1], wl_win[-1] - wl_win[0], np.inf],
            ),
            maxfev=800,
        )
        amplitude, center, sigma, background = popt
        y_fit   = gaussian(wl_win, *popt)
        r2      = r_squared(spec_win, y_fit)
        fwhm    = fwhm_from_sigma(sigma)
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
    min_r_squared: float = 0.9,
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
    min_r_squared : float
        Points where R² is below this threshold are marked as not converged
        (default 0.9).

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
    from lauexplore.emission import XEOL

    xeol = XEOL.from_h5(
        h5_path, scan_number,
        roi=wl_window,
        normalize_to_monitor=normalize_to_monitor,
        norm_zone=norm_zone,
    )

    wl      = xeol.wl_array
    spectra = xeol.spectra  # already treated: ref subtraction + monitor + norm_zone
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

        center, amplitude, fwhm, background, r2, ok = _fit_one(wl_win, spec_win)

        if ok and amplitude < min_amplitude:
            ok = False
            center = amplitude = fwhm = background = r2 = np.nan

        if ok and r2 < min_r_squared:
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


# ---------------------------------------------------------------------------
# InGaN composition from emission wavelength
# ---------------------------------------------------------------------------

def wl_to_In_fraction(
    wl_nm: float | np.ndarray,
    *,
    eg_gan: float = 3.44,
    eg_inn: float = 0.77,
    bowing: float = 1.43,
) -> float | np.ndarray:
    """Convert emission peak wavelength to InGaN In fraction via Vegard's law + bowing.

    Solves  E_g(x) = (1-x)*eg_gan + x*eg_inn - bowing*x*(1-x) = E_peak
    for x ∈ [0, 1].  Returns NaN for wavelengths outside the physical range.

    Parameters
    ----------
    wl_nm : float or array
        Peak wavelength in nm.
    eg_gan : float
        GaN bandgap in eV (default 3.44 eV at RT).
    eg_inn : float
        InN bandgap in eV (default 0.77 eV).
    bowing : float
        Empirical bowing parameter in eV (default 1.43 eV).
    """
    wl_nm = np.asarray(wl_nm, dtype=float)
    scalar = wl_nm.ndim == 0
    wl_nm = np.atleast_1d(wl_nm)

    E_peak = 1239.84 / wl_nm  # eV, E = hc/λ

    # Rearranged quadratic: bowing·x² + (eg_inn - eg_gan - bowing)·x + (eg_gan - E_peak) = 0
    a_coef = bowing
    b_coef = eg_inn - eg_gan - bowing
    c_coef = eg_gan - E_peak

    discriminant = b_coef ** 2 - 4.0 * a_coef * c_coef
    x = np.where(
        discriminant >= 0,
        (-b_coef - np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * a_coef),
        np.nan,
    )
    x = np.where((x >= 0.0) & (x <= 1.0), x, np.nan)

    return float(x[0]) if scalar else x


def add_In_fraction(
    df: pd.DataFrame,
    *,
    eg_gan: float = 3.44,
    eg_inn: float = 0.77,
    bowing: float = 1.43,
) -> pd.DataFrame:
    """Add an ``in_fraction`` column to a DataFrame returned by ``fit_xeol_peak_map``.

    Non-converged points and wavelengths outside the physical InGaN range are NaN.
    Modifies *df* in place and returns it.

    Parameters
    ----------
    df : DataFrame
        Output of ``fit_xeol_peak_map``.
    eg_gan, eg_inn, bowing : float
        Vegard's law parameters passed to ``wl_to_In_fraction``.
    """
    x = wl_to_In_fraction(df["peak_wl"].values, eg_gan=eg_gan, eg_inn=eg_inn, bowing=bowing)
    df["in_fraction"] = x
    df.loc[~df["converged"], "in_fraction"] = np.nan
    return df


def plot_In_content_map(
    df: pd.DataFrame,
    *,
    percentile_clip: tuple[float, float] = (2, 98),
    cmap: str = "plasma",
    figsize: tuple[float, float] = (6, 5),
    title: str = "In fraction  x  (In$_x$Ga$_{1-x}$N)",
) -> plt.Figure:
    """Plot a spatial map of the In fraction derived from the XEOL peak wavelength.

    Requires an ``in_fraction`` column — run ``add_In_fraction(df)`` first.

    Parameters
    ----------
    df : DataFrame
        Output of ``fit_xeol_peak_map`` after calling ``add_In_fraction``.
    percentile_clip : (lo, hi)
        Colour scale percentiles.
    cmap : str
        Colormap (default "plasma").
    figsize : (w, h)
        Figure size in inches.
    title : str
        Plot title.
    """
    if "in_fraction" not in df.columns:
        raise ValueError("df has no 'in_fraction' column — run add_In_fraction(df) first.")

    x_um = np.sort(df["x_um"].unique())
    y_um = np.sort(df["y_um"].unique())
    extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]

    i_min, j_min = df["i"].min(), df["j"].min()
    nbx = df["i"].nunique()
    nby = df["j"].nunique()

    grid = np.full((nbx, nby), np.nan)
    for _, row in df.iterrows():
        grid[int(row["i"] - i_min), int(row["j"] - j_min)] = row["in_fraction"]

    data = grid.T
    lo = np.nanpercentile(data[np.isfinite(data)], percentile_clip[0])
    hi = np.nanpercentile(data[np.isfinite(data)], percentile_clip[1])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, origin="lower", aspect="equal",
                   extent=extent, cmap=cmap, vmin=lo, vmax=hi)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("In fraction  x", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    plt.tight_layout()
    return fig
