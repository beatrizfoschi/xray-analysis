from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
import skimage.filters as filters
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_closing
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from skimage.segmentation import watershed
from skimage.filters import sobel
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis, mode
import scipy.stats

from utils.fitting import fwhm_from_sigma, gaussian
from utils.regions import elliptical_masks_um


def segment_leds(
    image: np.ndarray,
    min_area: int = 300,
    opening_radius: int = 2,
    fill_gaps: bool = False,
    plot: bool = False,
    extent: tuple[float, float, float, float] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Segment LED regions from a 2-D map (emission map or metric map).

    Parameters
    ----------
    image               : 2-D array — emission map or metric map (e.g. from
                          make_metric_grid). NaN values are treated as background
                          (set to 0).
    min_area            : minimum region size in pixels. Also used as the
                          block_size of threshold_local, so it must be odd.
    opening_radius      : disk radius for morphological opening (separates
                          touching LEDs and removes thin bridges).
    fill_gaps           : if True, use watershed to grow each label back to
                          the full extent before opening (useful when LED
                          projections touch each other).
    plot                : if True, show side-by-side plots of the map and
                          the resulting segmentation.
    extent              : [xmin, xmax, ymin, ymax] for axis labels in µm.
                          If None, pixel indices are used.
    figsize             : figure size. Auto-sized when None.

    Returns
    -------
    labels  : 2-D int array — 0 = background, 1..N = LED regions.
    regions : list of skimage regionprops objects.
    """
    import matplotlib.colors as mcolors

    img = np.where(np.isfinite(image), image, 0.0).astype(float)

    nonzero = img[img > 0]
    if len(nonzero) == 0:
        raise ValueError("image has no positive finite values — check the map.")

    # Former global-percentile threshold, replaced by the local adaptive one below.
    # To restore it, re-add `threshold_percentile: float = 30.0` to the signature:
    #     thr = float(np.percentile(nonzero, threshold_percentile))
    thr = filters.threshold_local(img, block_size=min_area)

    binary_full = img >= thr

    binary = binary_opening(binary_full, disk(opening_radius))
    binary = remove_small_objects(binary, min_size=min_area)

    labels = label(binary)

    if fill_gaps:
        binary_full = remove_small_objects(binary_full, min_size=min_area // 4)
        labels = watershed(sobel(img), markers=labels, mask=binary_full)

    regions = regionprops(labels, intensity_image=img)

    if plot:
        n_leds    = int(labels.max())
        imshow_kw = dict(origin='lower', interpolation='none',
                         extent=extent if extent is not None else None)
        xlabel    = 'x (µm)' if extent is not None else 'col (px)'
        ylabel    = 'y (µm)' if extent is not None else 'row (px)'

        cmap_led  = plt.cm.get_cmap('tab20', max(n_leds, 1))
        colors    = [(0.85, 0.85, 0.85, 1.0)] + [cmap_led(k) for k in range(n_leds)]
        cmap_disc = mcolors.ListedColormap(colors)
        led_ids   = list(range(1, n_leds + 1))
        bounds    = [-0.5] + [v - 0.5 for v in led_ids] + [led_ids[-1] + 0.5]
        norm      = mcolors.BoundaryNorm(bounds, cmap_disc.N)

        if figsize is None:
            figsize = (12, 5)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Map
        ax = axes[0]
        finite_vals = image[np.isfinite(image)]
        vmin = float(np.nanpercentile(finite_vals, 2))  if len(finite_vals) else 0
        vmax = float(np.nanpercentile(finite_vals, 98)) if len(finite_vals) else 1
        im = ax.imshow(image, cmap='viridis', vmin=vmin, vmax=vmax, **imshow_kw)
        plt.colorbar(im, ax=ax, label='Value')
        ax.set_title('Map')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Segmentation
        ax = axes[1]
        im = ax.imshow(labels, cmap=cmap_disc, norm=norm, **imshow_kw)
        cbar = plt.colorbar(im, ax=ax, ticks=led_ids)
        cbar.set_label('LED ID')
        ax.set_title(f'Segmentation — {n_leds} regions')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # LED ID labels at centroid
        pe = __import__('matplotlib.patheffects', fromlist=['withStroke'])
        stroke = pe.withStroke(linewidth=2, foreground='black')
        for reg in regions:
            r_c, c_c = reg.centroid   # (row, col) in pixel coords
            if extent is not None:
                nby, nbx = labels.shape
                xmin, xmax, ymin, ymax = extent
                xc = xmin + (c_c / nbx) * (xmax - xmin)
                yc = ymin + (r_c / nby) * (ymax - ymin)
            else:
                xc, yc = c_c, r_c
            ax.text(xc, yc, str(reg.label),
                    ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white',
                    path_effects=[stroke])

        plt.tight_layout()
        plt.show()
        return labels, regions, fig

    return labels, regions


def extract_led_pixels(labels, image):
    """Return a dict mapping each label id to the pixel values of that LED."""
    led_pixels = {}
    for lab in np.unique(labels):
        if lab == 0:
            continue
        led_pixels[lab] = image[labels == lab]
    return led_pixels


def plot_global_led_histogram(led_pixels, log=True, bins=50, density=False, edgecolor='black'):
    """Plot a histogram of all pixel intensities across all LEDs combined."""
    data = np.concatenate(list(led_pixels.values()))

    if log:
        data = np.log(data[data > 0])
        xlabel = "log(Integrated emission)"
    else:
        xlabel = "Integrated emission"

    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, edgecolor=edgecolor, alpha=0.8, density=density)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.show()


def plot_histograms_per_led(
    led_pixels,
    ncols=4,
    log=True,
    bins=30,
    density=False,
    show_stats_plot=True,
    show_median_vs_max=True,
    pmin=None,
    pmax=None,
    show_mean_line=True,
    moment_box=True,
    edgecolor=None,
    figsize=None
):
    """
    Plot per-LED intensity histograms with optional summary statistics.

    Generates one histogram subplot per LED, then optionally plots:
      - LED index vs median ± IQR and max intensity
      - Median vs max scatter

    Parameters
    ----------
    led_pixels : dict
        {label_id: pixel_values array}
    log : bool
        If True, plot log(intensity).
    show_mean_line : bool
        Draw a vertical dashed line at the mean on each histogram.
    moment_box : bool
        Draw a text box with mode, mean, std, skewness, and kurtosis.
    show_stats_plot : bool
        Show the summary median ± IQR + max plot.
    show_median_vs_max : bool
        Show a scatter plot of median vs max per LED.
    """
    led_ids = list(led_pixels.keys())
    n = len(led_ids)
    nrows = int(np.ceil(n / ncols))

    medians, q25, q75, maxima = [], [], [], []

    if pmin is None and pmax is None:
        pmin = min(np.min(v[v > 0]) for v in led_pixels.values())
        pmax = max(np.max(v[v > 0]) for v in led_pixels.values())
        if log:
            pmin = np.log(pmin)
            pmax = np.log(pmax)

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    plt.figure(figsize=figsize)

    for i, lab in enumerate(led_ids):
        ax = plt.subplot(nrows, ncols, i + 1)

        data = np.asarray(led_pixels[lab], dtype=float)
        data = data[data > 0]

        if log:
            data_plot = np.log(data)
            xlabel = "log(I)"
        else:
            data_plot = data
            xlabel = "Intensity (a. u.)"

        med = np.median(data_plot)
        q1 = np.percentile(data_plot, 25)
        q3 = np.percentile(data_plot, 75)
        mx = np.max(data_plot)

        medians.append(med)
        q25.append(q1)
        q75.append(q3)
        maxima.append(mx)

        _mode = scipy.stats.mode(data_plot)
        mu = np.mean(data_plot)
        var = np.std(data_plot, ddof=1)
        sk = skew(data_plot, bias=False)
        ku = kurtosis(data_plot, fisher=True, bias=False)

        ax.hist(data_plot, bins=bins, alpha=0.8, density=density, edgecolor=edgecolor)
        ax.axvline(med, color="k", linestyle="--", linewidth=1, label="median" if i == 0 else None)

        if show_mean_line:
            ax.axvline(mu, color="k", linestyle=":", linewidth=1, label="mean" if i == 0 else None)

        ax.set_title(f"LED {lab}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if density else "Counts")
        ax.set_xlim(pmin, pmax)

        if moment_box:
            txt = (
                f"Mode: {_mode[0]:.2f}\n"
                f"Mean: {mu:.2f}\n"
                f"Std: {var:.2f}\n"
                f"Skew: {sk:.2f}\n"
                f"Kurt: {ku:.2f}"
            )
            ax.text(
                0.98, 0.98, txt,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor=edgecolor)
            )

    plt.tight_layout()
    plt.show()

    if show_stats_plot:
        medians = np.asarray(medians)
        q25 = np.asarray(q25)
        q75 = np.asarray(q75)
        maxima = np.asarray(maxima)

        x = np.arange(1, len(led_ids) + 1)

        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.errorbar(x, medians, yerr=[medians - q25, q75 - medians], fmt="o", capsize=4)
        ax1.set_xlabel("LED index")
        ax1.set_ylabel("Median ± IQR of log(I)" if log else "Median ± IQR of I")
        ax1.plot(x, maxima, marker="s", linestyle="-", c="red", label="Max intensity (log)" if log else "Max intensity")
        ax1.set_title("LED-to-LED variability: median ± IQR and max")
        fig.legend(loc="upper center")
        plt.xticks([i for i in range(1, n + 1)])
        plt.show()

    if show_median_vs_max:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(medians, maxima)
        ax.set_xlabel("Median log(I)" if log else "Median I")
        ax.set_ylabel("Max log(I)" if log else "Max I")
        ax.set_title("Median vs max emission per LED")
        fig.tight_layout()
        plt.show()




def fit_peak_get_fwhm(wl, y, wl_roi=(440, 520), half_window_nm=6.0, min_snr=5.0):
    """
    Fit a narrow emission peak using a local Gaussian model with linear baseline.

    The model fitted is:

        I(λ) = A * exp(-(λ - λ0)^2 / (2σ^2)) + (mλ + b)

    where:
        A      = peak amplitude
        λ0     = peak center (continuous)
        σ      = Gaussian width parameter
        m, b   = linear baseline parameters

    The FWHM is computed as:

        FWHM = 2 * sqrt(2 ln 2) * σ ≈ 2.3548 σ

    Parameters
    ----------
    wl : array_like
        Wavelength axis in nm (can be non-uniformly spaced).
    y : array_like
        Intensity values for one spectrum.
    wl_roi : tuple(float, float)
        Spectral region of interest (nm) where the peak is expected.
    half_window_nm : float
        Half-width (in nm) of the local fitting window around the discrete peak maximum.
    min_snr : float
        Minimum signal-to-noise ratio required to attempt fitting.

    Returns
    -------
    wl0 : float
        Continuous peak position (nm). np.nan if fit fails.
    fwhm : float
        Full width at half maximum (nm). np.nan if fit fails.
    success_flag : bool
        True if fit converged and passed sanity checks.
    """
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)

    if wl.size != y.size or wl.size < 5:
        return np.nan, np.nan, False
    if not np.all(np.isfinite(y)):
        return np.nan, np.nan, False

    mroi = (wl >= wl_roi[0]) & (wl <= wl_roi[1])
    if not np.any(mroi):
        return np.nan, np.nan, False

    wl_r = wl[mroi]
    y_r = y[mroi]

    i0_local = int(np.argmax(y_r))
    wl0_guess = float(wl_r[i0_local])

    mwin = (wl >= wl0_guess - half_window_nm) & (wl <= wl0_guess + half_window_nm)
    xw = wl[mwin]
    yw = y[mwin]

    if xw.size < 7:
        return np.nan, np.nan, False

    k = max(2, xw.size // 6)
    edge_vals = np.concatenate([yw[:k], yw[-k:]])
    b0 = float(np.median(edge_vals))
    A0 = float(yw.max() - b0)
    if A0 <= 0:
        return np.nan, np.nan, False

    noise = float(np.std(edge_vals - np.median(edge_vals)))
    if noise > 0 and (A0 / noise) < min_snr:
        return np.nan, np.nan, False

    step = float(np.median(np.diff(xw)))
    sigma0 = max(step, half_window_nm / 6.0)
    # order follows utils.fitting.gaussian: amplitude, centre, sigma, background, slope
    p0 = [A0, wl0_guess, sigma0, b0, 0.0]

    lower = [0.0, xw.min(), step / 20, -np.inf, -np.inf]
    upper = [np.inf, xw.max(), half_window_nm, np.inf, np.inf]

    try:
        popt, _ = curve_fit(gaussian, xw, yw, p0=p0, bounds=(lower, upper), maxfev=4000)
        A, wl0, sigma, b, m = popt
        fwhm = fwhm_from_sigma(sigma)

        if not np.isfinite(wl0) or not np.isfinite(fwhm) or fwhm <= 0:
            return np.nan, np.nan, False

        return float(wl0), float(fwhm), True

    except Exception:
        return np.nan, np.nan, False


def refine_peak_parabola_nonuniform(wl, y, idx):
    """
    Quadratic (parabolic) sub-pixel peak refinement using 3 points in wavelength space.

    Works for non-uniform wavelength sampling.
    Returns (wl_peak, success_flag).
    """
    if idx <= 0 or idx >= len(y) - 1:
        return float(wl[idx]), False

    x = np.array([wl[idx - 1], wl[idx], wl[idx + 1]], dtype=float)
    z = np.array([y[idx - 1], y[idx], y[idx + 1]], dtype=float)

    if not np.all(np.isfinite(z)):
        return float(wl[idx]), False

    a, b, c = np.polyfit(x, z, deg=2)

    if a == 0 or not np.isfinite(a) or not np.isfinite(b):
        return float(wl[idx]), False

    wl_peak = -b / (2 * a)

    if wl_peak < x.min() or wl_peak > x.max():
        return float(wl[idx]), False

    return float(wl_peak), True






def extract_spectra_from_circles(
    xeol,
    centers_um: dict,
    radius_um: float,
    x_points: np.ndarray,
    y_points: np.ndarray,
    img_shape: tuple,
    *,
    labels=None,
    cmap: str = "tab20",
    figsize: tuple = (5, 5),
    vline_nm: float | list[float] | None = None,
    sharey: bool = False,
) -> tuple[dict, plt.Figure]:
    """Extract mean spectra from circular regions around user-defined centres.

    Parameters
    ----------
    xeol : XEOL
        XEOL object with `spectra` (Npoints, Nchannels) and `wl_array`.
    centers_um : dict
        ``{led_id: (x_um, y_um)}`` — centre of each circle in physical units (µm).
    radius_um : float
        Circle radius in µm.
    x_points, y_points : 1D arrays
        Physical axis arrays used to build the scan grid.
    img_shape : (nrows, ncols)
        Shape of the 2D map array (same as used in ``xeol.data.reshape(...)``).
    labels : 2D int array, optional
        Segmentation map drawn as background (for visual reference).
    cmap : str
        Colormap for the segmentation overlay (default "tab20").
    figsize : (w, h)
        Figure size in inches.
    vline_nm : float or list of float, optional
        Wavelength(s) in nm where a vertical dashed line is drawn on the
        individual and overlaid spectrum plots (e.g. ``460.0`` or ``[423.0, 460.0]``).

    Returns
    -------
    mean_spectra : dict
        ``{led_id: spectrum array}`` — mean spectrum within each circle.
    fig : plt.Figure
        Map with circles overlaid at the specified centres.

    Example
    -------
    >>> centers = {
    ...     1: (3.0,  3.0),
    ...     2: (13.0, 3.0),
    ...     3: (23.0, 3.0),
    ... }
    >>> spectra, fig = extract_spectra_from_circles(
    ...     xeol, centers, radius_um=2.0,
    ...     x_points=x_points, y_points=y_points,
    ...     img_shape=(101, 201), labels=labels,
    ... )
    """
    masks = elliptical_masks_um(centers_um, radius_um, x_points, y_points, img_shape)

    # ── extract spectra ───────────────────────────────────────────────────────
    mean_spectra = {}

    for led_id, mask in masks.items():
        specs = xeol.spectra[mask.flatten(), :]
        mean_spectra[led_id] = specs.mean(axis=0) if specs.shape[0] > 0 \
                               else np.full(xeol.wl_array.shape, np.nan)

    # ── plot ──────────────────────────────────────────────────────────────────
    img_data = xeol.data.reshape(img_shape)
    extent   = [x_points[0], x_points[-1], y_points[-1], y_points[0]]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_data, cmap='viridis', alpha=1, extent=extent, vmin=0)
    if labels is not None:
        ax.imshow(labels, cmap=cmap, alpha=0.45, extent=extent)

    for led_id, (x_c_um, y_c_um) in centers_um.items():
        ellipse = mpatches.Ellipse(
            (x_c_um, y_c_um), width=2 * radius_um, height=2 * radius_um,
            fill=False, edgecolor='white', linewidth=1.5, linestyle='--'
        )
        ax.add_patch(ellipse)
        ax.text(x_c_um, y_c_um, str(led_id),
                ha='center', va='center', fontsize=8,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4))

    ax.invert_yaxis()
    ax.set_aspect(1)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title(f'Circle extraction  r={radius_um} µm')
    plt.tight_layout()

    # ── espectros sobrepostos ──────────────────────────────────────────────────
    wl         = xeol.wl_array
    led_ids    = sorted(mean_spectra)
    tab_colors = plt.get_cmap('tab20').colors
    vlines     = ([vline_nm] if isinstance(vline_nm, (int, float))
                  else (vline_nm or []))

    def _draw_vlines(ax):
        for v in vlines:
            ax.axvline(v, color='k', linewidth=0.8, linestyle='--', alpha=0.6)

    fig_ov, ax_ov = plt.subplots(figsize=(10, 5))
    for led_id in led_ids:
        ax_ov.plot(wl, mean_spectra[led_id],
                   color=tab_colors[(led_id - 1) % 20],
                   label=f'LED {led_id}', lw=1.2)
    _draw_vlines(ax_ov)
    ax_ov.set_xlabel('Wavelength (nm)')
    ax_ov.set_ylabel('Intensity (a.u.)')
    ax_ov.set_title(f'Mean spectrum per LED — r={radius_um} µm')
    ax_ov.legend(fontsize=8, ncol=2)
    fig_ov.tight_layout()

    # ── subplots individuais ───────────────────────────────────────────────────
    n         = len(led_ids)
    ncols_plt = 4
    nrows_plt = (n + ncols_plt - 1) // ncols_plt
    fig_ind, axes = plt.subplots(nrows_plt, ncols_plt,
                                 figsize=(ncols_plt * 3.5, nrows_plt * 2.5),
                                 sharex=True, sharey=sharey)
    for ax2, led_id in zip(axes.flat, led_ids):
        ax2.plot(wl, mean_spectra[led_id],
                 color=tab_colors[(led_id - 1) % 20], lw=1.0)
        _draw_vlines(ax2)
        ax2.set_title(f'LED {led_id}', fontsize=9)
        ax2.set_xlabel('λ (nm)', fontsize=7)
        ax2.tick_params(labelsize=7)
    for ax2 in axes.flat[n:]:
        ax2.axis('off')
    fig_ind.suptitle(f'Mean spectrum per LED — r={radius_um} µm', fontsize=11)
    fig_ind.tight_layout()

    return mean_spectra, fig, fig_ov, fig_ind
