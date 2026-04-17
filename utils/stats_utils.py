import numpy as np
from scipy import ndimage as ndi
import skimage.filters as filters
from skimage.morphology import remove_small_objects, binary_opening, disk, binary_closing
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis, mode
import scipy.stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def segment_leds(
    image,
    threshold_percentile=30,
    otsu=False,
    min_area=300,
    opening_radius=2
):
    """
    Segment LEDs from a 2D emission map.

    Parameters
    ----------
    image : 2D array
        Emission map (LEDs only).
    threshold_percentile : float
        Percentile of non-zero pixels used to compute the threshold value,
        then passed as block_size to threshold_local.
    min_area : int
        Minimum region size in pixels (used for remove_small_objects and as
        block_size for threshold_local).
    opening_radius : int
        Radius for morphological opening disk.

    Returns
    -------
    labels : 2D int array
        Labeled image (0 = background, 1..N LEDs).
    regions : list
        regionprops objects for each labeled LED.
    """
    img = np.asarray(image, dtype=float)

    thr = np.percentile(img[img > 0], threshold_percentile)
    thr = filters.threshold_local(thr, block_size=min_area)
    if otsu:
        thr = filters.threshold_otsu(img)
    binary = img >= thr

    binary = binary_opening(binary, disk(opening_radius))
    binary = remove_small_objects(binary, min_size=min_area)

    labels = label(binary)
    regions = regionprops(labels, intensity_image=img)

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


def gauss_linbaseline(x, A, x0, sigma, m, b):
    """Gaussian peak with linear baseline: A * exp(-(x-x0)^2 / 2σ^2) + m*x + b."""
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + (m * x + b)


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
    p0 = [A0, wl0_guess, sigma0, 0.0, b0]

    lower = [0.0, xw.min(), step / 20, -np.inf, -np.inf]
    upper = [np.inf, xw.max(), half_window_nm, np.inf, np.inf]

    try:
        popt, _ = curve_fit(gauss_linbaseline, xw, yw, p0=p0, bounds=(lower, upper), maxfev=4000)
        A, wl0, sigma, m, b = popt
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma)

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


def plot_strain_histograms_plotly_1d(
    strain_1d: dict,
    *,
    components=None,
    component_titles=None,
    nbins=60,
    histnorm="probability",
    x_range=None,
    robust_percentile=99.5,
    show_stats=True,
    rows=2,
    cols=2,
    height=650,
    width=900,
    title="Deviatoric strain distribution (µLaue)"
):
    """
    Plot interactive histograms of deviatoric strain components using Plotly.

    Parameters
    ----------
    strain_1d : dict
        Maps component name to 1D array of strain values.
        Example keys: "e_xx_yy_over2", "e_zz", "e_xy", "e_xz_yz_over2".
    components : list, optional
        Subset of keys to plot. Defaults to all keys in strain_1d.
    component_titles : dict, optional
        Maps component keys to HTML-formatted axis titles.
    nbins : int
        Number of histogram bins.
    histnorm : str or None
        Plotly histnorm: "probability", "probability density", or None (counts).
    x_range : tuple(float, float), optional
        Common x-axis range for all panels. Auto-computed from robust_percentile if None.
    robust_percentile : float
        Percentile used to auto-compute x_range (default: 99.5).
    show_stats : bool
        Annotate each panel with mean, std, and N; draw a mean line.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if components is None:
        components = list(strain_1d.keys())

    if component_titles is None:
        component_titles = {
            "e_xx_yy_over2": r"ε<sub>(xx+yy)/2</sub>",
            "e_zz": r"ε<sub>zz</sub>",
            "e_xy": r"ε<sub>xy</sub>",
            "e_xz_yz_over2": r"ε<sub>(xz+yz)/2</sub>",
        }

    data = {}
    for c in components:
        v = np.asarray(strain_1d[c]).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError(f"No finite values found for component '{c}'.")
        data[c] = v

    if x_range is None:
        allv = np.concatenate([data[c] for c in components], axis=0)
        lim = np.nanpercentile(np.abs(allv), robust_percentile)
        if not np.isfinite(lim) or lim == 0:
            lim = np.nanmax(np.abs(allv))
        x_range = (-lim, lim)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[component_titles.get(c, c) for c in components],
        horizontal_spacing=0.12,
        vertical_spacing=0.18
    )

    bar_fill = "rgba(120,120,120,0.55)"
    bar_line = "rgba(50,50,50,0.9)"

    for i, c in enumerate(components):
        r = i // cols + 1
        co = i % cols + 1
        v = data[c]

        fig.add_trace(
            go.Histogram(
                x=v,
                nbinsx=nbins,
                histnorm=histnorm,
                marker=dict(color=bar_fill, line=dict(color=bar_line, width=1)),
                hovertemplate="ε = %{x:.2e}<br>%{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=r, col=co
        )

        fig.add_vline(x=0, line_width=2, line_color="rgba(255,0,0,1)", row=r, col=co)
        fig.update_xaxes(range=list(x_range), row=r, col=co)

        xa = "x" if i == 0 else f"x{i+1}"
        ya = "y" if i == 0 else f"y{i+1}"

        if show_stats:
            mu = np.nanmean(v)
            sig = np.nanstd(v)
            fig.add_annotation(
                x=0.02, y=0.98,
                xref=f"{xa} domain",
                yref=f"{ya} domain",
                text=f"μ={mu:.0e}<br>σ={sig:.1e}<br>N={v.size}",
                showarrow=False,
                align="left",
                font=dict(size=11, color="rgba(30,30,30,0.85)"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                borderpad=4
            )
            fig.add_vline(x=mu, line_width=2, line_color="rgba(0,0,255,1)", line_dash="dot", row=r, col=co)

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgba(0,0,255,1)", dash="dot"),
            name="μ",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        bargap=0.05,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        width=width,
        margin=dict(l=70, r=30, t=80, b=60),
        font=dict(family="Arial", size=14, color="rgba(20,20,20,1)")
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False,
        ticks="outside", ticklen=5, tickcolor="rgba(0,0,0,0.35)", title_text="strain"
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False,
        ticks="outside", ticklen=5, tickcolor="rgba(0,0,0,0.35)",
        title_text="probability" if histnorm else "counts"
    )

    return fig
