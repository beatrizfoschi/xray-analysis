"""
mean_strain.py — Deviatoric strain per LED: means, variability, distributions.

Shares the circular-region geometry with ``emission.stats_utils`` via
``utils.regions.elliptical_masks_um``, but operates on a lauexplore Dataset
rather than an XEOL object.

Three entry points:
    mean_strain              mean tensor per LED + strain map with regions overlaid
    plot_strain_variability  std of each component per LED (matplotlib)
    plot_strain_histograms   interactive per-component histograms (plotly)

Usage
-----
>>> from laue.mean_strain import mean_strain
>>> df, fig = mean_strain(
...     dataset,
...     centers_um = {1: (3.0, 3.0), 2: (13.0, 3.0)},
...     radius_um  = 2.0,
... )
>>> print(df)   # mean strain components × 1e-4 per LED
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import skew, kurtosis, mode

from lauexplore.plots.base import _as_grid

from utils.regions import elliptical_masks_um


_COMPONENTS = {
    "e_xx": (0, 0),
    "e_yy": (1, 1),
    "e_zz": (2, 2),
    "e_xy": (0, 1),
    "e_xz": (0, 2),
    "e_yz": (1, 2),
}


def mean_strain(
    dataset,
    centers_um: dict,
    radius_um: float,
    *,
    ref_frame: str = "crystal",
    map_component: str = "e_zz",
    symmetric_cbar: bool = True,
    percentile_clip: tuple[float, float] = (2, 98),
    labels=None,
    cmap_strain: str = "RdBu_r",
    cmap_seg: str = "tab20",
    figsize: tuple[float, float] = (5, 5),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Compute mean deviatoric strain per LED from circular regions.

    Parameters
    ----------
    dataset : Dataset
        lauexplore Dataset with ``scan`` and strain data loaded.
    centers_um : dict
        ``{led_id: (x_um, y_um)}`` — centre of each circle in µm.
    radius_um : float
        Circle radius in µm.
    ref_frame : "crystal" or "sample"
        Reference frame for the deviatoric strain tensor (default "crystal").
    map_component : str
        Strain component displayed in the background map.
        One of: ``"e_xx"``, ``"e_yy"``, ``"e_zz"``, ``"e_xy"``, ``"e_xz"``, ``"e_yz"``.
    symmetric_cbar : bool
        If True (default), colour scale is symmetric around zero — appropriate
        for deviatoric strain which can be positive or negative.
    percentile_clip : (lo, hi)
        Percentiles used to set the colour scale limits.
    labels : 2D int array, optional
        Segmentation overlay drawn for visual reference.
    cmap_strain : str
        Colormap for the strain background map (default "RdBu_r").
    cmap_seg : str
        Colormap for the segmentation overlay (default "tab20").
    figsize : (w, h)
        Figure size in inches.

    Returns
    -------
    df : pd.DataFrame
        Indexed by ``led_id`` with columns ``n_points`` and
        ``mean_e_xx / yy / zz / xy / xz / yz`` in units of × 1e-4.
    fig : plt.Figure
        Strain map with circle extraction regions overlaid.
    """
    scan = dataset.scan
    if scan is None:
        raise ValueError("dataset.scan must not be None.")

    if ref_frame == "crystal":
        strain_tensors = dataset.deviatoric_strain_crystal_frame   # (N, 3, 3)
    elif ref_frame == "sample":
        strain_tensors = dataset.deviatoric_strain_sample_frame
    else:
        raise ValueError(f"ref_frame must be 'crystal' or 'sample', got '{ref_frame}'.")

    if map_component not in _COMPONENTS:
        raise ValueError(f"map_component must be one of {list(_COMPONENTS)}, got '{map_component}'.")

    # ── spatial grid ──────────────────────────────────────────────────────────
    ny, nx   = scan.nbypoints, scan.nbxpoints
    x_points = scan.xpoints * 1e3   # mm → µm
    y_points = scan.ypoints * 1e3
    # flat pixel index → scan index (safe for horizontal AND vertical scans)
    flat_to_scan = np.full(ny * nx, -1, dtype=int)
    for idx in range(scan.length):
        i, j = scan.index_to_ij(idx)
        flat_to_scan[j * nx + i] = idx

    masks = elliptical_masks_um(centers_um, radius_um, x_points, y_points, (ny, nx))

    # ── mean strain per LED ───────────────────────────────────────────────────
    r_hist, c_hist = _COMPONENTS[map_component]

    rows        = []
    raw_values  = {}   # {led_id: 1D array of map_component values × 1e-4}

    for led_id, mask in masks.items():
        x_c_um, y_c_um = centers_um[led_id]
        scan_indices = flat_to_scan[mask.flatten()]
        valid        = scan_indices[scan_indices >= 0]
        n            = len(valid)

        row = {"led_id": led_id, "n_points": n}
        if n > 0:
            tensors = strain_tensors[valid, :, :]   # (n, 3, 3)
            for comp, (r, c) in _COMPONENTS.items():
                vals_comp = tensors[:, r, c] * 1e4
                row[f"mean_{comp}"] = float(np.nanmean(vals_comp))
                row[f"std_{comp}"]  = float(np.nanstd(vals_comp, ddof=1))
            raw_values[led_id] = tensors[:, r_hist, c_hist] * 1e4
        else:
            for comp in _COMPONENTS:
                row[f"mean_{comp}"] = np.nan
                row[f"std_{comp}"]  = np.nan
            raw_values[led_id] = np.array([])

        rows.append(row)

    df = pd.DataFrame(rows).set_index("led_id")

    # ── background strain map ─────────────────────────────────────────────────
    r_idx, c_idx  = _COMPONENTS[map_component]
    strain_flat   = strain_tensors[:, r_idx, c_idx] * 1e4
    strain_grid   = _as_grid(strain_flat, scan)          # (ny, nx)

    finite        = strain_grid[np.isfinite(strain_grid)]
    lo            = float(np.nanpercentile(finite, percentile_clip[0]))
    hi            = float(np.nanpercentile(finite, percentile_clip[1]))
    if symmetric_cbar:
        bound = max(abs(lo), abs(hi))
        lo, hi = -bound, bound

    extent = [x_points[0], x_points[-1], y_points[-1], y_points[0]]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(strain_grid, cmap=cmap_strain, extent=extent,
                   vmin=lo, vmax=hi, origin="upper")
    if labels is not None:
        _labels = np.asarray(labels)
        if not np.issubdtype(_labels.dtype, np.number):
            raise TypeError(
                f"labels must be a numeric (integer) array from segment_leds, "
                f"got dtype {_labels.dtype}."
            )
        ax.imshow(_labels.astype(float), cmap=cmap_seg, alpha=0.3,
                  extent=extent, origin="upper")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=f"{map_component}  (× 1e-4)")

    for led_id, (x_c_um, y_c_um) in centers_um.items():
        ellipse = mpatches.Ellipse(
            (x_c_um, y_c_um), width=2 * radius_um, height=2 * radius_um,
            fill=False, edgecolor="white", linewidth=1.5, linestyle="--"
        )
        ax.add_patch(ellipse)
        ax.text(x_c_um, y_c_um, str(led_id),
                ha="center", va="center", fontsize=8,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.4))

    ax.invert_yaxis()
    ax.set_aspect(1)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(
        f"Mean strain per LED — {map_component}  r={radius_um} µm\n"
        f"ref frame: {ref_frame}"
    )
    plt.tight_layout()

    # ── per-LED strain histograms for map_component ───────────────────────────
    led_ids = sorted(centers_um.keys())
    n_leds  = len(led_ids)
    ncols_h = 4
    nrows_h = (n_leds + ncols_h - 1) // ncols_h

    # shared x-limits from all LEDs (robust 1–99 percentile)
    all_finite = np.concatenate([
        v[np.isfinite(v)] for v in raw_values.values() if v.size > 0
    ])
    if all_finite.size > 0:
        pmin = float(np.percentile(all_finite, 1))
        pmax = float(np.percentile(all_finite, 99))
    else:
        pmin, pmax = None, None

    fig_hist, axes_h = plt.subplots(
        nrows_h, ncols_h,
        figsize=(4 * ncols_h, 3 * nrows_h),
        squeeze=False,
    )

    for ax2, led_id in zip(axes_h.flat, led_ids):
        vals = raw_values.get(led_id, np.array([]))
        vals = vals[np.isfinite(vals)]

        if vals.size > 1:
            # FD bins computed on data clipped to the global range
            clipped = vals[(vals >= pmin) & (vals <= pmax)] if pmin is not None else vals
            edges   = np.histogram_bin_edges(clipped if clipped.size >= 2 else vals, bins="fd")
            nbins   = max(len(edges) - 1, 5)

            ax2.hist(vals, bins=nbins, alpha=0.8, edgecolor=None)

            med = float(np.median(vals))
            mu  = float(vals.mean())
            sd  = float(vals.std(ddof=1))
            sk  = float(skew(vals, bias=False))
            ku  = float(kurtosis(vals, fisher=True, bias=False))
            mo  = float(mode(vals).mode)

            ax2.axvline(med, color="k", linestyle="--", linewidth=1)
            ax2.axvline(mu,  color="k", linestyle=":",  linewidth=1)

            txt = (
                f"Mode: {mo:.2f}\n"
                f"Mean: {mu:.2f}\n"
                f"Std:  {sd:.2f}\n"
                f"Skew: {sk:.2f}\n"
                f"Kurt: {ku:.2f}"
            )
            ax2.text(
                0.98, 0.98, txt,
                transform=ax2.transAxes, ha="right", va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor=None),
            )

        if pmin is not None:
            ax2.set_xlim(pmin, pmax)
        ax2.set_title(f"LED {led_id}")
        ax2.set_xlabel(f"{map_component}  (× 1e-4)")
        ax2.set_ylabel("Counts")
        ax2.tick_params(labelsize=7)

    for ax2 in axes_h.flat[n_leds:]:
        ax2.axis("off")

    fig_hist.suptitle(
        f"Strain distribution per LED — {map_component}   r = {radius_um} µm",
        fontsize=11,
    )
    fig_hist.tight_layout()

    return df, fig, fig_hist


# ---------------------------------------------------------------------------
# Strain variability per LED
# ---------------------------------------------------------------------------

def plot_strain_variability(
    df: pd.DataFrame,
    *,
    ncols: int = 3,
    figsize_overview: tuple[float, float] | None = None,
    figsize_ind: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    """Plot std of each strain component per LED (from the circular extraction regions).

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``mean_strain`` — indexed by ``led_id``, must contain
        ``std_e_xx / yy / zz / xy / xz / yz`` columns (× 1e-4).
    ncols : int
        Number of columns for the individual-component subplots (default 3).
    figsize_overview : (w, h), optional
        Figure size for the overview plot (default auto).
    figsize_ind : (w, h), optional
        Figure size for the individual subplots (default auto).

    Returns
    -------
    fig_ov : plt.Figure
        Overview — all components on one axes.
    fig_ind : plt.Figure
        Individual subplot per component.
    """
    std_cols = [f"std_{c}" for c in _COMPONENTS]
    missing  = [c for c in std_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing columns: {missing}. "
            "Run mean_strain() with the current version to get std columns."
        )

    led_ids = list(df.index)
    n       = len(led_ids)
    x       = np.arange(1, n + 1)

    markers  = ["o", "s", "^", "D", "v", "P"]
    colors   = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ── overview: all components ──────────────────────────────────────────────
    if figsize_overview is None:
        figsize_overview = (max(6, n * 0.5 + 2), 4)

    fig_ov, ax_ov = plt.subplots(figsize=figsize_overview)

    for k, (comp, col) in enumerate(zip(_COMPONENTS, std_cols)):
        stds = df[col].values.astype(float)
        ax_ov.plot(
            x, stds,
            marker=markers[k % len(markers)],
            linestyle="-",
            color=colors[k % len(colors)],
            label=comp,
            markersize=5,
        )
        mean_std = float(np.nanmean(stds))
        ax_ov.axhline(
            mean_std,
            color=colors[k % len(colors)],
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

    ax_ov.set_xlabel("LED index")
    ax_ov.set_ylabel("Std of strain  (× 1e-4)")
    ax_ov.set_title("LED-to-LED strain variability — all components")
    ax_ov.legend(fontsize=8, ncol=2)
    ax_ov.set_xticks(x)
    ax_ov.set_xticklabels(led_ids)
    fig_ov.tight_layout()

    # ── individual subplots per component ─────────────────────────────────────
    nrows_ind = (len(_COMPONENTS) + ncols - 1) // ncols
    if figsize_ind is None:
        figsize_ind = (ncols * 4, nrows_ind * 3)

    fig_ind, axes_ind = plt.subplots(
        nrows_ind, ncols,
        figsize=figsize_ind,
        squeeze=False,
    )

    for k, (comp, col) in enumerate(zip(_COMPONENTS, std_cols)):
        ax = axes_ind.flat[k]
        stds     = df[col].values.astype(float)
        mean_std = float(np.nanmean(stds))

        ax.plot(x, stds, marker=markers[k % len(markers)],
                linestyle="-", color=colors[k % len(colors)], markersize=5)
        ax.axhline(
            mean_std,
            color="red",
            linestyle="-",
            linewidth=1.2,
            label=f"Mean = {mean_std:.2f}",
        )
        ax.set_title(comp)
        ax.set_xlabel("LED index")
        ax.set_ylabel("Std  (× 1e-4)")
        ax.set_xticks(x)
        ax.set_xticklabels(led_ids, fontsize=7)
        ax.legend(fontsize=7)

    for ax in axes_ind.flat[len(_COMPONENTS):]:
        ax.axis("off")

    fig_ind.suptitle("LED-to-LED strain variability — per component", fontsize=11)
    fig_ind.tight_layout()

    return fig_ov, fig_ind


# ── Interactive strain histograms (Plotly) ────────────────────────────────────
#
# Moved here from emission/stats_utils.py on 2026-08-11: deviatoric strain is
# Laue physics, and this was the only thing pulling plotly into the emission
# package. Unlike plot_strain_variability above, this takes raw 1-D arrays per
# component rather than the mean_strain DataFrame, and returns a plotly Figure.

def _nbins_for(nbins_spec, v, x_range, idx):
    """Resolve nbins for a single panel.

    nbins_spec can be:
      int         — same for all panels
      str         — numpy rule applied per panel: "auto", "fd", "sturges", "sqrt"
      list/tuple  — one value per panel (int or str), indexed by idx
    """
    spec = nbins_spec[idx] if isinstance(nbins_spec, (list, tuple)) else nbins_spec
    if isinstance(spec, str):
        clipped = v[(v >= x_range[0]) & (v <= x_range[1])]
        if clipped.size < 2:
            return 30
        edges = np.histogram_bin_edges(clipped, bins=spec)
        return max(len(edges) - 1, 5)
    return int(spec)
def plot_strain_histograms(
    strain_1d: dict,
    *,
    components=None,
    component_titles=None,
    nbins="fd",
    histnorm="probability",
    x_range=None,
    robust_percentile=99.5,
    show_stats=True,
    center_on_mean=False,
    show_zero_line=True,
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
    nbins : int, str, or list/tuple
        Number of histogram bins. Options:
        - ``int``         — same count for every panel.
        - ``str``         — numpy rule applied per panel: ``"auto"``, ``"fd"``
                            (Freedman-Diaconis, default), ``"sturges"``, ``"sqrt"``.
        - ``list``/``tuple`` — one value (int or str) per component.
    histnorm : str or None
        Plotly histnorm: "probability", "probability density", or None (counts).
    x_range : tuple(float, float), optional
        Common x-axis range for all panels. Auto-computed from robust_percentile if None.
    robust_percentile : float
        Percentile used to auto-compute x_range (default: 99.5).
    show_stats : bool
        Annotate each panel with mean, std, and N; draw a mean line.
    center_on_mean : bool
        If True, each panel's x-range is centered on that component's own mean
        (± robust_percentile spread around the mean) instead of sharing a single
        x_range symmetric around 0.
    show_zero_line : bool
        If True, draw a vertical red line at x=0 in each panel.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

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

    panel_x_range = {}
    if x_range is None:
        if center_on_mean:
            for c in components:
                v = data[c]
                mu = np.nanmean(v)
                spread = np.nanpercentile(np.abs(v - mu), robust_percentile)
                if not np.isfinite(spread) or spread == 0:
                    spread = np.nanmax(np.abs(v - mu))
                panel_x_range[c] = (mu - spread, mu + spread)
        else:
            allv = np.concatenate([data[c] for c in components], axis=0)
            lim = np.nanpercentile(np.abs(allv), robust_percentile)
            if not np.isfinite(lim) or lim == 0:
                lim = np.nanmax(np.abs(allv))
            x_range = (-lim, lim)
            for c in components:
                panel_x_range[c] = x_range
    else:
        for c in components:
            panel_x_range[c] = x_range

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
        rng = panel_x_range[c]

        fig.add_trace(
            go.Histogram(
                x=v,
                nbinsx=_nbins_for(nbins, v, rng, i),
                histnorm=histnorm,
                marker=dict(color=bar_fill, line=dict(color=bar_line, width=1)),
                hovertemplate="ε = %{x:.2e}<br>%{y:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=r, col=co
        )

        if show_zero_line:
            fig.add_vline(x=0, line_width=2, line_color="rgba(255,0,0,1)", row=r, col=co)
        fig.update_xaxes(range=list(rng), row=r, col=co)

        xa = "x" if i == 0 else f"x{i+1}"
        ya = "y" if i == 0 else f"y{i+1}"

        if show_stats:
            mu = np.nanmean(v)
            sig = np.nanstd(v)
            fig.add_annotation(
                x=0.02, y=0.98,
                xref=f"{xa} domain",
                yref=f"{ya} domain",
                text=f"μ={mu:.2f}<br>σ={sig:.2f}<br>N={v.size}",
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
