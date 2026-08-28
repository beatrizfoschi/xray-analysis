"""
spot_fit_viewer.py — Click a position on a `spot_fit` map to see the fit behind it.

    %matplotlib widget
    from laue.spot_fit_viewer import interactive_fit_map

    interactive_fit_map(df, aligned, metric='n_resolved')

Unlike `satellite.visualize.interactive_map`, which re-runs its detection on the
clicked position, this rebuilds the model from the row the map was drawn from.
The panel therefore cannot disagree with the map, there are no fit parameters to
keep in sync between the two calls, and the click is instant.

What to look for
----------------
The markers separate the two counts that `spot_fit` reports. Filled crosses are
the components that resolved — `n_resolved`, the physical sub-peaks. Hollow grey
ones are the rest of `n_fitted`: what the fit needed to describe the ROI, mostly
Gaussians absorbing a heavy tail. A row of hollow markers strung along a streak
is the signature of a spot that is elongated rather than split.

They are drawn but not discarded from the model: the residual panel shows the
full fitted model, because the dropped components are doing real work in it —
refitting without them measurably degrades the recovered separation.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from laue.spot_fit import model_from_result

_INDEX_COLS = ("i", "j", "frame_idx", "x_um", "y_um", "status")


def metric_grid(
    df: pd.DataFrame,
    metric: str,
) -> Tuple[np.ndarray, list, Tuple[float, float, float, float]]:
    """A metric as a 2-D array over the scan grid, plus its imshow extent.

    Returns
    -------
    grid : (nby, nbx) array, ready for ``imshow(..., origin='lower')``
    lookup : row index of each grid cell, -1 where the scan has no row
    extent : (x_min, x_max, y_min, y_max) in microns
    """
    if metric not in df.columns:
        available = [c for c in df.columns if c not in _INDEX_COLS]
        raise ValueError(f"Column {metric!r} not in df. Available: {available}")
    for col in ("i", "j"):
        if col not in df.columns:
            raise ValueError(
                f"df has no {col!r} column — it needs the grid coordinates that "
                "`read_roi_stack` returns as `index`, or that `run_spot_pipeline` "
                "puts in itself."
            )

    i_vals = np.sort(df["i"].unique())
    j_vals = np.sort(df["j"].unique())
    i_min, j_min = int(i_vals.min()), int(j_vals.min())

    grid = np.full((len(j_vals), len(i_vals)), np.nan)
    lookup = np.full((len(j_vals), len(i_vals)), -1, dtype=int)
    for row_pos, (_, row) in enumerate(df.iterrows()):
        gi = int(row["i"]) - i_min
        gj = int(row["j"]) - j_min
        grid[gj, gi] = row[metric]
        lookup[gj, gi] = row_pos

    if "x_um" in df.columns and "y_um" in df.columns:
        x, y = np.sort(df["x_um"].unique()), np.sort(df["y_um"].unique())
        extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    else:
        extent = (i_min - 0.5, i_min + len(i_vals) - 0.5,
                  j_min - 0.5, j_min + len(j_vals) - 0.5)
    return grid, lookup, extent


def draw_fit_panels(
    axes: Sequence[plt.Axes],
    roi: np.ndarray,
    row: dict,
    *,
    shared_sigma: bool = True,
    rotation: bool = False,
) -> None:
    """Draw data / model / residual for one position onto three axes."""
    for ax in axes:
        ax.clear()
        ax.axis("off")

    n_fitted = int(row.get("n_fitted", 0) or 0)
    if n_fitted < 1:
        axes[0].imshow(roi, cmap="viridis", origin="lower")
        axes[0].set_title("no fit at this position", fontsize=9)
        return

    model = model_from_result(row, roi.shape, shared_sigma, rotation)
    diff = roi - model
    vmax = float(np.nanmax(roi))

    axes[0].imshow(roi, cmap="viridis", vmin=0, vmax=vmax, origin="lower")
    axes[1].imshow(model, cmap="viridis", vmin=0, vmax=vmax, origin="lower")
    lim = float(np.nanmax(np.abs(diff))) or 1.0
    axes[2].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")

    n_res = int(row.get("n_resolved", 0) or 0)
    for c in range(1, n_fitted + 1):
        style = (dict(marker="+", color="red") if c <= n_res
                 else dict(marker="x", color="0.6"))
        for ax in axes[:2]:
            ax.plot(row[f"x{c}"], row[f"y{c}"], linestyle="none",
                    ms=10, mew=2, **style)

    axes[0].set_title(f"data — fitted {n_fitted}, resolved {n_res}", fontsize=9)
    axes[1].set_title("model", fontsize=9)
    axes[2].set_title(f"residual  (χ² = {row.get('chi2', float('nan')):.1f})",
                      fontsize=9)


def summarise(row: dict) -> str:
    """One-block text summary of a fitted position."""
    def g(key, fmt="{:.2f}"):
        v = row.get(key, float("nan"))
        try:
            return "—" if v is None or np.isnan(v) else fmt.format(v)
        except (TypeError, ValueError):
            return str(v)

    n_fit = int(row.get("n_fitted", 0) or 0)
    n_res = int(row.get("n_resolved", 0) or 0)
    lines = [
        f"position  (i, j) = ({int(row['i'])}, {int(row['j'])})",
        f"fitted {n_fit} Gaussian(s), {n_res} resolved",
        f"χ² = {g('chi2', '{:.1f}')}   bg = {g('bg', '{:.0f}')}",
        "",
    ]
    if n_res >= 2:
        lines += [
            f"separation  {g('separation')} px",
            f"orientation {g('orientation', '{:.1f}')}°",
            f"A₂/(A₁+A₂)  {g('ratio', '{:.3f}')}",
        ]
    else:
        lines.append("no separation — fewer than two components resolved")
    lines.append("")
    for c in range(1, n_fit + 1):
        mark = "*" if c <= n_res else " "
        lines.append(
            f" {mark} peak {c}: ({g(f'x{c}')}, {g(f'y{c}')})  "
            f"A = {g(f'A{c}', '{:.0f}')}"
        )
    if n_fit > n_res:
        lines.append("")
        lines.append("(unmarked peaks did not resolve)")
    return "\n".join(lines)


def interactive_fit_map(
    df: pd.DataFrame,
    stack: np.ndarray,
    *,
    metric: str = "n_resolved",
    shared_sigma: bool = True,
    rotation: bool = False,
    cmap: str = "viridis",
    percentile_clip: Tuple[float, float] = (2, 98),
    figsize: Tuple[float, float] = (13, 4.5),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """Map a `spot_fit` column; click a position to inspect the fit there.

    Parameters
    ----------
    df : DataFrame
        Output of `analyse_stack` or `run_spot_pipeline` with ``fit_spot``.
        Needs the ``i``/``j`` grid coordinates.
    stack : (n_positions, h, w) array
        The ROIs the fit was run on, in the row order of ``df``. When ``df`` came
        from ``analyse_stack(aligned, index)``, that is ``aligned`` — pass the
        *same* array, aligned and normalised the same way, or the residual panel
        will be drawn against a different image than the fit saw.
    metric : str
        Column to map. ``n_resolved`` by default: the sub-peak count.
    shared_sigma, rotation : bool
        Must match what the fit was run with, since they decide how the stored
        parameters are read back.
    vmin, vmax : float, optional
        Fixed colour limits; otherwise ``percentile_clip`` sets them. Useful for
        an integer metric, where ``vmin=1`` keeps the colours comparable between
        spots.

    Needs the ``ipympl`` backend (``%matplotlib widget``) to receive clicks;
    without it the map still draws and the panels stay empty.
    """
    stack = np.asarray(stack)
    if len(stack) != len(df):
        raise ValueError(
            f"stack has {len(stack)} ROIs but df has {len(df)} rows — pass the "
            "array the fit was run on, in the same order."
        )

    grid, lookup, extent = metric_grid(df, metric)
    lo = vmin if vmin is not None else np.nanpercentile(grid, percentile_clip[0])
    hi = vmax if vmax is not None else np.nanpercentile(grid, percentile_clip[1])

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 1, 1, 1, 0.9])
    ax_map = fig.add_subplot(gs[0, 0])
    panels = [fig.add_subplot(gs[0, k]) for k in (1, 2, 3)]
    ax_txt = fig.add_subplot(gs[0, 4])
    ax_txt.axis("off")

    im = ax_map.imshow(grid, origin="lower", extent=extent, aspect="equal",
                       cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    ax_map.set_title(f"{metric} — click a position", fontsize=10)
    ax_map.set_xlabel("x (µm)")
    ax_map.set_ylabel("y (µm)")

    marker, = ax_map.plot([], [], "s", mfc="none", mec="red", ms=9, mew=1.6)
    text = ax_txt.text(0.0, 1.0, "click a position on the map",
                       va="top", ha="left", fontsize=8, family="monospace",
                       transform=ax_txt.transAxes)

    nby, nbx = grid.shape
    x0, x1, y0, y1 = extent
    # imshow with an extent puts cell centres at the ends, so a click maps back
    # through the cell width rather than by a plain rescale of the span.
    dx = (x1 - x0) / max(nbx - 1, 1)
    dy = (y1 - y0) / max(nby - 1, 1)

    def _on_click(event):
        if event.inaxes is not ax_map or event.xdata is None:
            return
        gi = int(round((event.xdata - x0) / dx)) if nbx > 1 else 0
        gj = int(round((event.ydata - y0) / dy)) if nby > 1 else 0
        if not (0 <= gi < nbx and 0 <= gj < nby):
            return
        row_pos = int(lookup[gj, gi])
        if row_pos < 0:
            return

        row = df.iloc[row_pos].to_dict()
        draw_fit_panels(panels, np.asarray(stack[row_pos], dtype=float), row,
                        shared_sigma=shared_sigma, rotation=rotation)
        marker.set_data([x0 + gi * dx], [y0 + gj * dy])
        text.set_text(summarise(row))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    return fig
