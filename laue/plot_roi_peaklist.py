"""
plot_roi_peaklist.py — Views for the simulation-driven COM peak list.

Companion to `laue.roi_peaklist`, kept separate the way `plot_sapphire_filter`
is kept apart from `sapphire_peak_filter`.

    plot_background_comparison   raw / background / difference, side by side
    plot_overview                whole detector: ROIs, forbidden zones, COMs
    plot_roi_zoom                one ROI: masked pixels, prediction, COM
    roi_panel                    the two above, driven by sliders
    plot_com_vs_boxsize          is the COM converged in boxsize?
    plot_com_vs_intensity        is the COM shift correlated with intensity?

The last two are diagnostics, not decoration. A COM displacement that has not
converged in boxsize, or that tracks the intensity map, is not a measurement of
lattice rotation — see the header of `laue.roi_peaklist`.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from laue._imaging import extract_crop
from laue.roi_peaklist import build_peaklist, exclusion_halves, measure_roi


def _clim(img, valid=None, lo=1.0, hi=99.5):
    data = img[valid] if valid is not None else img
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 1.0
    return float(np.percentile(data, lo)), float(np.percentile(data, hi))


# ── Background ────────────────────────────────────────────────────────────────

def plot_background_comparison(image, background, valid_mask=None, *,
                               figsize=(16, 5.5), log_scale=True):
    """Raw frame, estimated background, and the difference, on shared axes.

    The middle panel should look like the fluorescence envelope with no trace
    of individual spots. If spots are visible in it, sigma is too small and the
    subtraction is eating signal; if the right panel still has a broad gradient,
    sigma is too large.
    """
    image = np.asarray(image, dtype=float)
    background = np.asarray(background, dtype=float)
    diff = image - background

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)

    vmin, vmax = _clim(image, valid_mask)
    norm = LogNorm(vmin=max(vmin, 1.0), vmax=vmax) if log_scale else None
    im0 = axs[0].imshow(image, cmap="inferno", norm=norm,
                        vmin=None if norm else vmin, vmax=None if norm else vmax)
    axs[0].set_title("raw frame")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(background, cmap="inferno")
    axs[1].set_title(f"background\n(should show no spots)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    dmin, dmax = _clim(diff, valid_mask, 1.0, 99.9)
    im2 = axs[2].imshow(diff, cmap="inferno", vmin=0, vmax=dmax)
    axs[2].set_title("raw − background\n(should be flat between spots)")
    fig.colorbar(im2, ax=axs[2], fraction=0.046)

    fig.tight_layout()
    # A tuple, like every other function here. Returning a bare Figure makes
    # Jupyter render it a second time when the call is a cell's last
    # expression, since a Figure has a rich repr and a tuple does not.
    return fig, axs


# ── Whole-detector overview ───────────────────────────────────────────────────

def plot_overview(
    image,
    peaks,
    *,
    valid_mask=None,
    blacklist_xy=None,
    boxsize=15,
    exclusion_half=None,
    ax=None,
    figsize=(11, 10),
    show_rejected=True,
    max_boxes=2000,
    title=None,
):
    """The detector with material ROIs, substrate exclusion zones and COMs.

    Green boxes are the ROIs actually measured; grey boxes are ROIs that failed
    an acceptance test; red boxes are the forbidden zones around the simulated
    substrate spots. A white ``+`` marks each accepted centre of mass and a
    faint dot the prediction it came from, so the displacement being measured
    is visible directly.
    """
    image = np.asarray(image, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    vmin, vmax = _clim(image, valid_mask, 1.0, 99.97)
    ax.imshow(image, cmap="Greys_r", vmin=vmin, vmax=vmax, interpolation="nearest")

    if blacklist_xy is not None and len(blacklist_xy) and exclusion_half is not None:
        hx, hy = exclusion_halves(exclusion_half)
        for x, y in np.asarray(blacklist_xy)[:max_boxes]:
            ax.add_patch(Rectangle((x - hx - 0.5, y - hy - 0.5),
                                   2 * hx + 1, 2 * hy + 1,
                                   fill=True, facecolor="red", alpha=0.18,
                                   edgecolor="red", linewidth=0.4))

    acc = peaks[peaks["accepted"]]
    rej = peaks[~peaks["accepted"]]

    b = int(boxsize)

    def _boxes(df, colour, alpha):
        for x, y in df[["X_pred", "Y_pred"]].to_numpy()[:max_boxes]:
            ax.add_patch(Rectangle((x - b - 0.5, y - b - 0.5), 2 * b + 1, 2 * b + 1,
                                   fill=False, edgecolor=colour, linewidth=0.6,
                                   alpha=alpha))

    if show_rejected and len(rej):
        _boxes(rej, "0.55", 0.7)
    if len(acc):
        _boxes(acc, "lime", 0.9)
        ax.plot(acc["X_pred"], acc["Y_pred"], ".", color="deepskyblue", ms=2.5,
                alpha=0.8, label="simulated")
        ax.plot(acc["X"], acc["Y"], "+", color="white", ms=6, mew=0.9,
                label="centre of mass")

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title(title or (f"{len(acc)} accepted / {len(peaks)} predicted   "
                           f"boxsize={b} px"))
    ax.legend(loc="upper right", framealpha=0.6, fontsize=8)
    fig.tight_layout()
    return fig, ax


# ── Single ROI ────────────────────────────────────────────────────────────────

def plot_roi_zoom(image, valid_mask, center_xy, boxsize, *, ax=None,
                  measurement=None, figsize=(5.5, 5.5), **measure_kwargs):
    """One ROI, with the masked pixels, the prediction and the measured COM.

    Masked pixels are hatched in red: those are detector gaps and substrate
    exclusion zones, and they contributed nothing to the COM shown.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    m = measurement or measure_roi(image, valid_mask, center_xy, boxsize,
                                   **measure_kwargs)

    cx, cy = float(center_xy[0]), float(center_xy[1])
    centre_px = (int(round(cx)), int(round(cy)))
    crop, (row0, col0) = extract_crop(image, centre_px, int(boxsize), coords="numpy")
    vm, _ = extract_crop(valid_mask.astype(np.float32), centre_px, int(boxsize),
                         coords="numpy")
    vm = vm > 0.5

    extent = (col0 - 0.5, col0 + crop.shape[1] - 0.5,
              row0 + crop.shape[0] - 0.5, row0 - 0.5)
    vmin, vmax = _clim(crop, vm, 1.0, 99.5)
    ax.imshow(crop, cmap="inferno", vmin=vmin, vmax=vmax, extent=extent,
              interpolation="nearest")

    if (~vm).any():
        blocked = np.ma.masked_where(vm, np.ones_like(crop))
        ax.imshow(blocked, cmap="autumn", alpha=0.45, extent=extent,
                  interpolation="nearest")

    ax.plot(cx, cy, "o", mfc="none", mec="deepskyblue", ms=11, mew=1.6,
            label="simulated")
    if np.isfinite(m["X"]):
        ax.plot(m["X"], m["Y"], "+", color="white", ms=14, mew=2.0,
                label="centre of mass")

    state = "accepted" if m["accepted"] else f"REJECTED — {m['reject_reason']}"
    ax.set_title(
        f"{state}\n"
        f"dX={m['dX']:+.2f}  dY={m['dY']:+.2f} px   valid={100 * m['valid_frac']:.0f}%\n"
        f"I={m['total_counts']:.3g}  snr={m['snr']:.1f}  "
        f"aspect={m['aspect_ratio']:.2f}  theta={m['theta']:.0f}°",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    fig.tight_layout()
    return fig, ax


# ── Diagnostics ───────────────────────────────────────────────────────────────

def plot_com_vs_boxsize(image, valid_mask, center_xy, *, boxsizes=range(5, 41, 2),
                        ax=None, figsize=(7, 4.5), **measure_kwargs):
    """COM displacement as a function of ROI size — the convergence test.

    A trustworthy COM reaches a plateau: past the point where the box contains
    the whole spot, adding background-only pixels should not move it. A curve
    that keeps drifting means a pedestal is still present and is dragging the
    centre towards the middle of the box, which makes the measured displacement
    depend on the box rather than on the sample.
    """
    boxsizes = list(boxsizes)
    dx, dy, rejected = [], [], []
    for b in boxsizes:
        m = measure_roi(image, valid_mask, center_xy, b, **measure_kwargs)
        dx.append(m["dX"])
        dy.append(m["dY"])
        rejected.append(not m["accepted"])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(boxsizes, dx, "o-", label="dX")
    ax.plot(boxsizes, dy, "s-", label="dY")

    # A growing box eventually swallows a detector gap or an exclusion zone,
    # and past that point the curve is the COM of a truncated spot. It is still
    # drawn — the shape is informative — but never unmarked, or a plateau read
    # off this plot could be one the pipeline would refuse to measure.
    rej = np.asarray(rejected)
    if rej.any():
        b = np.asarray(boxsizes)
        ax.plot(b[rej], np.asarray(dx)[rej], "x", color="crimson", ms=8, mew=1.5)
        ax.plot(b[rej], np.asarray(dy)[rej], "x", color="crimson", ms=8, mew=1.5,
                label="rejected by measure_roi")

    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("boxsize (px)")
    ax.set_ylabel("COM − prediction (px)")
    ax.set_title(f"COM convergence at ({center_xy[0]:.0f}, {center_xy[1]:.0f})\n"
                 "flat = trustworthy; drifting = residual background",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_com_vs_intensity(peaks, *, figsize=(12, 4)):
    """Does the COM displacement correlate with intensity or background?

    It should not. The COM shift being tracked is a lattice rotation, which has
    no reason to follow how bright a spot is. A visible correlation means part
    of the displacement is produced by the measurement — an incompletely
    removed pedestal, or a threshold biting differently into a fainter spot —
    and the amount of it is the part that cannot be read as rotation.
    """
    df = peaks[peaks["accepted"]]
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    for ax, col, label in (
        (axs[0], "total_counts", "integrated intensity"),
        (axs[1], "bg_level", "ROI background level"),
        (axs[2], "valid_frac", "fraction of valid pixels"),
    ):
        x = df[col].to_numpy(dtype=float)
        y = df["dR"].to_numpy(dtype=float)
        ax.plot(x, y, ".", ms=4, alpha=0.6)
        if col == "total_counts":
            ax.set_xscale("log")
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() > 2:
            r = np.corrcoef(x[good], y[good])[0, 1]
            ax.set_title(f"r = {r:+.2f}", fontsize=10)
        ax.set_xlabel(label)
        ax.set_ylabel("|COM − prediction| (px)")
        ax.grid(alpha=0.3)

    fig.suptitle("COM displacement vs. measurement conditions — "
                 "a strong correlation means part of the shift is artefact",
                 fontsize=10)
    fig.tight_layout()
    return fig, axs


# ── Interactive panel ─────────────────────────────────────────────────────────

def roi_panel(
    image,
    sim,
    valid_mask,
    *,
    blacklist_xy=None,
    exclusion_half=None,
    boxsize=15,
    boxsize_range=(5, 40),
    figsize=(16, 8),
    **measure_kwargs,
):
    """Overview and per-spot zoom, with a boxsize slider and a spot selector.

    Requires ipywidgets and an interactive matplotlib backend
    (``%matplotlib widget``). Returns ``(fig, controls)``; keep a reference to
    both or the widget stops responding.

    The boxsize slider re-measures every ROI, so the whole peak list — and the
    acceptance counts in the title — update together with the zoom.
    """
    import ipywidgets as widgets
    from IPython.display import display

    state = {"peaks": None}

    fig = plt.figure(figsize=figsize)
    ax_ov = fig.add_subplot(1, 2, 1)
    ax_zoom = fig.add_subplot(1, 2, 2)

    spot_slider = widgets.IntSlider(
        value=0, min=0, max=max(len(sim) - 1, 0), step=1,
        description="spot", continuous_update=False,
        layout=widgets.Layout(width="45%"),
    )
    box_slider = widgets.IntSlider(
        value=int(boxsize), min=int(boxsize_range[0]), max=int(boxsize_range[1]),
        step=1, description="boxsize", continuous_update=False,
        layout=widgets.Layout(width="45%"),
    )
    info = widgets.HTML()

    def _recompute(b):
        state["peaks"] = build_peaklist(image, sim, valid_mask, b, **measure_kwargs)

    def _redraw(_=None):
        b = box_slider.value
        if state["peaks"] is None or int(state["peaks"]["boxsize"].iloc[0]) != b:
            _recompute(b)
        peaks = state["peaks"]

        ax_ov.clear()
        plot_overview(image, peaks, valid_mask=valid_mask,
                      blacklist_xy=blacklist_xy, boxsize=b,
                      exclusion_half=exclusion_half, ax=ax_ov)

        i = spot_slider.value
        row = peaks.iloc[i]
        ax_ov.plot(row["X_pred"], row["Y_pred"], "o", mfc="none", mec="yellow",
                   ms=14, mew=1.8)

        ax_zoom.clear()
        plot_roi_zoom(image, valid_mask, (row["X_pred"], row["Y_pred"]), b,
                      ax=ax_zoom, measurement=row.to_dict())

        hkl = (f"({int(row['h'])} {int(row['k'])} {int(row['l'])})"
               if "h" in peaks.columns else "")
        energy = f"{row['Energy']:.2f} keV" if "Energy" in peaks.columns else ""
        info.value = (f"<b>spot {i}</b> {hkl} {energy} &nbsp;|&nbsp; "
                      f"accepted: {int(peaks['accepted'].sum())} / {len(peaks)}")
        fig.canvas.draw_idle()

    spot_slider.observe(_redraw, names="value")
    box_slider.observe(_redraw, names="value")
    _redraw()

    controls = widgets.VBox([widgets.HBox([spot_slider, box_slider]), info])
    display(controls)
    return fig, controls
