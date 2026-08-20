#!/usr/bin/env python3
"""
plot_sapphire_filter.py — Visualize which Laue peaks were classified as
sapphire (excluded) vs GaN (kept) on top of a diffraction image.

QA companion to sapphire_peak_filter.py: recomputes the same tolerance-based
match against a simulated sapphire blacklist and overlays both classes of
peaks on the raw frame, so the choice of --tol (and of the simulation
parameters) can be checked by eye.

Typical workflow (Jupyter notebook)
-----------------------------------
from laue.sapphire_peak_filter import simulate_sapphire_blacklist
from laue.plot_sapphire_filter import plot_frame

my_dict_mat = {"Sapphire_epi": ["Sapphire_epi", [4.758, 4.758, 12.991, 90, 90, 120], "Al2O3"]}
blacklist_xy = simulate_sapphire_blacklist(
    "Sapphire_epi",
    fit_path = "blc16817_paper_data/sapphire/fitfiles_sub_new/frame_00000_g0.fit",
    material_dictionary = my_dict_mat,
    Emin = 5, Emax = 25,
)

fig, ax = plot_frame(
    image        = "blc16817_paper_data/Pedro_SEG/eiger4m_0000.h5",
    h5_key       = "entry_0000/CRGIF/eiger4m/data",
    datfile      = "blc16817_paper_data/Pedro_SEG/datfiles_new/frame_00000.dat",
    blacklist_xy = blacklist_xy,
    tol          = 3.0,
)

A CLI entry point (``python plot_sapphire_filter.py --image ... --datfile ...
--fitfile ... --material ...``) is also available — see ``main()`` below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

try:
    # Package import (e.g. `from laue.plot_sapphire_filter import plot_frame`
    # in a notebook, with the repo root — the parent of laue/ — on sys.path).
    from .sapphire_peak_filter import (
        find_sapphire_mask, load_dat, load_fit_peaklist, simulate_sapphire_blacklist,
    )
except ImportError:
    # Standalone script (`python plot_sapphire_filter.py`): the script's own
    # directory is on sys.path[0], so the plain module name resolves.
    from sapphire_peak_filter import (
        find_sapphire_mask, load_dat, load_fit_peaklist, simulate_sapphire_blacklist,
    )


# ── Image loading ──────────────────────────────────────────────────────────────

from laue.readers import find_image_key as _find_image_key, imshow_detector as _imshow_detector


def load_image(path, h5_key=None, frame_index: int = 0):
    """Detector frame as float64 — the dtype this module's plots assumed."""
    from laue.readers import load_frame
    import numpy as np
    return load_frame(path, h5_key=h5_key, frame_index=frame_index, dtype=np.float64)




# ── Plot ───────────────────────────────────────────────────────────────────────



def plot_filter_result(
    image: np.ndarray,
    df,
    mask_removed: np.ndarray,
    *,
    blacklist_xy: np.ndarray | None = None,
    x_col: str = "peak_X",
    y_col: str = "peak_Y",
    vmin: float = 10.0,
    vmax: float | None = None,
    log_scale: bool = True,
    cmap: str = "Greys",
    kept_color: str = "#3fa34d",
    removed_color: str = "#e05252",
    blacklist_color: str = "#4a90d9",
    marker_size: float = 70,
    blacklist_marker_size: float = 25,
    roi: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    ax=None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a diffraction image with kept (GaN) and excluded (sapphire) peaks overlaid.

    Parameters
    ----------
    image : (H, W) array — the raw detector frame.
    df : DataFrame of GaN peaks (from sapphire_peak_filter.load_dat), unfiltered.
    mask_removed : bool array, True for peaks classified as sapphire.
    blacklist_xy : optional (N, 2) array of *all* simulated sapphire positions
        (not just the ones matched to a real peak) — plotted as small blue
        crosses. Useful to check by eye whether the simulation geometry lines
        up with real bright spots in the image: if the blue crosses sit on
        top of visible spots, the geometry is right; if they're offset from
        every real spot, the UB matrix, calibration, pixel_size or
        frame_shape passed to ``simulate_sapphire_blacklist`` is wrong.
    vmin / vmax : colour-scale bounds in counts. vmax=None -> 99.9th percentile.
    roi : (x0, x1, y0, y1) pixel region to zoom into (default: full frame).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure

    _imshow_detector(ax, image, vmin=vmin, vmax=vmax, log_scale=log_scale, cmap=cmap)

    if blacklist_xy is not None:
        blacklist_xy = np.asarray(blacklist_xy, dtype=float)
        if len(blacklist_xy) > 0:
            ax.scatter(blacklist_xy[:, 0], blacklist_xy[:, 1],
                       s=blacklist_marker_size, marker="+", color=blacklist_color,
                       linewidths=0.9, alpha=0.8, zorder=2,
                       label=f"Simulated sapphire spots ({len(blacklist_xy)})")

    x, y = df[x_col].to_numpy(), df[y_col].to_numpy()
    kept = ~mask_removed
    n_kept, n_removed = int(kept.sum()), int(mask_removed.sum())

    ax.scatter(x[kept], y[kept], s=marker_size, facecolors="none",
               edgecolors=kept_color, linewidths=1.2, zorder=3,
               label=f"GaN — kept ({n_kept})")
    ax.scatter(x[mask_removed], y[mask_removed], s=marker_size, marker="x",
               color=removed_color, linewidths=1.4, zorder=3,
               label=f"Sapphire — excluded ({n_removed})")

    if roi is not None:
        x0, x1, y0, y1 = roi
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)  # inverted: origin="upper" has row 0 at the top

    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    return fig, ax


def plot_frame(
    image: str | Path,
    datfile: str | Path,
    blacklist_xy: np.ndarray,
    *,
    h5_key: str | None = None,
    frame_index: int = 0,
    tol: float = 3.0,
    x_col: str = "peak_X",
    y_col: str = "peak_Y",
    show_blacklist: bool = True,
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """One-call convenience wrapper for notebook use: load image + peaks, classify, plot.

    Parameters
    ----------
    image : diffraction image (.h5/.hdf5 or fabio-readable format).
    datfile : GaN .dat peak file (unfiltered).
    blacklist_xy : (N, 2) array of sapphire (x, y) pixel positions — typically
        from ``sapphire_peak_filter.simulate_sapphire_blacklist``.
    h5_key : dataset key inside the H5 file (default: auto-detect).
    frame_index : frame index if the H5 dataset is 3D.
    tol : matching tolerance in pixels (default 3.0).
    show_blacklist : if True (default), also overlay every simulated
        sapphire position as a small blue cross — not just the ones that
        matched a real peak. If the crosses don't sit on real bright spots
        in the image, the simulation geometry (UB/calib/pixel_size/
        frame_shape) is wrong, even if the matching count looks plausible.
    **plot_kwargs : forwarded to plot_filter_result (e.g. roi, vmin, vmax, title).

    Returns
    -------
    (fig, ax)
    """
    img = load_image(image, h5_key=h5_key, frame_index=frame_index)
    _, df = load_dat(datfile)
    blacklist_xy = np.asarray(blacklist_xy, dtype=float)
    peaks_xy = df[[x_col, y_col]].to_numpy(dtype=float)
    mask_removed = find_sapphire_mask(peaks_xy, blacklist_xy, tol)

    plot_kwargs.setdefault("title", f"{Path(datfile).name}   (tol = {tol} px)")
    if show_blacklist:
        plot_kwargs.setdefault("blacklist_xy", blacklist_xy)
    return plot_filter_result(img, df, mask_removed, x_col=x_col, y_col=y_col, **plot_kwargs)


def plot_fit_frame(
    image: str | Path,
    fit_path: str | Path,
    *,
    h5_key: str | None = None,
    frame_index: int = 0,
    use: str = "exp",
    label_hkl: bool = False,
    color: str = "#e05252",
    marker_size: float = 70,
    vmin: float = 10.0,
    vmax: float | None = None,
    log_scale: bool = True,
    cmap: str = "Greys",
    roi: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    ax=None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a diffraction image with the indexed spots from a .fit file overlaid.

    Quick visual QA for a LaueTools indexation result: check that the fitted
    (h k l) reflections line up with real spots on the raw frame.

    Parameters
    ----------
    image : diffraction image (.h5/.hdf5 or fabio-readable format).
    fit_path : LaueTools .fit indexation file.
    h5_key, frame_index : image loading options, as in ``plot_frame``.
    use : {"exp", "theo"} — plot the experimental (peak-search) or
        theoretical (model-predicted) spot positions (default "exp"). A
        large exp/theo mismatch on the same spot flags a poor fit.
    label_hkl : if True, annotate each spot with its (h k l) Miller indices.
    roi : (x0, x1, y0, y1) pixel region to zoom into (default: full frame).

    Returns
    -------
    (fig, ax)
    """
    img = load_image(image, h5_key=h5_key, frame_index=frame_index)
    peaks = load_fit_peaklist(fit_path)
    x_key, y_key = ("Xexp", "Yexp") if use == "exp" else ("Xtheo", "Ytheo")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure
    _imshow_detector(ax, img, vmin=vmin, vmax=vmax, log_scale=log_scale, cmap=cmap)

    x = peaks[x_key].to_numpy(dtype=float)
    y = peaks[y_key].to_numpy(dtype=float)
    ax.scatter(x, y, s=marker_size, facecolors="none", edgecolors=color,
               linewidths=1.2, zorder=3, label=f"Indexed spots ({len(peaks)})")

    if label_hkl:
        for xi, yi, row in zip(x, y, peaks[["h", "k", "l"]].itertuples(index=False)):
            ax.annotate(f"({int(row.h)} {int(row.k)} {int(row.l)})",
                        xy=(xi, yi), xytext=(xi + 12, yi - 12),
                        fontsize=7, color=color, ha="left", va="center")

    if roi is not None:
        x0, x1, y0, y1 = roi
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)  # inverted: origin="upper" has row 0 at the top

    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_title(title or f"{Path(fit_path).name}   ({use})")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
    return fig, ax


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image", required=True,
                     help="Diffraction image (.h5/.hdf5 or fabio-readable format).")
    ap.add_argument("--h5-key", default=None,
                     help="Dataset key inside the H5 file (default: auto-detect).")
    ap.add_argument("--frame-index", type=int, default=0,
                     help="Frame index if the H5 dataset is 3D (default: 0).")
    ap.add_argument("--datfile", required=True, help="GaN .dat peak file (unfiltered).")
    ap.add_argument("--material", required=True,
                     help="Material key in LaueTools' built-in materials dictionary. "
                          "For a custom material, use the Python API instead: "
                          "simulate_sapphire_blacklist(..., material_dictionary={...}).")
    ap.add_argument("--fitfile", default=None,
                     help="Sapphire .fit file to derive UB matrix, calibration, pixel size "
                          "and frame shape from (optional if all of --ub/--calib/"
                          "--pixel-size/--frame-shape are given explicitly).")
    ap.add_argument("--ub", type=float, nargs=9, default=None,
                     metavar=("UB11", "UB12", "UB13", "UB21", "UB22", "UB23", "UB31", "UB32", "UB33"),
                     help="3x3 UB matrix, row-major (overrides --fitfile).")
    ap.add_argument("--calib", type=float, nargs=5, default=None,
                     metavar=("DIST", "XCEN", "YCEN", "XBET", "XGAM"),
                     help="Detector calibration parameters (overrides --fitfile).")
    ap.add_argument("--pixel-size", type=float, default=None,
                     help="Detector pixel size in mm (overrides --fitfile / --camera-label).")
    ap.add_argument("--frame-shape", type=int, nargs=2, default=None,
                     metavar=("N_ROWS", "N_COLS"),
                     help="Detector frame shape in pixels (overrides --fitfile / --camera-label).")
    ap.add_argument("--camera-label", default=None,
                     help="LaueTools camera key (e.g. EIGER_4M), used only if pixel size / "
                          "frame shape are still unknown after --fitfile / explicit overrides.")
    ap.add_argument("--emin", type=float, default=5.0,
                     help="Minimum simulated energy in keV (default: 5.0).")
    ap.add_argument("--emax", type=float, default=25.0,
                     help="Maximum simulated energy in keV (default: 25.0).")
    ap.add_argument("--tol", type=float, default=3.0,
                     help="Matching tolerance in pixels (default: 3.0).")
    ap.add_argument("--x-col", default="peak_X")
    ap.add_argument("--y-col", default="peak_Y")
    ap.add_argument("--roi", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                     default=None, help="Zoom to a pixel region (default: full frame).")
    ap.add_argument("--vmin", type=float, default=10.0)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--linear", action="store_true",
                     help="Use a linear colour scale instead of log (default: log).")
    ap.add_argument("--hide-blacklist", action="store_true",
                     help="Do not overlay all simulated sapphire positions "
                          "(default: shown, as small blue crosses).")
    ap.add_argument("--output", default=None,
                     help="Save figure to this path instead of showing it interactively.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 9})

    ub_matrix = np.array(args.ub).reshape(3, 3) if args.ub else None
    frame_shape = tuple(args.frame_shape) if args.frame_shape else None

    blacklist_xy = simulate_sapphire_blacklist(
        args.material,
        fit_path=args.fitfile,
        ub_matrix=ub_matrix,
        calibration_parameters=args.calib,
        pixel_size=args.pixel_size,
        frame_shape=frame_shape,
        camera_label=args.camera_label,
        Emin=args.emin, Emax=args.emax,
    )

    fig, ax = plot_frame(
        args.image, args.datfile, blacklist_xy,
        h5_key=args.h5_key, frame_index=args.frame_index,
        tol=args.tol, x_col=args.x_col, y_col=args.y_col,
        show_blacklist=not args.hide_blacklist,
        vmin=args.vmin, vmax=args.vmax, log_scale=not args.linear,
        roi=args.roi,
    )

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
