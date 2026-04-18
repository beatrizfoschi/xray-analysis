"""
roi_viewer.py — Interactive ROI viewer across a sequence of Laue images.

Shows a cropped region of the detector centred on a given spot, with two
panels: global scale (for frame-to-frame comparison) and local scale (to
reveal internal spot structure).  Navigate with a slider or by typing the
image index.

Usage
-----
>>> from laue.roi_viewer import roi_viewer
>>> roi_viewer(
...     img_source="path/to/tifs",
...     roi_center=(534, 993),     # (x, y) in XMAS 1-based coordinates
...     boxsize=20,
... )

>>> roi_viewer(
...     img_source="scan.h5",
...     h5_img_key="2.1/instrument/detector/data",
...     roi_center=(534, 993),
...     boxsize=20,
... )
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from lauexplore.image import read as read_image


def roi_viewer(
    img_source: str | Path,
    roi_center: tuple[int, int],
    boxsize: int,
    *,
    img_prefix: str = "img_",
    img_suffix: str = ".tif",
    img_index_pad: int = 4,
    h5_img_key: str | None = None,
    coords: str = "xmas",
    n_sample_vmax: int = 50,
    workers: int = 8,
    cmap: str = "hot",
    figsize: tuple[float, float] = (11, 5),
) -> tuple[plt.Figure, widgets.Widget]:
    """Interactive ROI viewer across a Laue image sequence.

    Parameters
    ----------
    img_source : str or Path
        TIF folder path, or HDF5 file containing detector images.
    roi_center : (x, y)
        Centre of the ROI.  Interpretation depends on ``coords``:
        - ``"xmas"``: 1-based (column, row) as reported by XMAS software.
        - ``"numpy"``: 0-based (column, row) in array convention.
    boxsize : int
        Half-side of the ROI: the crop is ``(2*boxsize+1) × (2*boxsize+1)``.
        Clips to image borders and zero-pads if the ROI extends outside.
    img_prefix, img_suffix, img_index_pad :
        TIF filename format: ``{prefix}{index:0>{pad}d}{suffix}``.
    h5_img_key : str, optional
        Dataset key inside ``img_source`` when it is an HDF5 file.
        Shape must be ``(n_images, height, width)``.
    coords : {"xmas", "numpy"}
        Coordinate convention for ``roi_center`` (default ``"xmas"``).
    n_sample_vmax : int
        Number of images sampled to compute the global colour scale.
    workers : int
        Number of threads for parallel sampling.
    cmap : str
        Matplotlib colormap (default ``"hot"``).
    figsize : (width, height)
        Figure size in inches.

    Returns
    -------
    (fig, ui) : plt.Figure, ipywidgets.Widget
    """
    img_source = Path(img_source)
    is_h5      = img_source.suffix in ('.h5', '.hdf5')

    if is_h5:
        if h5_img_key is None:
            raise ValueError("h5_img_key must be set when img_source is an HDF5 file.")
        with h5py.File(img_source, "r") as h5f:
            n_imgs = h5f[h5_img_key].shape[0]
    else:
        img_files = sorted(img_source.glob(f"{img_prefix}*{img_suffix}"))
        n_imgs    = len(img_files)
        if n_imgs == 0:
            raise FileNotFoundError(
                f"No files matching '{img_prefix}*{img_suffix}' found in {img_source}"
            )

    print(f"{n_imgs} images found.")

    # ── Coordinate conversion ─────────────────────────────────────────────────
    x, y      = roi_center
    cen_col   = (x - 1) if coords == "xmas" else x
    cen_row   = (y - 1) if coords == "xmas" else y

    # ── Image loading ─────────────────────────────────────────────────────────
    def _load_raw(index: int) -> np.ndarray:
        if is_h5:
            with h5py.File(img_source, "r") as h5f:
                return h5f[h5_img_key][index].astype(float)
        fname = img_source / f"{img_prefix}{index:0>{img_index_pad}d}{img_suffix}"
        return read_image(fname).astype(float)

    def _crop_roi(img: np.ndarray) -> np.ndarray:
        r0 = max(0, cen_row - boxsize)
        r1 = min(img.shape[0], cen_row + boxsize + 1)
        c0 = max(0, cen_col - boxsize)
        c1 = min(img.shape[1], cen_col + boxsize + 1)
        roi = img[r0:r1, c0:c1]
        pad_top    = max(0, boxsize - cen_row)
        pad_bottom = max(0, (cen_row + boxsize + 1) - img.shape[0])
        pad_left   = max(0, boxsize - cen_col)
        pad_right  = max(0, (cen_col + boxsize + 1) - img.shape[1])
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            roi = np.pad(roi, ((pad_top, pad_bottom), (pad_left, pad_right)))
        return roi

    def _load_roi(index: int) -> np.ndarray:
        return _crop_roi(_load_raw(index))

    # ── Global colour scale ───────────────────────────────────────────────────
    print(f"Computing global colour scale from {n_sample_vmax} images...")
    sample_idx = np.linspace(0, n_imgs - 1, min(n_sample_vmax, n_imgs), dtype=int)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sample_maxima = list(pool.map(lambda i: _load_roi(i).max(), sample_idx))
    vmax_global = float(np.percentile(sample_maxima, 95))
    print(f"Scale: 0 – {vmax_global:.0f} counts")

    # ── Widgets ───────────────────────────────────────────────────────────────
    slider  = widgets.IntSlider(
        value=0, min=0, max=n_imgs - 1, step=1,
        description="image:",
        layout=widgets.Layout(width="80%"),
        continuous_update=False,
    )
    idx_box = widgets.BoundedIntText(
        value=0, min=0, max=n_imgs - 1,
        description="index:",
        layout=widgets.Layout(width="150px"),
    )
    widgets.jslink((slider, "value"), (idx_box, "value"))

    # ── Figure ────────────────────────────────────────────────────────────────
    extent = [-boxsize - 0.5, boxsize + 0.5, boxsize + 0.5, -boxsize - 0.5]
    fig, (ax_global, ax_local) = plt.subplots(1, 2, figsize=figsize)
    plt.tight_layout(pad=2)

    def _update(change):
        idx = slider.value
        roi = _load_roi(idx)

        if is_h5:
            name = f"index {idx}"
        else:
            name = img_files[idx].name

        peak_pos = np.unravel_index(roi.argmax(), roi.shape)
        pr = peak_pos[0] - boxsize
        pc = peak_pos[1] - boxsize

        for ax in (ax_global, ax_local):
            ax.cla()

        ax_global.imshow(roi, origin="upper", cmap=cmap,
                         vmin=0, vmax=vmax_global, extent=extent)
        ax_global.axhline(0, color="cyan", lw=0.5, alpha=0.5)
        ax_global.axvline(0, color="cyan", lw=0.5, alpha=0.5)
        ax_global.plot(pc, pr, "+", color="lime", ms=10, mew=1.5)
        ax_global.set_title(f"{name}  |  global scale", fontsize=9)
        ax_global.set_xlabel("Δcol (px)")
        ax_global.set_ylabel("Δrow (px)")

        ax_local.imshow(roi, origin="upper", cmap=cmap,
                        vmin=np.percentile(roi, 5),
                        vmax=np.percentile(roi, 99.5),
                        extent=extent)
        ax_local.axhline(0, color="cyan", lw=0.5, alpha=0.5)
        ax_local.axvline(0, color="cyan", lw=0.5, alpha=0.5)
        ax_local.plot(pc, pr, "+", color="lime", ms=10, mew=1.5)
        ax_local.set_title(f"{name}  |  local scale", fontsize=9)
        ax_local.set_xlabel("Δcol (px)")

        fig.suptitle(
            f"ROI  X={roi_center[0]}, Y={roi_center[1]}  |  "
            f"boxsize={boxsize}  |  max={roi.max():.0f}  |  "
            f"img {idx}/{n_imgs - 1}",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    slider.observe(_update, names="value")
    _update(None)

    ui = widgets.HBox([slider, idx_box])
    display(ui)

    return fig, ui
