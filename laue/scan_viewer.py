"""
scan_viewer.py
==============

Interactive viewer: fluorescence map + diffraction image for a Laue raster scan.

Usage
-----
>>> from laue.scan_viewer import scan_viewer
>>> scan_viewer(
...     h5_path="scan_001.h5",
...     fluo_element="Ga",
...     img_source="path/to/tifs",          # TIF folder
...     roi_y=slice(1360, 1420),
...     roi_x=slice(1230, 1280),
... )

>>> scan_viewer(
...     h5_path="scan_001.h5",
...     fluo_element="Ga",
...     img_source="scan_001.h5",           # H5 with images
...     h5_img_key="2.1/instrument/detector/data",
...     roi_y=slice(1360, 1420),
...     roi_x=slice(1230, 1280),
... )
"""

from __future__ import annotations

from pathlib import Path

import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from IPython.display import display

from lauexplore.emission import Fluorescence
from lauexplore.image import read as read_image
from lauexplore.plots.base import _as_grid
from lauexplore.scan import Scan


def scan_viewer(
    h5_path: str | Path,
    fluo_element: str,
    img_source: str | Path,
    roi_y: slice | None = None,
    roi_x: slice | None = None,
    *,
    scan_number: int = 1,
    img_prefix: str = "img_",
    img_suffix: str = ".tif",
    img_index_pad: int = 4,
    h5_img_key: str | None = None,
    normalize_to_monitor: bool = True,
    sigmoid_cutoff: float = 0.5,
    sigmoid_gain: float = 5.0,
) -> tuple[plt.Figure, widgets.Widget]:
    """Interactive fluorescence map + diffraction image viewer.

    Parameters
    ----------
    h5_path:
        Path to the scan HDF5 file.
    fluo_element:
        Element symbol as stored in the HDF5 (e.g. ``"Ga"``).
    img_source:
        TIF folder path, or HDF5 file path containing detector images.
    roi_y, roi_x:
        Detector ROI slices in [y, x] (row, column) convention.
        Pass ``None`` (default) to use the full image.
    scan_number:
        Scan entry number inside the HDF5 (default 1).
    img_prefix, img_suffix, img_index_pad:
        TIF filename format: ``{prefix}{index:0>{pad}d}{suffix}``.
    h5_img_key:
        Dataset key inside ``img_source`` when it is an HDF5 file.
        Indexed as ``h5f[h5_img_key][file_index]``.
    normalize_to_monitor:
        Divide fluorescence by monitor counts (default True).
    sigmoid_cutoff, sigmoid_gain:
        Parameters for ``skimage.exposure.adjust_sigmoid``.
    """
    h5_path  = Path(h5_path)
    img_source = Path(img_source)

    scan = Scan.from_h5(h5_path, scan_number)
    fluo = Fluorescence.from_h5(h5_path, fluo_element, scan_number,
                                normalize_to_monitor=normalize_to_monitor)

    fluo_grid = _as_grid(fluo.data, scan)
    motor_x   = scan.xpoints * 1e3          # mm → µm, shape (nbxpoints,)
    motor_y   = scan.ypoints * 1e3          # mm → µm, shape (nbypoints,)

    def _load_image(file_index: int) -> np.ndarray:
        if img_source.suffix in ('.h5', '.hdf5'):
            if h5_img_key is None:
                raise ValueError("h5_img_key must be set when img_source is an HDF5 file.")
            with h5py.File(img_source) as h5f:
                raw = h5f[h5_img_key][file_index]
        else:
            fname = img_source / f"{img_prefix}{file_index:0>{img_index_pad}d}{img_suffix}"
            raw = read_image(fname)
        crop = raw[roi_y, roi_x] if (roi_y is not None or roi_x is not None) else raw
        return sk.exposure.adjust_sigmoid(crop, cutoff=sigmoid_cutoff, gain=sigmoid_gain)

    def _calc_lims(im: np.ndarray, m: float = 3.0) -> tuple[float, float]:
        return im.mean() - m * im.std(), im.mean() + m * im.std()

    row_slider = widgets.IntSlider(
        value=0, min=0, max=scan.nbypoints - 1, step=1, description="Row"
    )
    col_slider = widgets.IntSlider(
        value=0, min=0, max=scan.nbxpoints - 1, step=1, description="Col"
    )

    fig, (ax_map, ax_img) = plt.subplots(
        2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [1, 5]}
    )

    def _update(row: int, col: int) -> None:
        ax_map.cla()
        ax_img.cla()

        ax_map.set_aspect("equal")
        ax_map.set_xlabel("Position [μm]")
        ax_map.set_ylabel("Position [μm]")
        ax_map.set_title(f"{fluo_element} fluorescence")
        ax_map.pcolormesh(motor_x, motor_y, fluo_grid, cmap="inferno")
        ax_map.hlines(motor_y[row], motor_x.min(), motor_x.max(), color="blue")
        ax_map.vlines(motor_x[col], motor_y.min(), motor_y.max(), color="blue")

        file_index = scan.ij_to_index(col, row)
        img = _load_image(file_index)
        imin, imax = _calc_lims(img, m=1)

        ax_img.set_aspect("equal")
        ax_img.set_xlabel("X pixel")
        ax_img.set_ylabel("Y pixel")
        ax_img.set_title(f"File index: {file_index}")
        ny, nx = img.shape
        x0 = roi_x.start if roi_x is not None else 0
        x1 = roi_x.stop  if roi_x is not None else nx
        y0 = roi_y.start if roi_y is not None else 0
        y1 = roi_y.stop  if roi_y is not None else ny
        ax_img.imshow(
            img, vmin=imin, vmax=imax, cmap="seismic",
            extent=[x0, x1, y1, y0],
        )
        fig.canvas.draw_idle()

    def _on_key(event) -> None:
        row, col = row_slider.value, col_slider.value
        deltas = {"up": (1, 0), "down": (-1, 0), "right": (0, 1), "left": (0, -1)}
        if event.key not in deltas:
            return
        dr, dc = deltas[event.key]
        row_slider.value = int(np.clip(row + dr, row_slider.min, row_slider.max))
        col_slider.value = int(np.clip(col + dc, col_slider.min, col_slider.max))

    row_slider.observe(lambda c: _update(c.new, col_slider.value), "value")
    col_slider.observe(lambda c: _update(row_slider.value, c.new), "value")
    fig.canvas.mpl_connect("key_press_event", _on_key)

    ui = widgets.HBox([row_slider, col_slider])
    display(ui)
    _update(row_slider.value, col_slider.value)

    return fig, ui
