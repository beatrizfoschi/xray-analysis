"""
scan_viewer.py
==============

Interactive viewer: XEOL emission map + diffraction image for a Laue raster scan.

Usage
-----
>>> from laue.scan_viewer import scan_viewer
>>> scan_viewer(
...     h5_path="scan_001.h5",
...     xeol_roi=(350.0, 450.0),            # nm
...     img_source="path/to/tifs",          # TIF folder
...     roi_y=slice(1360, 1420),
...     roi_x=slice(1230, 1280),
... )

>>> scan_viewer(
...     h5_path="scan_001.h5",
...     xeol_roi=(350.0, 450.0),            # nm
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

from lauexplore.emission import XEOL
from lauexplore.image import read as read_image
from lauexplore.plots.base import _as_grid
from lauexplore.scan import Scan


def scan_viewer(
    h5_path: str | Path,
    xeol_roi: tuple[float, float],
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
    xeol_norm_zone: tuple[float, float] | None = None,
    sigmoid_cutoff: float = 0.5,
    sigmoid_gain: float = 5.0,
) -> tuple[plt.Figure, widgets.Widget]:
    """Interactive XEOL emission map + diffraction image viewer.

    Parameters
    ----------
    h5_path:
        Path to the scan HDF5 file.
    xeol_roi:
        Wavelength integration range in nm, e.g. ``(350.0, 450.0)``.
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
        Divide XEOL by monitor counts (default True).
    xeol_norm_zone:
        Wavelength range (nm) used as normalization reference, passed to
        ``XEOL.from_h5`` as ``norm_zone``.
    sigmoid_cutoff, sigmoid_gain:
        Parameters for ``skimage.exposure.adjust_sigmoid``.
    """
    h5_path  = Path(h5_path)
    img_source = Path(img_source)

    scan = Scan.from_h5(h5_path, scan_number)
    xeol = XEOL.from_h5(h5_path, scan_number,
                         roi=xeol_roi,
                         normalize_to_monitor=normalize_to_monitor,
                         norm_zone=xeol_norm_zone)

    fluo_grid = _as_grid(xeol.data, scan)
    map_label = f"XEOL {xeol_roi[0]:.0f}–{xeol_roi[1]:.0f} nm"
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

    # As setas esquerda/direita são mapeadas por padrão pelo matplotlib como
    # "back/forward view history" — isso conflita com a navegação do scan.
    plt.rcParams["keymap.back"]    = [k for k in plt.rcParams["keymap.back"]    if k != "left"]
    plt.rcParams["keymap.forward"] = [k for k in plt.rcParams["keymap.forward"] if k != "right"]

    # Zoom state: salvo apenas no button_release (pan/zoom com mouse).
    # Separado completamente da navegação por slider/teclado.
    _state: dict = {
        "default_map": None,   # limites auto da primeira draw (fixos)
        "default_img": None,
        "zoom_map":    None,   # limites definidos pelo usuário (None = full view)
        "zoom_img":    None,
    }

    def _lims_differ(a: tuple, b: tuple) -> bool:
        return not (np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]))

    def _on_mouse_release(_) -> None:
        if _state["default_map"] is None:
            return
        cur_map = (ax_map.get_xlim(), ax_map.get_ylim())
        cur_img = (ax_img.get_xlim(), ax_img.get_ylim())
        _state["zoom_map"] = cur_map if _lims_differ(cur_map, _state["default_map"]) else None
        _state["zoom_img"] = cur_img if _lims_differ(cur_img, _state["default_img"]) else None

    fig.canvas.mpl_connect("button_release_event", _on_mouse_release)

    def _update(row: int, col: int) -> None:
        ax_map.cla()
        ax_img.cla()

        ax_map.set_aspect("equal")
        ax_map.set_xlabel("Position [μm]")
        ax_map.set_ylabel("Position [μm]")
        ax_map.set_title(map_label)
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

        if _state["zoom_map"] is not None:
            ax_map.set_xlim(_state["zoom_map"][0])
            ax_map.set_ylim(_state["zoom_map"][1])
        if _state["zoom_img"] is not None:
            ax_img.set_xlim(_state["zoom_img"][0])
            ax_img.set_ylim(_state["zoom_img"][1])

        if _state["default_map"] is None:
            # Primeira draw: síncrona para capturar os limites auto do matplotlib
            fig.canvas.draw()
            _state["default_map"] = (ax_map.get_xlim(), ax_map.get_ylim())
            _state["default_img"] = (ax_img.get_xlim(), ax_img.get_ylim())
        else:
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
