"""
lattice_ratio_map.py — b/a and c/a maps over a Laue raster scan.

Companion to ``Dataset.strain_map`` in lauexplore, which plots the deviatoric
strain components but not the lattice parameter ratios. Uses the same
``tiles`` layout, so both figures look and zoom alike.

The colorbar can be centered on a reference value (e.g. relaxed GaN), either
on the raw ratio or on the relative deviation ``(r - ref) / ref`` in × 1e-4,
the same units as the strain maps.

Usage
-----
>>> from laue.lattice_ratio_map import lattice_ratio_map
>>> lattice_ratio_map(dataset, ref=[1.0, 1.6259])                  # raw, centered
>>> lattice_ratio_map(dataset, ref=[1.0, 1.6259], span=[2e-4, 1e-3])
>>> lattice_ratio_map(dataset, ref=[1.0, 1.6259], relative=True, span=5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from lauexplore.plots.base import _as_grid
from lauexplore.plots.composite import tiles
from lauexplore.plots.hovermenus import scan_hovermenu

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from lauexplore.dataset import Dataset

PanelValue = float | Sequence[float | None] | None


def _per_panel(value: PanelValue, n: int) -> list:
    """Escalar -> [value]*n ; lista/tupla -> ela mesma."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"Expected {n} values (one per panel), got {len(value)}.")
        return list(value)
    return [value] * n


def lattice_ratio_map(
    dataset: Dataset,
    mask: np.ndarray | None = None,
    *,
    ref: PanelValue = None,
    relative: bool = False,
    span: PanelValue = None,
    zmin: PanelValue = None,
    zmax: PanelValue = None,
    colorscale: str = "balance",
    width: int = 800,
    height: int = 400,
    **tiles_kwargs,
) -> go.Figure:
    """Plot b/a and c/a maps side by side.

    Every ``PanelValue`` argument takes a scalar (same for both panels) or a
    two-element list ``[b/a, c/a]``; ``None`` inside the list leaves that
    panel on its default.

    Parameters
    ----------
    dataset:
        lauexplore ``Dataset`` with ``scan`` set.
    mask:
        Boolean array (one value per scan point); False points become NaN.
    ref:
        Reference value placed at the center of the colorbar.
    relative:
        If True (requires ``ref`` for both panels), plot
        ``(r - ref) / ref × 1e4`` centered on 0.
    span:
        Half-width of the colorbar around its center, in plotted units
        (ratio, or × 1e-4 if ``relative``). None = symmetric range chosen
        automatically by plotly.
    zmin, zmax:
        Explicit color limits; take priority over ``ref``/``span``.
    colorscale, width, height, **tiles_kwargs:
        Forwarded to ``lauexplore.plots.composite.tiles``.
    """
    if dataset.scan is None:
        raise ValueError("In order to plot you must specify the scan object.")

    ratios = [dataset.boa, dataset.coa]
    titles = ["b/a", "c/a"]
    n = len(ratios)
    refs = _per_panel(ref, n)

    if relative:
        if any(r is None for r in refs):
            raise ValueError("relative=True requires ref for every panel.")
        ratios = [(r - r0) / r0 * 1e4 for r, r0 in zip(ratios, refs)]
        titles = [f"({t} − {r0:g}) / {r0:g}" for t, r0 in zip(titles, refs)]
        centers = [0.0] * n
        tiles_kwargs.setdefault("cbar_title", "× 1e-4")
    else:
        centers = refs

    if mask is not None:
        ratios = [np.where(mask, r, np.nan) for r in ratios]
    grids = [_as_grid(r, dataset.scan) for r in ratios]
    customdata, hovertemplate = scan_hovermenu(dataset.scan)

    fig = tiles(
        grids,
        x=dataset.scan.xpoints * 1e3,
        y=dataset.scan.ypoints * 1e3,
        nrows=1,
        ncols=n,
        colorscale=colorscale,
        width=width,
        height=height,
        customdata=customdata,
        hovertemplate=hovertemplate,
        subplot_titles=titles,
        mask=mask,
        **tiles_kwargs,
    )

    for trace, c, s, lo, hi in zip(
        fig.data, centers, _per_panel(span, n), _per_panel(zmin, n), _per_panel(zmax, n)
    ):
        if lo is not None or hi is not None:
            trace.update(zmin=lo, zmax=hi)
        elif c is not None and s is not None:
            trace.update(zmin=c - s, zmax=c + s)
        elif c is not None:
            trace.update(zmid=c)    # plotly escolhe a faixa simétrica em torno de c
    return fig
