"""
dislocation_analysis.py — Laue dislocation contrast analysis.

Simulates Laue diffraction patterns and computes the invisibility criteria
(g·b and g·(b×u)) for user-defined dislocation types.

Interpreting g·b
----------------
A dislocation with Burgers vector b produces contrast in a diffraction spot g
only when g·b ≠ 0. When g·b = 0, the lattice distortion caused by the
dislocation leaves no signature in that reflection — the spot appears sharp
and unaffected, as if the dislocation were not present.

    g·b = 0        →  dislocation invisible — spot unaffected
    g·b = ±1       →  weak contrast
    g·b = ±2, ±3   →  strong contrast — spot clearly perturbed

The sign of g·b indicates the direction of the lattice displacement relative
to g, but does not affect contrast intensity. In Laue, only |g·b| matters.

Identifying a dislocation type experimentally relies on finding pairs of
reflections where the effect appears (g·b ≠ 0) and disappears (g·b = 0).

Typical workflow
----------------
>>> spots  = simulate_laue("GaN", ub_matrix, calib_params, Emin=5, Emax=25)
>>> result = dislocation_contrast(spots, [
...     Dislocation([1/3, 1/3, -2/3, 0], label="a"),
...     Dislocation([0, 0, 0, 1],         label="c"),
...     Dislocation([1/3, 1/3, -2/3, 1],  label="c+a",
...                 line_direction=[0, 0, 0, 1]),
... ], tol=0.1)
>>> fig = plot_contrast(result, "a")
>>> fig.show()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure


# ── Index conversion ──────────────────────────────────────────────────────────

def _to_3index(v: np.ndarray) -> np.ndarray:
    """Convert a 3- or 4-component Miller(-Bravais) vector to 3-index form.

    For a 4-index vector [h, k, i, l], validates i == -(h+k) and returns
    [h, k, l]. For a 3-index vector, returns it unchanged.
    """
    v = np.asarray(v, dtype=float)
    if v.shape == (3,):
        return v
    if v.shape == (4,):
        h, k, i_idx, l = v
        if abs(i_idx + h + k) > 1e-6:
            raise ValueError(
                f"Invalid Miller-Bravais vector: i={i_idx} != -(h+k)={-(h+k):.6f}"
            )
        return np.array([h, k, l])
    raise ValueError(f"Vector must have 3 or 4 components, got shape {v.shape}")


# ── Dislocation specification ─────────────────────────────────────────────────

@dataclass
class Dislocation:
    """Dislocation type specification.

    Parameters
    ----------
    burgers : array-like, shape (3,) or (4,)
        Burgers vector in direct-lattice crystal coordinates.
        Accepts 3-index [b1, b2, b3] or Miller-Bravais 4-index [h, k, i, l]
        notation. For partial dislocations, fractional values are valid
        (e.g., [1/3, 1/3, -2/3, 0] for 1/3<11-20> in GaN).
    line_direction : array-like, shape (3,) or (4,), optional
        Dislocation line direction u in the same coordinate convention as
        ``burgers``. Required for the edge-component criterion g·(b×u).
        For a pure screw dislocation (b ∥ u), b×u = 0 and every spot
        satisfies g·(b×u) = 0, so ``fully_invisible`` reduces to g·b = 0.
    label : str
        Human-readable name used as column prefix and in plot legends
        (e.g. ``"a"``, ``"c"``, ``"c+a"``).
    """
    burgers: np.ndarray
    line_direction: np.ndarray | None = None
    label: str = ""

    def __post_init__(self) -> None:
        self.burgers = np.asarray(self.burgers, dtype=float)
        if self.burgers.shape not in ((3,), (4,)):
            raise ValueError(
                f"burgers must be a 3- or 4-component vector, got shape {self.burgers.shape}"
            )
        if self.line_direction is not None:
            self.line_direction = np.asarray(self.line_direction, dtype=float)
            if self.line_direction.shape not in ((3,), (4,)):
                raise ValueError(
                    f"line_direction must be a 3- or 4-component vector, "
                    f"got shape {self.line_direction.shape}"
                )


# ── Simulation wrapper ────────────────────────────────────────────────────────

def simulate_laue(
    material: str,
    ub_matrix: np.ndarray,
    calibration_parameters: list,
    **kwargs,
) -> pd.DataFrame:
    """Simulate Laue diffraction spots for a given material and orientation.

    Thin wrapper around ``lauexplore.peaks.simulate`` that validates the UB
    matrix shape and returns the standard spot DataFrame.

    Parameters
    ----------
    material : str
        Material key in the LaueTools material dictionary (e.g. ``"GaN"``).
    ub_matrix : np.ndarray, shape (3, 3)
        Crystal orientation / UB matrix in the lab frame.
    calibration_parameters : list
        Five-element detector calibration list:
        [distance, x_center, y_center, x_beta, x_gamma].
    **kwargs
        Forwarded to ``lauexplore.peaks.simulate``:
        ``Emin``, ``Emax``, ``material_dictionary``,
        ``camera_label``, ``detector_diameter``.

    Returns
    -------
    pd.DataFrame
        Columns: h, k, l, 2θ, χ, X, Y, Energy.
        Only spots that fall within the detector bounds are included.
    """
    import lauexplore.peaks as _peaks

    ub_matrix = np.asarray(ub_matrix, dtype=float)
    if ub_matrix.shape != (3, 3):
        raise ValueError(f"ub_matrix must be shape (3, 3), got {ub_matrix.shape}")

    return _peaks.simulate(material, ub_matrix, calibration_parameters, **kwargs)


# ── Dislocation contrast ──────────────────────────────────────────────────────

def dislocation_contrast(
    spots: pd.DataFrame,
    dislocations: list[Dislocation],
    *,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Compute dislocation contrast criteria for a set of simulated Laue spots.

    For each dislocation type, adds columns to the spot DataFrame:

    - ``{label}_gb``             : g·b value (float)
    - ``{label}_visible``        : |g·b| > tol (bool)
    - ``{label}_gbu``            : g·(b×u) value (float) — only when
                                   ``line_direction`` is set
    - ``{label}_fully_invisible``: |g·b| ≤ tol AND |g·(b×u)| ≤ tol (bool)
                                   — only when ``line_direction`` is set

    The g·b criterion is applied in direct-lattice fractional coordinates:
    g·b = h·b₁ + k·b₂ + l·b₃. The g·(b×u) values are also computed in
    fractional coordinates, which is the standard approximation used in
    Laue dislocation analysis for all crystal systems.

    Parameters
    ----------
    spots : pd.DataFrame
        Output of ``simulate_laue`` (must contain columns h, k, l).
    dislocations : list of Dislocation
        One or more dislocation types. Column names are derived from
        ``Dislocation.label``; if empty, the list index is used.
    tol : float
        Threshold below which g·b (or g·(b×u)) is treated as zero.
        Use ``tol=1e-6`` (default) for perfect dislocations.
        Use ``tol=0.1`` for partial dislocations where g·b = {0, ±1/3, ...}.

    Returns
    -------
    pd.DataFrame
        Copy of ``spots`` with additional columns for each dislocation type.
    """
    result = spots.copy()
    g = spots[["h", "k", "l"]].to_numpy(dtype=float)   # (N, 3)

    for idx, dislo in enumerate(dislocations):
        prefix = dislo.label if dislo.label else f"dislo_{idx}"
        b3 = _to_3index(dislo.burgers)                  # (3,)

        gb = g @ b3                                      # (N,)
        result[f"{prefix}_gb"]      = gb
        result[f"{prefix}_visible"] = np.abs(gb) > tol

        if dislo.line_direction is not None:
            u3  = _to_3index(dislo.line_direction)
            bxu = np.cross(b3, u3)                       # (3,)
            gbu = g @ bxu                                # (N,)
            result[f"{prefix}_gbu"]             = gbu
            result[f"{prefix}_fully_invisible"] = (
                (np.abs(gb) <= tol) & (np.abs(gbu) <= tol)
            )

    return result


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_contrast(
    spots: pd.DataFrame,
    dislocation_label: str,
    *,
    component: str = "gb",
    width: int = 700,
    height: int = 700,
    colorscale: str = "RdBu",
    title: str | None = None,
) -> "go.Figure":
    """Scatter plot of detector positions coloured by g·b (or g·(b×u)).

    Parameters
    ----------
    spots : pd.DataFrame
        Output of ``dislocation_contrast``.
    dislocation_label : str
        The label used when defining the ``Dislocation``.
    component : {"gb", "gbu"}
        Which column to use for colour coding.
        ``"gb"`` → g·b,  ``"gbu"`` → g·(b×u).
    width, height : int
        Figure dimensions in pixels.
    colorscale : str
        Plotly colorscale. ``"RdBu"`` centres at zero: red = visible
        (|g·b| > 0), blue = invisible (g·b = 0).
    title : str, optional
        Figure title. Defaults to ``"Dislocation contrast: {label}_{component}"``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    col = f"{dislocation_label}_{component}"
    if col not in spots.columns:
        raise KeyError(
            f"Column '{col}' not found. "
            f"Run dislocation_contrast with label='{dislocation_label}' first."
        )

    z = spots[col].to_numpy(dtype=float)
    abs_max = float(np.abs(z).max()) or 1.0

    hover = (
        "h=%{customdata[0]:.0f}, k=%{customdata[1]:.0f}, l=%{customdata[2]:.0f}<br>"
        f"{col}=%{{marker.color:.4f}}<br>"
        "E=%{customdata[3]:.3f} keV"
        "<extra></extra>"
    )

    fig = go.Figure(
        go.Scatter(
            x=spots["X"],
            y=spots["Y"],
            mode="markers",
            marker=dict(
                color=z,
                colorscale=colorscale,
                cmin=-abs_max,
                cmax=abs_max,
                colorbar=dict(title=col),
                size=8,
            ),
            customdata=spots[["h", "k", "l", "Energy"]].to_numpy(),
            hovertemplate=hover,
        )
    )
    fig.update_layout(
        title=title or f"Dislocation contrast: {col}",
        xaxis=dict(title="X (pixels)", range=[0, 2018]),
        yaxis=dict(title="Y (pixels)", range=[2018, 0]),
        width=width,
        height=height,
    )
    return fig


# ── Interactive comparison with experimental image ────────────────────────────

def plot_contrast_with_experiment(
    spots: pd.DataFrame,
    dislocation_label: str,
    experimental_image: np.ndarray | str | Path,
    *,
    component: str = "gb",
    zoom_boxsize: tuple[int, int] = (100, 100),
    h5_img_key: str | None = None,
    overlay_spots: bool = True,
    figsize: tuple[float, float] = (14, 7),
    cmap_exp: str = "gray",
) -> "matplotlib.figure.Figure":
    """Side-by-side view of simulated contrast and experimental detector image.

    Left panel: simulated spot positions coloured by g·b (or g·(b×u)).
    Right panel: experimental detector image.

    Clicking on a spot in the left panel zooms the right panel to a region of
    ``zoom_boxsize`` pixels centred on that spot's detector coordinates (X, Y).
    If ``overlay_spots`` is True, simulated spot positions are also drawn on
    the experimental image for direct comparison.

    Requires the ``%matplotlib widget`` (ipympl) backend in Jupyter.

    Parameters
    ----------
    spots : pd.DataFrame
        Output of ``dislocation_contrast`` (must contain X, Y, h, k, l columns).
    dislocation_label : str
        The label used when calling ``dislocation_contrast``.
    experimental_image : np.ndarray, str, or Path
        Detector image as a numpy array, a TIF/EDF file path, or an HDF5 file
        path (requires ``h5_img_key``).
    component : {"gb", "gbu"}
        Column to use for colour coding in the simulation panel.
    zoom_boxsize : (width, height)
        Size of the zoom region in the experimental panel (pixels).
    h5_img_key : str, optional
        Dataset key when ``experimental_image`` is an HDF5 file.
        Indexed as ``h5f[h5_img_key][...]``.
    overlay_spots : bool
        If True, draw simulated spot positions on the experimental image.
    figsize : (width, height)
        Figure size in inches.
    cmap_exp : str
        Matplotlib colormap for the experimental image.
    """
    import h5py
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from lauexplore.image import read as read_image

    col = f"{dislocation_label}_{component}"
    if col not in spots.columns:
        raise KeyError(
            f"Column '{col}' not found. "
            f"Run dislocation_contrast with label='{dislocation_label}' first."
        )

    # ── Load experimental image ───────────────────────────────────────────────
    if isinstance(experimental_image, np.ndarray):
        exp_img = experimental_image
    else:
        src = Path(experimental_image)
        if src.suffix in ('.h5', '.hdf5'):
            if h5_img_key is None:
                raise ValueError("h5_img_key must be set when experimental_image is an HDF5 file.")
            with h5py.File(src, "r") as h5f:
                exp_img = h5f[h5_img_key][()]
        else:
            exp_img = read_image(src)

    img_h, img_w = exp_img.shape[:2]

    # ── Colour mapping for g·b ────────────────────────────────────────────────
    z        = spots[col].to_numpy(dtype=float)
    abs_max  = float(np.abs(z).max()) or 1.0
    norm     = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap_sim = plt.get_cmap("RdBu")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax_sim, ax_exp) = plt.subplots(1, 2, figsize=figsize)

    # Left: simulated spots
    sc = ax_sim.scatter(
        spots["X"], spots["Y"],
        c=z, cmap=cmap_sim, norm=norm,
        s=30, linewidths=0.3, edgecolors="k",
    )
    plt.colorbar(sc, ax=ax_sim, label=col)
    ax_sim.set_xlim(0, img_w)
    ax_sim.set_ylim(img_h, 0)
    ax_sim.set_xlabel("X (pixels)")
    ax_sim.set_ylabel("Y (pixels)")
    ax_sim.set_title(f"Simulation — {col}")
    ax_sim.set_aspect("equal")

    # Right: experimental image (full view initially)
    img_vmax = float(np.percentile(exp_img, 99))
    ax_exp.imshow(exp_img, cmap=cmap_exp, vmin=0, vmax=img_vmax,
                  origin="upper", extent=[0, img_w, img_h, 0])
    if overlay_spots:
        ax_exp.scatter(spots["X"], spots["Y"],
                       c=z, cmap=cmap_sim, norm=norm,
                       s=20, linewidths=0.3, edgecolors="w", alpha=0.7)
    ax_exp.set_xlabel("X (pixels)")
    ax_exp.set_ylabel("Y (pixels)")
    ax_exp.set_title("Experimental — click a spot to zoom")
    ax_exp.set_aspect("equal")

    # Annotation showing spot info
    info = ax_sim.text(
        0.02, 0.98, "", transform=ax_sim.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    # ── Click handler ─────────────────────────────────────────────────────────
    spot_xy = spots[["X", "Y"]].to_numpy(dtype=float)
    hw, hh  = zoom_boxsize[0] / 2, zoom_boxsize[1] / 2

    def _on_click(event):
        if event.inaxes is not ax_sim or event.button != 1:
            return

        dists = np.hypot(spot_xy[:, 0] - event.xdata,
                         spot_xy[:, 1] - event.ydata)
        i     = int(np.argmin(dists))
        row   = spots.iloc[i]
        cx, cy = row["X"], row["Y"]

        ax_exp.set_xlim(cx - hw, cx + hw)
        ax_exp.set_ylim(cy + hh, cy - hh)

        h_, k_, l_ = int(row["h"]), int(row["k"]), int(row["l"])
        gb_val     = row[col]
        info.set_text(f"({h_} {k_} {l_})  {col} = {gb_val:.3f}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.tight_layout()

    return fig
