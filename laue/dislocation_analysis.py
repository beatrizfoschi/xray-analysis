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

import numpy as np
import pandas as pd


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
