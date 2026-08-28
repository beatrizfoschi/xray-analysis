"""
spot_fit.py — Parametric multi-Gaussian fit of a Laue spot.

Where `spot_metrics` describes a spot through moments (and measures how far it
is from a single Gaussian without ever fitting one), this module fits an
explicit model: a sum of N 2-D Gaussians over a flat background. That buys the
sub-peaks themselves — position, amplitude and width of each component — which
moments cannot separate.

The two modules answer different questions and are meant to be used together:
``analyze_spot`` says *whether* a spot has sub-structure (``gaussian_residual``,
``n_local_maxima``); ``fit_spot`` says *what* that sub-structure is.

Fitting the raw ROI, not a preprocessed one
-------------------------------------------
The background is a free parameter of the model, so the fit must see the raw
crop. Do not feed it ``spot_metrics.preprocess`` output: that already subtracts
the background and clips at zero, which double-subtracts and leaves ``bg``
degenerate against the amplitudes. The 20th percentile is used here only to
build the initial guess.

Choosing N
----------
``n_components="auto"`` fits N = 1, 2, … ``n_max`` and selects one. Adding a
component almost always lowers chi², so the selection has to penalise the extra
freedom:

    aic = n·ln(chi2 / n) + 2k
    bic = n·ln(chi2 / n) + k·ln(n)

with ``n`` the pixel count, ``k`` the number of free parameters and ``chi2`` the
*raw* weighted sum of squares (not the reduced one — dividing by the degrees of
freedom would penalise k a second time). Since ln(n_pixels) > 2 for any ROI worth
fitting, BIC is the more conservative of the two: it grows N only on clear
sub-structure, while AIC accepts a smaller improvement and will label more
positions as split.

The logarithm is what makes those criteria usable here. Their textbook form,
``chi2 + 2k``, assumes the weights are a correct noise model, i.e. that the
reduced chi² of a good fit is about 1. On real Laue spots it is not: the model
has no rotation term and real spots have non-Gaussian tails, so chi² is dominated
by systematic model error rather than by counting noise. Measured on the
reference data, a 20x20 ROI gives chi² ≈ 88000 against a BIC penalty of 72, and
going from N=2 to N=3 buys a chi² drop of 3000 against a penalty difference of
18 — a penalty two orders of magnitude too small to ever bite, so every position
would be assigned ``n_max``. The form above profiles the unknown noise scale out
of the likelihood instead, which makes the comparison scale-free and the penalty
meaningful whatever the reduced chi² happens to be.

``criterion="chi2"`` is the intuitive alternative: take the smallest N whose
*reduced* chi² falls below ``chi2_threshold``. It has no built-in defence against
overfitting, and its threshold is not portable — it has to be recalibrated by eye
whenever the noise level, the ROI size or the normalisation changes. It is here
to be compared against, not to be trusted by default.

Functions
---------
n_gaussians_2d      The model: N Gaussians + flat background
residuals           Weighted residual vector for least_squares
fit_n_gaussians     One fit at a fixed N
fit_spot            Entry point: fixed N or adaptive, flat dict out
model_from_result   Rebuild the fitted image from a ``fit_spot`` result

Notes
-----
Components are returned sorted by descending amplitude. That ordering is
ambiguous when two components have nearly equal amplitude: neighbouring scan
positions can swap labels, which shows up as speckle in the ``x1``/``x2`` maps
while ``separation`` and ``orientation`` stay smooth. Prefer the derived,
label-free quantities when mapping.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import least_squares

# Guard against a zero-width Gaussian blowing up the exponent mid-fit.
_SIGMA_FLOOR = 0.3

_CRITERIA = ("bic", "aic", "chi2")


# ── Model ─────────────────────────────────────────────────────────────────────

def _n_from_length(n_params: int, shared_sigma: bool) -> int:
    """Number of components implied by a parameter-vector length."""
    if shared_sigma:
        if n_params < 6 or (n_params - 3) % 3 != 0:
            raise ValueError(
                f"{n_params} parameters is not 3n+3 for any n >= 1 "
                "(shared_sigma=True)"
            )
        return (n_params - 3) // 3
    if n_params < 6 or (n_params - 1) % 5 != 0:
        raise ValueError(
            f"{n_params} parameters is not 5n+1 for any n >= 1 "
            "(shared_sigma=False)"
        )
    return (n_params - 1) // 5


def _unpack(params: np.ndarray, n: int, shared_sigma: bool):
    """Split a parameter vector into (centres+amplitudes, sx, sy, background).

    Layouts
    -------
    shared_sigma=True   [x1, y1, A1, ..., xn, yn, An, sx, sy, bg]     (3n + 3)
    shared_sigma=False  [x1, y1, A1, sx1, sy1, ..., bg]               (5n + 1)

    The n=1 case is the same vector either way, which is why the two layouts
    never collide when the length is used to infer n.
    """
    params = np.asarray(params, dtype=np.float64)
    if shared_sigma:
        comp = params[: 3 * n].reshape(n, 3)
        sx = np.full(n, params[3 * n])
        sy = np.full(n, params[3 * n + 1])
        bg = params[3 * n + 2]
    else:
        block = params[: 5 * n].reshape(n, 5)
        comp = block[:, :3]
        sx = block[:, 3]
        sy = block[:, 4]
        bg = params[5 * n]
    return comp, sx, sy, float(bg)


def n_gaussians_2d(
    params: Sequence[float],
    xx: np.ndarray,
    yy: np.ndarray,
    n_components: Optional[int] = None,
    shared_sigma: bool = True,
) -> np.ndarray:
    """Sum of ``n_components`` 2-D Gaussians over a flat background.

    Each Gaussian is axis-aligned with its own ``sx``/``sy`` (or a pair shared
    across components).  There is no rotation term: a tilted single spot is
    better described by ``spot_metrics.inertia_tensor``, and a tilted *pair* is
    represented by the positions of the two components, not by their shape.

    ``n_components`` is inferred from ``len(params)`` when omitted, so a
    9-element vector reproduces the two-Gaussian model exactly.
    """
    params = np.asarray(params, dtype=np.float64)
    if n_components is None:
        n_components = _n_from_length(params.size, shared_sigma)

    comp, sx, sy, bg = _unpack(params, n_components, shared_sigma)

    out = np.zeros(np.broadcast(xx, yy).shape, dtype=np.float64)
    for (x0, y0, amp), sxi, syi in zip(comp, sx, sy):
        sxi = max(sxi, _SIGMA_FLOOR)
        syi = max(syi, _SIGMA_FLOOR)
        out += amp * np.exp(
            -((xx - x0) ** 2 / (2 * sxi ** 2) + (yy - y0) ** 2 / (2 * syi ** 2))
        )
    return out + bg


def residuals(
    params: Sequence[float],
    xx: np.ndarray,
    yy: np.ndarray,
    data: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    shared_sigma: bool,
) -> np.ndarray:
    """Weighted (model − data), flattened for ``least_squares``."""
    model = n_gaussians_2d(params, xx, yy, n_components, shared_sigma)
    return (model - data).ravel() * weights.ravel()


# ── Initial guess ─────────────────────────────────────────────────────────────

def _initial_peaks(
    img: np.ndarray,
    n: int,
    *,
    min_sep: float = 2.0,
    threshold_rel: float = 0.1,
    filter_size: int = 3,
) -> list[tuple[float, float, float]]:
    """The ``n`` strongest well-separated local maxima, as (x, y, amplitude).

    Greedy suppression: take the brightest maximum, then the brightest remaining
    one at least ``min_sep`` pixels from every maximum already kept.

    When fewer than ``n`` survive, the list is padded with decoys offset from the
    strongest peak and progressively fainter, which gives ``least_squares`` a
    starting point it can either grow into a real component or shrink away.

    This deliberately does not call ``spot_metrics.n_local_maxima``: that counts
    maxima for a morphology indicator with a different default neighbourhood and
    a ``>=`` threshold, while this needs the maxima *and* their values under the
    exact recipe the fit was tuned against.
    """
    peak_mask = (img == maximum_filter(img, size=filter_size)) & (
        img > threshold_rel * img.max()
    )
    ys, xs = np.where(peak_mask)
    h, w = img.shape

    if len(ys) == 0:
        # Nothing stands out: seed everything at the centre, fainter each time.
        top = float(img.max())
        return [(w / 2.0, h / 2.0, top * (0.5 ** k)) for k in range(n)]

    vals = img[ys, xs]
    order = np.argsort(-vals, kind="stable")
    xs, ys, vals = xs[order], ys[order], vals[order]

    kept: list[tuple[float, float, float]] = []
    for x, y, v in zip(xs, ys, vals):
        if all(math.hypot(x - kx, y - ky) >= min_sep for kx, ky, _ in kept):
            kept.append((float(x), float(y), float(v)))
            if len(kept) == n:
                return kept

    x1, y1, a1 = kept[0]
    for k in range(1, n - len(kept) + 1):
        kept.append((x1 + k, y1 + k, a1 * (0.3 ** k)))
    return kept[:n]


def _build_p0_bounds(
    roi: np.ndarray,
    n: int,
    shared_sigma: bool,
    sigma_p0: float,
    sigma_bounds: tuple[float, float],
    min_sep: float,
    threshold_rel: float,
):
    """Initial vector and box bounds for one fit."""
    h, w = roi.shape
    top = float(roi.max())
    bg0 = float(np.percentile(roi, 20))
    signal = np.clip(roi - bg0, 0.0, None)
    peaks = _initial_peaks(signal, n, min_sep=min_sep, threshold_rel=threshold_rel)

    p0: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for x0, y0, amp in peaks:
        if shared_sigma:
            p0 += [x0, y0, amp]
            lo += [0.0, 0.0, 0.0]
            hi += [float(w), float(h), top * 2.0]
        else:
            p0 += [x0, y0, amp, sigma_p0, sigma_p0]
            lo += [0.0, 0.0, 0.0, sigma_bounds[0], sigma_bounds[0]]
            hi += [float(w), float(h), top * 2.0, sigma_bounds[1], sigma_bounds[1]]

    if shared_sigma:
        p0 += [sigma_p0, sigma_p0, bg0]
        lo += [sigma_bounds[0], sigma_bounds[0], 0.0]
        hi += [sigma_bounds[1], sigma_bounds[1], top]
    else:
        p0 += [bg0]
        lo += [0.0]
        hi += [top]

    p0 = np.asarray(p0, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    # A ROI carrying negatives (already background-subtracted upstream) would put
    # bg0 below its lower bound and least_squares would refuse the call outright.
    return np.clip(p0, lo, hi), lo, hi


# ── Single fit at fixed N ─────────────────────────────────────────────────────

def fit_n_gaussians(
    img: np.ndarray,
    n_components: int,
    *,
    shared_sigma: bool = True,
    sigma_p0: float = 2.0,
    sigma_bounds: tuple[float, float] = (0.5, 8.0),
    min_sep: float = 2.0,
    threshold_rel: float = 0.1,
    max_nfev: int = 200,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
) -> dict:
    """Fit exactly ``n_components`` Gaussians to one raw ROI.

    Pass the raw crop — the background is a fitted parameter (see module
    docstring).  Pixels are weighted by ``1/sqrt(max(counts, 1))``, the Poisson
    estimate of their standard deviation, so the returned ``chi2_raw`` is a
    proper chi² statistic and the model-selection criteria are well posed.  That
    weighting assumes the ROI is in counts; after a monitor normalisation, scale
    back to count units first or the reduced chi² loses its meaning.

    Returns
    -------
    dict with keys
        params      : (k,) fitted vector, components sorted by descending amplitude
        chi2_raw    : weighted sum of squared residuals
        chi2        : chi2_raw / (n_pixels − k), the reduced chi²
        aic, bic    : selection criteria (lower is better)
        n_params    : k
        success     : whether least_squares converged
    ``params`` is all-NaN and ``success`` False when the fit could not be run.
    """
    img = np.asarray(img, dtype=np.float64)
    n_pix = img.size
    k = (3 * n_components + 3) if shared_sigma else (5 * n_components + 1)

    def _failed() -> dict:
        return {
            "params": np.full(k, np.nan),
            "chi2_raw": np.nan,
            "chi2": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "n_params": k,
            "success": False,
        }

    # Bounds collapse to a point when the ROI is empty, and least_squares
    # requires lower < upper strictly.
    if not np.isfinite(img).all() or img.max() <= 0 or n_pix <= k:
        return _failed()

    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    weights = 1.0 / np.sqrt(np.clip(img, 1.0, None))

    p0, lo, hi = _build_p0_bounds(
        img, n_components, shared_sigma, sigma_p0, sigma_bounds, min_sep, threshold_rel
    )

    try:
        result = least_squares(
            residuals,
            p0,
            args=(xx, yy, img, weights, n_components, shared_sigma),
            bounds=(lo, hi),
            method="trf",
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
        )
    except Exception:
        return _failed()

    params = _sort_by_amplitude(result.x, n_components, shared_sigma)
    chi2_raw = float(np.sum(result.fun ** 2))
    # Gaussian log-likelihood with the noise scale profiled out. A perfect fit
    # sends this to -inf, which is harmless (it wins the comparison) but not
    # representable, so floor the ratio.
    log_term = n_pix * math.log(max(chi2_raw / n_pix, 1e-300))
    return {
        "params": params,
        "chi2_raw": chi2_raw,
        "chi2": chi2_raw / (n_pix - k),
        "aic": log_term + 2.0 * k,
        "bic": log_term + k * math.log(n_pix),
        "n_params": k,
        "success": bool(result.success),
    }


def _sort_by_amplitude(params: np.ndarray, n: int, shared_sigma: bool) -> np.ndarray:
    """Reorder components brightest-first, keeping the tail (sigmas, bg) in place.

    A stable sort leaves equal amplitudes in their fitted order, so a two-component
    vector is swapped exactly when the second is strictly brighter.
    """
    params = np.asarray(params, dtype=np.float64).copy()
    if n == 1:
        return params
    stride = 3 if shared_sigma else 5
    block = params[: stride * n].reshape(n, stride)
    order = np.argsort(-block[:, 2], kind="stable")
    params[: stride * n] = block[order].ravel()
    return params


# ── Entry point ───────────────────────────────────────────────────────────────

def fit_spot(
    img: np.ndarray,
    *,
    n_components: int | str = 2,
    n_max: int = 3,
    criterion: str = "bic",
    chi2_threshold: float = 1.5,
    shared_sigma: bool = True,
    min_counts: float = 0.0,
    sigma_p0: float = 2.0,
    sigma_bounds: tuple[float, float] = (0.5, 8.0),
    min_sep: float = 2.0,
    threshold_rel: float = 0.1,
    max_nfev: int = 200,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
) -> dict:
    """Fit one ROI and return a flat dict of parameters and derived quantities.

    Signature-compatible with ``spot_metrics.analyze_spot``: takes one 2-D crop,
    returns one flat dict, so ``scan_pipeline.run_spot_pipeline`` can be pointed
    at either through ``analysis_fn``.

    Parameters
    ----------
    n_components : int or "auto"
        Fixed component count, or ``"auto"`` to select one by ``criterion``.
    n_max : int
        Largest N tried when selecting, and the width of the output schema.
    criterion : {"bic", "aic", "chi2"}
        How ``"auto"`` picks N.  ``bic``/``aic`` take the lowest value over the
        fits that converged; ``chi2`` takes the smallest N whose reduced chi²
        falls below ``chi2_threshold``, falling back to the best reduced chi²
        when none does.  See the module docstring for what each implies.
    shared_sigma : bool
        One ``(sx, sy)`` pair for the whole spot, or one per component.  Shared
        is the more stable default: it assumes the sub-peaks have the same
        physical width, which is what makes N=3 fittable on a small ROI at all.
        Per-component widths are worth trying when the residual of a shared fit
        shows one sub-peak systematically broader (a diffuse V-pit beside a sharp
        core) — decide that on real residuals, not per pixel.
    min_counts : float
        Return the empty result when the raw ROI sums below this.

    Returns
    -------
    dict with a schema of fixed width, independent of the N actually chosen:
        x{1..n_max}, y{1..n_max}            component centres (px, crop frame)
        A{1..n_max}                          amplitudes
        sigma_x{1..n_max}, sigma_y{1..n_max} widths (repeated when shared)
        bg                                   flat background
        n_components                         N selected (0 when the fit failed)
        n_params, chi2, chi2_raw, aic, bic, success
        separation, orientation              between the two brightest components
        ratio                                A2 / (A1 + A2)
        total_amplitude, centroid_x, centroid_y
    Columns above the selected N are NaN, which keeps a DataFrame built from
    mixed-N positions rectangular.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")
    if isinstance(n_components, str):
        if n_components != "auto":
            raise ValueError(
                f'n_components must be an int or "auto", got {n_components!r}'
            )
        adaptive = True
    else:
        adaptive = False
        if not 1 <= n_components <= n_max:
            raise ValueError(
                f"n_components={n_components} outside 1..n_max={n_max}"
            )

    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"img must be 2-D, got shape {img.shape}")

    if img.size == 0 or not np.isfinite(img).all() or img.max() <= 0 \
            or img.sum() < min_counts:
        return _empty_result(n_max)

    fit_kw = dict(
        shared_sigma=shared_sigma,
        sigma_p0=sigma_p0,
        sigma_bounds=sigma_bounds,
        min_sep=min_sep,
        threshold_rel=threshold_rel,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
    )

    if not adaptive:
        best_n = n_components
        best = fit_n_gaussians(img, best_n, **fit_kw)
    else:
        candidates: list[tuple[int, dict]] = []
        for n in range(1, n_max + 1):
            res = fit_n_gaussians(img, n, **fit_kw)
            if res["success"]:
                candidates.append((n, res))
            if criterion == "chi2" and res["success"] and res["chi2"] <= chi2_threshold:
                # Smallest N that already explains the data: stop, do not pay for
                # fits that a threshold criterion would discard anyway.
                break
        if not candidates:
            return _empty_result(n_max)
        if criterion == "chi2":
            below = [(n, r) for n, r in candidates if r["chi2"] <= chi2_threshold]
            best_n, best = below[0] if below else min(
                candidates, key=lambda nr: nr[1]["chi2"]
            )
        else:
            best_n, best = min(candidates, key=lambda nr: nr[1][criterion])

    if not best["success"]:
        return _empty_result(n_max)

    return _flatten(best, best_n, n_max, shared_sigma)


def _empty_result(n_max: int) -> dict:
    out: dict = {}
    for k in range(1, n_max + 1):
        out[f"x{k}"] = np.nan
        out[f"y{k}"] = np.nan
        out[f"A{k}"] = np.nan
        out[f"sigma_x{k}"] = np.nan
        out[f"sigma_y{k}"] = np.nan
    out.update({
        "bg": np.nan,
        "n_components": 0,
        "n_params": 0,
        "chi2": np.nan,
        "chi2_raw": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "success": False,
        "separation": np.nan,
        "orientation": np.nan,
        "ratio": np.nan,
        "total_amplitude": np.nan,
        "centroid_x": np.nan,
        "centroid_y": np.nan,
    })
    return out


def _flatten(res: dict, n: int, n_max: int, shared_sigma: bool) -> dict:
    comp, sx, sy, bg = _unpack(res["params"], n, shared_sigma)

    out = _empty_result(n_max)
    for k in range(n):
        out[f"x{k + 1}"] = float(comp[k, 0])
        out[f"y{k + 1}"] = float(comp[k, 1])
        out[f"A{k + 1}"] = float(comp[k, 2])
        out[f"sigma_x{k + 1}"] = float(sx[k])
        out[f"sigma_y{k + 1}"] = float(sy[k])

    out["bg"] = bg
    out["n_components"] = n
    out["n_params"] = int(res["n_params"])
    out["chi2"] = float(res["chi2"])
    out["chi2_raw"] = float(res["chi2_raw"])
    out["aic"] = float(res["aic"])
    out["bic"] = float(res["bic"])
    out["success"] = bool(res["success"])

    amps = comp[:, 2]
    total = float(amps.sum())
    out["total_amplitude"] = total
    if total > 0:
        out["centroid_x"] = float((amps * comp[:, 0]).sum() / total)
        out["centroid_y"] = float((amps * comp[:, 1]).sum() / total)

    if n >= 2:
        dx = float(comp[1, 0] - comp[0, 0])
        dy = float(comp[1, 1] - comp[0, 1])
        out["separation"] = float(math.hypot(dx, dy))
        out["orientation"] = float(np.degrees(np.arctan2(dy, dx)))
        a1, a2 = float(amps[0]), float(amps[1])
        out["ratio"] = a2 / (a1 + a2 + 1e-9)
    return out


# ── Reconstruction ────────────────────────────────────────────────────────────

def model_from_result(result: dict, shape: tuple[int, int], shared_sigma: bool = True) -> np.ndarray:
    """Rebuild the fitted image from a ``fit_spot`` result, for residual plots.

    Returns an all-NaN array when the fit failed, so a residual panel shows the
    failure rather than a misleading flat model.
    """
    h, w = shape
    n = int(result.get("n_components", 0))
    if n < 1:
        return np.full(shape, np.nan)

    params: list[float] = []
    for k in range(1, n + 1):
        if shared_sigma:
            params += [result[f"x{k}"], result[f"y{k}"], result[f"A{k}"]]
        else:
            params += [
                result[f"x{k}"], result[f"y{k}"], result[f"A{k}"],
                result[f"sigma_x{k}"], result[f"sigma_y{k}"],
            ]
    if shared_sigma:
        params += [result["sigma_x1"], result["sigma_y1"], result["bg"]]
    else:
        params += [result["bg"]]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return n_gaussians_2d(np.asarray(params), xx, yy, n, shared_sigma)
