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

Counting sub-peaks: ``n_resolved``, not ``n_fitted``
-----------------------------------------------------
**The selected N is not the number of sub-peaks, and no choice of criterion makes
it one.** Model selection answers "how many Gaussians describe this best"; the
physical question is "how many sub-peaks are there". The two diverge because chi²
falls just as happily for a component thrown into a heavy tail as for one that
found a real peak, and BIC and AIC are chi² plus a penalty, so they inherit that.

Measured on the reference spot, over 400 positions: the BIC-selected N agrees
with a direct count of local maxima on **9%** of them, and calls 92% of positions
split where the direct count says 12.5%. Two attempts to fix it at the model level
both failed — giving the components a rotation angle moved single-maximum spots
from 8.9% to 15.1% of N=1, and a pseudo-Voigt profile with heavy tails beat the
best multi-Gaussian on 6% of positions.

So the count is taken from resolvability instead, by `_resolved_indices`: a
component survives if it is bright enough relative to the brightest and if the
model dips between it and every component already kept. That agrees with the
direct maxima count on 90.5% of positions. ``n_fitted`` is kept as what it
actually is — a measure of how much structure the fit needed — and ``separation``,
``orientation`` and ``ratio`` are reported only where two components are actually
resolved.

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

def _shape_size(rotation: bool) -> int:
    """Number of shape parameters per shape block: sx, sy and optionally theta."""
    return 3 if rotation else 2


def _n_params(n: int, shared_sigma: bool, rotation: bool) -> int:
    s = _shape_size(rotation)
    return (3 * n + s + 1) if shared_sigma else ((3 + s) * n + 1)


def _n_from_length(n_params: int, shared_sigma: bool, rotation: bool = False) -> int:
    """Number of components implied by a parameter-vector length.

    Both flags are needed: 13 parameters is three shared-shape rotated components
    and also two per-component rotated ones, so the length alone does not say.
    """
    s = _shape_size(rotation)
    per, offset = (3, s + 1) if shared_sigma else (3 + s, 1)
    if n_params < per + offset or (n_params - offset) % per != 0:
        raise ValueError(
            f"{n_params} parameters is not {per}n+{offset} for any n >= 1 "
            f"(shared_sigma={shared_sigma}, rotation={rotation})"
        )
    return (n_params - offset) // per


def _unpack(params: np.ndarray, n: int, shared_sigma: bool, rotation: bool = False):
    """Split a parameter vector into (centres+amplitudes, sx, sy, theta, background).

    Layouts — ``…`` is the shape block ``sx, sy`` plus ``theta`` when rotating:

    shared_sigma=True   [x1, y1, A1, …, xn, yn, An, sx, sy, (theta,) bg]
    shared_sigma=False  [x1, y1, A1, sx1, sy1, (theta1,) …, bg]

    ``theta`` is in radians and turns the component's own axes, so ``sx`` is the
    width along that rotated axis rather than along the detector column.
    """
    params = np.asarray(params, dtype=np.float64)
    s = _shape_size(rotation)
    if shared_sigma:
        comp  = params[: 3 * n].reshape(n, 3)
        shape = params[3 * n: 3 * n + s]
        sx    = np.full(n, shape[0])
        sy    = np.full(n, shape[1])
        theta = np.full(n, shape[2] if rotation else 0.0)
        bg    = params[3 * n + s]
    else:
        block = params[: (3 + s) * n].reshape(n, 3 + s)
        comp  = block[:, :3]
        sx    = block[:, 3]
        sy    = block[:, 4]
        theta = block[:, 5] if rotation else np.zeros(n)
        bg    = params[(3 + s) * n]
    return comp, sx, sy, theta, float(bg)


def n_gaussians_2d(
    params: Sequence[float],
    xx: np.ndarray,
    yy: np.ndarray,
    n_components: Optional[int] = None,
    shared_sigma: bool = True,
    rotation: bool = False,
) -> np.ndarray:
    """Sum of ``n_components`` 2-D Gaussians over a flat background.

    With ``rotation=False`` (the default) each Gaussian is axis-aligned, which is
    the model notebook 02 used. With ``rotation=True`` the component axes turn by
    ``theta``, and a single tilted streak becomes one elongated Gaussian instead
    of a row of round ones — see `fit_spot` for why that changes what the
    component count means.

    ``n_components`` is inferred from ``len(params)`` when omitted, so a
    9-element vector reproduces the two-Gaussian model exactly.
    """
    params = np.asarray(params, dtype=np.float64)
    if n_components is None:
        n_components = _n_from_length(params.size, shared_sigma, rotation)

    comp, sx, sy, theta, bg = _unpack(params, n_components, shared_sigma, rotation)

    out = np.zeros(np.broadcast(xx, yy).shape, dtype=np.float64)
    for (x0, y0, amp), sxi, syi, ti in zip(comp, sx, sy, theta):
        sxi = max(sxi, _SIGMA_FLOOR)
        syi = max(syi, _SIGMA_FLOOR)
        dx, dy = xx - x0, yy - y0
        if rotation:
            ct, st = np.cos(ti), np.sin(ti)
            u, v = dx * ct + dy * st, -dx * st + dy * ct
        else:
            u, v = dx, dy
        out += amp * np.exp(-(u ** 2 / (2 * sxi ** 2) + v ** 2 / (2 * syi ** 2)))
    return out + bg


def residuals(
    params: Sequence[float],
    xx: np.ndarray,
    yy: np.ndarray,
    data: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    shared_sigma: bool,
    rotation: bool = False,
) -> np.ndarray:
    """Weighted (model − data), flattened for ``least_squares``."""
    model = n_gaussians_2d(params, xx, yy, n_components, shared_sigma, rotation)
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


def _streak_angle(signal: np.ndarray) -> float:
    """Initial ``theta`` (radians) from the intensity second moments.

    Reuses `spot_metrics`, whose inertia tensor already answers exactly this —
    the principal axis of the spot — so the rotated fit starts pointing along the
    streak rather than searching for it.
    """
    from laue.spot_metrics import center_of_mass, inertia_tensor

    x_com, y_com = center_of_mass(signal)
    *_, theta_deg = inertia_tensor(signal, x_com, y_com)
    return 0.0 if np.isnan(theta_deg) else float(np.radians(theta_deg))


def _build_p0_bounds(
    roi: np.ndarray,
    n: int,
    shared_sigma: bool,
    sigma_p0: float,
    sigma_bounds: tuple[float, float],
    min_sep: float,
    threshold_rel: float,
    rotation: bool = False,
):
    """Initial vector and box bounds for one fit."""
    h, w = roi.shape
    top = float(roi.max())
    bg0 = float(np.percentile(roi, 20))
    signal = np.clip(roi - bg0, 0.0, None)
    peaks = _initial_peaks(signal, n, min_sep=min_sep, threshold_rel=threshold_rel)

    # theta is periodic, so a box bound is only safe centred on the estimate;
    # +/- pi/2 covers every distinct orientation.
    t0 = _streak_angle(signal) if rotation else 0.0
    shape_p0 = [sigma_p0, sigma_p0] + ([t0] if rotation else [])
    shape_lo = [sigma_bounds[0], sigma_bounds[0]] + ([t0 - np.pi / 2] if rotation else [])
    shape_hi = [sigma_bounds[1], sigma_bounds[1]] + ([t0 + np.pi / 2] if rotation else [])

    p0: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for x0, y0, amp in peaks:
        p0 += [x0, y0, amp]
        lo += [0.0, 0.0, 0.0]
        hi += [float(w), float(h), top * 2.0]
        if not shared_sigma:
            p0 += shape_p0
            lo += shape_lo
            hi += shape_hi

    if shared_sigma:
        p0 += shape_p0 + [bg0]
        lo += shape_lo + [0.0]
        hi += shape_hi + [top]
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
    rotation: bool = False,
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
    k = _n_params(n_components, shared_sigma, rotation)

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
        img, n_components, shared_sigma, sigma_p0, sigma_bounds, min_sep,
        threshold_rel, rotation,
    )

    try:
        result = least_squares(
            residuals,
            p0,
            args=(xx, yy, img, weights, n_components, shared_sigma, rotation),
            bounds=(lo, hi),
            method="trf",
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
        )
    except Exception:
        return _failed()

    params = _sort_by_amplitude(result.x, n_components, shared_sigma, rotation)
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


def _sort_by_amplitude(params: np.ndarray, n: int, shared_sigma: bool,
                       rotation: bool = False) -> np.ndarray:
    """Reorder components brightest-first, keeping the tail (sigmas, bg) in place.

    A stable sort leaves equal amplitudes in their fitted order, so a two-component
    vector is swapped exactly when the second is strictly brighter.
    """
    params = np.asarray(params, dtype=np.float64).copy()
    if n == 1:
        return params
    stride = 3 if shared_sigma else (3 + _shape_size(rotation))
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
    rotation: bool = False,
    amp_frac: float = 0.25,
    dip_frac: float = 0.05,
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
    amp_frac, dip_frac : float
        Resolvability thresholds, applied after the fit to decide which
        components count as distinct sub-peaks: a component must reach
        ``amp_frac`` of the brightest amplitude, and the model must dip by
        ``dip_frac`` of the shallower end between it and every component already
        kept.  Physical choices, not statistical ones.

        ``dip_frac`` is deliberately small because the dip is *exactly zero*
        below the Sparrow limit — two equal Gaussians closer than 2 sigma sum to
        a single-peaked profile, so any dip at all already means they are
        separated.  Measured on the noiseless model, the dip runs:

            separation      2.0σ   2.5σ   3.0σ   4.0σ
            equal peaks     0.000  0.123  0.358  0.729
            A2/A1 = 0.6     0.000  0.000  0.198  0.654
            A2/A1 = 0.3     0.000  0.000  0.017  0.535

        so 0.05 admits a genuine doublet just past the limit while still
        rejecting everything below it.  Raising it to the Rayleigh criterion's
        ~26% — which is stated for *equal* peaks — would demand nearly 3 sigma of
        an equal pair and 4 sigma of a 3:1 one.

        ``amp_frac`` catches what the dip test cannot: a faint component thrown
        far into a tail does produce a dip, so only its amplitude gives it away.
        On the reference data the third component sits at a median 10% of the
        peak, which 0.25 rejects.  See the module docstring for why the count
        cannot come from the selection criterion instead.
    min_counts : float
        Return the empty result when the raw ROI sums below this.

    Returns
    -------
    dict with a schema of fixed width, independent of the N actually chosen:
        x{1..n_max}, y{1..n_max}            component centres (px, crop frame)
        A{1..n_max}                          amplitudes
        sigma_x{1..n_max}, sigma_y{1..n_max} widths (repeated when shared)
        theta{1..n_max}                      angles (deg), NaN unless rotation=True
        bg                                   flat background
        n_fitted                             Gaussians the fit used (0 on failure)
        n_resolved                           of those, how many are distinct
                                             sub-peaks — the physical count
        n_params, chi2, chi2_raw, aic, bic, success
        separation, orientation              between the two resolved sub-peaks,
                                             NaN when fewer than two are resolved
        ratio                                A2 / (A1 + A2) of that same pair
        total_amplitude, centroid_x, centroid_y   over every fitted component
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
        rotation=rotation,
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

    return _flatten(best, best_n, n_max, shared_sigma, rotation, amp_frac, dip_frac)


# ── Resolvability ─────────────────────────────────────────────────────────────

def _has_valley(
    params: np.ndarray,
    n: int,
    shared_sigma: bool,
    rotation: bool,
    bg: float,
    a: np.ndarray,
    b: np.ndarray,
    dip_frac: float,
    n_samples: int = 33,
) -> bool:
    """Does the model dip between the two centres ``a`` and ``b``?

    Walks the fitted model along the segment joining them and compares the lowest
    point to the shallower end. The background is taken off first: a pedestal
    lifts both the ends and the valley, and would otherwise make every dip look
    shallow on a bright ROI.

    The model is sampled rather than the data, so noise cannot dig a false valley
    between two halves of a single lobe.
    """
    t  = np.linspace(0.0, 1.0, n_samples)
    xs = a[0] + t * (b[0] - a[0])
    ys = a[1] + t * (b[1] - a[1])
    prof = n_gaussians_2d(params, xs, ys, n, shared_sigma, rotation) - bg

    ends = min(float(prof[0]), float(prof[-1]))
    if ends <= 0:
        return False
    return (ends - float(prof.min())) / ends >= dip_frac


def _resolved_indices(
    params: np.ndarray,
    n: int,
    shared_sigma: bool,
    rotation: bool,
    amp_frac: float,
    dip_frac: float,
) -> list[int]:
    """Which fitted components stand as distinct sub-peaks.

    Two tests, one for each way a component can be an artefact of the model
    rather than a feature of the spot:

    * **amplitude** — a component fainter than ``amp_frac`` of the brightest is
      dropped. This is what a Gaussian recruited to paint a heavy tail looks
      like: measured on the reference data, a third component sits at a median
      10% of the peak, 2.4 widths out, with only 6% of them within one width.
    * **valley** — a component with no dip between it and one already kept is
      dropped. That is the other failure: two components describing one lobe
      between them, which no amount of separation alone would catch.

    The valley test also handles unequal amplitudes for free. A faint peak has to
    lie further out than an equal one before it shows as a bump on the flank of
    the bright one, and asking for a dip encodes that; a plain "separated by more
    than k sigma" rule does not.

    Components arrive sorted by descending amplitude, so index 0 is the brightest
    and is always kept.
    """
    comp, _, _, _, bg = _unpack(params, n, shared_sigma, rotation)
    if n < 2:
        return [0]

    amps = comp[:, 2]
    if amps[0] <= 0:
        return [0]

    kept = [0]
    for c in range(1, n):
        if amps[c] < amp_frac * amps[0]:
            continue
        if all(_has_valley(params, n, shared_sigma, rotation, bg,
                           comp[k], comp[c], dip_frac)
               for k in kept):
            kept.append(c)
    return kept


def _empty_result(n_max: int) -> dict:
    out: dict = {}
    for k in range(1, n_max + 1):
        out[f"x{k}"] = np.nan
        out[f"y{k}"] = np.nan
        out[f"A{k}"] = np.nan
        out[f"sigma_x{k}"] = np.nan
        out[f"sigma_y{k}"] = np.nan
        out[f"theta{k}"] = np.nan
    out.update({
        "bg": np.nan,
        "n_fitted": 0,
        "n_resolved": 0,
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


def _flatten(res: dict, n: int, n_max: int, shared_sigma: bool,
             rotation: bool = False, amp_frac: float = 0.25,
             dip_frac: float = 0.05) -> dict:
    params = res["params"]
    comp, sx, sy, theta, bg = _unpack(params, n, shared_sigma, rotation)

    out = _empty_result(n_max)
    for k in range(n):
        out[f"x{k + 1}"] = float(comp[k, 0])
        out[f"y{k + 1}"] = float(comp[k, 1])
        out[f"A{k + 1}"] = float(comp[k, 2])
        out[f"sigma_x{k + 1}"] = float(sx[k])
        out[f"sigma_y{k + 1}"] = float(sy[k])
        # NaN rather than 0 when not rotating: the angle was never a parameter,
        # and reporting it as measured-and-zero would be a different claim.
        out[f"theta{k + 1}"] = float(np.degrees(theta[k])) if rotation else np.nan

    out["bg"] = bg
    out["n_fitted"] = n
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

    kept = _resolved_indices(params, n, shared_sigma, rotation, amp_frac, dip_frac)
    out["n_resolved"] = len(kept)

    # The pair quantities describe two *resolved* sub-peaks or nothing at all.
    # Left unguarded they would report the distance to a tail-patching component,
    # which is a property of the fit rather than of the spot — and on this data
    # that is the majority of positions, so the maps would be mostly artefact.
    if len(kept) >= 2:
        p, q = comp[kept[0]], comp[kept[1]]
        dx, dy = float(q[0] - p[0]), float(q[1] - p[1])
        out["separation"]  = float(math.hypot(dx, dy))
        out["orientation"] = float(np.degrees(np.arctan2(dy, dx)))
        a1, a2 = float(p[2]), float(q[2])
        out["ratio"] = a2 / (a1 + a2 + 1e-9)
    return out


# ── Reconstruction ────────────────────────────────────────────────────────────

def model_from_result(result: dict, shape: tuple[int, int], shared_sigma: bool = True,
                      rotation: bool = False) -> np.ndarray:
    """Rebuild the fitted image from a ``fit_spot`` result, for residual plots.

    Returns an all-NaN array when the fit failed, so a residual panel shows the
    failure rather than a misleading flat model.
    """
    h, w = shape
    n = int(result.get("n_fitted", 0))
    if n < 1:
        return np.full(shape, np.nan)

    def shape_block(k: int) -> list[float]:
        block = [result[f"sigma_x{k}"], result[f"sigma_y{k}"]]
        if rotation:
            block.append(np.radians(result[f"theta{k}"]))
        return block

    params: list[float] = []
    for k in range(1, n + 1):
        params += [result[f"x{k}"], result[f"y{k}"], result[f"A{k}"]]
        if not shared_sigma:
            params += shape_block(k)
    params += (shape_block(1) if shared_sigma else []) + [result["bg"]]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return n_gaussians_2d(np.asarray(params), xx, yy, n, shared_sigma, rotation)
