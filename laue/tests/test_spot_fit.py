"""Tests for the parametric multi-Gaussian spot fit.

The parity block is the important one. `spot_fit` generalises a two-Gaussian
fitter that had already produced results being interpreted (notebook
`02_parametric_fit`, spot r1972/c873 of H4_M5773W1_B04), so the generalisation
has to reproduce it exactly at N=2, not merely agree with it. `_reference_fit_pixel`
below is that original, copied verbatim, and the parity tests assert bit-identical
output rather than approximate agreement.

Checked once against the real data behind that notebook: over 60 ROIs drawn from
its 4941-position scan, `fit_n_gaussians(roi, 2)` matched `_reference_fit_pixel`
to max |Δparams| = 0 and max |Δchi²| = 0, and the full chain (align → fit)
reproduced the cached parameters to float32 precision in the positions and widths.
The amplitudes and background differ there by a single per-pixel factor of
0.976–1.036, which is the monitor (I₀) normalisation the notebook applies before
fitting and this test cannot reach without the beamline HDF5.
"""

from __future__ import annotations

import numpy as np
import pytest

from laue.spot_fit import (
    _initial_peaks,
    _n_from_length,
    fit_n_gaussians,
    fit_spot,
    model_from_result,
    n_gaussians_2d,
)


# ── The original two-Gaussian fitter, verbatim ────────────────────────────────

def _reference_two_gaussians_2d(params, xx, yy):
    x1, y1, A1, x2, y2, A2, sx, sy, bg = params
    sx = max(sx, 0.3); sy = max(sy, 0.3)
    g1 = A1 * np.exp(-((xx - x1)**2/(2*sx**2) + (yy - y1)**2/(2*sy**2)))
    g2 = A2 * np.exp(-((xx - x2)**2/(2*sx**2) + (yy - y2)**2/(2*sy**2)))
    return g1 + g2 + bg


def _reference_fit_pixel(args):
    from scipy.ndimage import maximum_filter
    from scipy.optimize import least_squares

    def residuals(params, xx, yy, data, weights):
        return (_reference_two_gaussians_2d(params, xx, yy) - data).ravel() * weights.ravel()

    def find_two_peaks(img, min_sep=2):
        peaks = (img == maximum_filter(img, size=3))
        peaks &= img > 0.1 * img.max()
        ys, xs = np.where(peaks)
        if len(ys) == 0:
            h, w = img.shape
            return (w/2, h/2, img.max()), (w/2, h/2, 0.5*img.max())
        vals = img[ys, xs]
        order = np.argsort(-vals)
        ys, xs, vals = ys[order], xs[order], vals[order]
        p1 = (xs[0], ys[0], vals[0])
        p2 = None
        for i in range(1, len(xs)):
            if np.hypot(xs[i] - p1[0], ys[i] - p1[1]) >= min_sep:
                p2 = (xs[i], ys[i], vals[i])
                break
        if p2 is None:
            h, w = img.shape
            p2 = (p1[0] + 1, p1[1] + 1, 0.3 * p1[2])
        return p1, p2

    idx, roi = args
    roi = roi.astype(np.float64)
    h, w = roi.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    bg0 = np.percentile(roi, 20)
    signal = np.clip(roi - bg0, 0, None)
    p1, p2 = find_two_peaks(signal)

    p0 = np.array([p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], 2.0, 2.0, bg0])
    lo = np.array([0, 0, 0, 0, 0, 0, 0.5, 0.5, 0])
    hi = np.array([w, h, roi.max()*2, w, h, roi.max()*2, 8.0, 8.0, roi.max()])
    weights = 1.0 / np.sqrt(np.clip(roi, 1.0, None))

    try:
        result = least_squares(residuals, p0, args=(xx, yy, roi, weights),
                               bounds=(lo, hi), method="trf",
                               max_nfev=200, ftol=1e-6, xtol=1e-6)
        params = result.x
        chi2 = np.sum(result.fun**2) / (roi.size - len(p0))
        success = result.success
    except Exception:
        params = np.full(9, np.nan)
        chi2 = np.nan
        success = False

    if not np.isnan(params[0]) and params[5] > params[2]:
        params = np.concatenate([params[3:6], params[0:3], params[6:]])
    return idx, params, chi2, success


# ── Synthetic ROIs ────────────────────────────────────────────────────────────

def _make_roi(peaks, *, shape=(20, 20), sigma=2.0, bg=100.0, seed=0, noise=True):
    """Poisson-noisy ROI holding the given (x, y, amplitude) Gaussians."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.full(shape, float(bg))
    for x0, y0, amp in peaks:
        img += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    if noise:
        img = np.random.default_rng(seed).poisson(img).astype(np.float64)
    return img


SINGLE = _make_roi([(10.0, 10.0, 5000.0)], seed=1)
DOUBLET = _make_roi([(7.0, 10.0, 5000.0), (13.0, 10.0, 3000.0)], seed=2)
TRIPLET = _make_roi(
    [(6.0, 10.0, 5000.0), (10.0, 6.0, 4000.0), (14.0, 13.0, 3500.0)], seed=3
)


# ── Parity with the original implementation ───────────────────────────────────

@pytest.mark.parametrize("roi", [SINGLE, DOUBLET, TRIPLET], ids=["single", "doublet", "triplet"])
def test_two_gaussian_fit_is_bit_identical_to_the_original(roi):
    _, ref_params, ref_chi2, ref_success = _reference_fit_pixel((0, roi.copy()))
    mine = fit_n_gaussians(roi, 2, shared_sigma=True)

    assert mine["success"] == bool(ref_success)
    assert np.array_equal(mine["params"], ref_params)
    assert mine["chi2"] == ref_chi2


def test_parity_holds_on_random_rois():
    """Random ROIs reach corners of the fit the shaped examples do not."""
    rng = np.random.default_rng(7)
    for _ in range(15):
        roi = rng.poisson(
            _make_roi(
                [(rng.uniform(4, 16), rng.uniform(4, 16), rng.uniform(500, 8000)),
                 (rng.uniform(4, 16), rng.uniform(4, 16), rng.uniform(500, 8000))],
                bg=rng.uniform(10, 500), noise=False,
            )
        ).astype(np.float64)
        _, ref_params, ref_chi2, ref_success = _reference_fit_pixel((0, roi.copy()))
        mine = fit_n_gaussians(roi, 2, shared_sigma=True)
        assert mine["success"] == bool(ref_success)
        assert np.array_equal(mine["params"], ref_params)


def test_model_matches_the_original_formula():
    yy, xx = np.mgrid[0:20, 0:20].astype(np.float64)
    params = np.array([7.0, 10.0, 5000.0, 13.0, 11.0, 3000.0, 2.0, 2.5, 120.0])
    assert np.array_equal(
        n_gaussians_2d(params, xx, yy), _reference_two_gaussians_2d(params, xx, yy)
    )


def test_n_components_is_inferred_from_the_parameter_length():
    yy, xx = np.mgrid[0:8, 0:8].astype(np.float64)
    params = np.array([4.0, 4.0, 100.0, 5.0, 5.0, 50.0, 2.0, 2.0, 1.0])
    assert np.array_equal(
        n_gaussians_2d(params, xx, yy), n_gaussians_2d(params, xx, yy, 2, True)
    )


@pytest.mark.parametrize(
    "n, shared, expected",
    [(1, True, 6), (2, True, 9), (3, True, 12), (1, False, 6), (2, False, 11), (3, False, 16)],
)
def test_parameter_vector_length(n, shared, expected):
    assert _n_from_length(expected, shared) == n


def test_ambiguous_length_is_rejected():
    with pytest.raises(ValueError):
        _n_from_length(10, shared_sigma=True)


# ── Recovering known parameters ───────────────────────────────────────────────

def test_fit_recovers_the_positions_it_was_built_from():
    res = fit_spot(DOUBLET, n_components=2)
    assert res["success"]
    # Brightest component first, so peak 1 is the amplitude-5000 one at x=7.
    assert res["x1"] == pytest.approx(7.0, abs=0.3)
    assert res["y1"] == pytest.approx(10.0, abs=0.3)
    assert res["x2"] == pytest.approx(13.0, abs=0.3)
    assert res["separation"] == pytest.approx(6.0, abs=0.4)
    assert res["orientation"] == pytest.approx(0.0, abs=5.0)
    assert res["ratio"] == pytest.approx(3000 / 8000, abs=0.06)


def test_components_come_back_brightest_first():
    """Same doublet with the amplitudes swapped must give the same peak 1."""
    faint_first = _make_roi([(7.0, 10.0, 3000.0), (13.0, 10.0, 5000.0)], seed=2)
    res = fit_spot(faint_first, n_components=2)
    assert res["A1"] > res["A2"]
    assert res["x1"] == pytest.approx(13.0, abs=0.3)


def test_shared_sigma_repeats_one_width_across_components():
    res = fit_spot(DOUBLET, n_components=2, shared_sigma=True)
    assert res["sigma_x1"] == res["sigma_x2"]
    assert res["sigma_y1"] == res["sigma_y2"]
    assert res["n_params"] == 9


def test_per_component_sigma_frees_the_widths():
    res = fit_spot(DOUBLET, n_components=2, shared_sigma=False)
    assert res["success"]
    assert res["n_params"] == 11


# ── Model selection ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("criterion", ["bic", "aic"])
def test_auto_picks_one_component_for_a_single_peak(criterion):
    res = fit_spot(SINGLE, n_components="auto", criterion=criterion)
    assert res["n_components"] == 1
    assert np.isnan(res["x2"])
    assert np.isnan(res["separation"])


@pytest.mark.parametrize("criterion", ["bic", "aic"])
def test_auto_finds_the_second_component_of_a_doublet(criterion):
    res = fit_spot(DOUBLET, n_components="auto", criterion=criterion)
    assert res["n_components"] == 2
    assert res["separation"] == pytest.approx(6.0, abs=0.5)


def test_bic_is_not_more_permissive_than_aic():
    """BIC penalises each parameter by ln(n_pixels), AIC by 2 — so BIC never grows N further."""
    for roi in (SINGLE, DOUBLET, TRIPLET):
        n_bic = fit_spot(roi, n_components="auto", criterion="bic")["n_components"]
        n_aic = fit_spot(roi, n_components="auto", criterion="aic")["n_components"]
        assert n_bic <= n_aic


def test_chi2_criterion_stops_at_the_first_acceptable_fit():
    """A generous threshold is met by N=1 even on a doublet; a strict one is not."""
    loose = fit_spot(DOUBLET, n_components="auto", criterion="chi2", chi2_threshold=1e9)
    assert loose["n_components"] == 1
    strict = fit_spot(DOUBLET, n_components="auto", criterion="chi2", chi2_threshold=1e-9)
    assert strict["n_components"] >= 2


def test_selection_criteria_profile_the_noise_scale_out():
    """The log form is what keeps the penalty comparable to the chi² it fights.

    With the textbook ``chi2 + 2k``, a reduced chi² of ~200 — which is what a
    two-Gaussian model actually achieves on a real streaked Laue spot — makes the
    penalty negligible and every position is assigned n_max.
    """
    res = fit_n_gaussians(DOUBLET, 2)
    k, n_pix = res["n_params"], DOUBLET.size
    log_term = n_pix * np.log(res["chi2_raw"] / n_pix)
    assert res["chi2"] == pytest.approx(res["chi2_raw"] / (n_pix - k))
    assert res["aic"] == pytest.approx(log_term + 2 * k)
    assert res["bic"] == pytest.approx(log_term + k * np.log(n_pix))


def test_criteria_are_invariant_to_rescaling_the_roi():
    """Profiling the noise scale out means a uniform gain change cannot move N.

    A monitor normalisation is exactly such a rescaling, so the selected N must
    not depend on whether it was applied.
    """
    n_plain = fit_spot(DOUBLET, n_components="auto", criterion="bic")["n_components"]
    n_scaled = fit_spot(DOUBLET * 4.0, n_components="auto", criterion="bic")["n_components"]
    assert n_plain == n_scaled


def _rotated_streak(seed=11):
    """A tilted elongated spot: no N of axis-aligned Gaussians fits it well."""
    h, w = 20, 20
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ang = np.radians(35.0)
    u = (xx - 10) * np.cos(ang) + (yy - 10) * np.sin(ang)
    v = -(xx - 10) * np.sin(ang) + (yy - 10) * np.cos(ang)
    streak = 200.0 + 6000.0 * np.exp(-(u ** 2 / (2 * 6.0 ** 2) + v ** 2 / (2 * 1.5 ** 2)))
    return np.random.default_rng(seed).poisson(streak).astype(np.float64)


def test_the_penalty_still_bites_when_chi2_is_dominated_by_model_error():
    """Regression for the criterion that could only ever return n_max.

    Real Laue spots are streaked and the model has no rotation term, so the
    residual stays far above the counting noise at every N. Under ``chi2 + 2k``
    the penalty is then negligible against the chi² differences and N is pinned
    to n_max; the selection has to stay meaningful anyway.
    """
    roi = _rotated_streak()
    assert fit_n_gaussians(roi, 2)["chi2"] > 10, "expected a badly-fitting ROI"

    two, three = fit_n_gaussians(roi, 2), fit_n_gaussians(roi, 3)
    gain = two["bic"] - three["bic"]
    penalty = (three["n_params"] - two["n_params"]) * np.log(roi.size)
    # The penalty is what a third component has to earn. It is worthless once it
    # is orders of magnitude below the gain it is meant to offset.
    assert abs(gain) < 100 * penalty


def test_the_selected_n_survives_a_gain_change():
    """A monitor normalisation rescales the ROI; it must not move N.

    This is the property the textbook ``chi2 + 2k`` form lacks: chi² scales with
    the data while the penalty does not, so a rescaled ROI would drift to n_max.
    """
    roi = _rotated_streak()
    for criterion in ("bic", "aic"):
        plain = fit_spot(roi, n_components="auto", criterion=criterion)["n_components"]
        scaled = fit_spot(roi * 1000.0, n_components="auto", criterion=criterion)["n_components"]
        assert plain == scaled


def test_n_max_caps_the_search():
    res = fit_spot(TRIPLET, n_components="auto", n_max=2, criterion="aic")
    assert res["n_components"] <= 2


def test_unknown_criterion_is_rejected():
    with pytest.raises(ValueError, match="criterion"):
        fit_spot(DOUBLET, n_components="auto", criterion="rsquared")


def test_n_components_outside_the_cap_is_rejected():
    with pytest.raises(ValueError, match="n_max"):
        fit_spot(DOUBLET, n_components=5, n_max=3)


# ── Output schema ─────────────────────────────────────────────────────────────

def test_schema_width_does_not_depend_on_the_selected_n():
    """A DataFrame built from mixed-N positions must stay rectangular."""
    keys = [set(fit_spot(roi, n_components="auto", n_max=3).keys())
            for roi in (SINGLE, DOUBLET, TRIPLET)]
    assert keys[0] == keys[1] == keys[2]
    assert "x3" in keys[0] and "sigma_y3" in keys[0]


def test_columns_above_the_selected_n_are_nan():
    res = fit_spot(SINGLE, n_components=1, n_max=3)
    assert res["n_components"] == 1
    for k in (2, 3):
        assert np.isnan(res[f"x{k}"]) and np.isnan(res[f"A{k}"])


def test_derived_quantities_use_every_component():
    res = fit_spot(TRIPLET, n_components=3)
    total = res["A1"] + res["A2"] + res["A3"]
    assert res["total_amplitude"] == pytest.approx(total)
    expected_cx = (res["A1"] * res["x1"] + res["A2"] * res["x2"]
                   + res["A3"] * res["x3"]) / total
    assert res["centroid_x"] == pytest.approx(expected_cx)


# ── Degenerate input ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "roi",
    [
        np.zeros((20, 20)),
        np.full((20, 20), np.nan),
        np.full((2, 2), 100.0),          # fewer pixels than free parameters
    ],
    ids=["empty", "all-nan", "too-small"],
)
def test_degenerate_rois_return_the_empty_result_without_raising(roi):
    res = fit_spot(roi, n_components=2)
    assert res["success"] is False
    assert res["n_components"] == 0
    assert np.isnan(res["x1"])


def test_min_counts_rejects_a_faint_roi():
    faint = np.full((20, 20), 1.0)
    assert fit_spot(faint, n_components=1, min_counts=1e6)["n_components"] == 0


def test_a_background_subtracted_roi_still_fits():
    """Negative pixels would put the initial background below its own bound."""
    res = fit_spot(DOUBLET - np.median(DOUBLET), n_components=2)
    assert res["success"]


def test_non_2d_input_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        fit_spot(np.zeros((3, 20, 20)))


# ── Reconstruction ────────────────────────────────────────────────────────────

def test_model_from_result_reproduces_the_fitted_image():
    res = fit_spot(DOUBLET, n_components=2)
    model = model_from_result(res, DOUBLET.shape)
    direct = fit_n_gaussians(DOUBLET, 2)["params"]
    yy, xx = np.mgrid[0:20, 0:20].astype(np.float64)
    assert np.allclose(model, n_gaussians_2d(direct, xx, yy))


def test_model_from_a_failed_fit_is_nan_not_a_flat_plane():
    model = model_from_result(fit_spot(np.zeros((20, 20))), (20, 20))
    assert np.isnan(model).all()


# ── Initial guess ─────────────────────────────────────────────────────────────

def test_initial_peaks_are_ordered_and_separated():
    peaks = _initial_peaks(DOUBLET - DOUBLET.min(), 2, min_sep=2.0)
    assert len(peaks) == 2
    assert peaks[0][2] >= peaks[1][2]
    assert np.hypot(peaks[0][0] - peaks[1][0], peaks[0][1] - peaks[1][1]) >= 2.0


def test_initial_peaks_pad_when_the_spot_has_only_one_maximum():
    peaks = _initial_peaks(SINGLE - SINGLE.min(), 3, min_sep=2.0)
    assert len(peaks) == 3
    assert peaks[1][2] < peaks[0][2]
