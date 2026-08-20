"""Regression tests freezing the current numerical behaviour of `emission`.

Written before merging duplicated helpers, so that a merge which changes results
fails loudly instead of silently shifting fitted peak positions and, through
``wl_to_In_fraction``, the reported In composition maps.

The frozen literals were produced by the code as it stood on 2026-08-11 with
numpy 1.x / scipy 1.x / scikit-learn 1.3.2. They are not analytically derived —
they are a snapshot. If one of them changes, the question to answer is *why*,
not whether to update the number.
"""

from __future__ import annotations

import numpy as np
import pytest

from emission.NMF import nmf_sklearn_hyperspectral, run_nmf
from emission.nmf_per_seg import _run_nmf
from emission.stats_utils import fit_peak_get_fwhm
from emission.xeol_peak_map import _fit_one, wl_to_In_fraction
from utils.fitting import gaussian, r_squared


# ── Fixtures: one clean and one noisy synthetic emission peak ─────────────────

WL = np.linspace(440.0, 470.0, 121)
TRUE_CENTER, TRUE_AMP, TRUE_SIGMA, TRUE_BG = 455.3, 100.0, 3.7, 12.0

SPEC_CLEAN = TRUE_AMP * np.exp(-0.5 * ((WL - TRUE_CENTER) / TRUE_SIGMA) ** 2) + TRUE_BG
SPEC_NOISY = SPEC_CLEAN + np.random.default_rng(12345).normal(0.0, 0.5, WL.size)


# ── The three Gaussian models ────────────────────────────────────────────────

def test_gaussian_matches_frozen_value():
    assert gaussian(455.0, 100.0, 455.3, 3.7, 12.0) == pytest.approx(
        111.6718325648248, rel=0, abs=1e-12
    )


def test_omitting_the_baseline_terms_reproduces_the_simpler_models():
    """The unified model subsumes the two it replaced — exactly, not approximately.

    `xeol_peak_map._gaussian` was this with `slope=0`; `plot_histograms._gauss`
    was this with `background=slope=0`. This is what licensed merging them.
    """
    x = np.linspace(440.0, 470.0, 57)

    constant_baseline = 100.0 * np.exp(-0.5 * ((x - 455.3) / 3.7) ** 2) + 12.0
    np.testing.assert_array_equal(gaussian(x, 100.0, 455.3, 3.7, 12.0), constant_baseline)
    np.testing.assert_array_equal(
        gaussian(x, 100.0, 455.3, 3.7, 12.0, 0.0), constant_baseline
    )

    no_baseline = 100.0 * np.exp(-0.5 * ((x - 455.3) / 3.7) ** 2)
    np.testing.assert_array_equal(gaussian(x, 100.0, 455.3, 3.7), no_baseline)


def test_r_squared_matches_frozen_value():
    assert r_squared(SPEC_NOISY, SPEC_CLEAN) == pytest.approx(
        0.9998135201148269, rel=0, abs=1e-12
    )


# ── The two peak fits ────────────────────────────────────────────────────────

def test_fit_one_on_clean_peak_recovers_truth():
    center, amp, fwhm, bg, r2, ok = _fit_one(WL, SPEC_CLEAN)
    assert ok
    assert center == pytest.approx(455.3, abs=1e-9)
    assert amp == pytest.approx(100.0, abs=1e-9)
    assert fwhm == pytest.approx(8.712834166614513, abs=1e-9)
    assert bg == pytest.approx(12.0, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_fit_one_on_noisy_peak_matches_frozen_values():
    center, amp, fwhm, bg, r2, ok = _fit_one(WL, SPEC_NOISY)
    assert ok
    assert center == pytest.approx(455.2947661254667, abs=1e-9)
    assert amp == pytest.approx(99.79463190854123, abs=1e-9)
    assert fwhm == pytest.approx(8.718064096980694, abs=1e-9)
    assert bg == pytest.approx(12.047445615485447, abs=1e-9)
    assert r2 == pytest.approx(0.9998194421517689, abs=1e-12)


def test_fit_peak_get_fwhm_on_clean_peak_matches_frozen_values():
    wl0, fwhm, ok = fit_peak_get_fwhm(WL, SPEC_CLEAN, wl_roi=(440, 470))
    assert ok
    assert wl0 == pytest.approx(455.3, abs=1e-9)
    assert fwhm == pytest.approx(8.712834166614469, abs=1e-9)


def test_fit_peak_get_fwhm_on_noisy_peak_matches_frozen_values():
    wl0, fwhm, ok = fit_peak_get_fwhm(WL, SPEC_NOISY, wl_roi=(440, 470))
    assert ok
    # The literal is from before the baseline parameters were reordered to match
    # utils.fitting.gaussian. Reordering is mathematically a no-op but changes the
    # trust-region path, moving the fitted centre by 1.3e-7 nm — a fraction of the
    # 0.25 nm sampling step. Both orderings reach the same optimum: their sums of
    # squared residuals agree to 2e-11 and the fitted curves to 1.5e-6.
    assert wl0 == pytest.approx(455.283103700734, abs=1e-6)
    assert fwhm == pytest.approx(8.821637492823067, abs=1e-7)


def test_the_two_fits_agree_on_clean_data_but_not_on_noisy_data():
    """The two fits are NOT interchangeable — this is the merge hazard.

    On a noiseless peak the constant and linear baselines coincide. On noisy
    data the linear baseline of ``fit_peak_get_fwhm`` absorbs part of the wings,
    widening the FWHM by ~1.2 % relative to ``_fit_one``. Any merge must keep the
    baseline model an explicit choice rather than silently picking one, because
    the fitted centre feeds ``wl_to_In_fraction``.
    """
    _, _, fwhm_a, *_ = _fit_one(WL, SPEC_CLEAN)
    _, fwhm_b, _ = fit_peak_get_fwhm(WL, SPEC_CLEAN, wl_roi=(440, 470))
    assert fwhm_a == pytest.approx(fwhm_b, rel=1e-9)

    _, _, fwhm_a, *_ = _fit_one(WL, SPEC_NOISY)
    _, fwhm_b, _ = fit_peak_get_fwhm(WL, SPEC_NOISY, wl_roi=(440, 470))
    assert fwhm_b / fwhm_a == pytest.approx(1.0119, abs=5e-4)


# ── In composition ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "wl_nm, expected",
    [
        (400.0, 0.0855787624779975),
        (450.0, 0.17808580824111025),
        (500.0, 0.25731804487638477),
        (550.0, 0.32635367447623864),
    ],
)
def test_wl_to_In_fraction_matches_frozen_values(wl_nm, expected):
    got = wl_to_In_fraction(np.array([wl_nm]))[0]
    assert got == pytest.approx(expected, rel=0, abs=1e-12)


def test_In_fraction_increases_with_wavelength():
    x = wl_to_In_fraction(np.array([400.0, 450.0, 500.0, 550.0]))
    assert np.all(np.diff(x) > 0)


# ── NMF ──────────────────────────────────────────────────────────────────────

def _synthetic_cube():
    """Low-rank non-negative data with one negative entry, to exercise clipping."""
    rng = np.random.default_rng(7)
    nx, ny, n_channels, k = 6, 5, 40, 3
    H = np.abs(rng.normal(1.0, 0.3, (k, n_channels)))
    W = np.abs(rng.normal(1.0, 0.5, (nx * ny, k)))
    X = W @ H
    X[0, 0] = -1.0
    return X, (nx, ny), k


def test_run_nmf_clips_negatives_by_default_and_refuses_them_otherwise():
    """The clipping that `nmf_per_seg._clip_nonneg` used to do now lives in the core."""
    X, _, k = _synthetic_cube()
    assert X.min() < 0

    W, H, _, _, _ = run_nmf(X, k, random_state=0)
    assert np.all(W >= 0) and np.all(H >= 0)

    with pytest.raises(ValueError, match="non-negative"):
        run_nmf(X, k, random_state=0, clip_negative=False)


@pytest.mark.parametrize(
    "loss, max_iter",
    [("frobenius", 1000), ("kullback-leibler", 2000)],
)
def test_the_two_nmf_implementations_are_bit_identical(loss, max_iter):
    """`_run_nmf` and `nmf_sklearn_hyperspectral` share their whole core.

    Equality here is exact, not approximate — this is what makes merging them a
    pure deduplication rather than a numerical change.
    """
    X, map_shape, k = _synthetic_cube()
    nx, ny = map_shape

    W, H, rmse, _ = _run_nmf(X, k, loss=loss, max_iter=max_iter, random_state=0)
    W_maps, H_h, _, E_map, _, _, _ = nmf_sklearn_hyperspectral(
        X, map_shape, k, loss=loss, max_iter=max_iter, random_state=0,
        show_progress=False,
    )

    np.testing.assert_array_equal(W.reshape(nx, ny, k), W_maps)
    np.testing.assert_array_equal(H, H_h)
    np.testing.assert_array_equal(rmse.reshape(nx, ny), E_map)


@pytest.mark.parametrize(
    "loss, max_iter, w_sum, h_sum, rmse_mean",
    [
        ("frobenius", 1000, 37.883955832530, 255.613798109104, 0.025037608529),
        ("kullback-leibler", 2000, 50.134545980042, 180.442210964082, 0.049656220940),
    ],
)
def test_nmf_output_matches_frozen_values(loss, max_iter, w_sum, h_sum, rmse_mean):
    X, _, k = _synthetic_cube()
    W, H, rmse, _ = _run_nmf(X, k, loss=loss, max_iter=max_iter, random_state=0)
    assert W.sum() == pytest.approx(w_sum, abs=1e-9)
    assert H.sum() == pytest.approx(h_sum, abs=1e-9)
    assert rmse.mean() == pytest.approx(rmse_mean, abs=1e-9)


def test_nmf_defaults_differ_between_the_two_entry_points():
    """Documented hazard: merging must not unify these defaults by accident."""
    import inspect

    run_defaults = inspect.signature(_run_nmf).parameters
    hyper_defaults = inspect.signature(nmf_sklearn_hyperspectral).parameters

    assert run_defaults["loss"].default == "kullback-leibler"
    assert hyper_defaults["loss"].default == "frobenius"
    assert run_defaults["max_iter"].default == 2000
    assert hyper_defaults["max_iter"].default == 1000
