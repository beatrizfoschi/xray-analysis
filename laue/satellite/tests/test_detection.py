"""Unit tests for laue.satellite.detection using synthetic MQW spot images.

Run with:  pytest laue/satellite/tests/test_detection.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the parent package importable from anywhere

from laue.satellite.detection import (
    subtract_background,
    find_sl0_centroid,
    find_satellite_axis,
    extract_1d_profile,
    detect_satellites,
    make_synthetic_image,
    locate_sl0_by_local_max,
    clip_hot_pixels,
)
from laue.satellite.metrics import compute_metrics, metrics_to_flat_dict


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def synthetic_image_35deg():
    """Standard synthetic image: 3 satellite orders, axis at 35°."""
    return make_synthetic_image(
        n_satellites=3, spacing=22.0, axis_angle=35.0,
        sl0_amplitude=5000.0, envelope_decay=0.45,
        fwhm=4.0, noise_level=40.0, seed=0,
    )


@pytest.fixture(scope='module')
def detection_result_35deg(synthetic_image_35deg):
    return detect_satellites(
        synthetic_image_35deg,
        axis_angle=35.0,
        n_max=3,
        min_prominence=0.04,
        strip_width=6.0,
        bg_sigma=20.0,
    )


# ── Background subtraction ────────────────────────────────────────────────────

class TestSubtractBackground:
    def test_output_is_non_negative(self):
        rng = np.random.default_rng(1)
        image = rng.normal(300, 30, (80, 80)).astype(np.float32)
        out = subtract_background(image)
        assert float(out.min()) >= 0.0

    def test_peak_preserved_relative_to_background(self):
        image = np.full((100, 100), 500.0, dtype=np.float32)
        image[50, 50] = 20000.0
        out = subtract_background(image, sigma=20.0)
        assert out[50, 50] > out[20, 20] * 10

    def test_output_dtype(self):
        image = np.ones((50, 50), dtype=np.uint16) * 100
        out = subtract_background(image)
        assert out.dtype == np.float32


# ── SL₀ centroid ─────────────────────────────────────────────────────────────

class TestFindSL0Centroid:
    def test_centred_peak(self):
        image = np.zeros((100, 100), dtype=np.float32)
        image[48:53, 48:53] = 1000.0
        cy, cx = find_sl0_centroid(image)
        assert abs(cy - 50) < 2
        assert abs(cx - 50) < 2

    def test_off_centre_peak(self):
        image = np.zeros((100, 100), dtype=np.float32)
        image[20:25, 70:75] = 500.0
        cy, cx = find_sl0_centroid(image)
        assert abs(cy - 22) < 3
        assert abs(cx - 72) < 3

    def test_empty_image_returns_centre(self):
        image = np.zeros((80, 80), dtype=np.float32)
        cy, cx = find_sl0_centroid(image)
        assert abs(cy - 40) < 5
        assert abs(cx - 40) < 5


# ── Axis detection ────────────────────────────────────────────────────────────

class TestFindSatelliteAxis:
    def test_horizontal_stripe(self):
        image = np.zeros((100, 100), dtype=np.float32)
        image[50, 10:90] = 500.0
        angle = find_satellite_axis(image, center=(50, 50))
        assert abs(angle % 180) < 10 or abs((angle % 180) - 180) < 10

    def test_vertical_stripe(self):
        image = np.zeros((100, 100), dtype=np.float32)
        image[10:90, 50] = 500.0
        angle = find_satellite_axis(image, center=(50, 50))
        diff = min(abs(abs(angle) - 90), abs(abs(angle) - 90 - 180))
        assert diff < 15

    @pytest.mark.parametrize('true_angle', [0.0, 30.0, 60.0, 90.0, 135.0])
    def test_known_axis_angles(self, true_angle):
        image = make_synthetic_image(
            axis_angle=true_angle, n_satellites=3, spacing=22.0,
            envelope_decay=0.3, noise_level=20.0, seed=5,
        )
        detected = find_satellite_axis(image)
        # Allow 180° ambiguity (axis direction is undetermined)
        diff = abs(detected - true_angle) % 180
        diff = min(diff, 180 - diff)
        assert diff < 20, f"Detected {detected:.1f}°, expected ~{true_angle:.1f}°"


# ── Profile extraction ────────────────────────────────────────────────────────

class TestExtract1DProfile:
    def test_returns_arrays_of_equal_length(self):
        image = make_synthetic_image(axis_angle=0.0, seed=7)
        cy, cx = find_sl0_centroid(image)
        d, i = extract_1d_profile(image, (cy, cx), axis_angle=0.0)
        assert len(d) == len(i)
        assert len(d) > 0

    def test_peak_at_centre(self):
        image = np.zeros((100, 100), dtype=np.float32)
        image[50, 50] = 1000.0
        d, i = extract_1d_profile(image, (50.0, 50.0), axis_angle=0.0, strip_width=3.0)
        peak_bin = np.argmax(i)
        assert abs(d[peak_bin]) < 3, "Profile maximum should be near s=0 for centred peak"


# ── SL0 location (README.md, "The bright peak in the ROI is the bulk") ─────────────────────────────────────

class TestLocateSl0ByLocalMax:
    """SL0 sits on the bulk's flank as a few standout pixels, so it is located
    by a bare 8-neighbour test rather than by fitting a line shape."""

    def test_single_local_maximum_is_confirmed(self):
        image = np.zeros((100, 100), dtype=np.float32)
        image += np.arange(100, dtype=np.float32) * 2.0   # smooth background ramp
        image[50, 56] = 500.0                              # the true SL0 pixel
        sl0_center = (50.0, 50.0)
        out = locate_sl0_by_local_max(image, sl0_pos_along_axis=6.0,
                                      axis_angle=0.0, sl0_center=sl0_center,
                                      boxsize=3.0)
        assert out['sl0_confirmed'] is True
        assert out['sl0_measured_position_2d'] == (50, 56)
        assert out['sl0_measured_pos'] == pytest.approx(6.0)
        assert out['sl0_measured_amplitude'] == pytest.approx(500.0)

    def test_smooth_ramp_has_no_local_maximum(self):
        image = np.tile(np.arange(100, dtype=np.float32), (100, 1))  # varies by column only
        out = locate_sl0_by_local_max(image, sl0_pos_along_axis=6.0,
                                      axis_angle=0.0, sl0_center=(50.0, 50.0),
                                      boxsize=3.0)
        assert out['sl0_confirmed'] is False
        assert '0 local maxima' in out['reason']

    def test_two_candidates_is_ambiguous(self):
        image = np.zeros((100, 100), dtype=np.float32)
        image[50, 55] = 300.0
        image[50, 57] = 300.0
        out = locate_sl0_by_local_max(image, sl0_pos_along_axis=6.0,
                                      axis_angle=0.0, sl0_center=(50.0, 50.0),
                                      boxsize=3.0)
        assert out['sl0_confirmed'] is False
        assert '2 local maxima' in out['reason']
        assert sorted((r, c) for r, c, _ in out['candidates']) == [(50, 55), (50, 57)]

    def test_search_box_is_symmetric_about_the_prediction(self):
        # Regression: floor()/ceil() bounds rounded outward asymmetrically, so
        # a fractional centre made boxsize=4 search 10x10 px instead of 9x9 and
        # swept in noise maxima from outside the requested window.
        image = np.zeros((100, 100), dtype=np.float32)
        image[50, 55] = 100.0            # inside  ±4 of the prediction
        image[50, 60] = 900.0            # outside ±4, must not be seen
        sl0_center = (50.0, 50.2)        # fractional, as on real data
        out = locate_sl0_by_local_max(image, sl0_pos_along_axis=5.0,
                                      axis_angle=0.0, sl0_center=sl0_center,
                                      boxsize=4.0)
        assert out['sl0_confirmed'] is True
        assert out['sl0_measured_position_2d'] == (50, 55)

    def test_narrow_peak_survives_only_on_raw_counts(self):
        # Regression, and the exact failure seen on frame 7354: the caller used
        # to pre-clean the crop with clip_hot_pixels, whose noise scale is a MAD
        # over the whole crop. A genuine narrow SL0 clears n_sigma*noise, is
        # replaced by its own 3x3 median, and vanishes before this function
        # ever runs — reported back as "0 local maxima in the box".
        # SL0 rides the bulk's steep tail here, as it does on real data; the
        # gradient is what keeps noise from throwing up spurious maxima.
        rng = np.random.default_rng(0)
        rr, _ = np.mgrid[0:161, 0:161]
        image = (400.0 * np.exp(-(rr - 68.0) / 3.0)).astype(np.float32)
        image += rng.normal(0, 3.0, image.shape).astype(np.float32)
        image[68, 82] += 450.0

        out = locate_sl0_by_local_max(image, sl0_pos_along_axis=8.0,
                                      axis_angle=90.0, sl0_center=(60.0, 82.0),
                                      boxsize=3.0)
        assert out['sl0_confirmed'] is True
        assert out['sl0_measured_position_2d'] == (68, 82)

        clipped = clip_hot_pixels(image, n_sigma=8.0)
        spoiled = locate_sl0_by_local_max(clipped, sl0_pos_along_axis=8.0,
                                          axis_angle=90.0, sl0_center=(60.0, 82.0),
                                          boxsize=3.0)
        assert spoiled['sl0_confirmed'] is False, (
            'clip_hot_pixels destroys this peak — the caller must pass the raw '
            'crop, never a hot-pixel-clipped one')

    def test_box_outside_image_is_not_confirmed(self):
        image = np.zeros((20, 20), dtype=np.float32)
        out = locate_sl0_by_local_max(image, sl0_pos_along_axis=500.0,
                                      axis_angle=0.0, sl0_center=(10.0, 10.0),
                                      boxsize=3.0)
        assert out['sl0_confirmed'] is False
        assert 'outside the image' in out['reason']

    def test_projection_respects_axis_angle(self):
        # axis_angle=90 deg: row = sl0_center[0] + s, col = sl0_center[1] (sin=1, cos=0)
        image = np.zeros((100, 100), dtype=np.float32)
        image[57, 50] = 500.0
        out = locate_sl0_by_local_max(image, sl0_pos_along_axis=7.0,
                                      axis_angle=90.0, sl0_center=(50.0, 50.0),
                                      boxsize=3.0)
        assert out['sl0_confirmed'] is True
        assert out['sl0_measured_position_2d'] == (57, 50)
        assert out['sl0_measured_pos'] == pytest.approx(7.0)


# ── Satellite detection ───────────────────────────────────────────────────────

class TestDetectSatellites:
    def test_finds_sl0(self, detection_result_35deg):
        orders = [p['order'] for p in detection_result_35deg['peaks']]
        assert 0 in orders, 'SL₀ (order 0) not detected'

    def test_finds_first_order_satellites(self, detection_result_35deg):
        orders = {p['order'] for p in detection_result_35deg['peaks']}
        assert 1 in orders or -1 in orders, 'No first-order satellites detected'

    def test_sl0_is_brightest(self, detection_result_35deg):
        peaks = detection_result_35deg['peaks']
        amps = {p['order']: p['amplitude'] for p in peaks}
        sl0_amp = amps.get(0, 0.0)
        for order, amp in amps.items():
            if order != 0:
                assert amp <= sl0_amp * 1.05, (
                    f"Order {order} amplitude ({amp:.0f}) > SL₀ ({sl0_amp:.0f})"
                )

    def test_amplitude_decreases_with_order(self, detection_result_35deg):
        peaks = {p['order']: p['amplitude'] for p in detection_result_35deg['peaks']}
        sl0 = peaks.get(0, 0.0)
        for n in [1, 2, 3]:
            for sign in [1, -1]:
                if sign * n in peaks:
                    assert peaks[sign * n] < sl0, (
                        f'Order {sign*n} amp {peaks[sign*n]:.0f} ≥ SL₀ amp {sl0:.0f}'
                    )

    def test_result_keys(self, detection_result_35deg):
        expected = {'peaks', 'axis_angle', 'sl0_center', 'profile', 'image_sub'}
        assert expected.issubset(detection_result_35deg.keys())

    def test_profile_shape(self, detection_result_35deg):
        d, i = detection_result_35deg['profile']
        assert d.shape == i.shape
        assert len(d) > 10

    def test_empty_image_returns_no_peaks(self):
        result = detect_satellites(np.zeros((100, 100), dtype=np.float32))
        assert result['peaks'] == []

    def test_auto_axis_detection(self, synthetic_image_35deg):
        result = detect_satellites(synthetic_image_35deg, axis_angle=None, min_prominence=0.04)
        diff = abs(result['axis_angle'] - 35.0) % 180
        diff = min(diff, 180 - diff)
        assert diff < 20, f"Auto-detected axis {result['axis_angle']:.1f}° too far from 35°"

    @pytest.mark.parametrize('axis_angle', [0.0, 45.0, 90.0, -30.0, 120.0])
    def test_multiple_axis_angles(self, axis_angle):
        img = make_synthetic_image(axis_angle=axis_angle, noise_level=30.0, seed=10)
        result = detect_satellites(img, axis_angle=axis_angle, min_prominence=0.04)
        orders = [p['order'] for p in result['peaks']]
        assert 0 in orders, f'SL₀ not found for axis_angle={axis_angle}°'

    def test_n_max_filters_high_orders(self):
        img = make_synthetic_image(n_satellites=5, spacing=15.0, axis_angle=0.0)
        result = detect_satellites(img, axis_angle=0.0, n_max=2, min_prominence=0.03)
        for p in result['peaks']:
            assert abs(p['order']) <= 2, f'Order {p["order"]} exceeds n_max=2'


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_n_sat_count(self, detection_result_35deg):
        m = compute_metrics(detection_result_35deg['peaks'])
        orders = [p['order'] for p in detection_result_35deg['peaks']]
        expected_n_sat = sum(1 for o in orders if o != 0)
        assert m['n_sat'] == expected_n_sat

    def test_delta_q_positive_for_positive_spacing(self, detection_result_35deg):
        m = compute_metrics(detection_result_35deg['peaks'])
        if not np.isnan(m['delta_q']):
            # Expected spacing ~22 px; delta_q is positive (higher orders at +x)
            assert 10 < abs(m['delta_q']) < 40, f"Unexpected delta_q = {m['delta_q']}"

    def test_alpha_positive(self, detection_result_35deg):
        m = compute_metrics(detection_result_35deg['peaks'])
        if not np.isnan(m['alpha']):
            assert m['alpha'] > 0, 'Envelope decay α should be positive'

    def test_empty_peaks_returns_nans(self):
        m = compute_metrics([])
        assert m['n_sat'] == 0
        assert np.isnan(m['delta_q'])
        assert np.isnan(m['alpha'])

    def test_flat_dict_serialisation(self, detection_result_35deg):
        m = compute_metrics(detection_result_35deg['peaks'])
        flat = metrics_to_flat_dict(m, prefix='sat_')
        assert all(isinstance(v, float) for v in flat.values())
        assert 'sat_n_sat' in flat
        assert 'sat_alpha' in flat

    def test_bulk_pos_is_the_detected_order_zero_peak(self, detection_result_35deg):
        """`bulk_pos`, not `sl0_pos` — the bright peak is the bulk in an MQW.

        Renamed 2026-08-12.  The quantity is unchanged; the old name asserted
        something the data disproved (README.md, "The bright peak in the ROI is the bulk"), and the physical
        reading of the indicator ("mean strain relative to the bulk") belongs to
        `bulk_sl0_offset_px`, which the Laue routes add downstream.
        """
        peaks = detection_result_35deg['peaks']
        m = compute_metrics(peaks)

        assert 'sl0_pos' not in m, (
            'sl0_pos must not come out of compute_metrics: this layer cannot '
            'know where SL0 is, only where the order-0 peak was detected.')
        assert 'bulk_pos' in m

        order0 = [p for p in peaks if p['order'] == 0]
        if order0:
            assert m['bulk_pos'] == pytest.approx(order0[0]['pos_along_axis'])
        else:
            assert np.isnan(m['bulk_pos'])

    def test_empty_metrics_carries_the_new_name(self):
        m = compute_metrics([])
        assert 'bulk_pos' in m and 'sl0_pos' not in m
        assert np.isnan(m['bulk_pos'])
