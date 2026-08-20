"""Regression tests for ROI cropping, written before deduplicating it.

`_extract_crop` existed identically in `run_single_spot` and
`satellite/run_single_image`, and `scan_pipeline._crop_roi` did the same thing
with a different argument order and no origin returned. The literals below are
the behaviour as it stood on 2026-08-11.

The centre convention is the trap here: `extract_crop` takes ``(x, y)`` — column
first, XMAS 1-based by default — while `_crop_roi` takes ``(row, col)``, 0-based.
"""

from __future__ import annotations

import numpy as np
import pytest

from laue._imaging import extract_crop


FRAME = np.random.default_rng(3).integers(0, 1000, (60, 80)).astype(np.float32)


@pytest.mark.parametrize(
    "center, boxsize, coords, expected_sum, expected_origin",
    [
        ((40, 30), 5, "xmas", 62765.0, (24, 34)),
        ((40, 30), 5, "python", 64867.0, (25, 35)),
        # centre near the corner: the crop is zero-padded and the origin goes negative
        ((2, 2), 5, "xmas", 21711.0, (-4, -4)),
        ((2, 2), 5, "python", 29971.0, (-3, -3)),
    ],
)
def test_crop_matches_frozen_values(center, boxsize, coords, expected_sum, expected_origin):
    crop, origin = extract_crop(FRAME, center, boxsize, coords)
    assert crop.shape == (2 * boxsize + 1, 2 * boxsize + 1)
    assert crop.dtype == np.float32
    assert crop.sum() == pytest.approx(expected_sum)
    assert origin == expected_origin


def test_xmas_coordinates_are_one_based():
    """XMAS centres are 1-based, so the same spot sits one pixel earlier in numpy."""
    _, origin_xmas = extract_crop(FRAME, (40, 30), 5, "xmas")
    _, origin_python = extract_crop(FRAME, (40, 30), 5, "python")
    assert origin_python[0] - origin_xmas[0] == 1
    assert origin_python[1] - origin_xmas[1] == 1


def test_out_of_bounds_is_zero_padded_not_clipped():
    crop, _ = extract_crop(FRAME, (2, 2), 5, "python")
    assert crop.shape == (11, 11)
    assert np.any(crop == 0.0)


def test_centre_pixel_of_the_crop_is_the_requested_pixel():
    center_xy, boxsize = (40, 30), 5
    crop, _ = extract_crop(FRAME, center_xy, boxsize, "python")
    col, row = center_xy
    assert crop[boxsize, boxsize] == FRAME[row, col]


def test_origin_maps_crop_coordinates_back_to_the_frame():
    crop, (row0, col0) = extract_crop(FRAME, (40, 30), 5, "python")
    for dr in range(crop.shape[0]):
        for dc in range(crop.shape[1]):
            r, c = row0 + dr, col0 + dc
            if 0 <= r < FRAME.shape[0] and 0 <= c < FRAME.shape[1]:
                assert crop[dr, dc] == FRAME[r, c]
