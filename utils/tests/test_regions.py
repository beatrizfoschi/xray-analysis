"""Tests for the elliptical region geometry extracted from emission and laue.

Neither original call site can be imported without lauexplore, so the guarantee
here is that the extracted helper reproduces the inline formula they used,
bit for bit, on a non-square pixel grid.
"""

from __future__ import annotations

import numpy as np

from utils.regions import elliptical_masks_um


CENTERS = {1: (3.0, 3.0), 2: (13.0, 3.0), 7: (8.0, 11.5)}
RADIUS = 2.0
X_POINTS = np.linspace(0.0, 40.0, 201)
Y_POINTS = np.linspace(0.0, 20.0, 101)
SHAPE = (101, 201)


def _inline_reference():
    """The formula exactly as it stood in both modules before extraction."""
    nrows, ncols = SHAPE
    x_range = X_POINTS[-1] - X_POINTS[0]
    y_range = Y_POINTS[-1] - Y_POINTS[0]
    x_um_per_px = x_range / ncols
    y_um_per_px = y_range / nrows
    yy, xx = np.mgrid[0:nrows, 0:ncols]
    r_x_px = RADIUS / x_um_per_px
    r_y_px = RADIUS / y_um_per_px

    masks = {}
    for led_id, (x_c_um, y_c_um) in CENTERS.items():
        cx = (x_c_um - X_POINTS[0]) / x_um_per_px
        cy = (y_c_um - Y_POINTS[0]) / y_um_per_px
        masks[led_id] = ((xx - cx) / r_x_px) ** 2 + ((yy - cy) / r_y_px) ** 2 <= 1.0
    return masks


def test_matches_the_inline_formula_it_replaced():
    got = elliptical_masks_um(CENTERS, RADIUS, X_POINTS, Y_POINTS, SHAPE)
    expected = _inline_reference()
    assert got.keys() == expected.keys()
    for led_id in expected:
        np.testing.assert_array_equal(got[led_id], expected[led_id])


def test_masks_have_the_map_shape_and_are_non_empty():
    masks = elliptical_masks_um(CENTERS, RADIUS, X_POINTS, Y_POINTS, SHAPE)
    for mask in masks.values():
        assert mask.shape == SHAPE
        assert mask.any()


def test_region_is_an_ellipse_in_pixels_when_the_steps_differ():
    """A physically circular region is not circular in pixel space.

    On a grid stepping 0.199 µm in x but 0.488 µm in y, a 2 µm radius spans about
    10 px horizontally and 4 px vertically. Collapsing this to one pixel radius
    would stretch the region along whichever axis has the coarser step.
    """
    y_coarse = np.linspace(0.0, 20.0, 41)
    shape = (41, 201)
    masks = elliptical_masks_um({1: (20.0, 10.0)}, RADIUS, X_POINTS, y_coarse, shape)

    rows, cols = np.where(masks[1])
    width_px = cols.max() - cols.min() + 1
    height_px = rows.max() - rows.min() + 1
    assert width_px > 2 * height_px


def test_centre_pixel_is_inside_the_region():
    masks = elliptical_masks_um({1: (20.0, 10.0)}, RADIUS, X_POINTS, Y_POINTS, SHAPE)
    nrows, ncols = SHAPE
    cx = int(round((20.0 - X_POINTS[0]) / ((X_POINTS[-1] - X_POINTS[0]) / ncols)))
    cy = int(round((10.0 - Y_POINTS[0]) / ((Y_POINTS[-1] - Y_POINTS[0]) / nrows)))
    assert masks[1][cy, cx]


def test_a_larger_radius_covers_a_superset():
    small = elliptical_masks_um(CENTERS, 1.0, X_POINTS, Y_POINTS, SHAPE)
    large = elliptical_masks_um(CENTERS, 3.0, X_POINTS, Y_POINTS, SHAPE)
    for led_id in CENTERS:
        assert np.all(large[led_id][small[led_id]])
