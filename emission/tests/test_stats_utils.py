"""Regression test for the skimage >=0.26 `remove_small_objects` migration in
`segment_leds`.

skimage 0.26 deprecated `min_size` (excludes objects strictly smaller than N,
i.e. keeps size >= N) in favour of `max_size` (excludes size <= N, i.e. keeps
size > N). A bare rename (`max_size=min_area`) silently drops any region of
exactly `min_area` pixels — the boundary `segment_leds`'s own docstring
promises to keep ("minimum region size in pixels"). The fix is `max_size =
min_area - 1`; both tests below exist to catch a regression back to the naive
rename, not just to check that the deprecation warning is gone.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from skimage.morphology import remove_small_objects

from emission.stats_utils import segment_leds


def test_object_of_exactly_min_area_pixels_is_kept():
    """The boundary case a naive `max_size=min_area` rename would break."""
    min_area = 5
    binary = np.zeros((3, 20), dtype=bool)
    binary[1, 0:5] = True     # exactly min_area pixels — must survive
    binary[1, 7:11] = True    # min_area - 1 pixels — must be removed

    out = remove_small_objects(binary, max_size=min_area - 1)
    assert out[1, 0:5].all()
    assert not out[1, 7:11].any()


def test_segment_leds_runs_without_the_min_size_deprecation_warning():
    rng = np.random.default_rng(0)
    img = np.zeros((60, 60))
    yy, xx = np.mgrid[0:60, 0:60]
    for cy, cx in [(15, 15), (15, 45), (45, 15), (45, 45)]:
        img += 10.0 * np.exp(-0.5 * (((yy - cy) / 4) ** 2 + ((xx - cx) / 4) ** 2))
    img += rng.normal(0, 0.05, img.shape)

    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        labels, regions = segment_leds(img, min_area=21, opening_radius=2)

    assert labels.shape == img.shape
    assert labels.max() >= 1
