"""ROI cropping shared across the Laue entry points.

`run_single_spot`, `satellite/run_single_image` and `scan_pipeline` each carried
their own copy of this.

Frame loading lives in `laue.readers` and is re-exported here so the existing
`from laue._imaging import load_frame` keeps working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


from laue.readers import load_frame




def extract_crop(
    frame: np.ndarray,
    center: Tuple[int, int],
    boxsize: int,
    coords: str = 'xmas',
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Extract a (2*boxsize+1)² crop, zero-padded where it runs off the frame.

    `center` is ``(x, y)`` — column first — because that is what roi_viewer
    reports. With ``coords='xmas'`` it is treated as 1-based.

    Returns ``(crop, (row0, col0))``; the origin is the top-left pixel of the
    crop in the original frame and is negative when the crop overhangs the edge,
    which is what makes positions mappable back to detector coordinates.
    """
    col_c, row_c = int(center[0]), int(center[1])   # (x, y) = (col, row)
    if coords == 'xmas':      # XMAS convention is 1-based
        col_c -= 1
        row_c -= 1

    H, W = frame.shape
    half = boxsize

    row0, row1 = row_c - half, row_c + half + 1
    col0, col1 = col_c - half, col_c + half + 1

    # slice bounds clamped to the frame
    r0c, r1c = max(row0, 0), min(row1, H)
    c0c, c1c = max(col0, 0), min(col1, W)

    crop = np.zeros((row1 - row0, col1 - col0), dtype=np.float32)
    dst_r0 = r0c - row0
    dst_c0 = c0c - col0
    crop[dst_r0:dst_r0 + (r1c - r0c), dst_c0:dst_c0 + (c1c - c0c)] = frame[r0c:r1c, c0c:c1c]

    return crop, (row0, col0)


def crop_roi(img: np.ndarray, cen_row: int, cen_col: int, boxsize: int) -> np.ndarray:
    """`extract_crop` with (row, col) 0-based arguments and no origin returned.

    Kept as a separate name because the scan pipeline indexes by row/col rather
    than by the detector's (x, y).
    """
    crop, _ = extract_crop(img, (cen_col, cen_row), boxsize, coords='python')
    return crop
