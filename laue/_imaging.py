"""ROI cropping and stack alignment shared across the Laue entry points.

`run_single_spot`, `satellite/run_single_image` and `scan_pipeline` each carried
their own copy of the cropping.

Frame loading lives in `laue.readers` and is re-exported here so the existing
`from laue._imaging import load_frame` keeps working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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


# ── Sub-pixel stack alignment ─────────────────────────────────────────────────

def align_stack(
    stack: np.ndarray,
    *,
    reference: Optional[np.ndarray] = None,
    max_shift: Optional[float] = None,
    crop_half: Optional[int] = None,
    upsample_factor: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Register every ROI of a stack to a common reference, to sub-pixel accuracy.

    Phase cross-correlation gives the shift, which is then applied by linear
    interpolation.  The reference defaults to the pixel-wise median of the stack:
    a spot that wanders by a fraction of a pixel across the scan averages into a
    sharp template, while an outlier frame does not drag it.

    This is a *stack* operation — the reference depends on every frame — so it
    cannot be expressed as a per-position analysis function and does not belong
    inside `scan_pipeline.run_spot_pipeline`, which streams one ROI at a time.
    Run it as a pre-pass over a stack held in memory.

    Alignment only matters for quantities tied to the absolute position of the
    spot in the crop — `spot_fit`'s ``x{k}``/``y{k}``, `spot_metrics`' ``x_com``.
    ``separation``, ``orientation``, ``ratio``, the widths and chi² are all
    invariant to a translation of the crop and are unaffected either way.

    Parameters
    ----------
    stack : (n_frames, H, W) array
        ROIs cut at the same detector position, one per scan point.
    reference : (H, W) array, optional
        Template to register against.  Median of ``stack`` when omitted.
    max_shift : float, optional
        Clamp each shift to ±``max_shift`` pixels per axis.  A ROI cut with a
        margin of ``m`` pixels around the region actually wanted should pass
        ``max_shift=m``, so that no interpolated pixel is ever pulled in from
        outside the data.
    crop_half : int, optional
        Trim the aligned frames to the central ``2 * crop_half`` pixels, which
        discards the margin that ``max_shift`` was reserved for.  The centre is
        the crop centre, so pass the same ``crop_half`` the analysis expects.
    upsample_factor : int
        Sub-pixel resolution of the correlation: shifts are resolved to
        ``1 / upsample_factor`` of a pixel.

    Returns
    -------
    aligned : (n_frames, h, w) float32 array
    shifts : (n_frames, 2) array of the applied ``(dy, dx)``, before cropping.
    """
    from scipy.ndimage import shift as ndi_shift
    from skimage.registration import phase_cross_correlation

    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f'stack must be (n_frames, H, W), got shape {stack.shape}')

    n_frames, H, W = stack.shape
    ref = np.median(stack, axis=0) if reference is None else np.asarray(reference)
    if ref.shape != (H, W):
        raise ValueError(f'reference shape {ref.shape} does not match frames {(H, W)}')

    if crop_half is None:
        out_h, out_w = H, W
        r0 = c0 = 0
    else:
        out_h = out_w = 2 * crop_half
        if out_h > H or out_w > W:
            raise ValueError(
                f'crop_half={crop_half} needs {out_h}x{out_w} but frames are {H}x{W}'
            )
        r0 = (H - out_h) // 2
        c0 = (W - out_w) // 2

    shifts = np.zeros((n_frames, 2), dtype=np.float32)
    aligned = np.empty((n_frames, out_h, out_w), dtype=np.float32)

    for k in range(n_frames):
        (dy, dx), _, _ = phase_cross_correlation(
            ref, stack[k], upsample_factor=upsample_factor, normalization=None
        )
        if max_shift is not None:
            dy = float(np.clip(dy, -max_shift, max_shift))
            dx = float(np.clip(dx, -max_shift, max_shift))
        shifts[k] = (dy, dx)
        moved = ndi_shift(stack[k], (dy, dx), order=1, mode='constant', cval=0.0)
        aligned[k] = moved[r0:r0 + out_h, c0:c0 + out_w]

    return aligned, shifts
