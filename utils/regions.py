"""Region-of-interest geometry shared across scan maps.

`emission.stats_utils.extract_spectra_from_circles` and `laue.mean_strain` both
select a circular region per LED from a scan map. Neither package owns the
geometry, so it lives here.
"""

from __future__ import annotations

import numpy as np


def elliptical_masks_um(centers_um, radius_um, x_points, y_points, shape):
    """Boolean mask per LED for a circle of ``radius_um`` around each centre.

    The mask is an ellipse in pixel space, not a circle: scan steps in x and y
    are generally different, so a physically circular region maps to different
    pixel radii along each axis.

    Parameters
    ----------
    centers_um : dict
        ``{led_id: (x_um, y_um)}`` — centre of each region in physical units (µm).
    radius_um : float
        Region radius in µm.
    x_points, y_points : array_like
        Scan axis positions in µm; only the first and last entries are used, so
        the grid is assumed regular.
    shape : tuple(int, int)
        ``(nrows, ncols)`` of the map the masks index into.

    Returns
    -------
    dict
        ``{led_id: mask}`` with each mask of shape ``shape``.
    """
    nrows, ncols = shape
    x0 = x_points[0]
    y0 = y_points[0]
    x_um_per_px = (x_points[-1] - x0) / ncols
    y_um_per_px = (y_points[-1] - y0) / nrows

    r_x_px = radius_um / x_um_per_px
    r_y_px = radius_um / y_um_per_px

    yy, xx = np.mgrid[0:nrows, 0:ncols]

    masks = {}
    for led_id, (x_c_um, y_c_um) in centers_um.items():
        cx = (x_c_um - x0) / x_um_per_px
        cy = (y_c_um - y0) / y_um_per_px
        masks[led_id] = ((xx - cx) / r_x_px) ** 2 + ((yy - cy) / r_y_px) ** 2 <= 1.0

    return masks
