import numpy as np
import h5py
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def track_spot_h5(
    h5_path,
    dset_path,
    xc0, yc0,
    w=10, h=10,
    threshold=0.0,
    start=0,
    stop=None,
    checkpoint_path=None,
    checkpoint_every=2000,
    dtype_out=np.float32,
    line_length=None,
    reset_row_start_from_prev=True,
    max_jump=None,  # max allowed center displacement (pixels) per frame update
):

    """
    Track a Laue diffraction spot by iteratively recentering a fixed-size ROI on the
    brightest pixel, reading frames sequentially from an HDF5 dataset.
    
    The tracking is performed frame-by-frame. When the scan is acquired line-by-line
    (row-major order), the ROI center at the beginning of each new scan line can be
    reset to the same ROI center used at the beginning of the previous line, preventing
    loss of the spot between lines.
    
    Within each ROI, ties between pixels with identical maximum intensity are resolved
    using the centroid (center of mass) of all maximal pixels.
    
    Per-frame outputs
    -----------------
    For each frame, the following quantities are stored:
    
    - frame : index of the frame in the HDF5 dataset
    - xcen, ycen : current ROI center in global image coordinates
      (updated only when imax >= threshold)
    - imax : maximum pixel intensity inside the ROI
    - x_max, y_max : centroid of all pixels equal to imax (global coordinates)
    - x_com, y_com : intensity-weighted center of mass of the ROI (global coordinates)
    - ok : 1 if imax >= threshold, 0 otherwise
    
    Failure condition (imax < threshold)
    ------------------------------------
    - x_max, y_max, x_com, y_com are set to NaN
    - xcen, ycen remain unchanged (ROI is not updated and stays at the previous position)
    
    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file containing the diffraction images.
    dset_path : str
        Path inside the HDF5 file to the 3D dataset with shape
        (n_frames, ny, nx).
    xc0, yc0 : float
        Initial ROI center in global image coordinates for the first frame.
    w, h : int, optional
        Width and height of the ROI in pixels. Default is 10 x 10.
    threshold : float, optional
        Minimum required maximum intensity inside the ROI for a successful tracking
        update. If imax < threshold, the ROI center is not updated.
    start : int, optional
        Index of the first frame to process. Default is 0.
    stop : int or None, optional
        Index of the last frame to process (exclusive). If None, all frames
        from start to the end of the dataset are processed.
    checkpoint_path : str or None, optional
        If provided, intermediate results are periodically saved to this path
        using numpy.savez, allowing recovery in case of kernel interruption.
    checkpoint_every : int, optional
        Number of frames between checkpoint saves. Default is 2000.
    dtype_out : numpy dtype, optional
        Data type used to store the output array. Default is np.float32.
    line_length : int or None, optional
        Number of frames per scan line (e.g. number of columns in the 2D scan map).
        If provided, the function assumes a line-by-line (row-major) acquisition.
    reset_row_start_from_prev : bool, optional
        If True and line_length is provided, the ROI center at the beginning of each
        scan line is reset to the same ROI center used at the beginning of the
        previous scan line. This prevents spot loss when moving from one line
        to the next. Default is True.
        
    max_jump : float or None, optional
        Maximum allowed displacement (in pixels) for updating the ROI center from one
        frame to the next. If the candidate update (x_max, y_max) is farther than
        max_jump from the previous center (xcen, ycen), the update is rejected and
        the center is kept unchanged for that frame. Default is None (no constraint).
    
    Returns
    -------
    out : np.ndarray, shape (N, 9)
        Array containing the tracking results for N processed frames.
        Columns are ordered as:
    
            [frame,
             xcen, ycen,
             imax,
             x_max, y_max,
             x_com, y_com,
             ok]
    
        All coordinates are expressed in the global image coordinate system.
    meta : dict
        Dictionary containing metadata and parameters used for the tracking
        (HDF5 paths, ROI size, threshold, scan geometry, etc.).
    """

    def _clip_roi_center(xc, yc, w, h, nx, ny):
        hx = w // 2
        hy = h // 2
        x0 = int(np.floor(xc)) - hx
        x1 = x0 + w
        y0 = int(np.floor(yc)) - hy
        y1 = y0 + h
        x0c = max(0, x0); y0c = max(0, y0)
        x1c = min(nx, x1); y1c = min(ny, y1)
        return x0c, x1c, y0c, y1c

    def _com_intensity(patch, x0, y0):
        s = float(np.sum(patch))
        if s <= 0.0 or not np.isfinite(s):
            return np.nan, np.nan
        ys, xs = np.indices(patch.shape)
        x = x0 + xs
        y = y0 + ys
        xcom = float(np.sum(patch * x) / s)
        ycom = float(np.sum(patch * y) / s)
        return xcom, ycom

    def _com_of_maxima(patch, x0, y0, imax):
        mask = (patch == imax)
        if not np.any(mask):
            return np.nan, np.nan
        ys, xs = np.nonzero(mask)
        x_max = float(np.mean(x0 + xs))
        y_max = float(np.mean(y0 + ys))
        return x_max, y_max

    with h5py.File(h5_path, "r") as hin:
        dset = hin[dset_path]
        nframes = dset.shape[0]
        ny, nx = dset.shape[1], dset.shape[2]

        if stop is None:
            stop = nframes
        stop = min(stop, nframes)

        frames = np.arange(start, stop, dtype=np.int32)
        N = frames.size

        out = np.full((N, 9), np.nan, dtype=dtype_out)

        xcen = float(xc0)
        ycen = float(yc0)

        row_start_centers = {}  # row_index -> (xcen_start, ycen_start)

        pbar = tqdm(enumerate(frames), total=N, desc="Tracking ROI", leave=True)

        for i, fi in pbar:
            # reset ROI at the start of each scan line
            if line_length is not None and reset_row_start_from_prev:
                k = int(fi - start)
                if (k % int(line_length)) == 0:
                    row = k // int(line_length)
                    if row == 0:
                        row_start_centers[row] = (xcen, ycen)
                    else:
                        prev = row_start_centers.get(row - 1, (xc0, yc0))
                        xcen, ycen = float(prev[0]), float(prev[1])
                        row_start_centers[row] = (xcen, ycen)

            img = dset[fi].astype(np.float32, copy=False)

            x0, x1, y0, y1 = _clip_roi_center(xcen, ycen, w, h, nx, ny)
            patch = img[y0:y1, x0:x1]

            imax = float(np.max(patch)) if patch.size else -np.inf
            ok = int(imax >= threshold)

            out[i, 0] = float(fi)
            out[i, 1] = float(xcen)
            out[i, 2] = float(ycen)
            out[i, 3] = float(imax)
            out[i, 8] = float(ok)

            if ok and patch.size > 0 and np.isfinite(imax):
                x_max, y_max = _com_of_maxima(patch, x0, y0, imax)
                x_com, y_com = _com_intensity(patch, x0, y0)

                out[i, 4] = x_max
                out[i, 5] = y_max
                out[i, 6] = x_com
                out[i, 7] = y_com

                # --- NEW: reject "jump" if too large ---
                if np.isfinite(x_max) and np.isfinite(y_max):
                    if max_jump is None:
                        xcen, ycen = x_max, y_max
                    else:
                        dx = float(x_max) - float(xcen)
                        dy = float(y_max) - float(ycen)
                        if (dx * dx + dy * dy) <= float(max_jump) ** 2:
                            xcen, ycen = x_max, y_max
                        # else: keep previous xcen,ycen (do not update)

                out[i, 1] = float(xcen)
                out[i, 2] = float(ycen)

            if checkpoint_path and (i + 1) % int(checkpoint_every) == 0:
                np.savez(
                    checkpoint_path,
                    out=out,
                    meta=dict(
                        h5_path=h5_path,
                        dset_path=dset_path,
                        xc0=xc0, yc0=yc0, w=w, h=h,
                        threshold=threshold,
                        start=start, stop=stop,
                        line_length=line_length,
                        reset_row_start_from_prev=reset_row_start_from_prev,
                        max_jump=max_jump,
                    ),
                )

        meta = dict(
            h5_path=h5_path,
            dset_path=dset_path,
            xc0=xc0, yc0=yc0, w=w, h=h,
            threshold=threshold,
            start=start, stop=stop,
            nx=nx, ny=ny,
            line_length=line_length,
            reset_row_start_from_prev=reset_row_start_from_prev,
            max_jump=max_jump,
        )

        if checkpoint_path:
            np.savez(checkpoint_path, out=out, meta=meta)

    return out, meta


def _tile_from_center(img, xcen, ycen, w, h, fill_value=np.nan):
    """
    Extract a fixed-size (h, w) tile centered at (xcen, ycen) from img.
    Pads with fill_value if the ROI crosses the image boundary.
    """
    ny, nx = img.shape

    hx = w // 2
    hy = h // 2

    # ROI bounds in full-image coords (exclusive end)
    x0 = int(np.floor(xcen)) - hx
    x1 = x0 + w
    y0 = int(np.floor(ycen)) - hy
    y1 = y0 + h

    # clipped bounds for reading from image
    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(nx, x1)
    y1c = min(ny, y1)

    tile = np.full((h, w), fill_value, dtype=np.float32)

    if (x1c > x0c) and (y1c > y0c):
        patch = img[y0c:y1c, x0c:x1c]

        # where to paste inside tile
        tx0 = x0c - x0
        ty0 = y0c - y0
        tile[ty0:ty0 + patch.shape[0], tx0:tx0 + patch.shape[1]] = patch.astype(np.float32, copy=False)

    return tile


def roi_mosaic_from_track(
    h5_path,
    dset_path,
    track_out,
    grid_shape=(101, 201),
    w=10,
    h=10,
    start=0,
    fill_value=np.nan,
    use_centers_cols=(1, 2),  # (xcen_col, ycen_col) in track_out
    dtype=np.float32,
):
    """
    Build a mosaic (like LaueTools) where each frame ROI becomes a tile in a (M,N) grid.

    Parameters
    ----------
    track_out : np.ndarray
        Output from your tracking. Assumes rows align with frames starting at 'start'.
        Uses columns (xcen_col, ycen_col) for ROI centers.
    grid_shape : tuple[int,int]
        (M, N) e.g. (101, 201). Must satisfy M*N == number of frames you want to mosaic.
    start : int
        First frame index corresponding to track_out[0].
    fill_value : float
        Value used to pad tiles when ROI hits boundaries.

    Returns
    -------
    mosaic : np.ndarray, shape (M*h, N*w)
    """
    M, N = grid_shape
    n_tiles = M * N
    if track_out.shape[0] < n_tiles:
        raise ValueError(f"track_out has {track_out.shape[0]} rows, but grid needs {n_tiles} tiles.")

    xcol, ycol = use_centers_cols

    mosaic = np.full((M * h, N * w), fill_value, dtype=dtype)

    with h5py.File(h5_path, "r") as hin:
        dset = hin[dset_path]
        # we assume frames are contiguous: frame_idx = start + k
        for k in tqdm(range(n_tiles), desc="Building ROI mosaic", leave=True):
            frame_idx = start + k
            img = dset[frame_idx]  # keep as-is; tile func casts to float32

            xcen = float(track_out[k, xcol])
            ycen = float(track_out[k, ycol])

            tile = _tile_from_center(img, xcen, ycen, w=w, h=h, fill_value=fill_value)

            r = k // N
            c = k % N
            mosaic[r*h:(r+1)*h, c*w:(c+1)*w] = tile

    return mosaic


def plot_roi_mosaic(mosaic, title=None, cmap="viridis", vmin=None, vmax=None, invert_y=False, extent=None):
    """Display the ROI mosaic image built by roi_mosaic_from_track."""
    plt.figure(figsize=(12, 6))
    plt.imshow(mosaic, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", extent=extent)
    if title:
        plt.title(title)
    plt.axis("off")
    if invert_y:
        plt.gca().invert_yaxis()
    plt.show()

