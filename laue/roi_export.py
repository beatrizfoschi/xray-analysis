"""Extract just the analysed ROI from a scan's HDF5 image stack.

`run_satellite_pipeline` / `run_single_image` never touch the whole detector
frame — they read one small crop out of it per position, at `roi_center`'s
ABSOLUTE position on the detector (see
`laue.satellite.scan_pipeline._process_one`). `export_roi_h5` writes just
that crop into a new HDF5 file, at that same absolute position, so the
exported file is a drop-in replacement for the original: same `roi_center`,
same `boxsize`, same `frame_index` / `frame_idx` semantics — nothing else in
a notebook needs to change.

⚠ Read "Why the file looks huge but is not" below before touching the
exported dataset directly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import h5py
import numpy as np


def export_roi_h5(
    img_source,
    h5_img_key: str,
    out_path,
    roi_center: Tuple[int, int],
    boxsize: int,
    *,
    out_h5_key: Optional[str] = None,
    frame_indices: Optional[Union[Sequence[int], np.ndarray]] = None,
    dtype=None,
    compression: Optional[str] = 'gzip',
    compression_opts: Optional[int] = 4,
    chunk_frames: int = 200,
    verbose: bool = True,
) -> dict:
    """Copy only the ROI used by the satellite pipeline into a new HDF5 file,
    keeping every pixel at its ORIGINAL, absolute detector position.

    The output dataset has the same (H, W) as the source frame — only the
    (2*boxsize+1)**2 window around `roi_center` is ever written; everything
    else reads back as 0 and consumes no disk space (see below). Because the
    data sits at the same absolute pixel coordinates as in the original file,
    `run_satellite_pipeline` / `run_single_image` need no change at all: point
    `img_source` at the export and keep `roi_center`, `boxsize` and
    `frame_index` exactly as they were. This matters specifically for the
    Laue period routes — they convert `crop_origin_px` (derived from
    `roi_center - boxsize`) to (2θ, χ) via the detector calibration, which
    only makes sense in absolute detector coordinates. A version of this
    function that re-centred the crop to a small, local coordinate frame was
    tried first and broke exactly that: every angle came out wrong (parent
    reprojection off by tens of degrees, nearly every period fit railing
    against its bound), because the geometry code was silently handed a
    local pixel position instead of the true detector one.

    Why the file looks huge but is not
    -----------------------------------
    `f[key].shape` reads as the FULL original frame size for every exported
    frame — e.g. (20301, 2162, 2068) for the whole MLed scan — which looks
    like it should be the same ~360 GB as the source. It is not: HDF5 chunks
    that are never written are never allocated on disk and read back as 0
    (the dataset's `fillvalue`), so only the chunks touching the actual ROI
    exist. Measured on data shaped like this project's (flat low background,
    a few sparse peaks up to ~4·10^5 counts): ~460 MB on disk for all 20301
    frames of a boxsize=80 crop, in about 90 s to write.

    **The one real trap this design carries**: naively loading the whole
    dataset (``f[key][:]`` or ``np.array(f[key])``) forces HDF5 to
    materialise the FULL nominal shape in memory — for the MLed scan that is
    ~360 GB of RAM, regardless of the ~460 MB on disk. Anyone opening this
    file must always slice with the same `roi_center`/`boxsize` used to
    create it (exactly what `run_satellite_pipeline` / `run_single_image`
    already do); never index it with `[:]` or a bare `[i]` over the full
    frame. This is written explicitly into the dataset's own attrs (see
    below) precisely so a reader who inspects the file before coding against
    it has a chance to notice.

    Parameters
    ----------
    img_source : path to the source HDF5 (the full detector-frame stack).
    h5_img_key  : dataset path inside it, e.g. 'entry_0000/CRGIF/eiger4m/data'.
    out_path    : where to write the new, small HDF5 file.
    roi_center, boxsize : identical meaning and identical VALUES to what you
        already pass to `run_satellite_pipeline` / `run_single_image` — (col,
        row) centre, 0-based; the crop is (2*boxsize+1)**2. Keep passing the
        SAME roi_center to those functions afterwards; nothing changes there.
    out_h5_key  : dataset path in the OUTPUT file. Defaults to `h5_img_key`.
    frame_indices : export only these frames (e.g. a boolean XEOL mask, or the
        integer indices where it is True) instead of the whole stack. This
        still breaks the 1:1 correspondence between `frame_idx` and the scan
        grid that `scan.ij_to_index` relies on — unrelated to the position
        fix above — the source index of each exported frame is written to
        f'{out_h5_key}_source_frame_idx' for provenance, but nothing here
        rebuilds the (i, j) mapping. Leave as None (the default: every frame,
        indices unchanged) unless you're prepared to adapt that mapping.
    dtype : output dtype. Defaults to the SOURCE dataset's own dtype — do not
        narrow to uint16: real bulk-peak amplitudes in this data exceed 65535
        counts (frame 7354's bulk peak is ~436261), which uint16 would clip.
    compression, compression_opts : passed to `h5py.Dataset` (gzip level 4 by
        default), applied per chunk — only the chunks touching the ROI are
        ever written, so this compresses actual data, not the sparse region.
    chunk_frames : frames per read/write batch AND the chunk size along the
        frame axis. Writing in batches (not one frame at a time) is what
        keeps this fast — a naive per-frame loop against a full-frame-shaped
        dataset was measured at >100x slower for no difference in output
        size, because a batched write touches each chunk once.

    Returns
    -------
    dict with 'out_path', 'n_frames', 'crop_shape', 'dtype', 'bytes_on_disk'.
    """
    col_c, row_c = int(roi_center[0]), int(roi_center[1])
    row0, row1 = row_c - boxsize, row_c + boxsize + 1
    col0, col1 = col_c - boxsize, col_c + boxsize + 1
    crop_side = 2 * boxsize + 1

    out_h5_key = out_h5_key or h5_img_key
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(img_source, 'r') as src:
        ds = src[h5_img_key]
        n_total, H, W = ds.shape
        if not (0 <= row0 and row1 <= H and 0 <= col0 and col1 <= W):
            raise ValueError(
                f'ROI [{row0}:{row1}, {col0}:{col1}] falls outside the '
                f'{H}x{W} frame — run_satellite_pipeline would silently read '
                f'a smaller, off-centre crop here rather than pad it.'
            )
        out_dtype = np.dtype(dtype) if dtype is not None else ds.dtype

        if frame_indices is None:
            indices = np.arange(n_total)
            contiguous = True
        else:
            indices = np.asarray(frame_indices)
            if indices.dtype == bool:
                indices = np.flatnonzero(indices)
            indices = np.sort(indices)
            contiguous = False

        n_out = len(indices)
        t0 = time.time()

        with h5py.File(out_path, 'w') as dst:
            # Full (H, W) — a footprint the data never fills in practice. See
            # the module and function docstrings before reading this dataset
            # any other way than by slicing [., row0:row1, col0:col1].
            out_ds = dst.create_dataset(
                out_h5_key, shape=(n_out, H, W),
                dtype=out_dtype, chunks=(min(chunk_frames, n_out), crop_side, crop_side),
                compression=compression, compression_opts=compression_opts,
                fillvalue=0,
            )
            out_ds.attrs['roi_center'] = (col_c, row_c)
            out_ds.attrs['boxsize'] = boxsize
            out_ds.attrs['crop_origin_px'] = (row0, col0)
            out_ds.attrs['source_file'] = str(img_source)
            out_ds.attrs['WARNING_sparse_full_frame_shape'] = (
                "This dataset's shape matches the ORIGINAL detector frame, "
                "but only the (row0:row1, col0:col1) window given by "
                "crop_origin_px and boxsize (this attr's siblings) actually "
                "has data on disk. Do not load this dataset in full "
                "(ds[:] / np.array(ds)) -- always slice the same window, "
                "exactly as run_satellite_pipeline / run_single_image do."
            )

            if not contiguous:
                dst.create_dataset(f'{out_h5_key}_source_frame_idx', data=indices)

            for start in range(0, n_out, chunk_frames):
                stop = min(start + chunk_frames, n_out)
                batch = indices[start:stop]
                if contiguous:
                    chunk = ds[batch[0]:batch[-1] + 1, row0:row1, col0:col1]
                else:
                    # One read per frame: safe regardless of h5py's fancy-
                    # indexing support for a given version, at negligible cost
                    # since each read is only crop_side x crop_side.
                    chunk = np.stack(
                        [ds[int(idx), row0:row1, col0:col1] for idx in batch])
                out_ds[start:stop, row0:row1, col0:col1] = \
                    chunk.astype(out_dtype, copy=False)
                if verbose:
                    print(f'\r[export_roi_h5] {stop}/{n_out} frames',
                          end='', flush=True)

        if verbose:
            print(f'\n[export_roi_h5] wrote {out_path} in '
                  f'{time.time() - t0:.1f}s')

    size = out_path.stat().st_size
    return {'out_path': str(out_path), 'n_frames': n_out,
            'crop_shape': (crop_side, crop_side), 'dtype': str(out_dtype),
            'bytes_on_disk': size}
