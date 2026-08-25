"""Extract just the analysed ROI from a scan's HDF5 image stack.

`run_satellite_pipeline` / `run_single_image` never touch the whole detector
frame — they read one small crop out of it per position (see
`laue.satellite.scan_pipeline._process_one`). `export_roi_h5` copies exactly
that crop, frame by frame, into a new, self-contained HDF5 file: small enough
to share, and a drop-in replacement for the original — point `img_source` at
it and set `roi_center` to the crop's own centre, `(boxsize, boxsize)`, and
every existing notebook cell runs unchanged.
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
    """Copy only the ROI used by the satellite pipeline into a new HDF5 file.

    Mirrors the exact slicing `scan_pipeline._process_one` performs —
    ``roi_center`` = (col, row), 0-based numpy; ``row0 = row_c - boxsize``,
    ``row1 = row_c + boxsize + 1``, same for columns — so the exported crop is
    pixel-identical to what the batch pipeline would have read from the
    original file.

    Parameters
    ----------
    img_source : path to the source HDF5 (the full detector-frame stack).
    h5_img_key  : dataset path inside it, e.g. 'entry_0000/CRGIF/eiger4m/data'.
    out_path    : where to write the new, small HDF5 file.
    roi_center, boxsize : identical meaning to `run_satellite_pipeline` — (col,
        row) centre, 0-based; the crop is (2*boxsize+1)**2.
    out_h5_key  : dataset path in the OUTPUT file. Defaults to `h5_img_key`, so
        ``run_satellite_pipeline(img_source=out_path, h5_img_key=h5_img_key,
        roi_center=(boxsize, boxsize), boxsize=boxsize, ...)`` is the only
        change a notebook needs.
    frame_indices : export only these frames (e.g. a boolean XEOL mask, or the
        integer indices where it is True) instead of the whole stack. This
        breaks the 1:1 correspondence between ``frame_idx`` and the scan grid
        that `scan.ij_to_index` relies on — the source index of each exported
        frame is written to ``f'{out_h5_key}_source_frame_idx'`` for
        provenance, but nothing here rebuilds the (i, j) mapping. Leave as
        None (the default: export every frame, indices unchanged) unless
        you're prepared to adapt that mapping downstream.
    dtype : output dtype. Defaults to the SOURCE dataset's own dtype — do not
        narrow to uint16: real bulk-peak amplitudes in this data exceed 65535
        counts (frame 7354's bulk peak is ~436261), which uint16 would clip.
    compression, compression_opts : passed to `h5py.Dataset` (gzip level 4 by
        default). Counts data with a flat background compresses well, but the
        real ratio depends on the data — not assumed here.
    chunk_frames : frames per read/write batch, bounding memory use for a
        scan-sized export instead of loading everything at once.

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
            out_ds = dst.create_dataset(
                out_h5_key, shape=(n_out, crop_side, crop_side),
                dtype=out_dtype, chunks=(1, crop_side, crop_side),
                compression=compression, compression_opts=compression_opts,
            )
            out_ds.attrs['roi_center_original'] = (col_c, row_c)
            out_ds.attrs['boxsize'] = boxsize
            out_ds.attrs['crop_origin_px'] = (row0, col0)
            out_ds.attrs['source_frame_shape'] = (H, W)
            out_ds.attrs['source_file'] = str(img_source)

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
                out_ds[start:stop] = chunk.astype(out_dtype, copy=False)
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
