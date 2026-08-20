"""Readers for the file formats a Laue experiment produces.

Detector frames (HDF5, numpy, and anything fabio or PIL opens) and LaueTools
files (`.fit` indexation results, `.dat` peak lists).

`load_frame` replaces three separate loaders that had each grown a case the
others lacked: 4-D HDF5 layouts, automatic dataset discovery, nrxrdct `spot_*`
groups, and fabio formats. All four are handled here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_H5_SUFFIXES = ('.h5', '.hdf5', '.nxs')


# ── Detector frames ───────────────────────────────────────────────────────────

def find_image_key(h5f) -> str:
    """Key of the first dataset in `h5f` large enough to be a detector image."""
    import h5py

    candidates: list[str] = []

    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 \
                and obj.shape[-1] >= 256 and obj.shape[-2] >= 256:
            candidates.append(name)

    h5f.visititems(_visitor)
    if not candidates:
        raise KeyError('No detector image dataset found; pass h5_key explicitly.')
    return candidates[0]


def _frame_from_dataset(dset, frame_index: int) -> np.ndarray:
    ndim = len(dset.shape)
    if ndim == 2:
        return dset[()]
    if ndim == 3:
        return dset[frame_index]
    if ndim == 4:
        # (frames, 1, H, W) — some beamline writers add an extra dimension
        return dset[frame_index, 0]
    raise ValueError(f'Unexpected dataset shape {dset.shape}')


def _load_h5_frame(path: Path, h5_key, frame_index, spot_key) -> np.ndarray:
    import h5py

    with h5py.File(path, 'r') as f:
        if spot_key is not None:
            return np.asarray(f[spot_key]['image'][()])
        if h5_key is not None:
            return np.asarray(_frame_from_dataset(f[h5_key], frame_index))

        # nrxrdct writes one group per spot
        spot_groups = [k for k in f.keys() if k.startswith('spot_')]
        if spot_groups:
            return np.asarray(f[spot_groups[0]]['image'][()])
        if 'image' in f:
            return np.asarray(f['image'][()])
        return np.asarray(_frame_from_dataset(f[find_image_key(f)], frame_index))


def load_frame(
    path,
    h5_key: Optional[str] = None,
    frame_index: int = 0,
    spot_key: Optional[str] = None,
    dtype=np.float32,
) -> np.ndarray:
    """Load one 2-D detector frame.

    Parameters
    ----------
    path : HDF5 (`.h5`, `.hdf5`, `.nxs`), numpy `.npy`, any format fabio or PIL
        can open — TIFF from the older sCMOS camera among them — or an in-memory
        `np.ndarray`, 2-D or 3-D. The array case exists so a stack combined in a
        notebook can be analysed without a round trip through disk; it is copied,
        never aliased.
    h5_key : dataset path inside the HDF5 file. If omitted, an nrxrdct `spot_*`
        group is used when present, then a top-level ``image``, then the first
        dataset big enough to be a detector frame.
    frame_index : index into a stacked (3-D or 4-D) dataset.
    spot_key : explicit nrxrdct spot group, e.g. ``'spot_0000_0'``.
    dtype : output dtype. float32 by default, which is what the analysis paths
        expect; pass float64 for display code that assumed it.

    Notes
    -----
    Files with no recognised suffix are tried as HDF5 first, then handed to
    fabio, since beamline files are often extensionless.
    """
    if isinstance(path, np.ndarray):
        arr = path
        if arr.ndim == 3:
            arr = arr[frame_index]
        elif arr.ndim != 2:
            raise ValueError(f'array must be 2-D or 3-D, got shape {arr.shape}')
        # copy, so the caller's array is never aliased by later in-place work
        return arr.astype(dtype, copy=True)

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in _H5_SUFFIXES:
        return _load_h5_frame(p, h5_key, frame_index, spot_key).astype(dtype)

    if suffix == '.npy':
        arr = np.load(p)
        if arr.ndim == 3:
            arr = arr[frame_index]
        return arr.astype(dtype)

    if suffix:
        return _read_with_fabio_or_pil(p).astype(dtype)

    # extensionless: HDF5 is the more likely case on the beamline
    try:
        return _load_h5_frame(p, h5_key, frame_index, spot_key).astype(dtype)
    except Exception:
        return _read_with_fabio_or_pil(p).astype(dtype)


def _read_with_fabio_or_pil(path: Path) -> np.ndarray:
    """fabio covers EDF/CBF/TIFF/…; PIL is the fallback when fabio is absent."""
    try:
        import fabio
    except ImportError:
        pass
    else:
        return np.asarray(fabio.open(str(path)).data)

    try:
        from PIL import Image
    except ImportError:
        raise ValueError(
            f'Cannot read {path.suffix!r}: neither fabio nor PIL is installed.'
        ) from None
    return np.asarray(Image.open(path))


def imshow_detector(
    ax,
    image: np.ndarray,
    *,
    vmin: float = 10.0,
    vmax: Optional[float] = None,
    log_scale: bool = True,
    cmap: str = 'Greys',
) -> None:
    """Detector display: percentile-clipped log or linear scale, with colorbar.

    `vmin` is a floor for the log scale, not a data limit. The default suits the
    Eiger; on a lower-count detector it can blank the frame, so lower it.
    """
    import matplotlib.colors as mcolors

    pos = image[image > 0]
    _vmax = float(np.percentile(pos, 99.9)) if vmax is None else vmax
    norm = (mcolors.LogNorm(vmin=vmin, vmax=_vmax) if log_scale
            else mcolors.Normalize(vmin=0, vmax=_vmax))
    im = ax.imshow(image, cmap=cmap, norm=norm, origin='upper',
                   aspect='equal', interpolation='none')
    ax.figure.colorbar(im, ax=ax, fraction=0.042, pad=0.03, shrink=0.82,
                       label='Intensity (counts)')


# ── LaueTools files ───────────────────────────────────────────────────────────

def load_fit_peaklist(fit_path) -> pd.DataFrame:
    """Indexed peak table from a `.fit` file.

    Columns depend on the file, but typically include Intensity, h, k, l,
    Energy, Xexp, Yexp, Xtheo, Ytheo. Inspect ``.columns`` before relying on a
    name — LaueTools versions differ.
    """
    from lauexplore._parsers._fitfile import FitFile

    return FitFile(str(fit_path)).peaklist


def load_calibration_from_fit(fit_path) -> dict:
    """UB matrix and detector calibration from a `.fit` file.

    Returns keys ``ub_matrix`` (3×3), ``calibration_parameters``
    ``[distance, x_center, y_center, x_beta, x_gamma]``, ``pixel_size`` in mm,
    and ``frame_shape`` as ``(n_rows, n_cols)``.

    The UB here is the orientation of the crystal that was indexed; see
    `geometry.lab_vectors_from_UB` for applying the lattice metric to it.
    """
    from lauexplore._parsers._fitfile import FitFile

    fit = FitFile(str(fit_path))
    framedim = fit.CCDdict['framedim']
    return {
        'ub_matrix': fit.UB,
        'calibration_parameters': list(fit.CCDdict['DetectorParameters']),
        'pixel_size': float(fit.CCDdict['pixelsize']),
        'frame_shape': (int(float(framedim[0])), int(float(framedim[1]))),
    }


def load_dat(path) -> Tuple[str, pd.DataFrame]:
    """Load a LaueTools `.dat` peak file.

    Returns ``(header_line, df)``; the header is kept verbatim so the file can
    be written back unchanged by `write_dat`.
    """
    with open(path) as f:
        header_line = f.readline().rstrip('\n')
    columns = header_line.split()
    df = pd.read_csv(path, sep=r'\s+', skiprows=1, names=columns)
    return header_line, df


def write_dat(df: pd.DataFrame, path, header_line: str) -> None:
    """Write a peak DataFrame back to a LaueTools-compatible `.dat` file."""
    with open(path, 'w') as f:
        f.write(header_line.rstrip('\n') + '\n')
        df.to_csv(f, sep=' ', header=False, index=False, float_format='%.2f')
