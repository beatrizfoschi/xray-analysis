"""
scan_pipeline.py — Scan-level pipeline for Laue spot morphology analysis.

Typical workflow
----------------
# 1. (once) Build a virtual H5 stack from a folder of per-frame H5 files
stack = create_virtual_stack(
    folder    = "/data/.../RAW_DATA/scan/",
    output_h5 = "/data/.../stack.h5",
    h5_key    = "entry_0000/CRGIF/eiger4m/data",
)

# 2. Identify ROI with roi_viewer, then run pipeline on a subset
from lauexplore.scan import Scan
scan = Scan.from_h5("scan.h5")

df = run_pipeline(
    img_source   = stack,
    scan         = scan,
    roi_center   = (534, 993),    # (x, y) XMAS 1-based from roi_viewer
    boxsize      = 25,
    scan_subset  = (40, 125, 10, 50),   # (i0, i1, j0, j1) — one LED
)

# 3. Plot maps
plot_maps(df, scan)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from laue.spot_metrics import analyze_spot


# ── Virtual stack builder ─────────────────────────────────────────────────────

def create_virtual_stack(
    folder:     str | Path,
    output_h5:  str | Path,
    h5_key:     str = "entry_0000/CRGIF/eiger4m/data",
    pattern:    str = "*.h5",
) -> Path:
    """Create an HDF5 Virtual Dataset from a folder of per-frame H5 files.

    The resulting file has a single dataset ``frames`` of shape
    ``(n_frames, H, W)`` that links to the original files without copying data.

    Parameters
    ----------
    folder : Path
        Folder containing the individual H5 files.
    output_h5 : Path
        Output path for the virtual stack file.
    h5_key : str
        Dataset key inside each individual H5 (default Eiger key).
    pattern : str
        Glob pattern to match image files.

    Returns
    -------
    Path to the created virtual stack file.
    """
    folder    = Path(folder)
    output_h5 = Path(output_h5)

    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No '{pattern}' files found in {folder}")
    print(f"Found {len(files)} files.")

    with h5py.File(files[0], "r") as f:
        src_shape = f[h5_key].shape      # (H, W) or (1, H, W)
        dtype     = f[h5_key].dtype

    squeeze = len(src_shape) == 3        # True if each file stores (1, H, W)
    H, W    = src_shape[-2], src_shape[-1]
    n       = len(files)
    print(f"Stack shape: ({n}, {H}, {W})  dtype: {dtype}")

    layout = h5py.VirtualLayout(shape=(n, H, W), dtype=dtype)
    for i, path in enumerate(files):
        src = h5py.VirtualSource(path, h5_key, shape=src_shape)
        layout[i] = src[0] if squeeze else src

    with h5py.File(output_h5, "w") as f:
        f.create_virtual_dataset("frames", layout, fillvalue=0)
        f.attrs["source_folder"] = str(folder)
        f.attrs["source_key"]    = h5_key
        f.attrs["n_frames"]      = n

    print(f"Virtual stack → {output_h5}")
    return output_h5


# ── ROI crop ──────────────────────────────────────────────────────────────────

def _crop_roi(
    img:        np.ndarray,
    cen_row:    int,
    cen_col:    int,
    boxsize:    int,
) -> np.ndarray:
    """Crop a (2*boxsize+1) × (2*boxsize+1) region, zero-padding at borders."""
    r0 = max(0, cen_row - boxsize)
    r1 = min(img.shape[0], cen_row + boxsize + 1)
    c0 = max(0, cen_col - boxsize)
    c1 = min(img.shape[1], cen_col + boxsize + 1)
    roi = img[r0:r1, c0:c1]

    pad_top    = max(0, boxsize - cen_row)
    pad_bottom = max(0, (cen_row + boxsize + 1) - img.shape[0])
    pad_left   = max(0, boxsize - cen_col)
    pad_right  = max(0, (cen_col + boxsize + 1) - img.shape[1])
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        roi = np.pad(roi, ((pad_top, pad_bottom), (pad_left, pad_right)))
    return roi


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    img_source:    str | Path,
    scan,                               # lauexplore Scan object
    roi_center:    tuple[int, int],
    boxsize:       int,
    *,
    h5_img_key:    str = "frames",
    direct_h5_key: str = "entry_0000/CRGIF/eiger4m/data",
    coords:        str = "xmas",        # "xmas" (1-based) or "numpy" (0-based)
    scan_subset:   tuple[int, int, int, int] | None = None,
    workers:       int = 8,
    **spot_kwargs,
) -> pd.DataFrame:
    """Run the spot morphology pipeline over a (sub)set of scan positions.

    Parameters
    ----------
    img_source : Path
        Either:
        - A directory containing one H5 file per frame (direct mode — fastest).
          Files are sorted alphabetically; index 0 = first file.
        - A single H5 file with shape ``(n_frames, H, W)`` (virtual stack mode).
    scan : lauexplore.scan.Scan
        Scan object providing grid geometry and index ↔ (i, j) mapping.
    roi_center : (x, y)
        Centre of the detector ROI in XMAS 1-based (col, row) coordinates
        (same convention as roi_viewer).  Use coords="numpy" for 0-based.
    boxsize : int
        Half-side of the ROI crop.
    h5_img_key : str
        Dataset key when reading from a virtual/stacked H5 file (default "frames").
    direct_h5_key : str
        Dataset key inside each individual H5 file when img_source is a folder
        (default Eiger key "entry_0000/CRGIF/eiger4m/data").
    coords : {"xmas", "numpy"}
        Coordinate convention for roi_center.
    scan_subset : (i0, i1, j0, j1) or None
        Grid index range following lauexplore's convention (NOT numpy row/col):
          i = column index, x direction  (0..nbxpoints-1)
          j = row/line index, y direction (0..nbypoints-1)
        None = full scan.
    workers : int
        Number of parallel threads for H5 reading.
    **spot_kwargs
        Extra keyword arguments forwarded to ``analyze_spot``.

    Returns
    -------
    pd.DataFrame with columns:
        i, j, frame_idx, x_um, y_um,
        x_com, y_com, x_com_rel, y_com_rel,
        lambda1, lambda2, aspect_ratio, theta,
        streak_D50, streak_D95, core_tail_ratio
    """
    img_source = Path(img_source)
    direct_mode = img_source.is_dir()

    # Coordinate conversion
    x, y    = roi_center
    cen_col = (x - 1) if coords == "xmas" else x
    cen_row = (y - 1) if coords == "xmas" else y

    # Resolve file list and detector shape
    if direct_mode:
        files = sorted(
            img_source.glob("*.h5"),
            key=lambda p: int(p.stem.split("_")[-1])
        )
        if not files:
            raise FileNotFoundError(f"No .h5 files found in {img_source}")
        with h5py.File(files[0], "r") as h5f:
            src_shape = h5f[direct_h5_key].shape   # (1, H, W) or (H, W)
        squeeze = len(src_shape) == 3
        H, W    = src_shape[-2], src_shape[-1]
        print(f"Direct mode: {len(files)} files, detector {H}×{W}")
    else:
        files  = None
        squeeze = False
        with h5py.File(img_source, "r") as h5f:
            _, H, W = h5f[h5_img_key].shape

    # Pre-compute clamped ROI slices (read only ROI pixels, not the full frame)
    r0_src = max(0, cen_row - boxsize)
    r1_src = min(H, cen_row + boxsize + 1)
    c0_src = max(0, cen_col - boxsize)
    c1_src = min(W, cen_col + boxsize + 1)
    row_slice = slice(r0_src, r1_src)
    col_slice = slice(c0_src, c1_src)

    pad_top    = max(0, boxsize - cen_row)
    pad_bottom = max(0, (cen_row + boxsize + 1) - H)
    pad_left   = max(0, boxsize - cen_col)
    pad_right  = max(0, (cen_col + boxsize + 1) - W)
    needs_pad  = any([pad_top, pad_bottom, pad_left, pad_right])

    # Subset bounds
    if scan_subset is not None:
        i0, i1, j0, j1 = scan_subset
    else:
        i0, i1 = 0, scan.nbxpoints
        j0, j1 = 0, scan.nbypoints

    positions = [
        (i, j)
        for i in range(i0, i1)
        for j in range(j0, j1)
    ]
    n_pos = len(positions)
    mode_label = "direct" if direct_mode else "virtual stack"
    print(f"Running pipeline on {n_pos} positions  ({i1-i0} × {j1-j0})  [{mode_label}]...")

    def _process_one(ij: tuple[int, int]) -> dict:
        i, j       = ij
        idx        = scan.ij_to_index(i, j)
        x_um, y_um = scan.ij_to_xy(i, j)
        if direct_mode:
            with h5py.File(files[idx], "r") as h5f:
                ds  = h5f[direct_h5_key]
                roi = (ds[0, row_slice, col_slice] if squeeze
                       else ds[row_slice, col_slice]).astype(np.float64)
        else:
            with h5py.File(img_source, "r") as h5f:
                roi = h5f[h5_img_key][idx, row_slice, col_slice].astype(np.float64)
        if needs_pad:
            roi = np.pad(roi, ((pad_top, pad_bottom), (pad_left, pad_right)))
        metrics = analyze_spot(roi, **spot_kwargs)
        return {
            "i":         i,
            "j":         j,
            "frame_idx": idx,
            "x_um":      float(x_um) * 1e3,
            "y_um":      float(y_um) * 1e3,
            **metrics,
        }

    rows = [None] * n_pos
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, ij): k for k, ij in enumerate(positions)}
        with tqdm(total=n_pos, desc="Analysing spots", unit="spot") as pbar:
            for future in as_completed(futures):
                rows[futures[future]] = future.result()
                pbar.update(1)

    df = pd.DataFrame(rows)
    df.sort_values(["i", "j"], inplace=True, ignore_index=True)
    return df


# ── Scan-level: streak angle gradient ────────────────────────────────────────

def compute_theta_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``theta_gradient`` column: local rate of change of streak angle.

    Computes the magnitude of the spatial gradient of θ across the scan grid
    using central differences.  A smooth gradient marks screw-TD asterism
    zones; a sharp jump marks a GNB wall crossing.

    Parameters
    ----------
    df : DataFrame
        Output of ``run_pipeline`` (must contain i, j, theta columns).

    Returns
    -------
    DataFrame with an additional ``theta_gradient`` column (°/step).
    """
    df = df.copy()
    i_vals = np.sort(df["i"].unique())
    j_vals = np.sort(df["j"].unique())
    i_min, j_min = int(i_vals.min()), int(j_vals.min())

    grid = np.full((len(i_vals), len(j_vals)), np.nan)
    idx_map: dict[tuple, tuple] = {}
    for _, row in df.iterrows():
        gi = int(row["i"] - i_min)
        gj = int(row["j"] - j_min)
        grid[gi, gj] = row["theta"]
        idx_map[(int(row["i"]), int(row["j"]))] = (gi, gj)

    # Central-difference gradient (angle wrap-safe via complex trick)
    angle_rad = np.deg2rad(grid)
    cos_g, sin_g = np.cos(angle_rad), np.sin(angle_rad)
    dcos_di = np.gradient(cos_g, axis=0);  dsin_di = np.gradient(sin_g, axis=0)
    dcos_dj = np.gradient(cos_g, axis=1);  dsin_dj = np.gradient(sin_g, axis=1)
    # |dθ/di|² + |dθ/dj|²  in radians → degrees
    grad_mag = np.degrees(np.sqrt(
        (dcos_di ** 2 + dsin_di ** 2) + (dcos_dj ** 2 + dsin_dj ** 2)
    ))

    df["theta_gradient"] = df.apply(
        lambda r: float(grad_mag[idx_map[(int(r["i"]), int(r["j"]))]]),
        axis=1,
    )
    return df


# ── 2D map visualisation ──────────────────────────────────────────────────────

_DEFAULT_METRICS = [
    # Layer 1 — essential core
    ("streak_D95",       "Streak length D95 (px)",      "inferno"),
    ("theta",            "Streak angle θ (°)",          "hsv"),
    ("aspect_ratio",     "Aspect ratio λ₁/λ₂",         "plasma"),
    ("x_com_rel",        "COM displacement x (px)",     "RdBu"),
    ("y_com_rel",        "COM displacement y (px)",     "RdBu"),
    # Layer 2 — physical interpretation
    ("core_tail_ratio",  "Core-to-tail ratio R",        "viridis"),
    ("kurtosis_streak",  "Kurtosis (streak axis)",      "coolwarm"),
    ("skewness_streak",  "Skewness (streak axis)",      "coolwarm"),
    ("d95_d50_ratio",    "D95/D50 ratio",               "plasma"),
    # Layer 3 — refinement
    ("effective_radius", "Effective radius √(λ₁+λ₂)",  "inferno"),
    ("tail_decay_xi",    "Tail decay length ξ (px)",    "magma"),
    ("gaussian_residual","Gaussian residual (norm.)",   "hot"),
    ("peak_com_offset",  "Peak–COM offset (px)",        "plasma"),
    ("n_local_maxima",   "N local maxima",              "Reds"),
]


def plot_maps(
    df:                 pd.DataFrame,
    scan,
    metrics:            list[tuple[str, str, str]] | None = None,
    *,
    ncols:              int = 4,
    percentile_clip:    tuple[float, float] = (2, 98),
    figsize:            tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot 2D maps of morphology metrics over the scan grid.

    Parameters
    ----------
    df : DataFrame
        Output of run_pipeline.
    scan : lauexplore.scan.Scan
        Scan object for physical axis labels.
    metrics : list of (column, title, cmap) or None
        Which metrics to plot.  Defaults to all _DEFAULT_METRICS.
    ncols : int
        Maximum number of panels per row (default 4).
    percentile_clip : (lo, hi)
        Colour scale percentiles.
    figsize : (w, h) or None
        Auto-computed if None.

    Returns
    -------
    matplotlib Figure
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS

    n     = len(metrics)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    x_um = np.sort(df["x_um"].unique())
    y_um = np.sort(df["y_um"].unique())
    dx   = (x_um[-1] - x_um[0]) if len(x_um) > 1 else 1.0
    dy   = (y_um[-1] - y_um[0]) if len(y_um) > 1 else 1.0
    aspect = dx / dy if dy > 0 else 1.0

    panel_h = 4.0
    panel_w = max(3.0, panel_h * aspect)
    if figsize is None:
        figsize = (ncols * panel_w + ncols * 0.5, nrows * (panel_h + 1.5))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    i_min, j_min = df["i"].min(), df["j"].min()
    nbx, nby     = df["i"].nunique(), df["j"].nunique()
    grid_shape   = (nbx, nby)
    extent       = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]

    for idx, (col, title, cmap) in enumerate(metrics):
        ax = axes[idx // ncols, idx % ncols]

        grid = np.full(grid_shape, np.nan)
        for _, row in df.iterrows():
            grid[int(row["i"] - i_min), int(row["j"] - j_min)] = row[col]

        data = grid.T
        lo   = np.nanpercentile(data, percentile_clip[0])
        hi   = np.nanpercentile(data, percentile_clip[1])

        im = ax.imshow(data, origin="lower", aspect="equal",
                       extent=extent, cmap=cmap, vmin=lo, vmax=hi)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

    # Hide unused axes in the last row
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    return fig
