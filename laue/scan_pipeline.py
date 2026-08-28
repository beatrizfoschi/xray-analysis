"""
scan_pipeline.py — Scan-level pipeline for per-position Laue spot analysis.

Reads one ROI per scan position and hands it to an analysis function:
``spot_metrics.analyze_spot`` by default, ``spot_fit.fit_spot`` for the
parametric multi-Gaussian fit. Everything around that — HDF5 access in either
layout, the thread pool, the grid bookkeeping, the masked positions, the maps —
is shared, so a new per-ROI analysis is one ``analysis_fn=`` away.

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

df = run_spot_pipeline(
    img_source   = stack,
    scan         = scan,
    roi_center   = (534, 993),    # (x, y) XMAS 1-based from roi_viewer
    boxsize      = 25,
    scan_subset  = (40, 125, 10, 50),   # (i0, i1, j0, j1) — one LED
)

# 3. Plot maps
plot_spot_maps(df, scan)

# Same pipeline, parametric fit instead of moments: one ROI, N chosen per
# position, and the maps follow the columns the fit returns.
from laue.spot_fit import fit_spot

df_fit = run_spot_pipeline(
    img_source  = stack,
    scan        = scan,
    roi_center  = (534, 993),
    boxsize     = 25,
    analysis_fn = fit_spot,
    n_components= "auto",     # forwarded to fit_spot
    criterion   = "bic",
)
plot_spot_maps(df_fit, scan)
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

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

from laue._imaging import crop_roi as _crop_roi


# ── Worker plumbing ───────────────────────────────────────────────────────────
#
# All of this lives at module level because a worker pickles what it runs, and a
# closure cannot be pickled. Nothing here may hold an open HDF5 handle or the
# scan object either — positions are resolved to plain (index, x, y) tuples in
# the parent before they are handed out.

def _frame_number(path: Path) -> int:
    """Sort key for per-frame files named ``..._<index>.h5``."""
    return int(path.stem.split("_")[-1])


def _list_frame_files(folder: Path) -> list[Path]:
    files = sorted(Path(folder).glob("*.h5"), key=_frame_number)
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {folder}")
    return files


class _RoiReader:
    """Reads one ROI per frame index, holding its HDF5 handle open.

    Opening the file per call serialises workers behind repeated open overhead,
    so the handle is cached — but an h5py handle survives neither a pickle nor a
    fork, so the cache is deliberately dropped on serialisation and reopened on
    first use in whatever thread or process ends up doing the reading.

    In direct mode the file list is rebuilt in the worker rather than shipped to
    it. joblib pickles the reader once per batch, and a scan of twenty thousand
    frames would otherwise send its whole list of paths along every time; the
    glob is deterministic, so re-running it costs one directory listing per
    worker and nothing per batch.
    """

    def __init__(self, img_source, *, direct_mode, h5_img_key,
                 direct_h5_key, squeeze, row_slice, col_slice, pad):
        self.img_source    = Path(img_source)
        self.direct_mode   = direct_mode
        self.h5_img_key    = h5_img_key
        self.direct_h5_key = direct_h5_key
        self.squeeze       = squeeze
        self.row_slice     = row_slice
        self.col_slice     = col_slice
        self.pad           = pad          # (top, bottom, left, right) or None
        self._reset_transient()

    def _reset_transient(self) -> None:
        self._files: list[Path] | None = None
        self._local  = threading.local()
        self._handles: list = []
        self._lock   = threading.Lock()

    @property
    def files(self) -> list[Path]:
        if self._files is None:
            self._files = _list_frame_files(self.img_source)
        return self._files

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for key in ("_files", "_local", "_handles", "_lock"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._reset_transient()

    def read(self, idx: int) -> np.ndarray:
        if self.direct_mode:
            # One file per frame: nothing to keep open between calls.
            with h5py.File(self.files[idx], "r") as h5f:
                ds  = h5f[self.direct_h5_key]
                roi = (ds[0, self.row_slice, self.col_slice] if self.squeeze
                       else ds[self.row_slice, self.col_slice]).astype(np.float64)
        else:
            h5f = getattr(self._local, "h5f", None)
            if h5f is None:
                h5f = h5py.File(self.img_source, "r")
                self._local.h5f = h5f
                with self._lock:
                    self._handles.append(h5f)
            roi = h5f[self.h5_img_key][idx, self.row_slice, self.col_slice].astype(np.float64)

        if self.pad is not None:
            top, bottom, left, right = self.pad
            roi = np.pad(roi, ((top, bottom), (left, right)))
        return roi

    def close(self) -> None:
        with self._lock:
            for h5f in self._handles:
                h5f.close()
            self._handles.clear()


def _analyse_one(task, reader: _RoiReader, analysis_fn, spot_kwargs: dict) -> dict:
    i, j, idx, x_um, y_um = task
    metrics = analysis_fn(reader.read(idx), **spot_kwargs)
    return {
        "i":         i,
        "j":         j,
        "frame_idx": idx,
        "x_um":      x_um,
        "y_um":      y_um,
        "status":    "ok",
        **metrics,
    }


def _analyse_array(task, analysis_fn, spot_kwargs: dict) -> tuple[int, dict]:
    k, roi = task
    return k, analysis_fn(roi, **spot_kwargs)


def _use_processes(executor: str, analysis_fn) -> bool:
    if executor == "auto":
        return analysis_fn is not analyze_spot
    if executor in ("thread", "process"):
        return executor == "process"
    raise ValueError(
        f'executor must be "auto", "thread" or "process", got {executor!r}'
    )


def _run_parallel(jobs, *, total: int, workers: int, use_processes: bool, desc: str):
    """Run `jobs` and yield results as they finish, with a progress bar.

    joblib rather than `concurrent.futures.ProcessPoolExecutor`, which hangs when
    driven from a Jupyter kernel: on Linux the pool inherits the kernel by
    forking, and forking a process that already holds threads — HDF5's, BLAS's,
    the kernel's own — deadlocks the children before they run a single task. That
    is what a run stuck at 0/N with the workers alive looks like. joblib's loky
    backend starts workers cleanly instead of forking, which is the whole reason
    it exists, and it also serialises through cloudpickle, so an ``analysis_fn``
    defined in a notebook cell works where a pickled one would not.

    ``generator_unordered`` is what keeps the bar honest: results are yielded as
    each finishes rather than collected into a list at the end.
    """
    from joblib import Parallel

    backend = "loky" if use_processes else "threading"
    results = Parallel(
        n_jobs=workers,
        backend=backend,
        return_as="generator_unordered",
    )(jobs)
    with tqdm(total=total, desc=desc, unit="roi") as pbar:
        for item in results:
            yield item
            pbar.update(1)


def _prepare_reader(
    img_source,
    roi_center: tuple[int, int],
    boxsize: int,
    *,
    h5_img_key: str,
    direct_h5_key: str,
    coords: str,
) -> tuple[_RoiReader, str]:
    """Resolve the image source and ROI geometry into a reader.

    Shared by `run_spot_pipeline` and `read_roi_stack` so that a ROI means the
    same pixels either way — the clamping and the zero padding at a detector edge
    included.
    """
    img_source  = Path(img_source)
    direct_mode = img_source.is_dir()

    x, y    = roi_center
    cen_col = (x - 1) if coords == "xmas" else x
    cen_row = (y - 1) if coords == "xmas" else y

    if direct_mode:
        files = _list_frame_files(img_source)
        with h5py.File(files[0], "r") as h5f:
            src_shape = h5f[direct_h5_key].shape   # (1, H, W) or (H, W)
        squeeze = len(src_shape) == 3
        H, W    = src_shape[-2], src_shape[-1]
        print(f"Direct mode: {len(files)} files, detector {H}×{W}")
    else:
        squeeze = False
        with h5py.File(img_source, "r") as h5f:
            _, H, W = h5f[h5_img_key].shape

    # Read only the ROI pixels, never the whole frame.
    row_slice = slice(max(0, cen_row - boxsize), min(H, cen_row + boxsize + 1))
    col_slice = slice(max(0, cen_col - boxsize), min(W, cen_col + boxsize + 1))

    pad = (
        max(0, boxsize - cen_row),
        max(0, (cen_row + boxsize + 1) - H),
        max(0, boxsize - cen_col),
        max(0, (cen_col + boxsize + 1) - W),
    )
    reader = _RoiReader(
        img_source,
        direct_mode=direct_mode,
        h5_img_key=h5_img_key,
        direct_h5_key=direct_h5_key,
        squeeze=squeeze,
        row_slice=row_slice,
        col_slice=col_slice,
        pad=pad if any(pad) else None,
    )
    return reader, ("direct" if direct_mode else "virtual stack")


def _resolve_positions(scan, scan_subset, mask):
    """Turn the scan grid into worker tasks of plain numbers.

    Every call into the scan object happens here, in the parent process, so the
    scan itself never has to cross into a worker. Masked positions become rows
    carrying no metric keys at all; pandas fills those columns with NaN, which is
    what keeps the grid rectangular for plotting.
    """
    if scan_subset is not None:
        i0, i1, j0, j1 = scan_subset
    else:
        i0, i1 = 0, scan.nbxpoints
        j0, j1 = 0, scan.nbypoints

    active_tasks: list[tuple] = []
    masked_rows: list[dict] = []
    n_pos = 0
    for i in range(i0, i1):
        for j in range(j0, j1):
            n_pos += 1
            idx = int(scan.ij_to_index(i, j))
            x_um, y_um = scan.ij_to_xy(i, j)
            x_um, y_um = float(x_um) * 1e3, float(y_um) * 1e3
            if mask is not None and not mask[idx]:
                masked_rows.append({
                    "i": i, "j": j, "frame_idx": idx,
                    "x_um": x_um, "y_um": y_um, "status": "masked",
                })
            else:
                active_tasks.append((i, j, idx, x_um, y_um))

    return active_tasks, masked_rows, n_pos, (i0, i1, j0, j1)


# ── Reading a stack into memory ───────────────────────────────────────────────

def read_roi_stack(
    img_source:    str | Path,
    scan,
    roi_center:    tuple[int, int],
    boxsize:       int,
    *,
    h5_img_key:    str = "frames",
    direct_h5_key: str = "entry_0000/CRGIF/eiger4m/data",
    coords:        str = "xmas",
    scan_subset:   tuple[int, int, int, int] | None = None,
    workers:       int = 8,
    dtype=np.float32,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Read one ROI per scan position into a single array.

    `run_spot_pipeline` streams positions and never holds more than one crop, so
    it cannot support anything that needs to see the whole scan at once —
    sub-pixel alignment against a common reference above all. This is the way in
    for those: read once, operate on the stack, then analyse it with
    `analyse_stack`.

    Only worth it when something really is stack-level. A ROI of
    ``(2·boxsize+1)²`` float32 over a few thousand positions is small (a 41x41
    crop over 5000 positions is ~34 MB), but the whole point of the streaming
    pipeline is not needing that.

    Returns
    -------
    stack : (n_positions, h, w) array, in the row order of ``index``.
    index : DataFrame with ``i, j, frame_idx, x_um, y_um`` — the grid coordinates
        of each slice, ready to be concatenated with per-position results.
    """
    reader, mode_label = _prepare_reader(
        img_source, roi_center, boxsize,
        h5_img_key=h5_img_key, direct_h5_key=direct_h5_key, coords=coords,
    )
    tasks, _, n_pos, (i0, i1, j0, j1) = _resolve_positions(scan, scan_subset, None)
    print(f"Reading {n_pos} ROIs  ({i1-i0} × {j1-j0})  [{mode_label}]...")

    side  = 2 * boxsize + 1
    stack = np.empty((len(tasks), side, side), dtype=dtype)
    try:
        # Reads release the GIL, so threads overlap them; the ROIs would have to
        # be pickled back out of a process pool for no gain.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(reader.read, task[2]): k
                for k, task in enumerate(tasks)
            }
            with tqdm(total=len(tasks), desc="Reading ROIs", unit="roi") as pbar:
                for future in as_completed(futures):
                    stack[futures[future]] = future.result()
                    pbar.update(1)
    finally:
        reader.close()

    index = pd.DataFrame(
        [{"i": i, "j": j, "frame_idx": idx, "x_um": x, "y_um": y}
         for i, j, idx, x, y in tasks]
    )
    return stack, index


def analyse_stack(
    stack:       np.ndarray,
    index:       pd.DataFrame | None = None,
    *,
    analysis_fn: Callable[..., dict] = analyze_spot,
    workers:     int = 8,
    executor:    str = "auto",
    **spot_kwargs,
) -> pd.DataFrame:
    """Run a per-ROI analysis over a stack already in memory.

    The stack-level counterpart of `run_spot_pipeline`, for when the ROIs have
    been through something that needed all of them at once — `_imaging.align_stack`
    being the case this exists for. Same analysis functions, same executor
    trade-off, same DataFrame out.

    Parameters
    ----------
    stack : (n, h, w) array
    index : DataFrame, optional
        Grid coordinates for each slice, as returned by `read_roi_stack`. Joined
        onto the results so the output can be mapped; without it the rows carry
        only a ``frame`` counter and `plot_spot_maps` has nothing to place them on.
    executor : {"auto", "thread", "process"}
        As in `run_spot_pipeline`, including the ``if __name__ == "__main__":``
        a process pool needs when this is called from a script.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"stack must be (n, h, w), got shape {stack.shape}")
    if index is not None and len(index) != len(stack):
        raise ValueError(
            f"index has {len(index)} rows but the stack has {len(stack)} frames"
        )

    use_processes = _use_processes(executor, analysis_fn)
    print(f"Analysing {len(stack)} ROIs  "
          f"[{workers} {'processes' if use_processes else 'threads'}]...")

    from joblib import delayed

    results: list[dict | None] = [None] * len(stack)
    jobs = (
        delayed(_analyse_array)(
            (k, np.asarray(stack[k], dtype=np.float64)), analysis_fn, spot_kwargs
        )
        for k in range(len(stack))
    )
    for k, metrics in _run_parallel(jobs, total=len(stack), workers=workers,
                                    use_processes=use_processes,
                                    desc="Analysing ROIs"):
        results[k] = metrics

    df = pd.DataFrame(results)
    df.insert(0, "frame", np.arange(len(stack)))
    df["status"] = "ok"
    if index is not None:
        df = pd.concat([index.reset_index(drop=True), df.drop(columns="frame")], axis=1)
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_spot_pipeline(
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
    mask:          np.ndarray | None = None,
    analysis_fn:   Callable[..., dict] = analyze_spot,
    executor:      str = "auto",
    **spot_kwargs,
) -> pd.DataFrame:
    """Run a per-position spot analysis over a (sub)set of scan positions.

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
    mask : np.ndarray of bool, shape (n_frames,), optional
        True = process, False = skip. Indexed by the linear frame index
        (same as scan.ij_to_index). Skipped positions appear in the
        DataFrame with status="masked" and NaN metrics, keeping the grid
        complete for plotting.
    analysis_fn : callable
        What to run on each ROI: takes the 2-D crop plus ``**spot_kwargs`` and
        returns a flat dict, which becomes the metric columns of the row.
        ``spot_metrics.analyze_spot`` (the default) for moment-based morphology,
        ``spot_fit.fit_spot`` for the parametric multi-Gaussian fit.

        Whatever is passed has to be a *per-position* function. An operation that
        needs the whole stack — sub-pixel alignment against a common reference,
        for one — cannot go here, because positions are read and analysed one at
        a time and never held together. Run `_imaging.align_stack` as a pre-pass
        instead; see its docstring for which metrics that actually affects.

        With ``executor="process"`` the function is serialised to the workers.
        joblib's loky backend uses cloudpickle, so a function defined in a
        notebook cell works as well as an imported one.
    executor : {"auto", "thread", "process"}
        Threads share the interpreter, so they overlap the HDF5 reads but not
        Python-level computation — everything holding the GIL still runs one at a
        time. That is the right trade for ``analyze_spot``, which is dominated by
        the reads. An iterative fit is not: ``fit_spot`` measured ~4 positions/s
        against ~91 for ``analyze_spot`` through the same thread pool, and 3.5x
        that once the work is in separate processes.

        ``"auto"`` picks threads for ``analyze_spot`` and processes for anything
        else. Override it when the guess is wrong — a cheap custom ``analysis_fn``
        is better off in threads, where there is no serialisation and no start-up
        cost.

        From a **script**, put the call behind ``if __name__ == "__main__":``;
        workers re-import the module they were started from. Notebooks need no
        such guard.
    **spot_kwargs
        Extra keyword arguments forwarded to ``analysis_fn``. With
        ``executor="process"`` these are pickled too, so keep them to plain data.

    Returns
    -------
    pd.DataFrame with columns ``i, j, frame_idx, x_um, y_um, status`` plus
    whatever keys ``analysis_fn`` returns — the Layer 1-3 morphology indicators
    for ``analyze_spot``, the fitted components for ``fit_spot``.
    """
    reader, mode_label = _prepare_reader(
        img_source, roi_center, boxsize,
        h5_img_key=h5_img_key, direct_h5_key=direct_h5_key, coords=coords,
    )
    active_tasks, masked_rows, n_pos, extent = _resolve_positions(
        scan, scan_subset, mask
    )
    i0, i1, j0, j1 = extent

    n_active  = len(active_tasks)
    n_masked  = n_pos - n_active
    mask_info = f",  masked={n_masked}" if mask is not None else ""

    use_processes = _use_processes(executor, analysis_fn)
    pool_label = "processes" if use_processes else "threads"

    print(f"Running pipeline on {n_pos} positions  ({i1-i0} × {j1-j0})  "
          f"[{mode_label}{mask_info},  {workers} {pool_label}]...")

    rows = list(masked_rows)
    if active_tasks:
        from joblib import delayed

        try:
            jobs = (
                delayed(_analyse_one)(task, reader, analysis_fn, spot_kwargs)
                for task in active_tasks
            )
            rows.extend(_run_parallel(
                jobs, total=n_active, workers=workers,
                use_processes=use_processes, desc="Analysing spots",
            ))
        finally:
            # Only the parent's own handles; each worker closes its own on exit.
            reader.close()

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
        Output of ``run_spot_pipeline`` (must contain i, j, theta columns).

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
    ("fwhm1",            "FWHM major axis (px)",        "viridis"),
    ("fwhm2",            "FWHM minor axis (px)",        "viridis"),
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


# Maps for a run with analysis_fn=spot_fit.fit_spot.
#
# The label-free quantities come first: `separation`, `orientation` and `ratio`
# survive both a translation of the crop and a relabelling of the components,
# while the per-component positions do neither. Components are ordered by
# amplitude, so where two of them are nearly equally bright the labels can swap
# between neighbouring positions and speckle the x1/x2 maps without anything
# physical having changed.
_FIT_METRICS = [
    ("separation",      "Sub-peak separation (px)",   "magma"),
    ("orientation",     "Separation angle (°)",       "twilight"),
    ("ratio",           "A₂ / (A₁ + A₂)",             "RdBu_r"),
    ("n_components",    "N components",               "Reds"),
    ("total_amplitude", "Total amplitude",            "viridis"),
    ("sigma_x1",        "Width σₓ (px)",              "cividis"),
    ("sigma_y1",        "Width σᵧ (px)",              "cividis"),
    ("bg",              "Background",                 "viridis"),
    ("centroid_x",      "Centroid x (px)",            "viridis"),
    ("centroid_y",      "Centroid y (px)",            "viridis"),
    ("x1",              "Peak 1 x (px)",              "viridis"),
    ("y1",              "Peak 1 y (px)",              "viridis"),
    ("chi2",            "Reduced χ²",                 "hot"),
]


def plot_spot_maps(
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
        Output of run_spot_pipeline, from either analysis function.
    scan : lauexplore.scan.Scan
        Scan object for physical axis labels.
    metrics : list of (column, title, cmap) or None
        Which metrics to plot.  When None, picked from the columns present:
        the fit maps for a ``fit_spot`` run, the morphology maps otherwise.
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
        metrics = _FIT_METRICS if "n_components" in df.columns else _DEFAULT_METRICS
    metrics = [m for m in metrics if m[0] in df.columns]
    if not metrics:
        raise ValueError("None of the requested metric columns are in the DataFrame.")

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

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             constrained_layout=True)

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

    return fig
