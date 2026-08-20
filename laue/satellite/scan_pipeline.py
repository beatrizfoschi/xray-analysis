"""Batch satellite-peak analysis over a 2D XRD-CT scan.

Public API
----------
run_satellite_pipeline(img_source, scan, roi_center, ...)  → pd.DataFrame
make_metric_grid(df, col)                            → np.ndarray (nby, nbx)
plot_satellite_maps(df, ...)                               → matplotlib Figure
plot_detector_positions(df, ...)                     → Figure, or (Figure, Figure, pd.DataFrame) with segmentation
plot_satellite_stats(df, orders, ...)                → matplotlib Figure
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import skew, kurtosis

from laue.satellite.detection import detect_satellites, locate_sl0_by_local_max
from laue.satellite.metrics import compute_metrics, metrics_to_flat_dict, per_order_metrics
from laue.satellite.period import (
    layer_period_from_peaks, locate_sl0_from_ladder, resolve_and_apply_order_sign,
)

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _sl0_columns(peaks, period: Dict[str, Any], period_kw: Dict[str, Any],
                 result: Dict[str, Any], crop: np.ndarray,
                 sl0_boxsize: float = 3.0) -> Dict[str, Any]:
    """Where SL₀ is, given that the detected order-0 peak is the bulk (§4.13).

    SL₀ is first predicted from the satellite ladder, then searched for in the
    raw crop around that prediction.  ``sl0_pos`` is the prediction;
    ``sl0_measured_pos`` exists only where SL₀ was actually found.
    ``sl0_confirmed=False`` is an expected outcome across much of a scan — the
    bulk dominates its own flank by orders of magnitude — and is not a failure.

    ``bulk_sl0_offset_px`` is the physics output: the bulk→SL₀ separation
    carries the average out-of-plane strain of the stack.

    Runs per pixel inside the worker, so it must be cheap and must never raise:
    a failure here would cost the row every other metric.
    """
    out: Dict[str, Any] = {}
    try:
        lam = period.get('period_angstrom')
        if lam is None or not np.isfinite(lam):
            return out
        sl0 = locate_sl0_from_ladder(
            peaks,
            hkl=period_kw['hkl'], lattice=period_kw['lattice'],
            UB=period_kw['UB'], detector=period_kw['detector'],
            crop_origin_px=period_kw['crop_origin_px'],
            period_angstrom=float(lam), verbose=False)

        out['sl0_pos'] = float(sl0['sl0_pos_along_axis'])
        for key in ('bulk_sl0_offset_px', 'bulk_sl0_offset_deg', 'gap_pred_px',
                    'deg_per_px', 'amplitude_ratio_order0_to_n1'):
            if key in sl0:
                out[key] = float(sl0[key])
        if 'order0_is_sl0' in sl0:
            out['order0_is_sl0'] = bool(sl0['order0_is_sl0'])

        if sl0.get('confident'):
            found = locate_sl0_by_local_max(
                crop, sl0['sl0_pos_along_axis'],
                result['axis_angle'], result['sl0_center'],
                boxsize=sl0_boxsize)
            out['sl0_confirmed'] = bool(found['sl0_confirmed'])
            if found['sl0_confirmed']:
                for key in ('sl0_measured_pos', 'sl0_measured_amplitude'):
                    out[key] = float(found[key])
                # Measured, so it supersedes the predicted-position offset above.
                if 'detected_order0_pos' in sl0:
                    out['bulk_sl0_offset_px'] = float(
                        sl0['detected_order0_pos'] - found['sl0_measured_pos'])
    except Exception as exc:
        out['sl0_error'] = f'{type(exc).__name__}: {exc}'
    return out


# ── Worker ────────────────────────────────────────────────────────────────────
# Module-level function required for ProcessPoolExecutor pickling.

def _process_one(args: tuple) -> Dict[str, Any]:
    (i, j, frame_idx, x_um, y_um,
     img_source, h5_img_key, row0, row1, col0, col1,
     detect_kw, period_kw, sl0_boxsize) = args

    try:
        with h5py.File(img_source, 'r') as h5f:
            crop = np.asarray(
                h5f[h5_img_key][frame_idx, row0:row1, col0:col1],
                dtype=np.float32,
            )

        result = detect_satellites(crop, **detect_kw)

        # The sign of the detected orders is a detector-space convention and can
        # be inverted relative to the crystal (§4.11.2).  Settle it before the
        # metrics: the period would rail against a bound and say so, but the
        # asymmetry indicators would silently change sign across the map.
        sign_flat: Dict[str, Any] = {}
        if period_kw is not None and all(
                period_kw.get(k) is not None
                for k in ('hkl', 'lattice', 'UB', 'detector', 'crop_origin_px')):
            try:
                result['peaks'], verdict = resolve_and_apply_order_sign(
                    result['peaks'],
                    hkl=period_kw['hkl'], lattice=period_kw['lattice'],
                    UB=period_kw['UB'], detector=period_kw['detector'],
                    crop_origin_px=period_kw['crop_origin_px'])
                sign_flat = {'order_sign_inverted': bool(verdict['inverted']),
                             'order_sign_confident': bool(verdict['confident']),
                             'order_sign_arrow_cos': float(verdict['arrow_cos'])}
            except Exception as exc:
                sign_flat = {'order_sign_inverted': False,
                             'order_sign_confident': False,
                             'order_sign_error': f'{type(exc).__name__}: {exc}'}

        metrics = compute_metrics(result['peaks'])
        flat    = metrics_to_flat_dict(metrics)
        flat.update(sign_flat)

        flat['axis_angle'] = float(result['axis_angle'])

        # Mean amplitude of the ±1 satellite orders (both sides averaged)
        amps1 = [p['amplitude'] for p in result['peaks'] if abs(p['order']) == 1]
        flat['sl1_mean_intensity'] = float(np.mean(amps1)) if amps1 else float('nan')

        # 2-D detector position of each satellite (crop coordinates)
        for pk in result['peaks']:
            label = f'n{pk["order"]}'
            r, c = pk['position_2d']
            flat[f'pos_row_{label}'] = float(r)
            flat[f'pos_col_{label}'] = float(c)

        distances, intensities = result['profile']
        flat.update(per_order_metrics(result['peaks'], distances, intensities))

        if period_kw is not None:
            # A period failure must not discard the other metrics for this pixel:
            # the Laue routes raise when a position has fewer than two satellite
            # orders, which is common at the edge of a mesa.
            try:
                period = layer_period_from_peaks(result['peaks'], **period_kw)

                # A fit railed against a period bound is not a measurement — the
                # model could not reproduce the observed spacing at any period in
                # range.  Emitting the bound would paint whole regions of the map
                # with the same plausible-looking number, so the value goes to NaN
                # and the reason is recorded, exactly as for a raised exception.
                railed = bool(period.get('fit_at_bound'))
                flat['period_nm']       = float('nan') if railed else period['period_nm']
                flat['period_angstrom'] = float('nan') if railed else period['period_angstrom']
                flat['delta_q_inv_ang'] = float('nan') if railed else period['delta_q_inv_ang']
                flat['period_method']   = period['method']
                if railed:
                    lo, hi = period.get('period_bounds_angstrom',
                                        (float('nan'), float('nan')))
                    flat['period_error'] = (
                        f'fit_at_bound: railed against the [{lo:.0f}, {hi:.0f}] A '
                        f'search range; check parent_offset_deg and the UB frame')

                for key in ('gamma_deg', 'two_theta_measured', 'chi_measured',
                            'fit_rms_deg', 'parent_offset_deg', 'train_delta_deg',
                            'fit_at_bound'):
                    if key in period:
                        flat[key] = float(period[key])

                # Per-pair values, so the spacing can be inspected order by order
                # instead of only through the mean and delta_q_std.
                for pair in period['per_pair']:
                    n1, n2 = pair['orders']
                    sfx = f'n{n1}_n{n2}'
                    if 'delta_px' in pair:                     # monochromatic / analytic
                        flat[f'delta_px_{sfx}']  = float(pair['delta_px'])
                        flat[f'period_nm_{sfx}'] = float(pair['period_nm'])
                        if 'delta_q_inv_ang' in pair:
                            flat[f'delta_q_inv_ang_{sfx}'] = float(pair['delta_q_inv_ang'])
                    else:                                      # laue_forward, angular
                        flat[f'sep_meas_deg_{sfx}'] = float(pair['sep_meas_deg'])
                        flat[f'sep_pred_deg_{sfx}'] = float(pair['sep_pred_deg'])

                # Where SL0 actually is.  The detected order-0 peak is the bulk
                # reflection in an MQW (§4.13), so `bulk_pos` above is NOT SL0.
                # This needs the orientation, hence the Laue routes only; it is
                # cheap because Λ has just been fitted here.
                if not railed and all(period_kw.get(k) is not None
                                      for k in ('hkl', 'lattice', 'UB', 'detector')):
                    flat.update(_sl0_columns(result['peaks'], period, period_kw,
                                             result, crop, sl0_boxsize))
            except Exception as exc:
                flat['period_nm']       = float('nan')
                flat['period_angstrom'] = float('nan')
                flat['delta_q_inv_ang'] = float('nan')
                flat['period_error']    = f'{type(exc).__name__}: {exc}'

        return {
            'i': i, 'j': j, 'frame_idx': frame_idx,
            'x_um': x_um, 'y_um': y_um, 'status': 'ok',
            **flat,
        }

    except Exception as exc:
        return {
            'i': i, 'j': j, 'frame_idx': frame_idx,
            'x_um': x_um, 'y_um': y_um, 'status': f'error: {exc}',
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_satellite_pipeline(
    img_source,
    scan,
    roi_center: Tuple[int, int],
    boxsize: int,
    h5_img_key: str = 'frames',
    scan_subset: Optional[Tuple[int, int, int, int]] = None,
    workers: int = 8,
    coords: str = 'numpy',
    mask: Optional[np.ndarray] = None,
    # detection parameters
    axis_angle: Optional[float] = None,
    n_max: int = 3,
    min_prominence: float = 0.05,
    strip_width: float = 5.0,
    bg_sigma: float = 20.0,
    peak_min_width: Optional[float] = 2.0,
    hot_pixel_sigma: Optional[float] = 10.0,
    n_range: Optional[Tuple[int, int]] = None,
    spacing_px: Optional[float] = None,
    adaptive_fill_win: bool = False,
    # geometry — all required to compute the layer period
    pixel_size_mm: Optional[float] = None,
    detector_distance_mm: Optional[float] = None,
    wavelength_angstrom: Optional[float] = None,
    energy_kev: Optional[float] = None,
    two_theta_0_deg: float = 0.0,
    chi_deg: float = 0.0,
    # period route — 'monochromatic' (default, legacy) | 'laue_analytic' | 'laue_forward'
    period_method: str = 'monochromatic',
    hkl: Optional[Tuple[int, int, int]] = None,
    lattice: Optional[Tuple[float, float]] = None,
    UB: Optional[np.ndarray] = None,
    detector=None,
    det_file=None,
    satellite_axis_psi_deg: Optional[float] = None,
    sl0_boxsize: float = 3.0,
) -> pd.DataFrame:
    """Run satellite peak analysis for every position in a 2D XRD-CT scan.

    Parameters
    ----------
    img_source      : path to the stacked HDF5 file (dataset shape: n_frames × H × W).
    scan            : lauexplore Scan object — provides nbxpoints, nbypoints,
                      ij_to_index(i, j), ij_to_xy(i, j).
    roi_center      : (x, y) = (col, row) centre of the Laue spot on the detector (pixels).
                      Identical for all frames — the spot position on the detector
                      does not change with scan position.
    boxsize         : half-size of the crop: total = (2*boxsize+1) × (2*boxsize+1) px.
    h5_img_key      : HDF5 dataset path for the image stack.
    scan_subset     : (i0, i1, j0, j1) — restrict to a sub-region of the scan grid.
                      i = x-column index, j = y-row index (lauexplore convention).
                      Processes the full scan when None.
    workers         : number of parallel worker processes.
    coords          : 'numpy' (0-based) or 'xmas' (1-based, subtracts 1 from roi_center).
    mask            : boolean array shape (n_frames,) — True = process, False = skip.
                      Indexed by the linear frame index (same as scan.ij_to_index).
                      Skipped positions appear in the DataFrame with status='masked'
                      and NaN metrics, keeping the grid complete for plotting.
                      Typically built as ``xeol.data > threshold``.
    axis_angle      : satellite axis angle (degrees from +x).  Auto-detected if None.
    n_max           : maximum satellite order to accept.
    min_prominence  : peak prominence threshold as fraction of profile max.
    strip_width     : strip width (px) for the 1-D profile projection.
    bg_sigma        : Gaussian sigma (px) for background subtraction.
    peak_min_width  : minimum peak width (bins) to reject noise spikes.  None = skip.
    hot_pixel_sigma : n_sigma for hot-pixel removal.  None = skip.
    n_range         : (n_min, n_max) to restrict accepted satellite orders.
    spacing_px      : override automatic satellite spacing estimate.
    sl0_boxsize     : half-width (px) of the box searched for SL₀ around its
                      predicted position (Laue routes only).  A handful of px —
                      unrelated to `boxsize` above.  Keep the same value here as
                      in `run_single_image`, or the map and the single-image
                      figure answer different questions.

    Returns
    -------
    pd.DataFrame — one row per scan position, columns:
        i, j            grid indices (lauexplore convention)
        frame_idx       linear frame index in the HDF5 dataset
        x_um, y_um      physical coordinates (same units as scan.ij_to_xy)
        status          'ok' or 'error: <message>'
        axis_angle      refined satellite axis angle (degrees)
        n_sat           count of detected satellite orders
        delta_q         px / order (signed, from linear fit)
        delta_q_std     scatter around uniform spacing (px)
        alpha           intensity decay exponent
        alpha_r2        R² of the decay fit
        fwhm_slope      FWHM vs |n| slope (px / order)
        fwhm_slope_r2   R² of the FWHM fit
        bulk_pos        detected order-0 peak along axis (px); the BULK in an MQW
        sl0_pos         predicted SL₀ (Laue routes only; NaN otherwise)
        bulk_sl0_offset_px   bulk → SL₀ separation = mean out-of-plane strain
        asymmetry_intensity_n{1,2,3}   (I⁺ − I⁻) / (I⁺ + I⁻) per order
        asymmetry_position_n{1,2,3}    position asymmetry in Δq units
        sl1_mean_intensity              mean amplitude of SL+1 and SL-1

    When the geometry parameters are complete, three more columns hold the
    mean over all consecutive satellite pairs::

        period_nm, period_angstrom, delta_q_inv_ang

    plus one triplet per adjacent pair (n1, n2), both orders non-zero::

        delta_px_n{n1}_n{n2}
        period_nm_n{n1}_n{n2}
        delta_q_inv_ang_n{n1}_n{n2}

    e.g. ``period_nm_n1_n2``, ``delta_px_n-3_n-2``.  Pairs straddling SL0 are
    absent by construction (SL0 is the main Bragg peak, excluded from the
    period estimate), so with orders ±1…±3 the pairs are (-3,-2), (-2,-1),
    (1,2) and (2,3).  Columns are missing (NaN after the DataFrame is built)
    at positions where the corresponding orders were not detected.

    Period route
    ------------
    ``period_method`` selects how Λ is obtained:

    'monochromatic' (default)
        Legacy path, unchanged.  Uses ``two_theta_0_deg`` and ``chi_deg``.
    'laue_forward'
        Polychromatic route for white/pink-beam Laue, fitted in pixel space.
        Requires ``hkl``, ``lattice`` and a calibration — either
        ``det_file='...det'`` (which also supplies ``UB``) or ``detector=`` plus
        ``UB=``.  γ, |G₀| and 2θ are derived; ``two_theta_0_deg`` and ``chi_deg``
        are ignored.  NOT validated against experiment — see
        NOTES_laue_vs_mono_period.md.
    'laue_analytic'
        Small-angle cross-check, additionally needs ``satellite_axis_psi_deg``.

    The Laue routes add ``period_method``, ``gamma_deg``,
    ``two_theta_from_pixel``, ``fit_rms_px``, ``parent_offset_px`` and
    ``predicted_axis_deg`` columns.  A position whose period cannot be computed
    keeps all its other metrics and records why in ``period_error``.
    """
    img_source = str(Path(img_source).resolve())

    # Fixed crop bounds — same Laue spot position on detector for all frames
    col_c, row_c = int(roi_center[0]), int(roi_center[1])   # (x, y) = (col, row)
    if coords == 'xmas':
        col_c -= 1
        row_c -= 1
    row0, row1 = row_c - boxsize, row_c + boxsize + 1
    col0, col1 = col_c - boxsize, col_c + boxsize + 1

    nx, ny = scan.nbxpoints, scan.nbypoints
    i0, i1, j0, j1 = scan_subset if scan_subset is not None else (0, nx, 0, ny)
    n_pos = (i1 - i0) * (j1 - j0)

    detect_kw = dict(
        axis_angle=axis_angle, n_max=n_max, min_prominence=min_prominence,
        strip_width=strip_width, bg_sigma=bg_sigma, peak_min_width=peak_min_width,
        hot_pixel_sigma=hot_pixel_sigma, n_range=n_range,
        spacing_px=spacing_px, adaptive_fill_win=adaptive_fill_win, verbose=False,
    )

    if det_file is not None and detector is None:
        from laue.satellite.geometry import DetectorGeometry
        detector, UB_from_file = DetectorGeometry.from_det_file(det_file)
        if UB is None:
            UB = UB_from_file
        print(f'[INFO] calibration from {det_file}: {detector}')

    if period_method == 'monochromatic':
        if pixel_size_mm is None and detector is not None:
            pixel_size_mm = detector.pixelsize
        if detector_distance_mm is None and detector is not None:
            detector_distance_mm = detector.dd
        geo_complete = (pixel_size_mm is not None
                        and detector_distance_mm is not None
                        and (wavelength_angstrom is not None or energy_kev is not None))
        period_kw = dict(
            method='monochromatic',
            pixel_size_mm=pixel_size_mm,
            detector_distance_mm=detector_distance_mm,
            wavelength_angstrom=wavelength_angstrom,
            energy_kev=energy_kev,
            two_theta_0_deg=two_theta_0_deg,
            chi_deg=chi_deg,
        ) if geo_complete else None
    else:
        if hkl is None or lattice is None:
            raise ValueError(
                f"period_method={period_method!r} requires hkl=(h, k, l) and "
                f"lattice=(a, c): γ and |G₀| are derived from indexing, never "
                f"defaulted.  Do not pass LaueTools' detector-space chi as γ."
            )
        if wavelength_angstrom is None and energy_kev is None:
            raise ValueError('Provide either wavelength_angstrom or energy_kev.')
        period_kw = dict(
            method=period_method,
            wavelength_angstrom=wavelength_angstrom,
            energy_kev=energy_kev,
            hkl=tuple(hkl), lattice=tuple(lattice),
            detector=detector, UB=UB,
            crop_origin_px=(row0, col0),
        )
        if period_method == 'laue_analytic':
            period_kw.update(pixel_size_mm=pixel_size_mm or (detector.pixelsize
                                                             if detector else None),
                             detector_distance_mm=detector_distance_mm or (detector.dd
                                                                          if detector else None),
                             satellite_axis_psi_deg=satellite_axis_psi_deg)
        print(f'[INFO] period route: {period_method}  hkl={tuple(hkl)}  '
              f'lattice={tuple(lattice)}')

    task_args = []
    rows: List[Dict[str, Any]] = []

    for i in range(i0, i1):
        for j in range(j0, j1):
            x_um, y_um = scan.ij_to_xy(i, j)
            x_um, y_um = float(x_um) * 1e3, float(y_um) * 1e3   # scan.ij_to_xy returns mm
            frame_idx = scan.ij_to_index(i, j)
            if mask is not None and not mask[frame_idx]:
                rows.append({
                    'i': i, 'j': j, 'frame_idx': frame_idx,
                    'x_um': x_um, 'y_um': y_um, 'status': 'masked',
                })
            else:
                task_args.append(
                    (i, j, frame_idx, x_um, y_um,
                     img_source, h5_img_key, row0, row1, col0, col1,
                     detect_kw, period_kw, sl0_boxsize)
                )

    n_active  = len(task_args)
    n_masked  = n_pos - n_active
    mask_info = f',  masked={n_masked}' if mask is not None else ''
    print(f'[INFO] {n_pos} positions  (grid {i1-i0} × {j1-j0}'
          f',  workers={workers}{mask_info})')

    pbar = _tqdm(total=n_active, desc='satellite analysis') if _HAS_TQDM else None

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, args): args for args in task_args}
        for fut in as_completed(futures):
            rows.append(fut.result())
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    df = pd.DataFrame(rows).sort_values(['i', 'j']).reset_index(drop=True)

    n_err = int((df['status'].str.startswith('error')).sum())
    if n_err:
        print(f'[WARN] {n_err}/{n_active} positions failed — '
              f'inspect with df[df.status.str.startswith("error")]')
    print(f'[INFO] Done.  DataFrame shape: {df.shape}')
    return df


# ── Shared grid utility ───────────────────────────────────────────────────────

def make_metric_grid(df: pd.DataFrame, col: str) -> np.ndarray:
    """Build a 2-D map array from a run_satellite_pipeline() DataFrame column.

    Returns an array of shape (nby, nbx) with ``origin='lower'`` orientation —
    the same layout used by all plotting functions in this module.  Use this to
    build segmentation masks from any metric (e.g. ``period_nm``).

    Parameters
    ----------
    df  : DataFrame from run_satellite_pipeline().
    col : column name to map (e.g. ``'period_nm'``, ``'alpha'``).

    Returns
    -------
    np.ndarray, shape (nby, nbx), dtype float64.  NaN where data is missing.
    """
    i_min = int(df['i'].min())
    j_min = int(df['j'].min())
    nbx   = int(df['i'].nunique())
    nby   = int(df['j'].nunique())
    grid  = np.full((nbx, nby), np.nan)
    for _, row in df.iterrows():
        v = row.get(col, np.nan)
        if pd.notna(v):
            grid[int(row['i'] - i_min), int(row['j'] - j_min)] = float(v)
    return grid.T   # (nby, nbx) with origin='lower'


# ── Plotting ──────────────────────────────────────────────────────────────────

# (col, title, cmap)
_DEFAULT_METRICS: List[Tuple[str, str, str]] = [
    ('n_sat',                  'N_sat (orders)',              'Blues'),
    ('delta_q',                'Δq (px/order)',               'RdBu_r'),
    ('alpha',                  'α decay',                     'plasma'),
    ('fwhm_slope',             'FWHM slope (px/order)',       'RdBu_r'),
    ('asymmetry_intensity_n1', 'Asym. intensity ±1',          'RdBu_r'),
    ('asymmetry_position_n1',  'Asym. position ±1 (Δq)',      'RdBu_r'),
    ('bulk_pos',               'Bulk peak position (px)',     'viridis'),
    ('sl0_pos',                'SL₀ position, predicted (px)', 'viridis'),
    ('bulk_sl0_offset_px',     'Bulk → SL₀ offset (px)',      'coolwarm'),
    ('sl1_mean_intensity',     'SL±1 mean intensity (counts)','inferno'),
    ('axis_angle',             'Axis angle θ (°)',            'RdBu_r'),
    ('period_nm',              'Layer period (nm)',            'viridis'),
]


def plot_satellite_maps(
    df: pd.DataFrame,
    metrics: Optional[List[Tuple[str, str, str]]] = None,
    ncols: int = 4,
    percentile_clip: Tuple[float, float] = (2, 98),
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot 2D metric maps from a run_pipeline() DataFrame.

    Parameters
    ----------
    df              : DataFrame returned by run_pipeline().
    metrics         : list of (column, title, cmap) tuples to plot.
                      Defaults to all 9 standard metrics present in df.
    ncols           : maximum panels per row (default 4).
    percentile_clip : (lo, hi) percentiles for robust colour scaling.
    figsize         : figure size in inches.  Auto-sized when None.

    Returns
    -------
    matplotlib Figure.
    """
    if metrics is None:
        metrics = [(col, title, cmap)
                   for col, title, cmap in _DEFAULT_METRICS
                   if col in df.columns]

    n     = len(metrics)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    x_um = np.sort(df['x_um'].unique())
    y_um = np.sort(df['y_um'].unique())
    dx   = float(x_um[-1] - x_um[0]) if len(x_um) > 1 else 1.0
    dy   = float(y_um[-1] - y_um[0]) if len(y_um) > 1 else 1.0
    aspect = dx / dy if dy > 0 else 1.0

    panel_h = 4.0
    panel_w = max(3.0, panel_h * aspect)
    if figsize is None:
        figsize = (ncols * panel_w + ncols * 0.5, nrows * (panel_h + 1.5))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    i_min = int(df['i'].min())
    j_min = int(df['j'].min())
    nbx   = int(df['i'].nunique())
    nby   = int(df['j'].nunique())
    extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]

    for idx, (col, title, cmap) in enumerate(metrics):
        ax = axes[idx // ncols, idx % ncols]

        grid = np.full((nbx, nby), np.nan)
        for _, row in df.iterrows():
            grid[int(row['i'] - i_min), int(row['j'] - j_min)] = row.get(col, np.nan)
        data = grid.T   # shape (nby, nbx) → rows = y, cols = x

        norm = None
        if col == 'axis_angle':
            mean = np.nanmean(data)
            absmax = max(
                abs(np.nanpercentile(data, percentile_clip[0]) - mean),
                abs(np.nanpercentile(data, percentile_clip[1]) - mean),
            ) or 1.0
            norm = TwoSlopeNorm(vmin=mean - absmax, vcenter=mean, vmax=mean + absmax)
            lo = hi = None
        else:
            lo = np.nanpercentile(data, percentile_clip[0])
            hi = np.nanpercentile(data, percentile_clip[1])

        im = ax.imshow(data, origin='lower', aspect='equal',
                       extent=extent, cmap=cmap, norm=norm, vmin=lo, vmax=hi,
                       interpolation='none')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if col == 'axis_angle':
            ticks = np.linspace(norm.vmin, norm.vmax, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{t:.2f}' for t in ticks])
        elif col == 'period_nm':
            ticks = np.linspace(lo, hi, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{t:.2f}' for t in ticks])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.tight_layout()
    return fig


def plot_detector_positions(
    df: pd.DataFrame,
    orders: Optional[List[int]] = None,
    percentile_clip: Tuple[float, float] = (2, 98),
    figsize: Optional[Tuple[float, float]] = None,
    pixel_size_mm: Optional[float] = None,
    segmentation: Optional[np.ndarray] = None,
    hist_bins: int = 25,
    hist_leds: Optional[List[int]] = None,
    hist_ncols: int = 4,
    hist_density: bool = True,
):
    """Plot per-satellite deviation from the scan-mean position on the detector.

    For each order n, two maps are shown:
        Δcol  — deviation along detector x  (column direction)
        Δrow  — deviation along detector y  (row direction)

    Both are computed as value − nanmean across all scan positions.

    Parameters
    ----------
    df              : DataFrame from run_satellite_pipeline() (must contain
                      pos_row_n{order} and pos_col_n{order} columns).
    orders          : satellite orders to plot.  Defaults to all orders for
                      which pos_row_n{order} exists in df.
    percentile_clip : (lo, hi) percentiles for colour scaling of maps.
    figsize         : figure size for the map figure.  Auto-sized when None.
    pixel_size_mm   : if given, deviations are shown in µm instead of px.
    segmentation    : 2-D integer array with shape (nby, nbx) — same grid and
                      orientation as the maps (origin='lower').  Non-zero
                      integers identify LED regions; zero/NaN = background.
                      When provided, a second figure is returned with per-LED
                      histograms (one panel per LED, shared x-axis).
    hist_bins       : number of histogram bins per LED panel.
    hist_leds       : LED IDs to include in histograms (default: all).
    hist_ncols      : number of LED columns per (order × direction) sub-grid
                      in the histogram figure (default: 4).
    hist_density    : if True (default), each LED panel shows probability
                      density (comparable across LEDs with different point
                      counts).  If False, shows raw counts.

    Returns
    -------
    fig_maps                         : Figure with deviation maps.
    (fig_maps, fig_hists, df_hist)   : when segmentation is provided.  df_hist is
                          a long-format DataFrame, one row per histogram bin, with
                          columns: order, direction, led_id, bin_lo, bin_hi,
                          bin_center, count, density, unit — i.e. the exact bin
                          data plotted in fig_hists, for further stats or export.
    """
    import math
    import matplotlib.gridspec as gridspec

    if orders is None:
        present = sorted(
            {int(c.split('_n')[1]) for c in df.columns if c.startswith('pos_row_n')},
            key=lambda n: (abs(n), n),
        )
        orders = [n for n in present if n != 0]

    if not orders:
        raise ValueError("No pos_row_n{order} columns found in df. "
                         "Re-run run_satellite_pipeline — position columns are added automatically.")

    scale    = (pixel_size_mm * 1e3) if pixel_size_mm is not None else 1.0
    unit_lbl = 'µm' if pixel_size_mm is not None else 'px'

    n_ord = len(orders)

    x_um = np.sort(df['x_um'].unique())
    y_um = np.sort(df['y_um'].unique())
    dx   = float(x_um[-1] - x_um[0]) if len(x_um) > 1 else 1.0
    dy   = float(y_um[-1] - y_um[0]) if len(y_um) > 1 else 1.0
    aspect = dx / dy if dy > 0 else 1.0

    i_min = int(df['i'].min())
    j_min = int(df['j'].min())
    nbx   = int(df['i'].nunique())
    nby   = int(df['j'].nunique())
    extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]

    # ── Segmentation setup ────────────────────────────────────────────────────
    led_ids: List[int]  = []
    led_colors: dict    = {}
    hist_led_ids: List[int] = []
    df_led = df

    if segmentation is not None:
        seg = np.asarray(segmentation, dtype=float)
        unique_vals = np.unique(seg[~np.isnan(seg)])
        led_ids = sorted([int(v) for v in unique_vals if v > 0])
        cmap_leds = plt.cm.get_cmap('tab20', max(len(led_ids), 1))
        led_colors = {lid: cmap_leds(k) for k, lid in enumerate(led_ids)}
        hist_led_ids = [lid for lid in led_ids if lid in hist_leds] \
                       if hist_leds is not None else led_ids

        def _lookup_led(row) -> int:
            ji = int(row['j'] - j_min)
            ii = int(row['i'] - i_min)
            if 0 <= ji < seg.shape[0] and 0 <= ii < seg.shape[1]:
                v = seg[ji, ii]
                if not np.isnan(v) and int(v) > 0:
                    return int(v)
            return 0

        df_led = df.copy()
        df_led['_led_id'] = df_led.apply(_lookup_led, axis=1)

    # ── Grid builder ──────────────────────────────────────────────────────────
    def _make_grid(col: str) -> np.ndarray:
        grid = np.full((nbx, nby), np.nan)
        for _, row in df.iterrows():
            v = row.get(col, np.nan)
            if pd.notna(v):
                grid[int(row['i'] - i_min), int(row['j'] - j_min)] = float(v)
        return grid.T   # (nby, nbx) with origin='lower'

    # Pre-compute delta grids (shared between maps and histograms)
    delta_grids: dict = {}
    for n in orders:
        lbl = f'n{n}'
        for col_name, direction in [(f'pos_col_{lbl}', 'x'),
                                    (f'pos_row_{lbl}', 'y')]:
            if col_name in df.columns:
                raw = _make_grid(col_name)
                delta_grids[(n, direction)] = (raw - np.nanmean(raw)) * scale

    # ── Map figure ────────────────────────────────────────────────────────────
    panel_h = 4.0
    panel_w = max(3.0, panel_h * aspect)
    if figsize is None:
        figsize = (2 * panel_w + 1.5, n_ord * (panel_h + 1.2))

    fig_maps, axes = plt.subplots(n_ord, 2, figsize=figsize, squeeze=False)

    def _show_map(ax, data: np.ndarray, title: str) -> None:
        vals = data[~np.isnan(data)]
        if len(vals) == 0:
            ax.set_visible(False)
            return
        absmax = max(
            abs(float(np.nanpercentile(vals, percentile_clip[0]))),
            abs(float(np.nanpercentile(vals, percentile_clip[1]))),
        ) or 1.0
        im = ax.imshow(data, origin='lower', aspect='equal',
                       extent=extent, cmap='RdBu_r',
                       vmin=-absmax, vmax=absmax,
                       interpolation='none')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=f'Δ ({unit_lbl})')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')

    for row_idx, n in enumerate(orders):
        order_sign = '+' if n > 0 else ''
        label_str  = f'SL{order_sign}{n}'
        for ax_col, direction, axis_name in [
            (0, 'x', 'Δx (col)'),
            (1, 'y', 'Δy (row)'),
        ]:
            ax  = axes[row_idx, ax_col]
            key = (n, direction)
            if key not in delta_grids:
                ax.set_visible(False)
                continue
            _show_map(ax, delta_grids[key], f'{label_str}  {axis_name}')

    fig_maps.suptitle(
        f'Satellite detector-position deviation from scan mean  [{unit_lbl}]',
        fontsize=12,
    )
    fig_maps.tight_layout()

    if segmentation is None or not hist_led_ids:
        return fig_maps

    # ── Histogram figure — one panel per LED, shared xlim ────────────────────
    def _point_delta(row, key) -> float:
        if key not in delta_grids:
            return float('nan')
        ji = int(row['j'] - j_min)
        ii = int(row['i'] - i_min)
        g  = delta_grids[key]
        if 0 <= ji < g.shape[0] and 0 <= ii < g.shape[1]:
            return float(g[ji, ii])
        return float('nan')

    n_leds_plot  = len(hist_led_ids)
    h_ncols      = min(hist_ncols, n_leds_plot)
    h_nrows_sub  = math.ceil(n_leds_plot / h_ncols)

    directions = [(0, 'x', 'Δx (col)'), (1, 'y', 'Δy (row)')]
    outer_cols  = len(directions)   # 2
    outer_rows  = n_ord

    panel_sub_h = 2.5
    panel_sub_w = 2.8
    hist_fig_h  = outer_rows * h_nrows_sub * (panel_sub_h + 0.4) + outer_rows * 0.6
    hist_fig_w  = outer_cols * h_ncols * (panel_sub_w + 0.3) + 1.0
    fig_hists   = plt.figure(figsize=(hist_fig_w, hist_fig_h))

    outer_gs = gridspec.GridSpec(
        outer_rows, outer_cols,
        figure=fig_hists,
        hspace=0.7, wspace=0.4,
    )

    hist_rows: List[Dict[str, Any]] = []

    for row_idx, n in enumerate(orders):
        order_sign = '+' if n > 0 else ''
        label_str  = f'SL{order_sign}{n}'

        for ax_col, direction, axis_name in directions:
            key = (n, direction)

            # Collect data for all LEDs
            all_vals: List[float] = []
            led_data: dict = {}
            for lid in hist_led_ids:
                subset = df_led[df_led['_led_id'] == lid]
                vals   = [_point_delta(r, key) for _, r in subset.iterrows()]
                vals   = [v for v in vals if not np.isnan(v)]
                led_data[lid] = vals
                all_vals.extend(vals)

            if not all_vals:
                continue

            lo_v = float(np.nanpercentile(all_vals, 1))
            hi_v = float(np.nanpercentile(all_vals, 99))
            bins = np.linspace(lo_v, hi_v, hist_bins + 1)

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                h_nrows_sub, h_ncols,
                subplot_spec=outer_gs[row_idx, ax_col],
                hspace=0.55, wspace=0.35,
            )

            # Group title via invisible spanning axes
            title_ax = fig_hists.add_subplot(outer_gs[row_idx, ax_col])
            title_ax.set_title(f'{label_str}  {axis_name}', fontsize=11,
                               fontweight='bold', pad=14)
            title_ax.axis('off')

            for k, lid in enumerate(hist_led_ids):
                r = k // h_ncols
                c = k % h_ncols
                ax = fig_hists.add_subplot(inner_gs[r, c])

                vals  = led_data[lid]
                color = led_colors[lid]
                if vals:
                    ax.hist(vals, bins=bins,
                            histtype='stepfilled', alpha=0.55,
                            color=color, edgecolor=color, linewidth=0.8,
                            density=hist_density)
                    ax.axvline(float(np.mean(vals)), color=color,
                               linewidth=1.4, linestyle='--')

                    counts, edges = np.histogram(vals, bins=bins)
                    density, _    = np.histogram(vals, bins=bins, density=True)
                    for lo_e, hi_e, cnt, dens in zip(edges[:-1], edges[1:], counts, density):
                        hist_rows.append({
                            'order':      n,
                            'direction':  direction,
                            'led_id':     lid,
                            'bin_lo':     float(lo_e),
                            'bin_hi':     float(hi_e),
                            'bin_center': float((lo_e + hi_e) / 2.0),
                            'count':      int(cnt),
                            'density':    float(dens),
                            'unit':       unit_lbl,
                        })

                ax.axvline(0, color='k', linewidth=0.8, linestyle=':')
                ax.set_xlim(lo_v, hi_v)
                ax.set_title(f'LED {lid}', fontsize=8, pad=3)
                ax.tick_params(labelsize=7)
                if c == 0:
                    ax.set_ylabel('Density' if hist_density else 'Count', fontsize=7)
                if r == h_nrows_sub - 1:
                    ax.set_xlabel(f'Δ ({unit_lbl})', fontsize=7)

    fig_hists.suptitle(
        f'Satellite position per LED  [{unit_lbl}]',
        fontsize=12,
    )
    df_hist = pd.DataFrame(hist_rows)
    return fig_maps, fig_hists, df_hist


def plot_satellite_stats(
    df: pd.DataFrame,
    orders: Union[int, List[int]],
    percentile_clip: Tuple[float, float] = (2, 98),
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Per-pixel scan maps of amplitude and detector-position deviation, per satellite order.

    For each requested order n, one row of 3 maps over the scan grid (x_um, y_um) —
    same grid/extent as plot_satellite_maps / plot_detector_positions, so panels line up
    pixel-for-pixel with those maps and with any LED segmentation built from them:
        amp_n{n}  — raw amplitude at every scan pixel, so LEDs that emit more or
                    less can be compared directly on the sample map.
        Δcol      — detector-position deviation from the scan mean, x direction.
        Δrow      — detector-position deviation from the scan mean, y direction.

    The amplitude panel is annotated with the distribution's moments (mean, std,
    skewness, excess kurtosis) computed over all valid scan pixels for that order.

    Parameters
    ----------
    df              : DataFrame from run_satellite_pipeline(). Must contain amp_n{n},
                      pos_row_n{n}, pos_col_n{n} for every requested order —
                      these only exist for orders that survived n_max / n_range
                      in the run that produced df.
    orders          : single satellite order or list of orders (e.g. -1 or [1, 2, -1]).
    percentile_clip : (lo, hi) percentiles for robust colour scaling.
    figsize         : figure size in inches. Auto-sized when None.

    Returns
    -------
    matplotlib Figure, one row per order, 3 columns (amplitude, Δcol, Δrow).
    """
    if isinstance(orders, int):
        orders = [orders]

    missing = [n for n in orders if f'amp_n{n}' not in df.columns]
    if missing:
        raise ValueError(
            f"Order(s) {missing} not found in df (amp_n{{order}} column missing). "
            f"Check the n_max / n_range used in run_satellite_pipeline for this df."
        )

    x_um = np.sort(df['x_um'].unique())
    y_um = np.sort(df['y_um'].unique())
    dx = float(x_um[-1] - x_um[0]) if len(x_um) > 1 else 1.0
    dy = float(y_um[-1] - y_um[0]) if len(y_um) > 1 else 1.0
    aspect = dx / dy if dy > 0 else 1.0
    extent = [float(x_um.min()), float(x_um.max()), float(y_um.min()), float(y_um.max())]

    n_ord   = len(orders)
    panel_h = 4.0
    panel_w = max(3.0, panel_h * aspect)
    if figsize is None:
        figsize = (3 * panel_w + 3 * 0.5, n_ord * (panel_h + 1.2))
    fig, axes = plt.subplots(n_ord, 3, figsize=figsize, squeeze=False)

    def _show(ax, data, title, cmap, vmin, vmax, cbar_label):
        im = ax.imshow(data, origin='lower', aspect='equal', extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')

    for row_idx, n in enumerate(orders):
        order_sign = '+' if n > 0 else ''
        label_str  = f'SL{order_sign}{n}'

        amp_grid  = make_metric_grid(df, f'amp_n{n}')
        col_grid  = make_metric_grid(df, f'pos_col_n{n}')
        row_grid  = make_metric_grid(df, f'pos_row_n{n}')
        dcol_grid = col_grid - np.nanmean(col_grid)
        drow_grid = row_grid - np.nanmean(row_grid)

        # ── Amplitude map + moments ────────────────────────────────────────────
        ax_amp = axes[row_idx, 0]
        amp_valid = amp_grid[np.isfinite(amp_grid)]
        if amp_valid.size == 0:
            ax_amp.set_visible(False)
        else:
            lo = float(np.nanpercentile(amp_grid, percentile_clip[0]))
            hi = float(np.nanpercentile(amp_grid, percentile_clip[1]))
            _show(ax_amp, amp_grid, f'{label_str}  amplitude', 'inferno', lo, hi,
                 'Amplitude (counts)')
            stats_txt = (f'n = {amp_valid.size}\n'
                        f'mean = {np.mean(amp_valid):.1f}\n'
                        f'std = {np.std(amp_valid):.1f}\n'
                        f'skew = {float(skew(amp_valid)):.2f}\n'
                        f'kurt = {float(kurtosis(amp_valid)):.2f}')
            ax_amp.text(0.02, 0.02, stats_txt, transform=ax_amp.transAxes,
                       ha='left', va='bottom', fontsize=7, color='white',
                       bbox=dict(boxstyle='round', fc='black', alpha=0.5))

        # ── Detector-position deviation maps ──────────────────────────────────
        for ax, grid, name in [(axes[row_idx, 1], dcol_grid, 'Δcol'),
                               (axes[row_idx, 2], drow_grid, 'Δrow')]:
            vals = grid[np.isfinite(grid)]
            if vals.size == 0:
                ax.set_visible(False)
                continue
            absmax = max(
                abs(float(np.nanpercentile(vals, percentile_clip[0]))),
                abs(float(np.nanpercentile(vals, percentile_clip[1]))),
            ) or 1.0
            _show(ax, grid, f'{label_str}  {name}', 'RdBu_r', -absmax, absmax,
                 f'{name} (px)')

    fig.suptitle('Satellite amplitude and detector-position maps', fontsize=12)
    fig.tight_layout()
    return fig
