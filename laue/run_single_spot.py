"""Jupyter-friendly entry point for spot morphology analysis on a single image.

Usage
-----
    from laue.run_single_spot import run_single_spot

    result = run_single_spot(
        img_source  = Path('/data/eiger4m_0000.h5'),
        h5_img_key  = 'entry_0000/CRGIF/eiger4m/data',
        frame_index = 0,
        roi_center  = (534, 993),
        boxsize     = 25,
        coords      = 'xmas',
    )
    result['metrics']   # dict with all computed indicators
    result['crop']      # 2-D numpy array that was analysed
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from scipy.ndimage import maximum_filter

from laue.spot_metrics import preprocess, analyze_spot


# ── Image loading ─────────────────────────────────────────────────────────────

from laue._imaging import extract_crop as _extract_crop, load_frame as _load_frame


# ── Local maxima positions ────────────────────────────────────────────────────

def _find_maxima_positions(
    img: np.ndarray,
    min_distance: int = 3,
    threshold_rel: float = 0.1,
    min_separation: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) of local maxima after optional Euclidean NMS."""
    size = 2 * min_distance + 1
    local_max = (img == maximum_filter(img, size=size)) & \
                (img >= threshold_rel * img.max())
    rows, cols = np.where(local_max)
    if len(rows) == 0 or min_separation is None or min_separation <= 0:
        return rows, cols
    # Greedy Euclidean NMS: keep strongest; suppress neighbours within min_separation px
    intensities = img[rows, cols]
    order = np.argsort(intensities)[::-1]
    kept_r: list[int] = []
    kept_c: list[int] = []
    for idx in order:
        r, c = int(rows[idx]), int(cols[idx])
        if all(np.hypot(r - kr, c - kc) >= min_separation
               for kr, kc in zip(kept_r, kept_c)):
            kept_r.append(r)
            kept_c.append(c)
    return np.array(kept_r, dtype=int), np.array(kept_c, dtype=int)


# ── Profiles ─────────────────────────────────────────────────────────────────

def _streak_profile(
    img: np.ndarray,
    x_com: float,
    y_com: float,
    theta_deg: float,
    n_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean intensity vs absolute distance along the streak axis."""
    theta_rad = np.radians(theta_deg)
    sx, sy = np.cos(theta_rad), np.sin(theta_rad)
    gy, gx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    proj = np.abs((gx - x_com) * sx + (gy - y_com) * sy).ravel()
    v = img.ravel()
    edges = np.linspace(0, proj.max(), n_bins + 1)
    counts = np.zeros(n_bins)
    totals = np.zeros(n_bins)
    for i, val in zip(np.searchsorted(edges[1:], proj), v):
        if i < n_bins:
            totals[i] += val
            counts[i] += 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, np.where(counts > 0, totals / counts, 0.0)


def _radial_profile(
    img: np.ndarray,
    x_com: float,
    y_com: float,
    n_bins: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean intensity vs distance from COM, binned radially."""
    gy, gx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    r = np.sqrt((gx - x_com) ** 2 + (gy - y_com) ** 2).ravel()
    v = img.ravel()
    r_max = r.max()
    edges = np.linspace(0, r_max, n_bins + 1)
    counts = np.zeros(n_bins)
    totals = np.zeros(n_bins)
    idx = np.searchsorted(edges[1:], r)
    for i, val in zip(idx, v):
        if i < n_bins:
            totals[i] += val
            counts[i] += 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    mean_intensity = np.where(counts > 0, totals / counts, 0.0)
    return centres, mean_intensity


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot(
    raw: np.ndarray,
    proc: np.ndarray,
    metrics: dict,
    r_core: float,
    lm_min_distance: int,
    lm_threshold_rel: float,
    lm_min_separation: Optional[float],
    title: str,
    figsize: tuple,
) -> plt.Figure:
    x_com  = metrics['x_com']
    y_com  = metrics['y_com']
    eff_r  = metrics['effective_radius']
    ctr    = metrics['core_tail_ratio']
    nlm    = metrics['n_local_maxima']
    theta  = metrics['theta']
    d50    = metrics['streak_D50']
    d95    = metrics['streak_D95']

    has_streak = not any(np.isnan([theta, d50, d95]))

    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=11, fontweight='bold')

    # ── Panel 1: raw image ────────────────────────────────────────────────────
    ax = axes[0]
    pos = raw[raw > 0]
    vmin = float(pos.min()) if len(pos) else 1.0
    ax.imshow(raw, norm=LogNorm(vmin=max(vmin, 1.0), vmax=float(raw.max())),
              cmap='inferno', origin='lower')
    ax.set_title('Raw crop (log scale)', fontsize=10)
    ax.set_xlabel('col (px)')
    ax.set_ylabel('row (px)')

    # ── Panel 2: preprocessed + metric overlays ───────────────────────────────
    ax = axes[1]
    pos_p = proc[proc > 0]
    vmin_p = float(pos_p.min()) if len(pos_p) else 1.0
    ax.imshow(proc, norm=LogNorm(vmin=max(vmin_p, 1.0), vmax=float(proc.max())),
              cmap='inferno', origin='lower')
    ax.set_title('Preprocessed + metric overlays', fontsize=10)
    ax.set_xlabel('col (px)')

    if not np.isnan(x_com):
        # COM marker
        ax.plot(x_com, y_com, '+', color='white', ms=10, mew=1.5,
                label='COM', zorder=5)

        # Core circle (core_tail_ratio)
        ax.add_patch(mpatches.Circle(
            (x_com, y_com), r_core,
            fill=False, edgecolor='#2ca02c', linewidth=1.5,
            linestyle='-', label=f'core r={r_core:.0f}px  (CTR={ctr:.2f})',
            zorder=4,
        ))

        # Effective radius circle (dispersion)
        if not np.isnan(eff_r):
            ax.add_patch(mpatches.Circle(
                (x_com, y_com), eff_r,
                fill=False, edgecolor='#ff7f0e', linewidth=1.5,
                linestyle='--', label=f'eff. radius={eff_r:.1f}px',
                zorder=4,
            ))

        # Streak axis line
        if has_streak:
            rad = np.radians(theta)
            half = max(raw.shape) * 0.7
            ax.plot(
                [x_com - half * np.cos(rad), x_com + half * np.cos(rad)],
                [y_com - half * np.sin(rad), y_com + half * np.sin(rad)],
                '--', color='#17becf', lw=1.2, alpha=0.85,
                label=f'streak axis θ={theta:.1f}°', zorder=3,
            )

    # Local maxima
    lm_rows, lm_cols = _find_maxima_positions(proc, lm_min_distance, lm_threshold_rel, lm_min_separation)
    ax.scatter(lm_cols, lm_rows, marker='x', s=60, color='#d62728',
               linewidths=1.5, label=f'local maxima ({nlm})', zorder=6)

    ax.legend(loc='upper right', fontsize=7, framealpha=0.6)

    # ── Panel 3: radial intensity profile ────────────────────────────────────
    ax = axes[2]
    if not np.isnan(x_com):
        r_bins, i_mean = _radial_profile(proc, x_com, y_com)
        ax.plot(r_bins, i_mean, color='#333', lw=1.5)
        ax.axvline(r_core, color='#2ca02c', lw=1.5, linestyle='-',
                   label=f'core r={r_core:.0f}px')
        if not np.isnan(eff_r):
            ax.axvline(eff_r, color='#ff7f0e', lw=1.5, linestyle='--',
                       label=f'eff. radius={eff_r:.1f}px')
        ax.set_xlabel('Distance from COM (px)')
        ax.set_ylabel('Mean intensity (counts)')
        ax.set_title('Radial profile', fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    # ── Panel 4: streak axis profile with D50 / D95 ───────────────────────────
    ax = axes[3]
    if has_streak:
        s_bins, s_mean = _streak_profile(proc, x_com, y_com, theta)
        ax.plot(s_bins, s_mean, color='#333', lw=1.5)
        ax.axvline(d50, color='#9467bd', lw=1.5, linestyle='-',
                   label=f'D50={d50:.1f}px')
        ax.axvline(d95, color='#e377c2', lw=1.5, linestyle='--',
                   label=f'D95={d95:.1f}px')
        ax.set_xlabel('|Distance| from COM along axis (px)')
        ax.set_ylabel('Mean intensity (counts)')
        ax.set_title(f'Streak axis profile  (θ={theta:.1f}°)', fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

    return fig


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(metrics: dict, r_core: float) -> None:
    sep = '=' * 58
    print(sep)
    print('  SPOT MORPHOLOGY — SUMMARY')
    print(sep)
    print(f"  COM position     : ({metrics['x_com']:.1f},  {metrics['y_com']:.1f})  px")
    print(f"  Streak angle     : {metrics['theta']:.1f}°")
    print(f"  Effective radius : {metrics['effective_radius']:.2f}  px  "
          f"[sqrt(λ₁+λ₂), total dispersion]")
    print(f"  Streak D50       : {metrics['streak_D50']:.2f}  px  [core half-width along axis]")
    print(f"  Streak D95       : {metrics['streak_D95']:.2f}  px  [tail extent along axis]")
    print(f"  D95/D50          : {metrics['d95_d50_ratio']:.2f}  [tail dominance along axis]")
    print(f"  Core-to-tail     : {metrics['core_tail_ratio']:.3f}  "
          f"[intensity within circular r={r_core:.0f}px]")
    print(f"  Local maxima     : {metrics['n_local_maxima']}  "
          f"[1 = simple spot,  >1 = split / satellite]")
    print(sep)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_single_spot(
    img_source=None,
    h5_img_key: str = 'frames',
    frame_index: int = 0,
    roi_center: Optional[Tuple[int, int]] = None,
    boxsize: int = 25,
    coords: str = 'numpy',
    # preprocessing params
    r_core: float = 3.0,
    smooth_sigma: float = 0.0,
    bg_method: str = 'corners',
    bg_percentile: float = 5.0,
    corner_size: int = 5,
    noise_nsigma: float = 0.0,
    min_counts: float = 10.0,
    lm_min_distance: int = 3,
    lm_threshold_rel: float = 0.1,
    lm_min_separation: Optional[float] = None,
    # display
    figsize: tuple = (18, 5),
    show_plot: bool = True,
) -> dict:
    """Run spot morphology analysis on a single image and display results.

    Parameters
    ----------
    img_source      : path to HDF5 or .npy file.  None = synthetic test spot.
    h5_img_key      : HDF5 dataset key.
    frame_index     : frame to use when the dataset is a 3-D stack.
    roi_center      : (x, y) = (col, row) of the spot centre on the detector.
                      None = use the full frame.
    boxsize         : half-size of the crop: total = (2*boxsize+1) × (2*boxsize+1) px.
    coords          : 'numpy' (0-based) or 'xmas' (1-based).
    r_core          : radius of the core region for core_tail_ratio (px).
    smooth_sigma    : Gaussian smoothing sigma before analysis (0 = disabled).
    bg_method       : 'corners' or 'percentile' — background estimation method.
    bg_percentile   : percentile for bg_method='percentile'.
    corner_size     : corner patch size for bg_method='corners'.
    noise_nsigma    : threshold in units of bg std below which pixels are zeroed.
    min_counts      : minimum total counts; returns NaN metrics if below.
    lm_min_distance   : minimum pixel separation between local maxima (L∞ filter).
    lm_threshold_rel  : local maxima below this fraction of the peak are ignored.
    lm_min_separation : if set, discard any maximum within this Euclidean distance
                        (px) of a stronger one.  None = disabled.
    figsize         : figure size in inches.
    show_plot       : display the figure inline.

    Returns
    -------
    dict with keys:
        'metrics'  : dict — all indicators from analyze_spot()
        'crop'     : 2-D numpy array (raw crop that was analysed)
        'proc'     : 2-D numpy array (preprocessed crop)
        'fig'      : matplotlib Figure (None if show_plot=False)
    """
    # ── Load image ────────────────────────────────────────────────────────────
    if img_source is None:
        print('[INFO] img_source=None — generating synthetic test spot.')
        rng = np.random.default_rng(0)
        cx, cy = 30, 30
        gy, gx = np.mgrid[0:61, 0:61]
        crop = (5000 * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * 4 ** 2))
                + rng.poisson(30, (61, 61))).astype(np.float32)
        crop_origin = (0, 0)
        title = 'Spot morphology — synthetic test'
    else:
        print(f'[INFO] Loading frame {frame_index} from {Path(img_source).name}')
        frame = _load_frame(img_source, h5_img_key, frame_index)
        print(f'[INFO] Frame shape: {frame.shape}  '
              f'range: [{frame.min():.0f}, {frame.max():.0f}]')
        if roi_center is not None:
            crop, _ = _extract_crop(frame, roi_center, boxsize, coords)
            print(f'[INFO] Crop: center={roi_center} ({coords}), '
                  f'boxsize={boxsize} → shape={crop.shape}')
        else:
            crop, _ = frame, (0, 0)
            print('[INFO] roi_center=None — using full frame.')
        title = (f'Spot morphology  |  frame {frame_index}  |  '
                 f'roi={roi_center}  boxsize={boxsize}')

    # ── Preprocess & analyse ──────────────────────────────────────────────────
    proc = preprocess(
        crop,
        bg_method=bg_method,
        bg_percentile=bg_percentile,
        corner_size=corner_size,
        smooth_sigma=smooth_sigma,
        noise_nsigma=noise_nsigma,
    )

    metrics = analyze_spot(
        crop,
        r_core=r_core,
        smooth_sigma=smooth_sigma,
        bg_method=bg_method,
        bg_percentile=bg_percentile,
        corner_size=corner_size,
        noise_nsigma=noise_nsigma,
        min_counts=min_counts,
        lm_min_distance=lm_min_distance,
        lm_threshold_rel=lm_threshold_rel,
        lm_min_separation=lm_min_separation,
    )

    _print_summary(metrics, r_core)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = None
    if show_plot:
        fig = _plot(
            raw=crop.astype(np.float64),
            proc=proc,
            metrics=metrics,
            r_core=r_core,
            lm_min_distance=lm_min_distance,
            lm_threshold_rel=lm_threshold_rel,
            lm_min_separation=lm_min_separation,
            title=title,
            figsize=figsize,
        )
        plt.show()

    return {'metrics': metrics, 'crop': crop, 'proc': proc, 'fig': fig}
