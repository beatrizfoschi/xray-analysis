"""Jupyter-friendly entry point for satellite peak analysis on a single image.

Drop into a notebook cell:

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path('caminho/para/xray-analysis')))
    from laue.satellite.run_single_image import run_single_image

    result = run_single_image(
        img_source  = Path('/data/eiger4m_0000.h5'),
        h5_img_key  = 'entry_0000/CRGIF/eiger4m/data',
        frame_index = 0,
        roi_center  = (1913, 1263),   # (x, y) = (col, row), pixels 0-based
        boxsize     = 60,             # half-size: crop = 121 x 121 px
    )

    result['peaks_df']    # DataFrame com as ordens detectadas
    result['metrics']     # dict com N_sat, delta_q, alpha, ...
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Allow importing sibling modules even when called from another directory
from laue.satellite.detection import (
    detect_satellites, make_synthetic_image, locate_sl0_by_local_max,
    extract_1d_profile,
)
from laue.satellite.metrics import compute_metrics
from laue.satellite.period import (
    layer_period_from_peaks, locate_sl0_from_ladder, resolve_and_apply_order_sign,
)


from laue._imaging import extract_crop as _extract_crop, load_frame as _load_frame


# ── Visualisation ─────────────────────────────────────────────────────────────

from laue.satellite._orders import order_color as _order_color, SL0_COLOR as _SL0_COLOR


def _order0_is_bulk(sl0: dict | None) -> bool:
    """Is the detected order-0 peak the bulk reflection rather than SL0?

    Shared by `_plot` and `_print_summary` so the figure and the printout can
    never disagree about the same peak.
    """
    return bool(sl0 and sl0.get('order0_is_sl0') is False)


def _order_label(order: int, bulk_is_not_sl0: bool) -> str:
    if order == 0:
        return 'BULK' if bulk_is_not_sl0 else 'SL+0'
    return f'SL{order:+d}'


def _sl0_position(sl0: dict | None) -> Optional[float]:
    """Position of SL0 along the axis, or None when it was not located."""
    if sl0 and sl0.get('sl0_confirmed'):
        return float(sl0['sl0_measured_pos'])
    return None


def _plot(
    image: np.ndarray,
    result: dict,
    figsize: tuple,
    title: str,
    profile_log: bool = False,
    show_linear: bool = True,
    sl0: dict | None = None,
    strip_width: float = 5.0,
    profile_source: str = 'subtracted',
) -> plt.Figure:
    peaks = result['peaks']
    axis_angle = result['axis_angle']
    sl0_center = result['sl0_center']
    image_sub = result['image_sub']

    if profile_source not in ('subtracted', 'raw'):
        raise ValueError(
            f"profile_source={profile_source!r}: use 'subtracted' (what the "
            f"detection saw) or 'raw' (measured counts)."
        )

    # The order-0 peak is the bulk Bragg reflection whenever the ladder test
    # places it off the superlattice ladder (README.md, "The bright peak in the ROI is the bulk"), so the
    # label follows that verdict rather than the order index.
    bulk_is_not_sl0 = _order0_is_bulk(sl0)
    _label = lambda order: _order_label(order, bulk_is_not_sl0)
    sl0_pos = _sl0_position(sl0)

    # Default to the background-subtracted profile, because that is the one the
    # detection actually worked on: the Gaussian amplitudes and sigmas drawn over
    # it were fitted there, so on the raw profile they have to be propped up on a
    # local baseline to line up at all.  On a bright parent the raw profile also
    # buries the outer satellites in its tail, which is exactly when the figure is
    # needed.  'raw' stays available for reading measured counts off the curve.
    profile_image = image if profile_source == 'raw' else image_sub
    distances, intensities = extract_1d_profile(
        profile_image, sl0_center, axis_angle, strip_width=strip_width)

    # Layout: 2 image panels + 1 or 2 profile panels
    if profile_log and show_linear:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
        ax_raw  = fig.add_subplot(gs[:, 0])
        ax_sub  = fig.add_subplot(gs[:, 1])
        ax_lin  = fig.add_subplot(gs[0, 2])
        ax_log  = fig.add_subplot(gs[1, 2], sharex=ax_lin)
        profile_axes = [(ax_lin, False), (ax_log, True)]
    else:
        fig, (ax_raw, ax_sub, ax_prof) = plt.subplots(1, 3, figsize=figsize,
                                                       constrained_layout=True)
        profile_axes = [(ax_prof, profile_log)]

    # Same crop, same pixel coordinates on both image panels: link them so
    # zooming/panning one (e.g. with %matplotlib widget) follows on the other —
    # the two panels are compared directly, not the profile axes alongside them.
    ax_sub.sharex(ax_raw)
    ax_sub.sharey(ax_raw)

    fig.suptitle(title, fontsize=11, fontweight='bold')

    def _log_imshow(ax, img, title_str, max_percent):
        pos = img[img > 0]
        vmin = float(pos.min()) if len(pos) else 1.0
        vmax = np.percentile(img,max_percent) if img.max() > 0 else 1.0
        ax.imshow(img, norm=LogNorm(vmin=max(vmin, 1e-3), vmax=max(vmax, 1.0)),
                  cmap='inferno', origin='lower')
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel('col (px)')
        ax.set_ylabel('row (px)')

    _log_imshow(ax_raw, image, 'Raw crop (log)', 99.9)
    _log_imshow(ax_sub, image_sub, 'Background-subtracted + peaks', 100)

    # Satellite axis line
    rad = np.radians(axis_angle)
    half = float(min(image.shape)) * 0.5
    cx_px, cy_px = sl0_center[1], sl0_center[0]
    ax_sub.plot(
        [cx_px - half * np.cos(rad), cx_px + half * np.cos(rad)],
        [cy_px - half * np.sin(rad), cy_px + half * np.sin(rad)],
        '--', color='white', alpha=0.4, lw=1,
    )

    # Label offsets go perpendicular to the train axis: SL0 sits only about
    # half a satellite step from the bulk peak, so an along-axis offset would
    # walk the two labels into each other.
    perp = np.array([np.cos(rad), -np.sin(rad)])   # (d_row, d_col), unit length
    label_off = 9.0

    def _mark(row, col_px, colour, text, marker='o'):
        ax_sub.plot(col_px, row, marker, color=colour, ms=11, mew=2, fillstyle='none')
        ax_sub.text(col_px + label_off * perp[1], row + label_off * perp[0],
                    text, color=colour, fontsize=8,
                    fontweight='bold', ha='left', va='center')

    for pk in peaks:
        r, c = pk['position_2d']
        _mark(r, c, _order_color(pk['order']), _label(pk['order']),
              marker='s' if (pk['order'] == 0 and bulk_is_not_sl0) else 'o')

    if sl0_pos is not None:
        _mark(sl0_center[0] + sl0_pos * np.sin(rad),
              sl0_center[1] + sl0_pos * np.cos(rad),
              _SL0_COLOR, 'SL0')

    # Profile floor: 1 count so log scale works
    prof_floor = max(float(intensities[intensities > 0].min()) * 0.5
                     if (intensities > 0).any() else 1.0, 1.0)

    for ax_prof, use_log in profile_axes:
        y = intensities if not use_log else np.maximum(intensities, prof_floor)
        ax_prof.plot(distances, y, color='#333', lw=1.5, zorder=2)

        ref_level = float(intensities.max())
        for pk in peaks:
            col = _order_color(pk['order'])
            s, sigma = pk['pos_along_axis'], pk['sigma']
            x_fit = np.linspace(s - 4 * sigma, s + 4 * sigma, 200)
            # The fit was made on the subtracted profile, so it sits on zero
            # there.  Only the raw curve needs a local baseline under it, and
            # that is a drawing aid, not part of the fit.
            if profile_source == 'raw':
                near = np.abs(distances - s) < sigma * 5
                bkg = float(intensities[near].min()) if near.any() else 0.0
            else:
                bkg = 0.0
            y_fit = pk['amplitude'] * np.exp(-0.5 * ((x_fit - s) / sigma) ** 2) + bkg
            if use_log:
                y_fit = np.maximum(y_fit, prof_floor)
                bkg = max(bkg, prof_floor)
            ax_prof.fill_between(x_fit, bkg, y_fit, alpha=0.25, color=col)
            ax_prof.axvline(s, color=col, lw=1.5, alpha=0.85)
            # Label at top of plot
            lvl = ref_level * (1.02 if not use_log else 1.5)
            ax_prof.text(s, lvl, ' ' + _label(pk['order']),
                         color=col, fontsize=8, ha='center', va='bottom')

        if sl0_pos is not None:
            ax_prof.axvline(sl0_pos, color=_SL0_COLOR, lw=1.8, alpha=0.9, zorder=3)
            ax_prof.text(sl0_pos, ref_level * (1.02 if not use_log else 1.5),
                         'SL0 ', color=_SL0_COLOR, fontsize=8,
                         ha='right', va='bottom', fontweight='bold')

        src = 'background-subtracted' if profile_source == 'subtracted' else 'raw'
        ax_prof.set_xlabel('Distance from the profile centroid (px)')
        ax_prof.set_ylabel('Intensity (counts, background-subtracted)'
                           if profile_source == 'subtracted' else 'Intensity (counts)')
        if use_log:
            ax_prof.set_yscale('log')
            ax_prof.set_title(f'Profile, {src} (log scale)', fontsize=10)
        else:
            ax_prof.set_ylim(bottom=0)
            ax_prof.set_title(f'Profile, {src} (linear scale)', fontsize=10)

    return fig


# A reprojection this far off is not a tolerance — it is the documented
# signature of a UB in the wrong laboratory frame (README.md, "A sample UB from
# indexing may arrive in a beam-∥-x frame"), which puts the reflection tens of
# degrees away while leaving |G| and γ right.
PARENT_OFFSET_WARN_DEG = 1.0

# The train direction is predicted from the orientation with no free parameter,
# and `train_direction_delta_deg` folds it into [0, 90] — it compares the two as
# LINES.  A few degrees is the fitting scatter; tens of degrees means the row of
# peaks that was segmented does not lie along the predicted superlattice
# direction at all, which no choice of Λ and no relabelling can repair.
TRAIN_DELTA_WARN_DEG = 5.0


def _print_fit_diagnostics(period: dict) -> None:
    """The numbers that say whether the orientation, not Λ, is what failed."""
    rms    = period.get('fit_rms_deg')
    offset = period.get('parent_offset_deg')
    train  = period.get('train_delta_deg')
    if rms is None and offset is None:
        return                       # monochromatic route: no model to diagnose

    print()
    print('  Fit diagnostics')
    if period.get('orders_used') is not None:
        print(f"    orders used      : {period['orders_used']}")
    if rms is not None and np.isfinite(rms):
        print(f"    fit_rms          : {rms * 1000:.2f} mdeg")
    if offset is not None and np.isfinite(offset):
        flag = '   <-- CHECK THE UB FRAME' if offset > PARENT_OFFSET_WARN_DEG else ''
        print(f"    parent_offset    : {offset:.4f} deg{flag}")
        if offset > PARENT_OFFSET_WARN_DEG:
            print( "                       The indexed reflection is predicted "
                   "that far from the")
            print( "                       measured order-0 peak.  Indexing output "
                   "often has the beam")
            print( "                       along +x while LaueTools uses +y — try "
                   "UB = ub_from_beam_x_frame(UB),")
            print( "                       and confirm with diagnose_ub_frame(). "
                   "A UB in the wrong frame")
            print( "                       still simulates a correct-looking "
                   "pattern, so a simulation")
            print( "                       overlaying the image does not clear it.")

    if train is not None and np.isfinite(train):
        flag = '   <-- TRAIN OFF THE PREDICTED LINE' if train > TRAIN_DELTA_WARN_DEG else ''
        print(f"    train_delta      : {train:.3f} deg{flag}")
        if train > TRAIN_DELTA_WARN_DEG:
            print( "                       The segmented row of peaks does not "
                   "lie along the direction")
            print( "                       the orientation predicts for the "
                   "satellites.  With parent_offset")
            print( "                       small, the reflection is the right one "
                   "— so what was segmented")
            print( "                       is probably not the satellite train: a "
                   "streak, a neighbouring")
            print( "                       spot, or structure in the flank.  No Λ "
                   "and no relabelling")
            print( "                       repairs a wrong line.")


def _print_order_sign(order_sign: dict | None) -> None:
    """The relabelling must never be invisible — it changes what every sign means."""
    if not order_sign:
        return
    if order_sign.get('inverted'):
        print(f"  Order sign   : INVERTED vs the model, labels corrected "
              f"(arrow_cos = {order_sign['arrow_cos']:+.3f})")
        print( "                 Detection's +n was on the far side of the parent. "
               "Every signed")
        print( "                 quantity below — delta_q, the asymmetries — now "
               "follows the crystal.")
    elif not order_sign.get('confident'):
        print(f"  Order sign   : not resolved — {order_sign.get('reason', '')}")


def _print_summary(result: dict, metrics: dict, period: dict | None = None,
                   sl0: dict | None = None, period_error: str | None = None,
                   order_sign: dict | None = None) -> None:
    peaks = result['peaks']
    bulk_is_not_sl0 = _order0_is_bulk(sl0)
    sl0_pos = _sl0_position(sl0)
    sep = '=' * 62
    print(sep)
    print('  SATELLITE PEAK DETECTION - SUMMARY')
    print(sep)
    sl0_r, sl0_c = result['sl0_center']
    print(f"  Axis angle   : {result['axis_angle']:.2f} deg")
    print(f"  Profile centroid : row = {sl0_r:.1f},  col = {sl0_c:.1f}  (crop coords)")
    print(f"  Peaks found  : {len(peaks)}")
    _print_order_sign(order_sign)

    if not np.isnan(metrics.get('delta_q', float('nan'))):
        print(f"  Delta-q      : {abs(metrics['delta_q']):.2f} px / order")
    if not np.isnan(metrics.get('alpha', float('nan'))):
        print(f"  Decay alpha  : {metrics['alpha']:.3f}  (R2 = {metrics.get('alpha_r2', float('nan')):.3f})")

    if period_error is not None:
        print()
        print('  Layer period : NOT COMPUTED')
        # Wrapped by hand rather than by textwrap: the guard messages carry
        # their own sentence structure and reflow reads worse than a margin.
        for line in period_error.split('. '):
            line = line.strip()
            if line:
                print(f"    {line}{'' if line.endswith('.') else '.'}")
        print('    (detection above is unaffected — the figure shows what was '
              'segmented)')

    if period is not None:
        p_ang = period['period_angstrom']
        p_nm  = period['period_nm']
        dq    = period.get('delta_q_inv_ang')
        # A fit railed against a period bound is not a measurement: the residual
        # is on centred directions, so a model pointing the wrong way shrinks its
        # own spread to zero and lands on the bound.  Printing the bound as a
        # period reads as a result and has cost a debugging session before.
        if period.get('fit_at_bound'):
            lo, hi = period.get('period_bounds_angstrom', (float('nan'),) * 2)
            which = 'lower' if abs(p_ang - lo) < abs(p_ang - hi) else 'upper'
            print()
            print(f"  Layer period : NOT MEASURED — the fit railed against the "
                  f"{which} bound")
            print(f"                 ({p_ang:.1f} Å is the search bound "
                  f"[{lo:.0f}, {hi:.0f}], not a fitted value)")
            print( "                 The model cannot reproduce the observed "
                   "spacing at any period")
            print( "                 in range.  Check the orientation first: see "
                   "the diagnostics below.")
        elif np.isfinite(p_ang):
            dq_str = f'Δq = {dq:.4f} Å⁻¹, ' if dq is not None else ''
            print(f"  Layer period : {p_ang:.1f} Å  =  {p_nm:.2f} nm"
                  f"   ({dq_str}mean over {len(period['per_pair'])} pair(s))")
            for pp in period['per_pair']:
                n1, n2 = pp['orders']
                if 'delta_px' in pp:                        # monochromatic / analytic
                    print(f"    SL{n1:+d}→SL{n2:+d} : Δx={pp['delta_px']:.2f} px"
                          f"  →  Λ = {pp['period_angstrom']:.1f} Å  "
                          f"({pp['period_nm']:.2f} nm)")
                else:                                        # laue_forward, angular
                    print(f"    SL{n1:+d}→SL{n2:+d} : measured={pp['sep_meas_deg']:.6f} deg"
                          f"  predicted={pp['sep_pred_deg']:.6f} deg")

        _print_fit_diagnostics(period)

    if peaks:
        rows = [(_order_label(pk['order'], bulk_is_not_sl0), pk['pos_along_axis'],
                 pk['amplitude'], pk['fwhm']) for pk in peaks]
        if sl0_pos is not None:
            rows.append(('SL0', sl0_pos, sl0['sl0_measured_amplitude'], float('nan')))
        rows.sort(key=lambda r: r[1])

        print()
        print(f"  {'Order':>7}  {'Pos (px)':>9}  {'Amplitude':>11}  {'FWHM (px)':>9}")
        print('  ' + '-' * 44)
        for label, pos, amplitude, fwhm in rows:
            fwhm_str = f'{fwhm:9.2f}' if np.isfinite(fwhm) else f"{'-':>9}"
            print(f"  {label:>7}  {pos:+9.2f}  {amplitude:11.1f}  {fwhm_str}")
    else:
        print('\n  No peaks detected. Try a lower min_prominence or an explicit axis_angle.')

    # The bulk-to-SL0 separation carries the mean out-of-plane strain of the
    # stack (README.md, "The bright peak in the ROI is the bulk"), so it is the physics output of this figure.
    if sl0_pos is not None and 'detected_order0_pos' in sl0:
        print()
        print(f"  Bulk -> SL0  : {sl0['detected_order0_pos'] - sl0_pos:+.2f} px"
              f"   (mean out-of-plane strain of the stack)")
    print(sep)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_single_image(
    img_source=None,
    h5_img_key: str = 'frames',
    frame_index: int = 0,
    roi_center: Optional[Tuple[int, int]] = None,
    boxsize: int = 60,
    coords: str = 'numpy',
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
    pixel_size_mm: Optional[float] = None,
    detector_distance_mm: Optional[float] = None,
    wavelength_angstrom: Optional[float] = None,
    energy_kev: Optional[float] = None,
    two_theta_0_deg: float = 0.0,
    chi_deg: float = 0.0,
    # ── Laue route: turns this into the bulk/SL0 segmentation test ──
    period_method: str = 'monochromatic',
    hkl: Optional[Tuple[int, int, int]] = None,
    lattice: Optional[Tuple[float, float]] = None,
    UB=None,
    detector=None,
    sl0_boxsize: float = 3.0,
    verbose: bool = False,
    quiet: bool = False,
    profile_log: bool = True,
    show_linear_profile: bool = True,
    profile_source: str = 'subtracted',
    print_summary: bool = True,
    figsize: tuple = (16, 5.5),
    show_plot: bool = True,
) -> Dict[str, Any]:
    """Run satellite peak detection on a single spot image and display results.

    Parameters
    ----------
    img_source      : path to an HDF5, .npy, TIFF or other fabio-readable file, or
                      an in-memory np.ndarray (2-D, or 3-D indexed by frame_index)
                      so a stack combined in a notebook needs no round trip
                      through disk. None = synthetic test image.
    h5_img_key      : HDF5 dataset key, e.g. 'entry_0000/CRGIF/eiger4m/data'.
    frame_index     : frame to use when the HDF5 dataset is a 3-D stack.
    roi_center      : (x, y) = (col, row) of the Laue spot on the detector (pixels).
                      If None, uses the full frame (or the synthetic image).
    boxsize         : half-size of the crop: total = (2*boxsize+1) x (2*boxsize+1) px.
                      Rule of thumb: boxsize > 2 * n_max * satellite_spacing.
    coords          : 'numpy' (0-based) or 'xmas' (1-based, subtracts 1 from roi_center).
    axis_angle      : satellite axis in degrees from +x. Auto-detected via PCA if None.
    n_max           : maximum satellite order to accept (e.g. 3 for SL+-3).
    min_prominence  : peak prominence threshold as fraction of profile max (0-1).
    strip_width     : strip width (px) perpendicular to axis for 1-D profile.
    bg_sigma        : Gaussian sigma (px) for background subtraction.
                      Set to 0 to skip subtraction (good when image is already clean).
                      Rule of thumb: bg_sigma ~ 0.5 * boxsize to remove only the
                      large-scale diffuse background without touching satellite peaks.
    peak_min_width  : minimum 1-D peak width in profile bins (~px) to reject
                      single-pixel hot-pixel spikes.  None = no width filter.
    hot_pixel_sigma : remove pixels > hot_pixel_sigma * local noise before
                      background subtraction.  None = skip.
    n_range         : (n_min, n_max) restrict which satellite orders are kept;
                      overrides n_max when provided.
    spacing_px      : if provided, overrides automatic spacing estimation (useful when
                      only one satellite is visible or spacing is strongly non-uniform).
    pixel_size_mm   : detector pixel size in mm (e.g. 0.075 for Eiger 4M).
                      Required to compute the layer period.
    detector_distance_mm : sample-to-detector distance in mm.
                      Required to compute the layer period.
    wavelength_angstrom  : X-ray wavelength in Angstroms.  Alternative to energy_kev.
    energy_kev           : X-ray energy in keV (e.g. 17.06).  Converted to Å via
                      λ = 12.3984 / E.  Takes precedence over wavelength_angstrom.
                      Provide either this or wavelength_angstrom to compute the period.
    two_theta_0_deg : approximate 2θ of the SL0 reflection in degrees (default 0).
                      Used to apply the flat-detector Lorentz cos² correction.
    chi_deg         : angle between this reflection's Q and the true growth axis
                      (default 0 = symmetric reflection). Corrects the layer
                      period for tilted/asymmetric reflections — see
                      metrics.layer_period_from_peaks.
    sl0_boxsize     : half-width (px) of the box searched for SL0 around its
                      predicted position.  Unrelated to `boxsize` above (the
                      crop half-size): this is a handful of px.  Keep it well
                      inside the gap to the nearest satellite and to the bulk
                      peak, or either can supply a second local maximum and
                      the box is rejected as ambiguous.
    verbose         : print step-by-step diagnostic messages from detect_satellites().
    profile_log     : show the 1-D profile in BOTH linear and log scales (default True).
                      The log panel reveals weak satellites that are invisible on linear.
    profile_source  : which image the 1-D profile is taken from.
                      'subtracted' (default) = the background-subtracted crop,
                      i.e. what the detection actually worked on, so the fitted
                      Gaussians drawn over it sit on the data they were fitted to
                      and the outer orders are not buried in the parent's tail.
                      'raw' = the untouched crop, for reading measured counts.
                      Both panels always come from the same image.
    figsize         : matplotlib figure size (width, height) in inches.
    show_plot       : display the figure inline (set False to suppress).

    Returns
    -------
    dict with keys:
        'peaks_df'   : pd.DataFrame — one row per detected peak
        'metrics'    : dict — N_sat, delta_q, alpha, fwhm_slope, asymmetry, bulk_pos
        'period'     : dict — layer period.  None when the geometry parameters
                       were not supplied, and also when the period calculation
                       raised: the figure and the detection are produced either
                       way, so segmentation can be tuned before the period can
                       possibly work.
        'period_error' : str or None — why 'period' is None, when it is because
                       the calculation failed rather than because it was not asked
                       for.  Detection and the figure are unaffected.
        'sl0'        : dict — SL0 position and the bulk-to-SL0 offset, or None
                       on the non-Laue routes.  `sl0_confirmed` says whether
                       SL0 was found in the image; `sl0_measured_pos` and
                       `sl0_measured_position_2d` are only present when it was.
        'result'     : raw output of detect_satellites()
        'crop'       : 2-D numpy array of the image that was analysed
        'fig'        : matplotlib Figure (None if show_plot=False)
    """
    # ── Warn if boxsize looks too small ──────────────────────────────────────
    if boxsize < 30 and img_source is not None:
        warnings.warn(
            f'boxsize={boxsize} gives a {2*boxsize+1}x{2*boxsize+1} px crop, '
            'which may be too small to see satellite peaks. '
            'For satellite analysis with typical 15-50 px inter-order spacing, '
            'boxsize >= 50 is recommended.',
            stacklevel=2,
        )

    # ── Load image ───────────────────────────────────────────────────────────
    if img_source is None:
        if not quiet:
            print('[INFO] img_source=None — generating synthetic test image (3 satellites, 22 px spacing, 35 deg)')
        crop = make_synthetic_image(n_satellites=3, spacing=22.0, axis_angle=35.0,
                                    envelope_decay=0.5, noise_level=50.0)
        crop_origin = (0, 0)
        title = 'Satellite detection — synthetic test image'
    else:
        if not quiet:
            if isinstance(img_source, np.ndarray):
                origin = f'in-memory array {img_source.shape}'
            else:
                origin = f'{Path(img_source).name}  key={h5_img_key!r}'
            print(f'[INFO] Loading frame {frame_index} from {origin}')
        frame = _load_frame(img_source, h5_img_key, frame_index)
        if not quiet:
            print(f'[INFO] Frame shape: {frame.shape}  range: [{frame.min():.0f}, {frame.max():.0f}]')

        if roi_center is not None:
            crop, crop_origin = _extract_crop(frame, roi_center, boxsize, coords)
            if not quiet:
                xc, yc = roi_center
                print(f'[INFO] Crop: center=({xc},{yc}) ({coords}), boxsize={boxsize} -> '
                      f'crop shape={crop.shape}  origin={crop_origin}')
        else:
            crop = frame
            crop_origin = (0, 0)
            if not quiet:
                print('[INFO] roi_center=None — using full frame')

        title = (f'Satellite detection  |  frame {frame_index}  |  '
                 f'roi=({roi_center})  boxsize={boxsize}')

    # ── Detect satellites ─────────────────────────────────────────────────────
    if not quiet:
        print(f'[INFO] Detecting: axis_angle={axis_angle}, n_max={n_max}, '
              f'prominence={min_prominence}, bg_sigma={bg_sigma}, '
              f'hot_pixel_sigma={hot_pixel_sigma}, peak_min_width={peak_min_width}, '
              f'spacing_px={spacing_px}')
    result = detect_satellites(
        crop,
        axis_angle=axis_angle,
        n_max=n_max,
        min_prominence=min_prominence,
        strip_width=strip_width,
        bg_sigma=bg_sigma,
        peak_min_width=peak_min_width,
        hot_pixel_sigma=hot_pixel_sigma,
        n_range=n_range,
        spacing_px=spacing_px,
        adaptive_fill_win=adaptive_fill_win,
        verbose=verbose,
    )

    geo_complete = (pixel_size_mm is not None
                    and detector_distance_mm is not None
                    and (wavelength_angstrom is not None or energy_kev is not None))
    laue_ready = all(v is not None for v in (hkl, lattice, UB, detector))

    # ── Order sign ─────────────────────────────────────────────────────────────
    # Detection cannot know which side of the parent is +n; only the orientation
    # can.  Settle it here, before anything consumes the labels: the period
    # announces an inverted sign by railing against a bound, but the asymmetry
    # indicators would silently come out with the opposite sign.
    order_sign = None
    if laue_ready:
        try:
            result['peaks'], order_sign = resolve_and_apply_order_sign(
                result['peaks'], hkl=tuple(hkl), lattice=tuple(lattice),
                UB=UB, detector=detector, crop_origin_px=crop_origin)
        except Exception as exc:
            order_sign = {'inverted': False, 'confident': False,
                          'reason': f'{type(exc).__name__}: {exc}',
                          'arrow_cos': float('nan'),
                          'train_delta_deg': float('nan'),
                          'parent_offset_deg': float('nan')}
        if order_sign.get('inverted') and not quiet:
            print('[SIGN] order labels inverted relative to the model '
                  f"(arrow_cos = {order_sign['arrow_cos']:+.3f}) — corrected")

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics(result['peaks'])

    # A period failure must not cost the figure.  This function is the bench
    # for tuning segmentation, and the period is the step most likely to raise
    # while the parameters are still wrong — too few orders, γ = 0, a UB in the
    # wrong frame.  Aborting here would hide the very image the caller needs in
    # order to fix them.  `scan_pipeline` already keeps its other metrics on the
    # same grounds; this mirrors it, and the reason is reported rather than lost.
    period = None
    period_error = None
    try:
        if laue_ready:
            period = layer_period_from_peaks(
                result['peaks'], method=period_method,
                wavelength_angstrom=wavelength_angstrom, energy_kev=energy_kev,
                hkl=tuple(hkl), lattice=tuple(lattice),
                detector=detector, UB=UB, crop_origin_px=crop_origin,
            )
        elif geo_complete:
            period = layer_period_from_peaks(
                result['peaks'],
                pixel_size_mm=pixel_size_mm or 1.0,
                detector_distance_mm=detector_distance_mm or 1.0,
                wavelength_angstrom=wavelength_angstrom,
                energy_kev=energy_kev,
                two_theta_0_deg=two_theta_0_deg,
                chi_deg=chi_deg,
            )
    except Exception as exc:
        period_error = f'{type(exc).__name__}: {exc}'

    # SL0. In an MQW the detected order-0 peak is the bulk reflection
    # (README.md, "The bright peak in the ROI is the bulk"), so SL0 is first predicted from the satellite
    # ladder — which needs the orientation, hence the Laue routes only — and
    # then located in the raw image around that prediction.
    # A railed Λ must not reach the ladder: it would predict SL0 at a wrong
    # position, and `locate_sl0_by_local_max` would then confirm whatever pixel
    # happens to sit there, turning a failed fit into an invented measurement.
    sl0 = None
    if (laue_ready and period is not None
            and np.isfinite(period['period_angstrom'])
            and not period.get('fit_at_bound')):
        try:
            sl0 = locate_sl0_from_ladder(
                result['peaks'], hkl=tuple(hkl), lattice=tuple(lattice),
                UB=UB, detector=detector, crop_origin_px=crop_origin,
                period_angstrom=float(period['period_angstrom']),
                verbose=verbose)
        except Exception as exc:
            if not quiet:
                print(f'[WARN] SL0 prediction failed: {type(exc).__name__}: {exc}')

        if sl0 is not None and sl0.get('confident'):
            try:
                # The untouched raw crop — background subtraction and hot-pixel
                # clipping both erase this feature; see the function's docstring.
                sl0.update(locate_sl0_by_local_max(
                    crop, sl0['sl0_pos_along_axis'],
                    result['axis_angle'], result['sl0_center'],
                    boxsize=sl0_boxsize))
            except Exception as exc:
                if not quiet:
                    print(f'[WARN] SL0 search failed: {type(exc).__name__}: {exc}')

    # ── Build output DataFrame ────────────────────────────────────────────────
    rows = []
    r0, c0 = crop_origin
    for pk in result['peaks']:
        row_det, col_det = pk['position_2d']
        rows.append({
            'order':           pk['order'],
            'pos_along_axis':  pk['pos_along_axis'],
            'amplitude':       pk['amplitude'],
            'fwhm':            pk['fwhm'],
            'sigma':           pk['sigma'],
            # Position in crop coordinates
            'crop_row':        row_det,
            'crop_col':        col_det,
            # Position in original detector coordinates
            'det_row':         row_det + r0,
            'det_col':         col_det + c0,
            'fit_success':     pk['fit_success'],
        })
    peaks_df = pd.DataFrame(rows)

    # ── Print summary ──────────────────────────────────────────────────────────
    if print_summary:
        _print_summary(result, metrics, period, sl0=sl0,
                       period_error=period_error, order_sign=order_sign)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = _plot(crop, result, figsize, title,
                profile_log=profile_log, show_linear=show_linear_profile,
                sl0=sl0, strip_width=strip_width, profile_source=profile_source)
    if show_plot:
        plt.show()

    return {
        'peaks_df':     peaks_df,
        'metrics':      metrics,
        'period':       period,
        'period_error': period_error,
        'order_sign':   order_sign,
        'sl0':          sl0,
        'result':       result,
        'crop':         crop,
        'fig':          fig,
    }
