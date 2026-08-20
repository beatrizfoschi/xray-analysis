"""Jupyter-friendly examples for the polychromatic (Laue) period routes.

The Laue routes are NOT validated against experiment — run ``check_geometry()``
first and only trust a period once the reprojection and axis diagnostics pass.

Drop into a notebook cell::

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path('caminho/para/xray-analysis')))

    from laue.satellite.example_laue_period import (
        load_calibration, check_geometry, period_from_image, demo_synthetic,
    )

    # 0) try it with no data at all
    demo_synthetic()

    # 1) load the .det (geometry + orientation in one call)
    geom, UB = load_calibration('/data/calib/mu_led.det')
    geom

    # 2) validate the geometry BEFORE trusting any period
    check_geometry(
        geom, UB,
        hkl                = (1, 0, 5),
        lattice            = (3.189, 5.185),
        measured_parent_px = (1263.4, 1913.2),   # (col, row) of the indexed spot
        measured_axis_deg  = -84.4,
        measured_axis_fwhm = 1.48,
    )

    # 3) period on one image, all three methods side by side
    out = period_from_image(
        img_source  = Path('/data/eiger4m_0000.h5'),
        h5_img_key  = 'entry_0000/CRGIF/eiger4m/data',
        frame_index = 0,
        roi_center  = (1263, 1913),      # (x, y) = (col, row)
        boxsize     = 60,
        geom        = geom,
        UB          = UB,
        hkl         = (1, 0, 5),
        lattice     = (3.189, 5.185),
        wavelength_angstrom = 0.727,
    )
    out['comparison']       # DataFrame: one row per method
    out['forward']          # full dict from the recommended route
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from laue.satellite.detection import detect_satellites
from laue.satellite.geometry import B_matrix_hexagonal, DetectorGeometry, LAB_KI, angular_separation, assert_ub_material, diagnose_ub_frame, direction_to_2theta_chi, gamma_from_vectors, kf_hat, lab_vectors_from_UB, pixels_to_2theta_chi, psi_from_geometry, ub_from_beam_x_frame
from laue.satellite.period import compare_methods, layer_period_from_peaks


def expected_trend_sentence(orders) -> str:
    """What the polychromatic model predicts for the gaps, with the right sign.

    The predicted step scales as 1/(|G0| + n·q·cos γ), so the denominator shrinks
    for negative orders and the gaps **grow**; for positive orders they contract.
    The canonical "~1.8 %/step contraction" is the positive-order statement, and
    quoting it on a negative-order train inverts the sign the user is asked to
    look for.
    """
    ns = {n for n in orders if n != 0}
    if ns and all(n < 0 for n in ns):
        return 'polychromatic expects a ~1.8 %/step GROWTH on this negative-order train;'
    if ns and all(n > 0 for n in ns):
        return 'polychromatic expects a ~1.8 %/step CONTRACTION on this positive-order train;'
    return ('polychromatic expects ~1.8 %/step: GROWTH on the negative side, '
            'CONTRACTION on the positive side;')


# ── 1. Calibration ────────────────────────────────────────────────────────────

def load_calibration(det_path, expected_material: Optional[str] = None
                     ) -> Tuple[DetectorGeometry, np.ndarray]:
    """Read a LaueTools ``.det``.  Returns (DetectorGeometry, UB_calib).

    Reminder: ``(xcen, ycen)`` is the detector reference point (PONI), not a beam
    center — in reflection geometry nothing is visible there.

    Pass ``expected_material`` (e.g. ``'GaN'``) to have the material mismatch
    raise instead of being discovered later as a wrong period.
    """
    geom, UB = DetectorGeometry.from_det_file(det_path)
    print(f'[INFO] {geom}')
    print(f'[INFO] framedim = {geom.framedim}   kf_direction = {geom.kf_direction}')
    print(f'[INFO] CCD = {geom.ccd_label}   calibrated with: {geom.material}')

    det = float(np.linalg.det(UB))
    ortho = float(np.abs(UB @ UB.T - np.eye(3)).max())
    print(f'[INFO] det(UB) = {det:.7f}   max|UB·UBᵀ − I| = {ortho:.2e}   '
          f'({"pure rotation, as expected" if abs(det - 1) < 1e-4 and ortho < 1e-5 else "NOT a clean rotation"})')

    if geom.material is not None:
        print(f'\n[WARN] This UB belongs to {geom.material}, the crystal the detector '
              f'was\n       calibrated with. Use this file for the GEOMETRY. For the '
              f'sample UB,\n       load the indexation of the sample\'s own Laue '
              f'pattern (.fit / .res).')

    if expected_material is not None:
        assert_ub_material(geom, expected_material)

    return geom, UB


# ── 2. Geometry validation — run this before trusting a period ───────────────

def check_geometry(
    geom: DetectorGeometry,
    UB: np.ndarray,
    *,
    hkl: tuple,
    lattice: tuple,
    measured_parent_px: Optional[Tuple[float, float]] = None,
    peaks=None,
    crop_origin_px: Optional[Tuple[float, float]] = None,
    measured_axis_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """The independent check of the geometry chain (ADDENDUM 2 §5.2).

    Everything else in the pipeline is self-consistent by construction and cannot
    detect a wrong UB or a mismatched angular convention.  This can: the parent
    reflection's (2θ, χ) is predicted from ``UB @ B @ [hkl]`` and the Laue
    condition, and compared with the measurement converted through LaueTools and
    the ``.det``.

    2θ alone needs **no UB** — it follows from the pixel position and the
    ``.det``.  A 2θ match therefore already validates the calibration, the
    wavelength and the (hkl) assignment; only χ tests the orientation.

    Measured parent position
    ------------------------
    Preferred: pass ``peaks`` (from ``detect_satellites``) with
    ``crop_origin_px``; the SL0 centroid is then taken from the fit.

    ``measured_parent_px`` is accepted directly but is easy to get wrong — **the
    ROI centre is not the parent peak**.  A segmentation ROI is placed to cover
    SL0 *and* the satellite train, so with one-sided orders (e.g.
    ``n_range=(-3, 0)``) its centre sits about 1.5 × spacing from SL0.  At this
    geometry that is ~0.5° of 2θ, ten times the pass criterion.

    Run it for several indexed reflections spread across the frame, not just
    central ones — azimuth convention errors grow away from the reference point.
    If it fails, nothing downstream is meaningful.
    """
    G_lab, z_lab = lab_vectors_from_UB(hkl, lattice, UB)

    naive = float(np.linalg.norm(np.asarray(UB, dtype=float) @ np.asarray(hkl, float)))

    tt_pred, chi_pred = direction_to_2theta_chi(kf_hat(G_lab, LAB_KI))
    gamma_deg = math.degrees(gamma_from_vectors(G_lab, z_lab))

    print(f'\n── Crystallography ({hkl[0]}{hkl[1]}{hkl[2]}) ──')
    print(f'  |G| = {np.linalg.norm(G_lab):.5f} A^-1     '
          f'(UB @ hkl alone would give {naive:.5f} — the B-omission error)')
    print(f'  gamma = {gamma_deg:.4f} deg   (crystallographic, NOT LaueTools chi)')

    out: Dict[str, Any] = {
        'G_magnitude': float(np.linalg.norm(G_lab)),
        'gamma_deg': gamma_deg,
        'predicted_two_theta_deg': tt_pred,
        'predicted_chi_deg': chi_pred,
    }

    # Prefer the fitted SL0 centroid over a hand-supplied pixel: an ROI centre is
    # offset from the parent by roughly half the satellite train.
    source = None
    if peaks is not None:
        if crop_origin_px is None:
            raise ValueError('peaks= also needs crop_origin_px=(row0, col0).')
        parent = next((p for p in peaks if p['order'] == 0), None)
        if parent is None:
            raise ValueError(
                'no SL0 (order 0) among the peaks — the reprojection check needs '
                'the parent reflection.  Widen n_range so order 0 is detected.'
            )
        row0, col0 = crop_origin_px
        measured_parent_px = (col0 + parent['position_2d'][1],
                              row0 + parent['position_2d'][0])
        source = 'fitted SL0 centroid'
    elif measured_parent_px is not None:
        source = 'measured_parent_px as given'

    print(f'\n── Reprojection in angular space (§5.2) ──')
    print(f'  predicted parent: 2theta = {tt_pred:.4f} deg   chi = {chi_pred:.4f} deg')
    if measured_parent_px is not None:
        mcol, mrow = measured_parent_px
        print(f'  source: {source}  (col={mcol:.2f}, row={mrow:.2f})')
        tt_m, chi_m = pixels_to_2theta_chi(mcol, mrow, geom)
        tt_m, chi_m = float(tt_m[0]), float(chi_m[0])
        err = math.degrees(float(angular_separation(tt_pred, chi_pred, tt_m, chi_m)))
        out.update(measured_two_theta_deg=tt_m, measured_chi_deg=chi_m,
                   reprojection_error_deg=err)
        verdict = 'PASS' if err < 0.05 else 'FAIL'
        print(f'  measured  parent: 2theta = {tt_m:.4f} deg   chi = {chi_m:.4f} deg')
        print(f'  separation = {err:.4f} deg   → {verdict}')
        print(f'    d(2theta) = {tt_m - tt_pred:+.4f}   d(chi) = {chi_m - chi_pred:+.4f}')
        if err >= 0.05:
            print('    2theta needs no UB, so split the diagnosis:')
            print('      d(2theta) large  -> .det, wavelength, or wrong (hkl)')
            print('      d(2theta) small but d(chi) large -> the UB or its convention')
            if source == 'measured_parent_px as given':
                print('      ...and check this is the SL0 centroid, not the ROI centre:')
                print('         an ROI covering SL0 + the train is offset by about')
                print('         1.5 x spacing, which is ~0.5 deg of 2theta here.')

            frames = diagnose_ub_frame(UB, hkl, lattice, tt_m, chi_m)
            best = frames['best']
            print('\n    laboratory-frame candidates for the UB:')
            for name, sep in frames.items():
                if name == 'best':
                    continue
                mark = '  <== best' if name == best else ''
                shown = 'off detector' if math.isinf(sep) else f'{sep:8.3f} deg'
                print(f'      {name:34s} {shown}{mark}')
            if frames[best] < 1.0 and best != 'as given (beam ∥ y, LaueTools)':
                print(f'\n    -> "{best}" agrees while the others do not.')
                print('       Fix the frame where the UB is loaded, e.g.')
                print('           UB = ub_from_beam_x_frame(UB_raw)')
                print('       so the correction stays visible.')
            out['ub_frame_candidates'] = frames
    else:
        print('  (no parent position given — pass peaks= + crop_origin_px= to validate)')

    if measured_axis_deg is not None:
        print(f'\n[NOTE] axis_angle ({measured_axis_deg} deg) is deliberately NOT used '
              f'here.\n       ADDENDUM 2 §4 withdraws that comparison: axis_angle is an '
              f'ad-hoc\n       metric of train inclination, not a calibrated geometric '
              f'quantity.\n       The calibrated equivalent is train_delta_deg, reported '
              f'by the fit.')

    return out


# ── 3. Period on one image, all methods ──────────────────────────────────────

def period_from_image(
    img_source,
    *,
    h5_img_key: str,
    frame_index: int,
    roi_center: Tuple[int, int],
    boxsize: int,
    geom: DetectorGeometry,
    UB: np.ndarray,
    hkl: tuple,
    lattice: tuple,
    wavelength_angstrom: float,
    coords: str = 'numpy',
    detect_kw: Optional[dict] = None,
) -> Dict[str, Any]:
    """Detect satellites on one crop and run all three period routes on them.

    ``roi_center`` is ``(x, y) = (col, row)``, the project-wide convention.
    """
    import h5py

    col_c, row_c = int(roi_center[0]), int(roi_center[1])
    if coords == 'xmas':
        col_c -= 1
        row_c -= 1
    row0, col0 = row_c - boxsize, col_c - boxsize

    with h5py.File(img_source, 'r') as f:
        crop = np.asarray(
            f[h5_img_key][frame_index,
                          row0:row_c + boxsize + 1,
                          col0:col_c + boxsize + 1],
            dtype=np.float32,
        )

    result = detect_satellites(crop, **(detect_kw or {}))
    peaks = result['peaks']
    print(f'[INFO] {len(peaks)} peaks, orders '
          f'{sorted(p["order"] for p in peaks)}, '
          f'axis_angle = {result["axis_angle"]:.2f} deg')

    return period_from_peaks(
        peaks, crop_origin_px=(row0, col0), geom=geom, UB=UB,
        hkl=hkl, lattice=lattice, wavelength_angstrom=wavelength_angstrom,
        measured_axis_deg=float(result['axis_angle']),
    )


def period_from_peaks(
    peaks,
    *,
    crop_origin_px: Tuple[int, int],
    geom: DetectorGeometry,
    UB: np.ndarray,
    hkl: tuple,
    lattice: tuple,
    wavelength_angstrom: float,
    measured_axis_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """Run monochromatic + both Laue routes on an existing peak list.

    Disagreement between the routes is the expected outcome — they read opposite
    components of the modulation.  The comparison is a diagnostic, not a vote.
    """
    forward = layer_period_from_peaks(
        peaks, method='laue_forward',
        wavelength_angstrom=wavelength_angstrom,
        hkl=hkl, lattice=lattice,
        detector=geom, UB=UB, crop_origin_px=crop_origin_px,
    )

    # laue_analytic is the only route still needing ψ by hand.  Derive it from the
    # measured spot position and the detected train direction rather than passing
    # 0 blindly, which inflates Λ whenever the train is not radial.
    row0, col0 = crop_origin_px
    sats = sorted((p for p in peaks if p['order'] != 0), key=lambda p: p['order'])
    p_first, p_last = sats[0]['position_2d'], sats[-1]['position_2d']
    train_deg = math.degrees(math.atan2(p_last[0] - p_first[0],
                                        p_last[1] - p_first[1]))
    mid_row = row0 + 0.5 * (p_first[0] + p_last[0])
    mid_col = col0 + 0.5 * (p_first[1] + p_last[1])
    psi_deg = psi_from_geometry(mid_row, mid_col, geom.ycen, geom.xcen, train_deg)

    comparison = compare_methods(
        peaks,
        wavelength_angstrom=wavelength_angstrom,
        hkl=hkl, lattice=lattice,
        detector=geom, UB=UB, crop_origin_px=crop_origin_px,
        satellite_axis_psi_deg=psi_deg,
        two_theta_0_deg=forward['two_theta_measured'],
    )

    df = pd.DataFrame([
        {'method': name,
         'period_nm': v['period_nm'],
         'period_angstrom': v['period_angstrom'],
         'error': v['error']}
        for name, v in comparison.items()
    ])

    print('\n── Period by method ──')
    print(df.to_string(index=False))
    print(f'\n── laue_forward diagnostics (all angular) ──')
    print(f'  fit rms residual   : {forward["fit_rms_deg"]*1e3:.4f} mdeg')
    print(f'  parent reprojection: {forward["parent_offset_deg"]:.4f} deg')
    print(f'  train direction    : {forward["train_delta_deg"]:.3f} deg '
          f'(predicted vs measured)')
    print(f'  gamma              : {forward["gamma_deg"]:.3f} deg')
    print(f'  2theta from indexing: {forward["two_theta_deg"]:.4f} deg')
    if np.isfinite(forward['two_theta_measured']):
        print(f'  2theta of the parent: {forward["two_theta_measured"]:.4f} deg  '
              f'(delta {forward["two_theta_measured"] - forward["two_theta_deg"]:+.4f})')
    print(f'  2theta of satellites: {forward["two_theta_satellites"]:.4f} deg  '
          f'(offset along the train by design — not comparable with indexing)')
    print(f'  fit at bound       : {forward["fit_at_bound"]}')
    print(f'  orders used        : {forward["orders_used"]}')

    print('\n  consecutive separations (exact spherical formula):')
    seps = [p['sep_meas_deg'] for p in forward['per_pair']]
    for p, s in zip(forward['per_pair'], seps):
        print(f'    {str(p["orders"]):>10s}  measured {p["sep_meas_deg"]:.6f} deg   '
              f'predicted {p["sep_pred_deg"]:.6f} deg   '
              f'ratio to first {s/seps[0]:.4f}')
    if len(seps) > 1:
        print('    ' + expected_trend_sentence(
            n for p in forward['per_pair'] for n in p['orders']))
        print('    monochromatic expects uniform spacing.')

    return {'comparison': df, 'forward': forward, 'raw': comparison}


# ── 4. Diagnostic plot: pixel space → angular space (ADDENDUM 2 §6) ─────────

def plot_pixel_to_angular_diagnostic(
    roi_image,
    peaks,
    geom: DetectorGeometry,
    crop_origin_px: Tuple[int, int],
    *,
    UB=None, hkl=None, lattice=None,
    savepath=None,
):
    """Two-panel diagnostic: detector space vs angular space.

    Left  : the ROI in pixels with the detected peaks overlaid — what the
            segmentation sees.
    Right : the same peaks as (2θ, χ), parent reflection marked.

    Prints the consecutive angular separations from the exact spherical formula
    and their ratios relative to the first pair.  Purely diagnostic — nothing
    here feeds the fit.

    What to look for:

    1. same number of peaks, same ordering, in both panels;
    2. the train is generally NOT along constant χ — visual confirmation that
       Δ(2θ) alone would be insufficient;
    3. ratios changing by ~1.8 %/step support the polychromatic picture, uniform
       spacing supports the monochromatic one.  Mind the sign: the gaps GROW on
       the negative side and CONTRACT on the positive one;
    4. the parent lands where indexing predicts (needs UB, hkl, lattice).
    """
    row0, col0 = crop_origin_px
    ordered = sorted(peaks, key=lambda p: p['order'])
    orders  = [p['order'] for p in ordered]
    cols = np.array([col0 + p['position_2d'][1] for p in ordered], dtype=float)
    rows = np.array([row0 + p['position_2d'][0] for p in ordered], dtype=float)

    tt, chi = pixels_to_2theta_chi(cols, rows, geom)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    if roi_image is not None:
        finite = np.asarray(roi_image, dtype=float)
        lo, hi = np.nanpercentile(finite, [30, 99.7])
        ax.imshow(finite, origin='lower', cmap='inferno', vmin=lo, vmax=hi,
                  interpolation='none',
                  extent=[col0, col0 + finite.shape[1],
                          row0, row0 + finite.shape[0]])
    ax.plot(cols, rows, 'o', mfc='none', mec='cyan', ms=9, mew=1.4)
    for n, c, r in zip(orders, cols, rows):
        ax.annotate(f'{n:+d}', (c, r), textcoords='offset points',
                    xytext=(7, 5), color='cyan', fontsize=8)
    ax.set_xlabel('detector column [px]')
    ax.set_ylabel('detector row [px]')
    ax.set_title('Detector space')

    ax = axes[1]
    ax.plot(chi, tt, 'o-', color='slateblue', ms=6, lw=1)
    for n, c, t in zip(orders, chi, tt):
        ax.annotate(f'{n:+d}', (c, t), textcoords='offset points',
                    xytext=(7, 5), fontsize=8)
    zero = [k for k, n in enumerate(orders) if n == 0]
    if zero:
        k = zero[0]
        ax.plot(chi[k], tt[k], '*', color='crimson', ms=16, label='parent (SL0)')

    if UB is not None and hkl is not None and lattice is not None:
        G_lab, _ = lab_vectors_from_UB(hkl, lattice, UB)
        tt_p, chi_p = direction_to_2theta_chi(kf_hat(G_lab, LAB_KI))
        ax.plot(chi_p, tt_p, 'x', color='limegreen', ms=13, mew=2.5,
                label='indexing prediction')
        if zero:
            d = math.degrees(float(angular_separation(tt_p, chi_p, tt[k], chi[k])))
            print(f'parent vs indexing prediction: {d:.4f} deg')

    ax.set_xlabel('chi [deg]')
    ax.set_ylabel('2theta [deg]')
    ax.set_title('Angular space')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)

    print('\nconsecutive angular separations (exact spherical formula):')
    seps, labels = [], []
    for k in range(len(orders) - 1):
        if orders[k + 1] - orders[k] != 1:
            continue
        seps.append(math.degrees(float(
            angular_separation(tt[k], chi[k], tt[k + 1], chi[k + 1]))))
        labels.append(f'({orders[k]:+d},{orders[k+1]:+d})')
    for lab, s in zip(labels, seps):
        print(f'  {lab:>9s}  {s:.6f} deg   ratio to first {s/seps[0]:.4f}')
    if len(seps) > 1:
        step = 100 * (seps[-1] / seps[0]) ** (1 / (len(seps) - 1)) - 100
        print(f'  mean change per step: {step:+.2f} %   '
              f'(~0 % => monochromatic)')
        print('  ' + expected_trend_sentence(orders))

    # Train inclination, as a readout for the record only — NOT an input.
    d_tt  = tt[-1] - tt[0]
    d_chi = (chi[-1] - chi[0]) * math.sin(math.radians(float(np.mean(tt))))
    print(f'  train inclination from the radial direction: '
          f'{abs(math.degrees(math.atan2(d_chi, d_tt))):.1f} deg  (diagnostic only)')

    return fig


# ── 5. Runnable demo, no data required ───────────────────────────────────────

def demo_synthetic(period_angstrom: float = 97.0, verbose: bool = True) -> Dict[str, Any]:
    """End-to-end example on model-generated peaks — needs no files.

    Uses the µLED calibration values and GaN (105).  A crystal orientation is
    built so the parent spot sits at the Bragg angle for λ = 0.727 Å, satellites
    are projected to pixels, and the routes are then asked to recover the period
    that generated them.
    """
    from laue.satellite.geometry import G_magnitude, direction_to_pixel, predict_satellite_pixels, theta_from_G

    hkl, lattice, lam = (1, 0, 5), (3.189, 5.185), 0.727
    geom = DetectorGeometry(dd=69.984, xcen=1079.57, ycen=983.73,
                            xbet=0.173, xgam=0.382, pixelsize=0.075,
                            framedim=(2162, 2068))

    # Orientation: place Ĝ at the Bragg angle to the beam, so 2θ is implied by
    # the indexing rather than chosen — in Laue the spot position follows Ĝ and
    # λ adjusts, so picking a pixel would silently imply a different wavelength.
    theta = theta_from_G(G_magnitude(*hkl, *lattice), lam)
    az = math.radians(75.0)
    G_dir = np.array([math.cos(theta) * math.cos(az), -math.sin(theta),
                      math.cos(theta) * math.sin(az)])
    B = B_matrix_hexagonal(*lattice)
    g_hat = (B @ np.array(hkl, float))
    g_hat /= np.linalg.norm(g_hat)
    v = np.cross(g_hat, G_dir)
    s, c = np.linalg.norm(v), float(g_hat @ G_dir)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    UB = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)

    G_lab, z_lab = lab_vectors_from_UB(hkl, lattice, UB)
    orders = [-3, -2, -1, 0, 1, 2, 3]
    col, row = predict_satellite_pixels(G_lab, z_lab, period_angstrom, orders, geom)

    boxsize = 60
    pcol, prow = direction_to_pixel(kf_hat(G_lab, LAB_KI), geom)
    row0, col0 = prow - boxsize, pcol - boxsize

    # pos_along_axis is the signed projection onto the satellite axis, measured
    # from SL0 — that is what the monochromatic and analytic routes consume.
    col, row = np.asarray(col), np.asarray(row)
    axis = np.array([col[-1] - col[0], row[-1] - row[0]])
    axis /= np.linalg.norm(axis)
    zero = orders.index(0)

    peaks = [
        {'order': n,
         'pos_along_axis': float(np.array([c - col[zero], r - row[zero]]) @ axis),
         'amplitude': 100.0, 'fwhm': 4.0, 'sigma': 1.7,
         'position_2d': (r - row0, c - col0)}
        for n, c, r in zip(orders, col, row)
    ]

    if verbose:
        print(f'[DEMO] true period = {period_angstrom:.2f} A '
              f'= {period_angstrom / 10:.2f} nm')
        print('[DEMO] the orientation is invented (azimuth 75 deg): the geometry is\n'
              '       self-consistent, but nothing here is comparable to the real '
              'sample.')
        check_geometry(geom, UB, hkl=hkl, lattice=lattice,
                       measured_parent_px=(pcol, prow))

    out = period_from_peaks(
        peaks, crop_origin_px=(row0, col0), geom=geom, UB=UB,
        hkl=hkl, lattice=lattice, wavelength_angstrom=lam,
    )
    rec = out['forward']['period_angstrom']
    print(f'\n[DEMO] recovered {rec:.3f} A  '
          f'(error {100 * (rec / period_angstrom - 1):+.4f} %)')
    return out


if __name__ == '__main__':
    demo_synthetic()
