"""Superlattice period from satellite peak positions — three routes.

The setup is polychromatic (white/pink-beam Laue) but the legacy formula is
monochromatic, so both are available and selected by ``method=``:

    'monochromatic'   default, legacy, numerically frozen by a regression test
    'laue_analytic'   small-angle; known +0.94 % bias at first order, cross-check only
    'laue_forward'    exact forward model fitted in (2θ, χ); recommended Laue route

`compare_methods` runs all three side by side without choosing between them.

The Laue routes are **not validated against experiment** — do not use them to
revise reported values before confirming with the beamline team. See
``README.md`` for the routes and their validation status.
"""

from __future__ import annotations

import math

import numpy as np
from typing import Any, Dict, List, Optional

from laue.satellite.geometry import (
    DetectorGeometry,
    LAB_KI,
    _laue_geometry,
    angular_separation,
    assert_no_branch_crossing,
    assert_reflection_observable,
    dihedral_phi,
    direction_to_2theta_chi,
    gamma_from_vectors,
    kf_hat,
    lab_vectors_from_UB,
    pixel_to_angle,
    pixels_to_2theta_chi,
    predict_satellite_angles,
    theta_from_G,
    train_direction_delta_deg,
    two_theta_chi_to_direction,
)



# ── Layer period from angular separation ──────────────────────────────────────

def _layer_period_monochromatic(
    peaks: List[Dict[str, Any]],
    *,
    pixel_size_mm: float,
    detector_distance_mm: float,
    wavelength_angstrom: Optional[float] = None,
    energy_kev: Optional[float] = None,
    two_theta_0_deg: float = 0.0,
    chi_deg: float = 0.0,
) -> Dict[str, Any]:
    """Estimate the MQW superlattice period Λ from consecutive satellite spacings.

    LEGACY PATH — monochromatic geometry.  Reached via
    ``layer_period_from_peaks(..., method='monochromatic')``, which is the
    default.  The arithmetic here is frozen: ``test_monochromatic_path_unchanged``
    is a regression guard against any change to its numerical output.

    For each pair of adjacent orders (n, n+1), the pixel distance along the
    strip axis is converted to a scattering-vector increment Δq via the flat-
    detector geometry, and then to a real-space period Λ = 2π / Δq.

    Physics
    -------
    Flat-detector projection (coordinate Jacobian):
        Δ(2θ) = arctan(Δx_mm · cos²(2θ₀) / D)
    The cos²(2θ₀) term is a purely geometric coordinate transformation
    converting a metric displacement on the flat detector into an angular
    separation.  It is NOT the Lorentz factor, which is an intensity
    correction (dwell time of the reciprocal node on the Ewald sphere).
    Scattering-vector increment (projected onto Q):
        Δq_proj = (4π/λ) · cos(θ₀) · Δθ,    Δθ = Δ(2θ) / 2
    Off-axis correction:
        The superlattice wavevector 2π/Λ points along the true growth axis
        ẑ, not along Q in general. Only its projection onto Q is observable
        as a magnitude shift (first-order perturbation |Q+δQ| ≈ |Q| + Q̂·δQ):
            Δq_proj = (2π/Λ) · cos(γ),   γ = angle(Q, ẑ)
        so the growth-axis Δq is recovered as Δq = Δq_proj / cos(γ), with
        γ given by ``chi_deg`` (the angle, in the sample's own frame, between
        this reflection's Q and the growth axis — e.g. as reported by the
        Laue indexation for this spot). γ = 0 recovers the symmetric-
        reflection case (Q ∥ growth axis).
    Layer period:
        Λ = 2π / Δq

    The 2θ₀ correction terms vanish for 2θ₀ = 0 (transmission / near-forward
    scattering), giving the small-angle approximation Λ ≈ λ·D / Δx_mm (before
    the chi_deg correction).

    Assumes the satellite axis on the detector is the projection of the
    growth axis (out-of-plane MQW geometry) — this holds independently of
    chi_deg, which only corrects the Q-projection of the resulting Δq.

    Parameters
    ----------
    peaks : list of peak dicts from detect_satellites()['peaks'].
        Must contain at least two peaks at consecutive integer orders.
    pixel_size_mm : float
        Detector pixel size in mm (e.g. 0.075 for Eiger 4M at ESRF).
    detector_distance_mm : float
        Sample-to-detector distance in mm.
    wavelength_angstrom : float, optional
        X-ray wavelength in Angstroms.  Provide either this or ``energy_kev``.
    energy_kev : float, optional
        X-ray energy in keV (e.g. 17.06 keV).  Converted to Å via
        λ = 12.3984 / E.  Takes precedence over ``wavelength_angstrom``.
    two_theta_0_deg : float
        Approximate 2θ of the SL0 reflection in degrees.  Defaults to 0
        (small-angle / forward-scattering approximation).
    chi_deg : float
        Angle between this reflection's scattering vector Q and the true
        MQW growth axis (degrees). Defaults to 0 (symmetric reflection,
        Q ∥ growth axis). Divides the observed Δq by cos(chi_deg) to
        recover the true growth-axis periodicity for tilted/asymmetric
        reflections.

    Returns
    -------
    dict with keys
        'period_angstrom'  — Λ averaged over all consecutive pairs (Å)
        'period_nm'        — same in nm
        'delta_q_inv_ang'  — mean Δq in Å⁻¹
        'per_pair'         — list of dicts, one per adjacent pair::

                             {'orders': (n1, n2),
                              'delta_px': float,
                              'period_angstrom': float,
                              'period_nm': float,
                              'delta_q_inv_ang': float}

    All float values are NaN if fewer than two consecutive peaks are found.
    """
    if energy_kev is not None:
        wavelength_angstrom = 12.3984 / energy_kev
    if wavelength_angstrom is None:
        raise ValueError("Provide either wavelength_angstrom or energy_kev.")

    nan_result: Dict[str, Any] = {
        'period_angstrom':  float('nan'),
        'period_nm':        float('nan'),
        'delta_q_inv_ang':  float('nan'),
        'per_pair':         [],
        'method':           'monochromatic',
    }

    sorted_peaks = sorted(peaks, key=lambda p: p['order'])
    if len(sorted_peaks) < 2:
        return nan_result

    two_theta_0_rad = math.radians(two_theta_0_deg)
    theta_0_rad     = two_theta_0_rad / 2.0
    cos_sq_2t0      = math.cos(two_theta_0_rad) ** 2
    cos_t0          = math.cos(theta_0_rad)
    lam             = wavelength_angstrom  # Å
    cos_chi         = math.cos(math.radians(chi_deg))

    per_pair: List[Dict[str, Any]] = []
    for p1, p2 in zip(sorted_peaks, sorted_peaks[1:]):
        n1, n2 = p1['order'], p2['order']
        if n2 - n1 != 1:
            continue  # skip non-consecutive pairs
        if n1 == 0 or n2 == 0:
            continue  # SL0 is the main Bragg peak — exclude from period pairs

        delta_px  = abs(p2['pos_along_axis'] - p1['pos_along_axis'])
        delta_mm  = delta_px * pixel_size_mm

        # Flat-detector projection (coordinate Jacobian): converts a metric
        # displacement Δx on the detector plane into an angular separation Δ(2θ).
        # Purely geometric — NOT the Lorentz factor, which is an intensity
        # correction (dwell time of the reciprocal node on the Ewald sphere).
        delta_2theta = math.atan(delta_mm * cos_sq_2t0 / detector_distance_mm)
        delta_theta  = delta_2theta / 2.0

        delta_q_proj = (4.0 * math.pi / lam) * cos_t0 * delta_theta
        # Recover the growth-axis Δq from its projection onto Q (see Physics
        # note above) — cos_chi == 0 (Q ⊥ growth axis) leaves it undefined.
        delta_q = (delta_q_proj / cos_chi) if cos_chi != 0 else float('nan')
        period  = (2.0 * math.pi / delta_q) if delta_q > 0 else float('nan')

        per_pair.append({
            'orders':           (n1, n2),
            'delta_px':         delta_px,
            'period_angstrom':  period,
            'period_nm':        period / 10.0 if math.isfinite(period) else float('nan'),
            'delta_q_inv_ang':  delta_q,
        })

    if not per_pair:
        return nan_result

    valid_periods = [p['period_angstrom'] for p in per_pair if math.isfinite(p['period_angstrom'])]
    valid_dq      = [p['delta_q_inv_ang'] for p in per_pair if math.isfinite(p['delta_q_inv_ang'])]

    mean_period = float(np.mean(valid_periods)) if valid_periods else float('nan')
    mean_dq     = float(np.mean(valid_dq))      if valid_dq      else float('nan')

    return {
        'period_angstrom':  mean_period,
        'period_nm':        mean_period / 10.0 if math.isfinite(mean_period) else float('nan'),
        'delta_q_inv_ang':  mean_dq,
        'per_pair':         per_pair,
        'method':           'monochromatic',
    }


def _observed_pairs(peaks: List[Dict[str, Any]]) -> List[tuple]:
    """Consecutive satellite pairs (n1, n2, Δpx), SL0 excluded — as in the mono path."""
    sorted_peaks = sorted(peaks, key=lambda p: p['order'])
    pairs = []
    for p1, p2 in zip(sorted_peaks, sorted_peaks[1:]):
        n1, n2 = p1['order'], p2['order']
        if n2 - n1 != 1 or n1 == 0 or n2 == 0:
            continue
        pairs.append((n1, n2, abs(p2['pos_along_axis'] - p1['pos_along_axis'])))
    return pairs


# ── Route A — analytic, small-angle ───────────────────────────────────────────

def _period_laue_analytic(
    peaks, *, pixel_size_mm, detector_distance_mm, wavelength_angstrom,
    hkl, lattice, satellite_axis_psi_deg, phi_deg=None, UB=None, u_hat=None,
) -> Dict[str, Any]:
    """Λ = λ·S·sin γ / (sin θ · Δψ_det).

    Carries a known +0.94 % bias at first order (it drops the q·cos γ term of the
    exact tangent expression) and the bias grows with order index.  Cross-check
    only — ``laue_forward`` is the correct route.
    """
    geo = _laue_geometry(hkl, lattice, wavelength_angstrom)
    lam = wavelength_angstrom

    if UB is not None:
        # B before UB — the stored UB is a pure rotation and carries no metric.
        G0_lab, z_lab = lab_vectors_from_UB(hkl, lattice, UB)
        u_vec = LAB_KI if u_hat is None else np.asarray(u_hat, dtype=float)
        phi   = dihedral_phi(G0_lab, z_lab, u_vec)
    elif phi_deg is not None:
        phi = math.radians(phi_deg)
    else:
        raise ValueError(
            "laue_analytic needs the dihedral angle φ: pass either UB + u_hat "
            "(preferred, φ is then derived) or phi_deg explicitly.  φ enters "
            "through S and cannot be defaulted."
        )

    S = math.sqrt(math.cos(geo['theta_rad']) ** 2 * math.cos(phi) ** 2
                  + math.sin(geo['theta_rad']) ** 2)

    per_pair = []
    for n1, n2, delta_px in _observed_pairs(peaks):
        dpsi = pixel_to_angle(
            delta_px,
            pixel_size_mm=pixel_size_mm,
            detector_distance_mm=detector_distance_mm,
            two_theta_deg=geo['two_theta_deg'],
            psi_deg=satellite_axis_psi_deg,
        )
        if dpsi <= 0.0:
            continue
        period = lam * S * math.sin(geo['gamma_rad']) / (math.sin(geo['theta_rad']) * dpsi)
        per_pair.append({
            'orders':           (n1, n2),
            'delta_px':         delta_px,
            'delta_psi_rad':    dpsi,
            'period_angstrom':  period,
            'period_nm':        period / 10.0,
        })

    return _summarise_laue(per_pair, 'laue_analytic', geo, extra={'S': S,
                                                                 'phi_deg': math.degrees(phi)})


def locate_sl0_from_ladder(
    peaks, *, hkl, lattice, UB, detector: DetectorGeometry, crop_origin_px,
    period_angstrom: float, u_hat=None, verbose: bool = True,
) -> Dict[str, Any]:
    """Where should SL0 be, and is the detected order-0 peak actually it?

    In an MQW the bright peak is dominated by the **bulk** Bragg reflection; the
    superlattice zero order sits on its flank, displaced by the average strain of
    the stack, and is orders of magnitude weaker.  The detection picks the bright
    peak and calls it order 0 — so ``peaks[order == 0]`` is generally the bulk,
    not SL0.

    This locates SL0 without using that peak at all:

    1. Λ is fixed by the satellite-satellite gaps (the bulk never enters).
    2. The model then **predicts** the (0, n₁) angular separation.
    3. That prediction places SL0 on the axis.
    4. The detected order-0 peak is compared against it.

    Step 2 is a genuine prediction, not a restatement: the fitted Λ comes from
    gaps that exclude SL0 entirely.

    A large ``bulk_sl0_offset_px`` is not an error — it is the bulk-to-SL0
    separation, which carries the average out-of-plane strain of the stack.

    Note on sign: for **negative** orders the predicted gaps GROW with |n|
    (the denominator |G₀| + n·q·cos γ shrinks); for positive orders they
    contract.  The canonical fixture in the spec is positive-order, so its
    "≈1.8 % contraction" reads with the opposite sign on a negative-order train.
    """
    G_lab, z_lab = lab_vectors_from_UB(hkl, lattice, UB)
    u_vec = LAB_KI if u_hat is None else np.asarray(u_hat, dtype=float)
    u_vec = u_vec / np.linalg.norm(u_vec)
    assert_reflection_observable(G_lab, u_vec, hkl, 'locate_sl0_from_ladder: ')

    sats = sorted((p for p in peaks if p['order'] != 0), key=lambda p: abs(p['order']))
    if len(sats) < 2:
        if verbose:
            print('[SL0] needs at least two satellites to set the local scale.')
        return {'sl0_pos_along_axis': float('nan'), 'confident': False,
                'reason': 'needs >= 2 satellites'}

    n1 = sats[0]['order']
    p1 = sats[0]['pos_along_axis']

    # Local scale: prefer two satellites on the SAME side of the parent, so the
    # baseline does not straddle it.  deg/px is a ratio over one span, so a wider
    # baseline still gives a valid scale — it is just less local, which matters
    # because deg/px varies across the crop with the detector obliquity.
    same_side = [p for p in sats if (p['order'] > 0) == (n1 > 0)]
    scale_pair = same_side[:2] if len(same_side) >= 2 else sats[:2]
    n2 = scale_pair[1]['order']
    p2 = scale_pair[1]['pos_along_axis']

    row0, col0 = crop_origin_px
    tt, chi = pixels_to_2theta_chi(
        [col0 + p['position_2d'][1] for p in scale_pair],
        [row0 + p['position_2d'][0] for p in scale_pair], detector)
    gap_meas_deg = float(np.degrees(angular_separation(tt[0], chi[0], tt[1], chi[1])))
    gap_meas_px = abs(p2 - p1)
    deg_per_px = gap_meas_deg / gap_meas_px if gap_meas_px > 0 else float('nan')

    # The prediction: the (0, n1) separation, from Λ fixed by the satellites.
    tt_p, chi_p = predict_satellite_angles(G_lab, z_lab, period_angstrom,
                                           (0, n1), u_vec)
    gap_pred_deg = float(np.degrees(angular_separation(
        tt_p[0], chi_p[0], tt_p[1], chi_p[1])))
    gap_pred_px = gap_pred_deg / deg_per_px

    # SL0 lies one predicted step from the nearest satellite, toward order 0.
    # That direction is read from how position trends with order across ALL
    # satellites, not from the ordering of two of them: on a two-sided train the
    # two innermost are -1 and +1, and comparing them inverts the step, putting
    # SL0 one order OUTWARD — on top of the next satellite.  Λ never uses SL0, so
    # that failure leaves the period healthy and shows up only as a marker in an
    # absurd place.
    n_arr = np.array([p['order'] for p in sats], dtype=float)
    p_arr = np.array([p['pos_along_axis'] for p in sats], dtype=float)
    trend = float(np.sum((n_arr - n_arr.mean()) * (p_arr - p_arr.mean())))
    direction = -math.copysign(1.0, trend * n1)
    sl0_pos = p1 + direction * gap_pred_px

    parent = next((p for p in peaks if p['order'] == 0), None)
    out: Dict[str, Any] = {
        'sl0_pos_along_axis': sl0_pos,
        'gap_pred_deg':       gap_pred_deg,
        'gap_pred_px':        gap_pred_px,
        'gap_meas_deg':       gap_meas_deg,
        'gap_meas_px':        gap_meas_px,
        'deg_per_px':         deg_per_px,
        'nearest_order':      n1,
    }

    if parent is not None:
        offset = parent['pos_along_axis'] - sl0_pos
        out['detected_order0_pos'] = parent['pos_along_axis']
        out['bulk_sl0_offset_px'] = offset
        out['bulk_sl0_offset_deg'] = offset * deg_per_px
        out['order0_is_sl0'] = bool(abs(offset) < 0.25 * gap_pred_px)
        out['confident'] = True
        if parent['amplitude'] > 0 and sats[0]['amplitude'] > 0:
            out['amplitude_ratio_order0_to_n1'] = (parent['amplitude']
                                                   / sats[0]['amplitude'])
    else:
        out['confident'] = False

    if verbose:
        print(f'[SL0] local scale from the ({n1:+d},{n2:+d}) gap: '
              f'{gap_meas_px:.2f} px = {gap_meas_deg:.6f} deg '
              f'-> {deg_per_px:.5f} deg/px')
        print(f'[SL0] predicted (0,{n1:+d}) gap: {gap_pred_deg:.6f} deg '
              f'= {gap_pred_px:.2f} px')
        print(f'[SL0] => SL0 should sit at pos_along_axis = {sl0_pos:+.2f} px')
        if parent is not None:
            print(f'[SL0] detected order-0 peak is at   {parent["pos_along_axis"]:+.2f} px'
                  f'   (offset {out["bulk_sl0_offset_px"]:+.2f} px = '
                  f'{out["bulk_sl0_offset_px"]/gap_pred_px:+.2f} steps)')
            if out['order0_is_sl0']:
                print('[SL0] -> the detected peak IS SL0 (bulk not resolved '
                      'separately)')
            else:
                print('[SL0] -> the detected peak is NOT SL0. It is the bulk '
                      'reflection;\n'
                      '        SL0 is buried in its flank at the position above.')
                if 'amplitude_ratio_order0_to_n1' in out:
                    print(f'        amplitude ratio order0/n{n1:+d} = '
                          f'{out["amplitude_ratio_order0_to_n1"]:.0f}x '
                          f'(compare with the satellite-to-satellite ratio;\n'
                          f'        a far larger jump means the peak is '
                          f'bulk-dominated)')
    return out


# ── Order-label sign: the arrow the detection cannot know ────────────────────

# The arrow's DIRECTION does not depend on Λ — only its length does — so any
# period in range serves to read the sign off the model.
SIGN_REFERENCE_PERIOD_A = 100.0


def resolve_order_sign(
    peaks, *, hkl, lattice, UB, detector, crop_origin_px, u_hat=None,
    parent_offset_max_deg: float = 1.0,
    train_delta_max_deg: float = 5.0,
    arrow_cos_min: float = 0.3,
) -> Dict[str, Any]:
    """Does the detection's +n point the same way as the crystal's +n?

    Detection assigns the sign of an order from the sign of ``pos_along_axis``,
    which comes from an axis folded into (-90, 90] — a detector-space convention
    with no reference to the growth direction.  Whether it agrees with the
    crystallographic +n is therefore luck, and the luck changes from reflection
    to reflection: on one sample the (105) trains come out right and a (-1,-1,6)
    comes out inverted.

    Only the model can settle it, and only when the model has earned the right to
    speak.  Three gates, all of which must pass:

    1. ``parent_offset_deg`` small — the orientation must point at *this*
       reflection.  A UB in the wrong frame would otherwise get to decide the
       sign, which is how a frame error once hid behind a plausible period.
    2. ``train_delta_deg`` small — the predicted train must lie along the
       measured line.  If the line is wrong, asking about the arrow is
       meaningless.
    3. ``|arrow_cos|`` clearly away from zero — near 90° there is no answer, and
       inventing one is worse than declining.

    The verdict is read from the cosine between the two centred configurations,
    which is the quantity the period fit itself turns on: negative means the fit
    drives Λ to a bound instead of to a period.  It is NOT read from which sign
    fits better — that would be settling a convention by residual, against the
    rule that a smaller residual is not evidence of a right convention.

    Returns a dict with ``inverted``, ``confident``, ``arrow_cos``,
    ``train_delta_deg``, ``parent_offset_deg`` and ``reason``.  ``inverted`` is
    always False when ``confident`` is False.
    """
    sats = sorted((p for p in peaks if p['order'] != 0), key=lambda p: p['order'])
    parent = next((p for p in peaks if p['order'] == 0), None)

    def _decline(reason: str) -> Dict[str, Any]:
        return {'inverted': False, 'confident': False, 'reason': reason,
                'arrow_cos': float('nan'), 'train_delta_deg': float('nan'),
                'parent_offset_deg': float('nan')}

    if len(sats) < 2:
        return _decline('needs at least two non-zero satellite orders')
    if parent is None:
        return _decline('needs the order-0 peak to verify the reflection first')

    G_lab, z_lab = lab_vectors_from_UB(hkl, lattice, UB)
    u_vec = LAB_KI if u_hat is None else np.asarray(u_hat, dtype=float)
    u_vec = u_vec / np.linalg.norm(u_vec)
    assert_reflection_observable(G_lab, u_vec, hkl, 'resolve_order_sign: ')

    row0, col0 = crop_origin_px
    orders = [p['order'] for p in sats]
    tt_meas, chi_meas = pixels_to_2theta_chi(
        [col0 + p['position_2d'][1] for p in sats],
        [row0 + p['position_2d'][0] for p in sats], detector)
    uf_meas = two_theta_chi_to_direction(tt_meas, chi_meas)
    assert_no_branch_crossing(uf_meas, 'resolve_order_sign, measured: ')

    tt_m0, chi_m0 = pixels_to_2theta_chi(col0 + parent['position_2d'][1],
                                         row0 + parent['position_2d'][0], detector)
    tt_p0, chi_p0 = direction_to_2theta_chi(kf_hat(G_lab, u_vec))
    parent_offset = float(np.degrees(angular_separation(
        tt_p0, chi_p0, float(tt_m0[0]), float(chi_m0[0]))))

    tt_pred, chi_pred = predict_satellite_angles(
        G_lab, z_lab, SIGN_REFERENCE_PERIOD_A, orders, u_vec)
    uf_pred = two_theta_chi_to_direction(tt_pred, chi_pred)
    train_delta = float(train_direction_delta_deg(
        tt_pred, chi_pred, tt_meas, chi_meas))

    P = uf_pred - uf_pred.mean(axis=0)
    M = uf_meas - uf_meas.mean(axis=0)
    denom = float(np.linalg.norm(P) * np.linalg.norm(M))
    arrow_cos = float(np.sum(P * M) / denom) if denom > 0 else float('nan')

    out: Dict[str, Any] = {
        'inverted':          bool(arrow_cos < 0.0),
        'confident':         True,
        'reason':            '',
        'arrow_cos':         arrow_cos,
        'train_delta_deg':   train_delta,
        'parent_offset_deg': parent_offset,
    }

    if not math.isfinite(arrow_cos):
        out.update(inverted=False, confident=False,
                   reason='degenerate configuration: no arrow to compare')
    elif parent_offset > parent_offset_max_deg:
        out.update(inverted=False, confident=False, reason=(
            f'parent_offset {parent_offset:.3f} deg > {parent_offset_max_deg} deg: '
            f'the orientation does not point at this reflection, so it cannot '
            f'decide the order sign — check the UB frame first'))
    elif train_delta > train_delta_max_deg:
        out.update(inverted=False, confident=False, reason=(
            f'train_delta {train_delta:.3f} deg > {train_delta_max_deg} deg: the '
            f'predicted train is not along the measured line, so the arrow '
            f'question does not apply'))
    elif abs(arrow_cos) < arrow_cos_min:
        out.update(inverted=False, confident=False, reason=(
            f'|arrow_cos| {abs(arrow_cos):.3f} < {arrow_cos_min}: the two trains '
            f'are near perpendicular and the sign is not determined'))
    return out


def resolve_and_apply_order_sign(peaks, **kwargs):
    """``resolve_order_sign`` plus the relabelling, in one place.

    Both callers (``run_single_image`` and ``scan_pipeline``) go through this so
    the correction cannot be applied in one and forgotten in the other.  Returns
    ``(peaks, verdict)``; the peaks are the same objects when nothing changed.

    Only the labels move — positions are untouched.  Everything downstream must
    see the corrected labels, not just the period: the period announces an
    inverted sign by railing against a bound, but
    ``asymmetry_intensity``/``asymmetry_position`` would simply come out with the
    opposite sign and say nothing.
    """
    verdict = resolve_order_sign(peaks, **kwargs)
    if verdict['confident'] and verdict['inverted']:
        peaks = [dict(p, order=-p['order']) for p in peaks]
    return peaks, verdict


def _amplitude_envelope_excess(parent, sats) -> float:
    """How far the order-0 amplitude sits above the satellite envelope.

    Consecutive satellite orders fall by a roughly constant factor (I_n ~ e^-α|n|).
    Extrapolating that factor one step past the nearest satellite predicts what an
    order-0 peak would be if it were the superlattice zero order.  A large excess
    is independent evidence that the peak is bulk-dominated — independent because
    it uses intensities, while the ladder test uses positions.

    Returns NaN when the amplitudes cannot support the extrapolation.
    """
    if len(sats) < 2:
        return float('nan')
    a0 = float(parent.get('amplitude', 0.0))
    a1 = float(sats[0].get('amplitude', 0.0))
    a2 = float(sats[1].get('amplitude', 0.0))
    if min(a0, a1, a2) <= 0.0:
        return float('nan')
    return a0 / (a1 * (a1 / a2))


def check_order_assignment(
    peaks, *, hkl, lattice, UB, detector: DetectorGeometry, crop_origin_px,
    period_angstrom: float, u_hat=None, max_hidden: int = 3, verbose: bool = True,
) -> Dict[str, Any]:
    """Test whether the satellite nearest SL0 really carries order ±1.

    Why this matters only for the Laue route: the monochromatic period came from
    the *slope* of position vs order, which is invariant under a constant shift
    of the order index — mislabelling every satellite by one changed nothing.
    The forward model predicts **non-uniform** spacing, so the absolute index now
    enters, and one hidden order costs roughly 2 % in Λ.

    Why the spacing ratio cannot answer it: the contraction rate is almost
    constant along the train (it moves ~0.0002 per order), so the pattern looks
    the same wherever it starts.  With only two or three gaps the information is
    not in the data.

    What does answer it: the **SL0 → nearest satellite** gap, which differs by a
    factor of ~2 between "the first satellite is ±1" and "one order is hidden in
    the parent".  SL0 is excluded from the period pairs because the bulk Bragg
    contribution biases its centroid, but that bias is pixels — irrelevant to a
    factor-of-two question.

    The headline number is ``steps_from_sl0``: how many satellite steps separate
    the **detected order-0 peak** from the nearest detected satellite.  Read it
    directly —

        ~1.0  the labelling is right, nothing is hidden
        ~2.0  one order is buried in the parent
        ~0.5  the peak is not on the ladder at all: it is the **bulk** reflection

    That last case is the normal one for an MQW and is why this function no
    longer assumes the order-0 peak is SL0.  A half-integer cannot be produced by
    a hidden order — hidden orders move the peak by whole steps.  It means the
    bright peak belongs to the bulk Bragg reflection, with SL0 buried in its
    flank, displaced by the average out-of-plane strain of the stack.  The
    displacement is unconstrained, so **the hidden-order question is then not
    decidable from the gap** and ``implied_offset`` is returned as ``None``
    rather than as a number that would be misread.  ``locate_sl0_from_ladder``
    is called to say where SL0 actually is; its result is returned under ``sl0``.

    ``verdict`` is the machine-readable form: ``'labels_correct'``,
    ``'hidden_order'``, ``'order0_off_ladder'`` (the bulk case) or
    ``'insufficient_data'``.

    ``period_angstrom`` should be the **fitted** Λ, not a guess: the predicted
    gaps scale as 1/Λ.  The mild circularity (Λ was fitted under an assumed
    labelling) is harmless here, because the two labellings differ by ~2 % in Λ
    while the hypotheses they discriminate differ by ~100 %.
    """
    parent = next((p for p in peaks if p['order'] == 0), None)
    sats = sorted((p for p in peaks if p['order'] != 0),
                  key=lambda p: abs(p['order']))
    if parent is None or not sats:
        if verbose:
            print('[order check] needs an order-0 peak and at least one '
                  'satellite — nothing to compare.')
        return {'implied_offset': None, 'steps_from_sl0': float('nan'),
                'confident': False, 'verdict': 'insufficient_data',
                'reason': 'needs an order-0 peak and at least one satellite'}

    G_lab, z_lab = lab_vectors_from_UB(hkl, lattice, UB)
    u_vec = LAB_KI if u_hat is None else np.asarray(u_hat, dtype=float)
    u_vec = u_vec / np.linalg.norm(u_vec)
    assert_reflection_observable(G_lab, u_vec, hkl, 'check_order_assignment: ')

    row0, col0 = crop_origin_px
    tt, chi = pixels_to_2theta_chi(
        [col0 + parent['position_2d'][1], col0 + sats[0]['position_2d'][1]],
        [row0 + parent['position_2d'][0], row0 + sats[0]['position_2d'][0]],
        detector)
    measured_gap = float(np.degrees(angular_separation(tt[0], chi[0], tt[1], chi[1])))

    sign = 1 if sats[0]['order'] > 0 else -1
    predicted = {}
    for hidden in range(max_hidden + 1):
        n_true = sign * (1 + hidden)
        tt_p, chi_p = predict_satellite_angles(
            G_lab, z_lab, period_angstrom, (0, n_true), u_vec)
        predicted[hidden] = float(np.degrees(angular_separation(
            tt_p[0], chi_p[0], tt_p[1], chi_p[1])))

    # Where does the measured gap fall, in units of "satellite steps from SL0"?
    # Interpolate between the cumulative predicted gaps rather than dividing by
    # the first one: the steps are not exactly equal (that is the whole point of
    # the polychromatic model).
    ladder = [0.0] + [predicted[h] for h in sorted(predicted)]
    steps = float('nan')
    for k in range(len(ladder) - 1):
        if ladder[k] <= measured_gap <= ladder[k + 1]:
            span = ladder[k + 1] - ladder[k]
            steps = k + ((measured_gap - ladder[k]) / span if span > 0 else 0.0)
            break
    else:
        steps = float(len(ladder) - 1) if measured_gap > ladder[-1] else 0.0

    nearest_int = round(steps)
    distance_to_int = abs(steps - nearest_int)

    # A hidden order displaces the peak by a WHOLE step, so only a near-integer
    # result is a statement about labelling.  Anything else means the peak is not
    # on the superlattice ladder — in an MQW that is the bulk reflection (§4.13).
    on_ladder = bool(distance_to_int < 0.25 and nearest_int >= 1)

    out = {
        'steps_from_sl0':   steps,
        'implied_offset':   (nearest_int - 1) if on_ladder else None,
        'confident':        on_ladder,
        'verdict':          ('labels_correct' if on_ladder and nearest_int == 1
                             else 'hidden_order' if on_ladder
                             else 'order0_off_ladder'),
        'order0_is_sl0':    on_ladder and nearest_int == 1,
        'measured_gap_deg': measured_gap,
        'predicted_gaps':   predicted,
        'nearest_order':    sats[0]['order'],
    }

    # Delegate the "then where IS SL0?" question — it needs two satellites to fix
    # the local pixel scale, and it never touches the order-0 peak.
    if len(sats) >= 2:
        out['sl0'] = locate_sl0_from_ladder(
            peaks, hkl=hkl, lattice=lattice, UB=UB, detector=detector,
            crop_origin_px=crop_origin_px, period_angstrom=period_angstrom,
            u_hat=u_hat, verbose=False)
        out['amplitude_excess'] = _amplitude_envelope_excess(parent, sats)

    if verbose:
        print(f'[order check]  detected order-0 peak -> nearest satellite '
              f'(labelled {sats[0]["order"]:+d})')
        print(f'  measured gap : {measured_gap:.4f} deg')
        print(f'  predicted for Lambda = {period_angstrom:.2f} A:')
        for h in sorted(predicted):
            tag = 'labelling is right' if h == 0 else f'{h} hidden order(s)'
            print(f'    {h + 1} step(s) = {predicted[h]:.4f} deg   ({tag})')
        print(f'  -> the satellite sits {steps:.2f} steps from the order-0 peak')
        if on_ladder:
            print(f'  -> CONFIDENT: offset = {out["implied_offset"]}'
                  + ('  (nothing hidden; the order-0 peak is SL0)'
                     if out['implied_offset'] == 0
                     else f'  ({out["implied_offset"]} order(s) buried in SL0)'))
        else:
            print(f'  -> the order-0 peak is NOT ON THE LADDER '
                  f'({distance_to_int:.2f} steps from the nearest integer).')
            print('     A hidden order would displace it by a WHOLE step, so this '
                  'is not a\n     labelling problem: the bright peak is the bulk '
                  'Bragg reflection and SL0\n     is buried in its flank. The '
                  'hidden-order question cannot be answered\n     from this gap '
                  '— implied_offset is None on purpose.')
            if 'sl0' in out and np.isfinite(out['sl0']['sl0_pos_along_axis']):
                print(f'     SL0 is predicted at pos_along_axis = '
                      f'{out["sl0"]["sl0_pos_along_axis"]:+.2f} px, i.e. '
                      f'{out["sl0"]["bulk_sl0_offset_px"]:+.2f} px from the '
                      f'bright peak\n     (that offset is the average '
                      f'out-of-plane strain, not an error).')
            if np.isfinite(out.get('amplitude_excess', float('nan'))):
                print(f'     independent check: the order-0 amplitude is '
                      f'{out["amplitude_excess"]:.0f}x above the satellite '
                      f'envelope.')

    return out


def _period_laue_forward(
    peaks, *, wavelength_angstrom, hkl, lattice, detector, UB, crop_origin_px,
    u_hat=None, period_bounds_angstrom=(50.0, 200.0),
) -> Dict[str, Any]:
    """Fit Λ to the measured satellite pixel positions through the exact forward model.

    Exact — no small-angle approximation and no ±1 % bias — and nothing
    geometric is supplied by hand: γ and |G₀| come from (hkl) + lattice, 2θ from
    the pixel position via the ``.det`` calibration, and the satellite axis
    direction is an output rather than an input.  γ, θ and |G₀| are NOT
    re-fitted; Λ is the single free parameter, so this is a well-conditioned 1-D
    minimisation.

    The fit is on positions referred to the centroid of the detected satellites,
    which removes the absolute registration between prediction and measurement
    without adding a nuisance parameter and without leaning on SL0, whose fitted
    position is contaminated by the bulk Bragg contribution.  The absolute
    offset is reported separately as ``parent_offset_px`` — that is the §3.3
    reprojection residual, and it is a diagnostic of the UB and the tilts, not
    of Λ.
    """
    from scipy.optimize import minimize_scalar

    if detector is None or UB is None:
        raise ValueError(
            "laue_forward requires detector=DetectorGeometry(...) and UB (the "
            "orientation matrix from the .det file).  Both are needed to project "
            "the model onto the detector; neither is defaulted, since a wrong "
            "orientation or calibration silently rescales Λ."
        )
    if crop_origin_px is None:
        raise ValueError(
            "laue_forward requires crop_origin_px=(row0, col0), the absolute "
            "detector pixel of the crop corner, to place the measured peaks in "
            "the detector frame."
        )

    G_lab, z_lab = lab_vectors_from_UB(hkl, lattice, UB)
    gamma = gamma_from_vectors(G_lab, z_lab)
    if abs(math.sin(gamma)) < 1e-12:
        h, k, l = hkl
        raise ValueError(
            f"symmetric reflection ({h}{k}{l}): γ = 0, satellites are degenerate "
            f"in Laue geometry — they land on the same detector pixel and no "
            f"period can be extracted."
        )

    u_vec = LAB_KI if u_hat is None else np.asarray(u_hat, dtype=float)
    u_vec = u_vec / np.linalg.norm(u_vec)
    assert_reflection_observable(G_lab, u_vec, hkl, 'laue_forward: ')

    row0, col0 = crop_origin_px
    sats = sorted((p for p in peaks if p['order'] != 0), key=lambda p: p['order'])
    if len(sats) < 2:
        raise ValueError(
            f"laue_forward needs at least two non-zero satellite orders, got "
            f"{len(sats)}."
        )
    orders   = [p['order'] for p in sats]
    meas_col = np.array([col0 + p['position_2d'][1] for p in sats], dtype=float)
    meas_row = np.array([row0 + p['position_2d'][0] for p in sats], dtype=float)

    # Measurement enters angular space here, through LaueTools and the .det.
    tt_meas, chi_meas = pixels_to_2theta_chi(meas_col, meas_row, detector)
    uf_meas = two_theta_chi_to_direction(tt_meas, chi_meas)
    assert_no_branch_crossing(uf_meas, 'measured satellites: ')
    uf_meas_c = uf_meas - uf_meas.mean(axis=0)

    def _predicted_dirs(period: float) -> np.ndarray:
        tt, chi = predict_satellite_angles(G_lab, z_lab, period, orders, u_vec)
        return two_theta_chi_to_direction(tt, chi)

    def residual(period: float) -> float:
        p = _predicted_dirs(period)
        return float(np.sum((p - p.mean(axis=0) - uf_meas_c) ** 2))

    res = minimize_scalar(residual, bounds=period_bounds_angstrom, method='bounded')
    period_fit = float(res.x)

    # A fit that rails against a bound is not a measurement — it means the model
    # cannot reproduce the observed spacing at any period in range, usually
    # because the UB or the calibration is wrong.  Surface it rather than
    # returning the bound as if it were a result.
    lo, hi = period_bounds_angstrom
    span = hi - lo
    at_bound = (period_fit - lo) < 1e-3 * span or (hi - period_fit) < 1e-3 * span

    tt_pred, chi_pred = predict_satellite_angles(G_lab, z_lab, period_fit, orders, u_vec)
    uf_pred = two_theta_chi_to_direction(tt_pred, chi_pred)
    resid_ang = np.degrees(np.linalg.norm(
        uf_pred - uf_pred.mean(axis=0) - uf_meas_c, axis=1))

    # Reprojection of the parent reflection, in angular space: predicted (2θ, χ)
    # of the indexed reflection against the measured SL0.  Diagnostic of the UB
    # and the calibration, not of Λ.
    parent = next((p for p in peaks if p['order'] == 0), None)
    if parent is not None:
        tt_p0, chi_p0 = direction_to_2theta_chi(kf_hat(G_lab, u_vec))
        tt_m0, chi_m0 = pixels_to_2theta_chi(col0 + parent['position_2d'][1],
                                             row0 + parent['position_2d'][0],
                                             detector)
        tt_parent, chi_parent = float(tt_m0[0]), float(chi_m0[0])
        parent_offset_deg = float(np.degrees(angular_separation(
            tt_p0, chi_p0, tt_parent, chi_parent)))
    else:
        tt_parent = chi_parent = parent_offset_deg = float('nan')

    theta_rad = theta_from_G(float(np.linalg.norm(G_lab)), wavelength_angstrom)

    # Consecutive separations, always with the exact spherical formula.
    per_pair = []
    for k in range(len(orders) - 1):
        if orders[k + 1] - orders[k] != 1:
            continue
        per_pair.append({
            'orders':          (orders[k], orders[k + 1]),
            'sep_meas_deg':    float(np.degrees(angular_separation(
                tt_meas[k], chi_meas[k], tt_meas[k + 1], chi_meas[k + 1]))),
            'sep_pred_deg':    float(np.degrees(angular_separation(
                tt_pred[k], chi_pred[k], tt_pred[k + 1], chi_pred[k + 1]))),
            'period_angstrom': period_fit,
            'period_nm':       period_fit / 10.0,
        })

    per_order = [
        {'order': n, 'two_theta_meas': float(tm), 'chi_meas': float(cm),
         'two_theta_pred': float(tp), 'chi_pred': float(cp),
         'residual_deg': float(rd)}
        for n, tm, cm, tp, cp, rd in zip(orders, tt_meas, chi_meas,
                                         tt_pred, chi_pred, resid_ang)
    ]

    return {
        'period_angstrom':       period_fit,
        'period_nm':             period_fit / 10.0,
        'delta_q_inv_ang':       2.0 * np.pi / period_fit,
        'per_pair':              per_pair,
        'per_order':             per_order,
        'method':                'laue_forward',
        'gamma_deg':             math.degrees(gamma),
        'theta_deg':             math.degrees(theta_rad),
        'two_theta_deg':         2.0 * math.degrees(theta_rad),
        # Parent 2θ/χ — the quantity comparable with two_theta_deg from indexing.
        # NOT the satellite mean: the satellites sit ~|n| steps along the train,
        # so their mean is offset from the parent by design, and comparing it
        # against indexing reads as a discrepancy that is not one.
        'two_theta_measured':    tt_parent,
        'chi_measured':          chi_parent,
        'two_theta_satellites':  float(np.mean(tt_meas)),
        'chi_satellites':        float(np.mean(chi_meas)),
        'G_magnitude':           float(np.linalg.norm(G_lab)),
        'fit_rms_deg':           float(np.sqrt(np.mean(resid_ang ** 2))),
        'fit_success':           bool(res.success) and not at_bound,
        'fit_at_bound':          bool(at_bound),
        'period_bounds_angstrom': (float(lo), float(hi)),
        'parent_offset_deg':     parent_offset_deg,
        'train_delta_deg':       train_direction_delta_deg(
            tt_pred, chi_pred, tt_meas, chi_meas),
        'orders_used':           orders,
    }


def _summarise_laue(per_pair, method: str, geo: Dict[str, float],
                    extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    valid = [p['period_angstrom'] for p in per_pair
             if math.isfinite(p['period_angstrom'])]
    mean_period = float(np.mean(valid)) if valid else float('nan')
    out: Dict[str, Any] = {
        'period_angstrom':  mean_period,
        'period_nm':        mean_period / 10.0 if math.isfinite(mean_period) else float('nan'),
        'delta_q_inv_ang':  (2.0 * np.pi / mean_period) if math.isfinite(mean_period)
                            and mean_period > 0 else float('nan'),
        'per_pair':         per_pair,
        'method':           method,
        'gamma_deg':        geo['gamma_deg'],
        'theta_deg':        geo['theta_deg'],
        'G_magnitude':      geo['G_magnitude'],
    }
    if extra:
        out.update(extra)
    return out


# ── Public dispatcher ─────────────────────────────────────────────────────────

_METHODS = ('monochromatic', 'laue_analytic', 'laue_forward')


def layer_period_from_peaks(
    peaks: List[Dict[str, Any]],
    *,
    pixel_size_mm: Optional[float] = None,
    detector_distance_mm: Optional[float] = None,
    wavelength_angstrom: Optional[float] = None,
    energy_kev: Optional[float] = None,
    two_theta_0_deg: float = 0.0,
    chi_deg: float = 0.0,
    method: str = 'monochromatic',
    hkl: Optional[tuple] = None,
    lattice: Optional[tuple] = None,
    UB: Optional[np.ndarray] = None,
    u_hat: Optional[np.ndarray] = None,
    phi_deg: Optional[float] = None,
    satellite_axis_psi_deg: Optional[float] = None,
    detector: Optional[DetectorGeometry] = None,
    crop_origin_px: Optional[tuple] = None,
    period_bounds_angstrom: tuple = (50.0, 200.0),
) -> Dict[str, Any]:
    """Estimate the MQW superlattice period Λ, by one of three routes.

    ``method='monochromatic'`` (default) is the legacy path and is unchanged.
    The two Laue routes are NOT validated against experiment and must not be
    used to revise reported values until confirmed with the beamline team.

    Methods
    -------
    'monochromatic' : flat-detector monochromatic geometry.  Reads the q·cos γ
        component.  Uses ``two_theta_0_deg`` and ``chi_deg``; ignores all the
        Laue-only parameters.  Numerically frozen — see
        ``test_monochromatic_path_unchanged``.
    'laue_analytic' : Λ = λ·S·sin γ / (sin θ · Δψ_det).  Small-angle; carries a
        known +0.94 % bias at first order that grows with order index.
        Cross-check only.
    'laue_forward' : exact forward model fitted in PIXEL space through the
        LaueTools ``.det`` calibration.  Recommended Laue route — no small-angle
        approximation, and nothing geometric supplied by hand: γ and |G₀| from
        (hkl) + lattice, 2θ from the pixel position, and the satellite axis
        direction as an OUTPUT of the model rather than an input.

    Laue-only parameters
    --------------------
    hkl, lattice : required by both Laue routes.  γ and |G₀| are DERIVED from
        these, never defaulted — ``two_theta_0_deg`` and ``chi_deg`` are ignored
        by the Laue routes on purpose.  Passing LaueTools' detector-space
        ``chi`` where the crystallographic γ is expected is a known source of
        error.
    detector, UB, crop_origin_px : required by 'laue_forward'.  ``detector`` is a
        :class:`DetectorGeometry` (five ``.det`` parameters plus pixel size),
        ``UB`` the orientation matrix from the same file, and ``crop_origin_px``
        the ``(row0, col0)`` absolute pixel of the crop corner, which places the
        measured peaks in the detector frame.
    u_hat : incident beam direction.  Defaults to ``LAB_KI`` = [0, 1, 0], the
        LaueTools laboratory convention; override only for a different frame.
    phi_deg, satellite_axis_psi_deg, pixel_size_mm, detector_distance_mm :
        used by 'laue_analytic' only.  The forward route derives all of them.

    Returns
    -------
    dict — always carries ``'method'`` recording which route produced the value,
    so downstream results stay traceable.  Laue routes add ``gamma_deg``,
    ``theta_deg`` and ``G_magnitude``; ``laue_forward`` adds ``fit_rms_rad``.

    Raises
    ------
    ValueError
        For an unknown method, for missing Laue parameters, and for a symmetric
        reflection (γ = 0) where the Laue geometry is physically singular — the
        satellites are degenerate and no period exists to be extracted.  This is
        raised rather than returned as NaN.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}. Expected one of {_METHODS}.")

    if method == 'monochromatic':
        if pixel_size_mm is None or detector_distance_mm is None:
            if detector is None:
                raise ValueError(
                    "monochromatic needs pixel_size_mm and detector_distance_mm, "
                    "or a detector=DetectorGeometry(...) to take them from."
                )
            pixel_size_mm        = detector.pixelsize
            detector_distance_mm = detector.dd
        return _layer_period_monochromatic(
            peaks,
            pixel_size_mm=pixel_size_mm,
            detector_distance_mm=detector_distance_mm,
            wavelength_angstrom=wavelength_angstrom,
            energy_kev=energy_kev,
            two_theta_0_deg=two_theta_0_deg,
            chi_deg=chi_deg,
        )

    if energy_kev is not None:
        wavelength_angstrom = 12.3984 / energy_kev
    if wavelength_angstrom is None:
        raise ValueError("Provide either wavelength_angstrom or energy_kev.")

    if method == 'laue_forward':
        return _period_laue_forward(
            peaks,
            wavelength_angstrom=wavelength_angstrom,
            hkl=hkl, lattice=lattice,
            detector=detector, UB=UB, crop_origin_px=crop_origin_px,
            u_hat=u_hat, period_bounds_angstrom=period_bounds_angstrom,
        )

    # laue_analytic — small-angle cross-check, still needs ψ and φ by hand
    if satellite_axis_psi_deg is None:
        raise ValueError(
            "laue_analytic requires satellite_axis_psi_deg (angle between the "
            "satellite axis and the local radial direction on the detector). "
            "Use psi_from_geometry() to compute it, or pass 0.0 explicitly to "
            "declare a radial axis.  It is not defaulted because the pixel→angle "
            "scale depends on it.  Prefer method='laue_forward', where the axis "
            "direction is an output of the model and this parameter disappears."
        )
    if pixel_size_mm is None or detector_distance_mm is None:
        if detector is None:
            raise ValueError(
                "laue_analytic needs pixel_size_mm and detector_distance_mm, "
                "or a detector=DetectorGeometry(...) to take them from."
            )
        pixel_size_mm        = detector.pixelsize
        detector_distance_mm = detector.dd

    return _period_laue_analytic(
        peaks,
        pixel_size_mm=pixel_size_mm,
        detector_distance_mm=detector_distance_mm,
        wavelength_angstrom=wavelength_angstrom,
        hkl=hkl, lattice=lattice,
        satellite_axis_psi_deg=satellite_axis_psi_deg,
        phi_deg=phi_deg, UB=UB, u_hat=u_hat,
    )


def compare_methods(peaks: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """Run every available method on the same peaks and return the periods side by side.

    Diagnostic only — it does NOT choose between them.  The Laue routes are
    unvalidated; disagreement between the columns is the expected outcome and is
    itself the information (see §5 of NOTES_laue_vs_mono_period.md for the
    discriminating tests).

    Methods that cannot run with the parameters supplied are reported with their
    error message instead of raising, so a partial comparison is still returned.

    Returns
    -------
    dict: {method_name: {'period_nm': float, 'period_angstrom': float,
                         'error': str or None, 'result': full dict or None}}
    """
    out: Dict[str, Any] = {}
    for name in _METHODS:
        try:
            res = layer_period_from_peaks(peaks, method=name, **kwargs)
            out[name] = {
                'period_nm':       res['period_nm'],
                'period_angstrom': res['period_angstrom'],
                'error':           None,
                'result':          res,
            }
        except Exception as exc:
            out[name] = {
                'period_nm':       float('nan'),
                'period_angstrom': float('nan'),
                'error':           f'{type(exc).__name__}: {exc}',
                'result':          None,
            }
    return out
