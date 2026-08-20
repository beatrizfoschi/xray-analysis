"""Per-scan-position coherence indicators derived from detected satellite peaks.

Indicators (Stage 3 of the MQW satellite analysis pipeline)
------------------------------------------------------------
n_sat           Count of resolved satellite orders above noise threshold.
                Physical meaning: vertical coherence length of the MQW stack.

delta_q         Pixel spacing between consecutive satellite orders.
                Superlattice period Λ = 2π / (delta_q · q_scale) if a
                q-calibration factor is available.  Deviation from nominal
                indicates thickness error.

alpha           Exponential decay of satellite intensities with order:
                I_n = C · exp(−α · |n|)  (C is a free parameter, not I₀).
                SL0 is excluded from the fit because it contains both the
                superlattice (superrede) average and the bulk Bragg
                contribution, making its amplitude anomalously large relative
                to the satellite envelope.  Alpha is the slope of log(I_n) vs |n| and is
                independent of the normalisation choice.
                Faster decay = rougher MQW interfaces / larger interdiffusion.

fwhm_slope      Linear slope of FWHM vs |n|.
                Positive slope = random period/thickness fluctuations across
                wells (each successive order integrates more disorder).

asymmetry_*     ±n intensity and position asymmetry (per order n = 1, 2, 3…).
                Non-zero asymmetry = systematic composition/strain gradient
                through the stack (e.g. well 1 vs. well 4).

bulk_pos        Position of the DETECTED ORDER-0 PEAK in axis-distance pixels,
                relative to the profile centroid.  In an MQW that peak is the
                bulk Bragg reflection, not SL₀ (see README.md, "The bright peak in the ROI is the bulk") —
                hence the name.  The true SL₀ is predicted, not detected, and is
                added downstream by the Laue period routes as ``sl0_pos``.
                Renamed from ``sl0_pos`` on 2026-08-12; the old name measured
                this same quantity under a wrong label.

Usage
-----
    from laue.satellite.detection import detect_satellites
    from laue.satellite.metrics import compute_metrics, metrics_to_flat_dict

    result  = detect_satellites(image)
    metrics = compute_metrics(result['peaks'])
    row     = metrics_to_flat_dict(metrics)   # for a pandas DataFrame
"""


from __future__ import annotations

import math

import numpy as np
from typing import Any, Dict, List, Optional

def compute_metrics(peaks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all per-position satellite coherence indicators.

    Parameters
    ----------
    peaks : list of peak dicts returned by detect_satellites().
            Required keys: 'order', 'pos_along_axis', 'amplitude', 'fwhm'.

    Returns
    -------
    dict with keys listed in the module docstring.
    """
    if not peaks:
        return _empty_metrics()

    orders = np.array([p['order'] for p in peaks], dtype=int)
    positions = np.array([p['pos_along_axis'] for p in peaks])
    amplitudes = np.array([p['amplitude'] for p in peaks])
    fwhms = np.array([p['fwhm'] for p in peaks])

    m: Dict[str, Any] = {}

    # ── SL₀ reference ────────────────────────────────────────────────────────
    sl0_mask = orders == 0
    if sl0_mask.any():
        sl0_idx = int(np.where(sl0_mask)[0][0])
        m['bulk_pos'] = float(positions[sl0_idx])
        sl0_amp = float(amplitudes[sl0_idx])
    else:
        m['bulk_pos'] = float('nan')
        sl0_amp = float(amplitudes.max())

    # ── 1. N_sat ─────────────────────────────────────────────────────────────
    m['n_sat'] = int((orders != 0).sum())

    # ── 2. Δq — satellite spacing (px / order) ────────────────────────────────
    nonzero = orders != 0
    if nonzero.sum() >= 2:
        coeffs = np.polyfit(orders[nonzero].astype(float), positions[nonzero], 1)
        m['delta_q'] = float(coeffs[0])          # px per order (signed)
        resid = positions[nonzero] - np.polyval(coeffs, orders[nonzero].astype(float))
        m['delta_q_std'] = float(resid.std())    # scatter around uniform spacing
    elif nonzero.sum() == 1:
        n1 = int(abs(orders[nonzero][0]))
        m['delta_q'] = float(abs(positions[nonzero][0])) / max(n1, 1)
        m['delta_q_std'] = float('nan')
    else:
        m['delta_q'] = float('nan')
        m['delta_q_std'] = float('nan')

    # ── 3. Intensity envelope decay α ────────────────────────────────────────
    # Fit log(I_n) = log(C) - α·|n|  (free intercept, SL0 excluded).
    # SL0 is not used as an anchor because it includes the bulk Bragg
    # contribution in addition to the superlattice average, making it
    # anomalously large.  The slope alone (= -α) is the physical quantity.
    amp_mask = (orders != 0) & (amplitudes > 0) & (sl0_amp > 0)
    if amp_mask.sum() >= 2:
        abs_n = np.abs(orders[amp_mask]).astype(float)
        log_ratio = np.log(amplitudes[amp_mask] / sl0_amp)
        coeffs_a = np.polyfit(abs_n, log_ratio, 1)
        m['alpha'] = float(-coeffs_a[0])          # positive means decay
        pred = np.polyval(coeffs_a, abs_n)
        ss_res = float(((log_ratio - pred) ** 2).sum())
        ss_tot = float(((log_ratio - log_ratio.mean()) ** 2).sum())
        m['alpha_r2'] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    else:
        m['alpha'] = float('nan')
        m['alpha_r2'] = float('nan')

    # ── 4. FWHM vs order slope ────────────────────────────────────────────────
    if nonzero.sum() >= 2:
        abs_n_fwhm = np.abs(orders[nonzero]).astype(float)
        fwhms_sat = fwhms[nonzero]
        coeffs_f = np.polyfit(abs_n_fwhm, fwhms_sat, 1)
        m['fwhm_slope'] = float(coeffs_f[0])     # px / order
        pred_f = np.polyval(coeffs_f, abs_n_fwhm)
        ss_res_f = float(((fwhms_sat - pred_f) ** 2).sum())
        ss_tot_f = float(((fwhms_sat - fwhms_sat.mean()) ** 2).sum())
        m['fwhm_slope_r2'] = float(1.0 - ss_res_f / ss_tot_f) if ss_tot_f > 0 else float('nan')
    else:
        m['fwhm_slope'] = float('nan')
        m['fwhm_slope_r2'] = float('nan')

    # ── 5. ±n asymmetry ───────────────────────────────────────────────────────
    # For each order n: compare +n vs −n in intensity and position.
    asym_int: Dict[int, float] = {}
    asym_pos: Dict[int, float] = {}
    delta_q = abs(m.get('delta_q', 0.0)) or 1.0
    max_n = int(np.abs(orders).max())
    for n in range(1, max_n + 1):
        pos_idx = np.where(orders == n)[0]
        neg_idx = np.where(orders == -n)[0]
        if len(pos_idx) and len(neg_idx):
            i_pos = float(amplitudes[pos_idx[0]])
            i_neg = float(amplitudes[neg_idx[0]])
            p_pos = float(positions[pos_idx[0]])
            p_neg = float(positions[neg_idx[0]])
            # (I+ − I−) / (I+ + I−): +1 = only positive order visible, −1 = only negative
            asym_int[n] = (i_pos - i_neg) / (i_pos + i_neg + 1e-12)
            # (|d+| − |d−|) / Δq: deviation from symmetric spacing, in units of 1 order
            asym_pos[n] = (abs(p_pos) - abs(p_neg)) / delta_q

    m['asymmetry_intensity'] = asym_int
    m['asymmetry_position'] = asym_pos

    return m


# ── Serialisation ─────────────────────────────────────────────────────────────

def metrics_to_flat_dict(
    metrics: Dict[str, Any],
    prefix: str = '',
) -> Dict[str, float]:
    """Flatten nested metrics into a {column: value} dict for a DataFrame row.

    Nested dicts (asymmetry_intensity, asymmetry_position) are expanded as
    e.g. asymmetry_intensity_n1, asymmetry_intensity_n2, …
    """
    flat: Dict[str, float] = {}
    for k, v in metrics.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            for n, val in v.items():
                flat[f'{key}_n{n}'] = float(val)
        elif isinstance(v, (int, float)):
            flat[key] = float(v)
        else:
            try:
                flat[key] = float(v)
            except (TypeError, ValueError):
                pass
    return flat


# ── Per-order fit quality ─────────────────────────────────────────────────────

def per_order_metrics(
    peaks: List[Dict[str, Any]],
    distances: np.ndarray,
    intensities: np.ndarray,
) -> Dict[str, float]:
    """R², amplitude and FWHM per satellite order, as flat {col: value} dict.

    Column names: r2_n{order}, amp_n{order}, fwhm_n{order}
    (e.g. r2_n-1, amp_n-2, fwhm_n0).

    R² is computed in a ±2σ window around each peak using the background-
    subtracted 1-D profile.  A value close to 1 means the Gaussian describes
    the peak well; low values indicate noisy or asymmetric peaks.
    """
    flat: Dict[str, float] = {}
    for pk in peaks:
        n     = pk['order']
        mu    = pk['pos_along_axis']
        sigma = pk['sigma']
        amp   = pk['amplitude']
        label = f'n{n}'

        flat[f'amp_{label}']  = float(amp)
        flat[f'fwhm_{label}'] = float(pk['fwhm'])

        if sigma > 0 and len(distances) > 0:
            mask = np.abs(distances - mu) <= 2.0 * sigma
            if mask.sum() >= 3:
                y     = intensities[mask]
                x     = distances[mask]
                y_fit = amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                ss_res = float(np.sum((y - y_fit) ** 2))
                ss_tot = float(np.sum((y - float(y.mean())) ** 2))
                r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
                flat[f'r2_{label}'] = max(r2, 0.0) if not np.isnan(r2) else float('nan')
            else:
                flat[f'r2_{label}'] = float('nan')
        else:
            flat[f'r2_{label}'] = float('nan')

    return flat


# ── Empty / missing result ────────────────────────────────────────────────────

def _empty_metrics() -> Dict[str, Any]:
    return {
        'n_sat': 0,
        'delta_q': float('nan'),
        'delta_q_std': float('nan'),
        'alpha': float('nan'),
        'alpha_r2': float('nan'),
        'fwhm_slope': float('nan'),
        'fwhm_slope_r2': float('nan'),
        'asymmetry_intensity': {},
        'asymmetry_position': {},
        'bulk_pos': float('nan'),
    }
