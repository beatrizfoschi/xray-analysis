"""Interactive 2-D metric map for satellite peak batch results.

In a Jupyter notebook:
    %matplotlib widget
    from laue.satellite.visualize import interactive_map

    # monochromatic route
    interactive_map(df, img_source, roi_center=(1915, 1263), boxsize=38,
                    metric='alpha',
                    pixel_size_mm=0.075, detector_distance_mm=70.0,
                    wavelength_angstrom=0.7267)

    # Laue route — same map, same click panel, plus the fit diagnostics
    interactive_map(df, img_source, roi_center=(1915, 1263), boxsize=38,
                    metric='period_nm', period_method='laue_forward',
                    hkl=(1, 0, 5), lattice=(3.189, 5.185), UB=UB, detector=geom,
                    wavelength_angstrom=0.7267)

Click any pixel in the map to open the single-image diagnostic panel for that
scan position, re-running the detection with the same parameters used in the
batch.  Pass the same period parameters too, or the panel and the map it came
from would be showing different quantities.
"""

from __future__ import annotations

import sys
import traceback
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from laue.satellite.run_single_image import run_single_image

try:
    from IPython.display import display as _ipy_display
    _IN_JUPYTER = True
except ImportError:
    _IN_JUPYTER = False

try:
    import ipywidgets as _widgets
    _HAS_WIDGETS = True
except ImportError:
    _HAS_WIDGETS = False


def _print_compact_summary(result: dict, metrics: dict, period: dict | None = None,
                           period_error: str | None = None,
                           order_sign: dict | None = None) -> None:
    peaks = result['peaks']
    m = metrics
    dq   = abs(m.get('delta_q', float('nan')))
    alp  = m.get('alpha', float('nan'))
    ar2  = m.get('alpha_r2', float('nan'))
    nsat = m.get('n_sat', len(peaks))
    ang  = result['axis_angle']
    print(f"θ = {ang:.1f}°  |  n = {nsat}  |  Δq = {dq:.2f} px  |  α = {alp:.3f}  (R² = {ar2:.3f})")

    # The Laue route's verdicts have to travel with the panel, or a railed fit and
    # a corrected label set both look like an ordinary result at a glance.
    if order_sign and order_sign.get('inverted'):
        print(f"  order sign INVERTED vs the model — labels corrected "
              f"(arrow_cos = {order_sign['arrow_cos']:+.3f})")
    if period_error is not None:
        print(f"  Λ not computed — {period_error.split('.')[0]}")
    elif period is not None:
        if period.get('fit_at_bound'):
            lo, hi = period.get('period_bounds_angstrom', (float('nan'),) * 2)
            print(f"  Λ NOT MEASURED — fit railed against the [{lo:.0f}, {hi:.0f}] Å bound")
        elif np.isfinite(period.get('period_nm', float('nan'))):
            bits = [f"Λ = {period['period_nm']:.2f} nm"]
            for key, fmt in (('fit_rms_deg', 'rms = {:.1f} mdeg'),
                             ('parent_offset_deg', 'parent = {:.3f}°'),
                             ('train_delta_deg', 'train = {:.2f}°')):
                v = period.get(key)
                if v is not None and np.isfinite(v):
                    bits.append(fmt.format(v * 1000 if key == 'fit_rms_deg' else v))
            print('  ' + '  |  '.join(bits))

    if peaks:
        print(f"  {'Order':>6}  {'Pos (px)':>9}  {'Amplitude':>11}  {'FWHM (px)':>9}")
        for pk in peaks:
            print(f"  {'SL'+str(pk['order']):>6}  {pk['pos_along_axis']:+9.2f}"
                  f"  {pk['amplitude']:11.1f}  {pk['fwhm']:9.2f}")


_METRIC_LABELS: dict = {
    'n_sat':                   'N_sat (orders)',
    'delta_q':                 'Δq (px/order)',
    'alpha':                   'α decay',
    'fwhm_slope':              'FWHM slope (px/order)',
    'asymmetry_intensity_n1':  'Asym. intensity ±1',
    'asymmetry_position_n1':   'Asym. position ±1 (Δq)',
    'bulk_pos':                'Bulk peak position (px)',
    'sl0_pos':                 'SL₀ position, predicted (px)',
    'bulk_sl0_offset_px':      'Bulk → SL₀ offset (px)',
    'sl1_mean_intensity':      'SL±1 mean intensity (counts)',
    'axis_angle':              'Axis angle θ (°)',
    'period_nm':               'Layer period (nm)',
    'period_angstrom':         'Layer period (Å)',
    'delta_q_inv_ang':         'Δq (Å⁻¹)',
    # ── Laue routes only ──────────────────────────────────────────────────────
    'fit_rms_deg':             'Fit RMS (deg)',
    'parent_offset_deg':       'Parent reprojection offset (deg)',
    'train_delta_deg':         'Train direction mismatch (deg)',
    'fit_at_bound':            'Fit railed against a bound (0/1)',
    'gamma_deg':               'γ, G to growth axis (°)',
    'two_theta_measured':      '2θ of the parent, measured (°)',
    'chi_measured':            'χ of the parent, measured (°)',
    'order0_is_sl0':           'Order-0 peak is SL₀ (0/1)',
    'sl0_confirmed':           'SL₀ found in the raw crop (0/1)',
    'sl0_measured_pos':        'SL₀ position, measured (px)',
    'sl0_measured_amplitude':  'SL₀ amplitude, measured (counts)',
    'bulk_sl0_offset_deg':     'Bulk → SL₀ offset (deg)',
    'gap_pred_px':             'Predicted SL₀ → SL±1 gap (px)',
    'deg_per_px':              'Local angular scale (deg/px)',
    'amplitude_ratio_order0_to_n1': 'Amplitude ratio order-0 / n∓1',
    'order_sign_inverted':     'Order labels inverted vs model (0/1)',
    'order_sign_confident':    'Order sign resolved (0/1)',
    'order_sign_arrow_cos':    'Order sign cos (model vs measured)',
}


def _period_kwargs(
    *, period_method, pixel_size_mm, detector_distance_mm, wavelength_angstrom,
    energy_kev, two_theta_0_deg, chi_deg, hkl, lattice, UB, detector, sl0_boxsize,
) -> dict:
    """The period arguments to forward to ``run_single_image``, per route.

    Route selection mirrors ``run_single_image`` and ``scan_pipeline`` exactly:
    the clicked panel has to reproduce the batch, so it must decide the same way.
    The Laue set wins when complete; the monochromatic set is the fallback; an
    empty dict means no period is computed and only the detection is shown.

    An incomplete Laue set raises rather than falling back — a map made with one
    route and inspected with the other would put two different quantities side by
    side without saying so.
    """
    laue_ready = all(v is not None for v in (hkl, lattice, UB, detector))
    geo_complete = (pixel_size_mm is not None
                    and detector_distance_mm is not None
                    and (wavelength_angstrom is not None or energy_kev is not None))

    if period_method != 'monochromatic' and not laue_ready:
        missing = [n for n, v in (('hkl', hkl), ('lattice', lattice),
                                  ('UB', UB), ('detector', detector))
                   if v is None]
        raise ValueError(
            f"period_method={period_method!r} needs hkl, lattice, UB and detector; "
            f"missing {', '.join(missing)}.  Pass the same ones given to "
            f"run_satellite_pipeline, or leave period_method='monochromatic'.  "
            f"Falling back would show a different quantity from the map."
        )

    if laue_ready:
        return dict(
            period_method=period_method,
            hkl=tuple(hkl), lattice=tuple(lattice), UB=UB, detector=detector,
            wavelength_angstrom=wavelength_angstrom, energy_kev=energy_kev,
            sl0_boxsize=sl0_boxsize,
        )
    if geo_complete:
        return dict(
            pixel_size_mm=pixel_size_mm,
            detector_distance_mm=detector_distance_mm,
            wavelength_angstrom=wavelength_angstrom,
            energy_kev=energy_kev,
            two_theta_0_deg=two_theta_0_deg,
            chi_deg=chi_deg,
        )
    return {}


def interactive_map(
    df: pd.DataFrame,
    img_source,
    roi_center: Tuple[int, int],
    boxsize: int,
    metric: str = 'n_sat',
    h5_img_key: str = 'frames',
    coords: str = 'numpy',
    cmap: str = 'viridis',
    percentile_clip: Tuple[float, float] = (2, 98),
    figsize: Tuple[float, float] = (6, 6),
    # detection kwargs — must match what was used in run_pipeline
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
    # period — pass the same route and parameters used in the batch
    period_method: str = 'monochromatic',
    # ... monochromatic route
    pixel_size_mm: Optional[float] = None,
    detector_distance_mm: Optional[float] = None,
    wavelength_angstrom: Optional[float] = None,
    energy_kev: Optional[float] = None,
    two_theta_0_deg: float = 0.0,
    chi_deg: float = 0.0,
    # ... Laue routes
    hkl: Optional[Tuple[int, int, int]] = None,
    lattice: Optional[Tuple[float, float]] = None,
    UB=None,
    detector=None,
    sl0_boxsize: float = 3.0,
) -> None:
    """Plot an interactive 2-D metric map; click a pixel to inspect that position.

    Parameters
    ----------
    df              : DataFrame returned by run_satellite_pipeline().
    img_source      : path to the HDF5 file (same as passed to the pipeline).
    roi_center      : (x, y) = (col, row) of the Laue spot, as in the pipeline.
    boxsize         : crop half-size (same as the pipeline).
    metric          : column name to display (e.g. 'alpha', 'axis_angle', 'n_sat').
    h5_img_key      : HDF5 dataset key (same as the pipeline).
    coords          : 'numpy' or 'xmas' (same as the pipeline).
    cmap            : matplotlib colormap for the metric map.
    percentile_clip : (lo, hi) percentiles for robust colour scaling.
    figsize         : figure size in inches for the map panel.
    axis_angle … spacing_px : detection parameters — pass the same values used
                    in the pipeline so the clicked diagnostic matches the batch.
    period_method   : 'monochromatic' (default), 'laue_analytic' or 'laue_forward'.

    Both period routes are supported, and the click panel decides between them the
    same way ``run_single_image`` and ``scan_pipeline`` do:

    * **monochromatic** — give ``pixel_size_mm``, ``detector_distance_mm`` and a
      wavelength or energy.
    * **Laue** — give ``hkl``, ``lattice``, ``UB`` and ``detector``.  These take
      precedence when all four are present, and the panel then also reports the
      fit diagnostics, the order-sign verdict and SL₀, exactly as the batch did.

    Pass the same set the pipeline was run with.  A map made with one route and
    inspected with the other would show two different quantities side by side
    without saying so, which is why an incomplete Laue set raises here rather than
    falling back.
    """
    if metric not in df.columns:
        available = [c for c in df.columns
                     if c not in ('i', 'j', 'frame_idx', 'x_um', 'y_um', 'status')]
        raise ValueError(f"Column '{metric}' not found in df. Available: {available}")

    # ── Build 2-D grid ────────────────────────────────────────────────────────
    i_min = int(df['i'].min())
    j_min = int(df['j'].min())
    nbx   = int(df['i'].nunique())
    nby   = int(df['j'].nunique())

    x_um = np.sort(df['x_um'].unique())
    y_um = np.sort(df['y_um'].unique())
    extent = [float(x_um.min()), float(x_um.max()),
              float(y_um.min()), float(y_um.max())]

    grid = np.full((nbx, nby), np.nan)
    for _, row in df[df['status'] == 'ok'].iterrows():
        grid[int(row['i'] - i_min), int(row['j'] - j_min)] = row.get(metric, np.nan)
    data = grid.T   # (nby, nbx) — rows=y, cols=x for imshow

    lo = np.nanpercentile(data, percentile_clip[0])
    hi = np.nanpercentile(data, percentile_clip[1])
    label = _METRIC_LABELS.get(metric, metric)

    # ── Create output area for diagnostic panels ──────────────────────────────
    # ipywidgets.Output captures display() calls made inside callbacks correctly.
    if _HAS_WIDGETS:
        diag_out = _widgets.Output()
    else:
        diag_out = None

    # ── Draw map ──────────────────────────────────────────────────────────────
    # plt.ioff() suppresses the automatic widget display so we can show it
    # explicitly via display(fig.canvas) inside a VBox — avoids double-render.
    plt.close('interactive_map')
    with plt.ioff():
        fig, ax_map = plt.subplots(figsize=figsize, num='interactive_map')
    im = ax_map.imshow(data, origin='lower', aspect='equal',
                       extent=extent, cmap=cmap, vmin=lo, vmax=hi,
                       interpolation='none')
    plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label=label)
    ax_map.set_title(f'{label}\n(click a pixel to inspect)', fontsize=11)
    ax_map.set_xlabel('x (µm)')
    ax_map.set_ylabel('y (µm)')
    fig.tight_layout()

    # Crosshair marker — updated on each click
    (marker,) = ax_map.plot([], [], '+', color='white', ms=14, mew=2, zorder=5)

    detect_kw = dict(
        axis_angle=axis_angle, n_max=n_max, min_prominence=min_prominence,
        strip_width=strip_width, bg_sigma=bg_sigma, peak_min_width=peak_min_width,
        hot_pixel_sigma=hot_pixel_sigma, n_range=n_range, spacing_px=spacing_px,
        adaptive_fill_win=adaptive_fill_win,
    )

    period_kw = _period_kwargs(
        period_method=period_method,
        pixel_size_mm=pixel_size_mm, detector_distance_mm=detector_distance_mm,
        wavelength_angstrom=wavelength_angstrom, energy_kev=energy_kev,
        two_theta_0_deg=two_theta_0_deg, chi_deg=chi_deg,
        hkl=hkl, lattice=lattice, UB=UB, detector=detector,
        sl0_boxsize=sl0_boxsize,
    )

    # ── Click handler ─────────────────────────────────────────────────────────
    def _on_click(event):
        if event.inaxes is not ax_map or event.xdata is None:
            return

        dist = np.hypot(df['x_um'] - event.xdata, df['y_um'] - event.ydata)
        row  = df.loc[dist.idxmin()]

        status = str(row.get('status', 'ok'))
        val    = row.get(metric, float('nan'))
        i, j   = int(row['i']), int(row['j'])
        fidx   = int(row['frame_idx'])

        marker.set_data([row['x_um']], [row['y_um']])
        fig.canvas.draw_idle()

        if diag_out is None:
            return

        diag_out.clear_output(wait=True)
        with diag_out:
            print(f'[CLICK] i={i}, j={j}, frame={fidx}, '
                  f'x={row["x_um"]:.4g} µm, y={row["y_um"]:.4g} µm  |  '
                  f'{metric} = {val:.4g}  |  status = {status}')
            if status not in ('ok', 'masked'):
                print(f'  [WARN] Batch status: {status}')
            try:
                # plt.ioff() prevents the diagnostic figure from auto-displaying
                # outside this Output widget in %matplotlib widget mode
                with plt.ioff():
                    out = run_single_image(
                        img_source=img_source,
                        h5_img_key=h5_img_key,
                        frame_index=fidx,
                        roi_center=roi_center,
                        boxsize=boxsize,
                        coords=coords,
                        profile_log=True,
                        show_linear_profile=False,
                        print_summary=False,
                        quiet=True,
                        figsize=(20, 6),
                        show_plot=False,
                        **detect_kw,
                        **period_kw,
                    )
                _ipy_display(out['fig'])
                _print_compact_summary(out['result'], out['metrics'],
                                       period=out.get('period'),
                                       period_error=out.get('period_error'),
                                       order_sign=out.get('order_sign'))
            except Exception:
                traceback.print_exc()

    fig.canvas.mpl_connect('button_press_event', _on_click)

    # ── Display ───────────────────────────────────────────────────────────────
    # ipywidgets being importable is not enough: `fig.canvas` is only a widget
    # under the ipympl backend (%matplotlib widget).  Under any other backend the
    # VBox rejects the plain canvas with an opaque traitlets error, and clicks are
    # never delivered anyway — so say that instead.
    canvas_is_widget = (_HAS_WIDGETS
                        and isinstance(fig.canvas, _widgets.Widget))

    if canvas_is_widget and diag_out is not None:
        # fig.canvas is the ipympl widget; display it alongside the Output widget
        # so the diagnostic panel appears under the map.
        _ipy_display(_widgets.VBox([fig.canvas, diag_out]))
    else:
        if _IN_JUPYTER:
            warnings.warn(
                'interactive_map needs the ipympl backend to receive clicks — '
                'run "%matplotlib widget" in a cell before calling it.  The map '
                'is shown, but clicking a pixel will do nothing.',
                stacklevel=2,
            )
            _ipy_display(fig)
        else:
            plt.show()
