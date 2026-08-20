# laue.satellite — MQW satellite peak analysis

Characterise the vertical coherence of multi-quantum-well (MQW) stacks from the
satellite peaks that flank a Bragg reflection in Laue microdiffraction images.

A periodic stack of period Λ = t_QW + t_barrier adds reciprocal-lattice points
spaced by 2π/Λ along the growth direction. On the detector these appear as a
discrete, roughly equispaced series of spots around the parent reflection. Their
count, spacing, intensity decay and width encode coherence length, interface
roughness, period fluctuation and composition gradients.

The pipeline is sample-agnostic: nothing about a particular chip or geometry is
hard-coded.

## Layout

```
laue/satellite/
├── detection.py               1-D detection: background, axis, profile, Gaussian fits
├── metrics.py                 the coherence indicators, and nothing else
├── geometry.py                crystallography and detector: .det, (2θ, χ), UB frames
├── period.py                  the three period routes and the method= dispatcher
├── scan_pipeline.py           parallel analysis over a 2-D scan grid
├── run_single_image.py        notebook entry point for one image
├── diagnose_single_image.py   command-line diagnostic
├── example_laue_period.py     notebook entry point for the Laue routes
├── visualize.py               interactive map browser
└── tests/
```

`period` imports from `geometry`; `metrics` depends on neither.

## Requirements

`numpy`, `scipy`, `pandas`, `matplotlib`, `h5py`, and `LaueTools` for the Laue
routes. Imports assume the repository root is on `sys.path`:

```python
import sys; sys.path.insert(0, '/path/to/xray-analysis')
```

## Single image

```python
from laue.satellite.run_single_image import run_single_image

result = run_single_image(
    img_source  = '/data/eiger4m_0000.h5',
    h5_img_key  = 'entry_0000/CRGIF/eiger4m/data',
    frame_index = 0,
    roi_center  = (1913, 1263),   # (x, y) = (col, row)
    boxsize     = 60,             # half-size; crop is 121 x 121 px
)
result['peaks_df']   # detected orders
result['metrics']    # the indicators
```

A period failure does not cost the figure. When the period calculation raises —
too few orders, γ = 0, a UB in the wrong frame — detection, the summary and the
plot are produced anyway, `result['period']` is `None`, and `result['period_error']`
says why. That is the case you are in while the segmentation parameters are still
being tuned, and it is precisely when you need to see the image.

Or from the shell, on synthetic data when no file is at hand:

```
python -m laue.satellite.diagnose_single_image
```

## Whole scan

```python
from laue.satellite.scan_pipeline import run_satellite_pipeline, plot_satellite_maps

df = run_satellite_pipeline(
    img_source  = 'stack.h5',
    h5_img_key  = 'entry_0000/CRGIF/eiger4m/data',
    scan        = scan,               # lauexplore Scan
    roi_center  = (1263, 1913),
    boxsize     = 60,
    scan_subset = (47, 140, 54, 83),  # (i0, i1, j0, j1)
    workers     = 60,
)
fig = plot_satellite_maps(df, percentile_clip=(2, 98))
```

Each worker opens the HDF5 file independently, read-only. A position whose period
calculation fails keeps its other metrics and records the reason in `period_error`;
it is not dropped.

## Browsing a scan result

```python
%matplotlib widget
from laue.satellite.visualize import interactive_map

interactive_map(df, img_source, roi_center=(1263, 1913), boxsize=60,
                metric='period_nm', period_method='laue_forward',
                hkl=(1, 0, 5), lattice=(3.189, 5.185), UB=UB, detector=geom,
                wavelength_angstrom=0.7267)
```

Click a pixel to re-run that position and see the crop, the profile and the peak
table. Both period routes are served, and the panel picks between them the way
`run_single_image` does: give `hkl`, `lattice`, `UB` and `detector` for the Laue
routes, or `pixel_size_mm`, `detector_distance_mm` and a wavelength for the
monochromatic one. Pass the same set the batch used — an incomplete Laue set
raises rather than falling back, since the panel and the map would otherwise be
showing different quantities. On the Laue routes the panel also reports Λ, the fit
diagnostics and the order-sign verdict, and says so when a fit railed against a
bound. The map needs the `ipympl` backend to receive clicks.

## Indicators

| output | physical meaning |
|---|---|
| `n_sat` | number of resolved orders — vertical coherence length of the stack |
| `delta_q` → `period_nm` | superlattice period; departure from nominal is a thickness error |
| `alpha` | decay of `I_n ~ C·exp(−α|n|)` — interface roughness / interdiffusion |
| `fwhm_slope` | FWHM vs \|n\| — random period fluctuation between wells |
| `asymmetry_intensity_n{1,2,3}` | systematic composition or strain gradient through the stack |
| `asymmetry_position_n{1,2,3}` | position asymmetry, in units of `delta_q` |
| `bulk_pos` | position of the detected order-zero peak along the axis |
| `sl0_pos` | position of SL₀ as predicted from the satellite ladder (Laue routes only) |
| `bulk_sl0_offset_px` | bulk → SL₀ separation — mean out-of-plane strain/composition |
| `axis_angle` | refined satellite axis, in degrees |

`alpha` excludes the order-zero peak from the fit: it carries both the stack
average and the bulk contribution, so its amplitude is not on the satellite
envelope.

## Period: three routes

The illumination is polychromatic (white or pink beam), while the legacy formula
is monochromatic. Both are available, selected by `method=`:

| `method` | status |
|---|---|
| `'monochromatic'` | default; frozen by a regression test |
| `'laue_analytic'` | small-angle approximation, known +0.94 % bias at first order; cross-check only |
| `'laue_forward'` | exact forward model fitted in (2θ, χ); the recommended Laue route |

```python
from laue.satellite.period import layer_period_from_peaks, compare_methods
from laue.satellite.geometry import DetectorGeometry

geom, UB = DetectorGeometry.from_det_file('calibration.det')

layer_period_from_peaks(
    peaks, method='laue_forward', wavelength_angstrom=0.7267,
    hkl=(1, 0, 5), lattice=(3.189, 5.185),   # γ and |G₀| derive from these
    detector=geom, UB=UB,
    crop_origin_px=(row0, col0),             # absolute pixel of the crop corner
)

compare_methods(peaks, **kw)   # all three side by side, without choosing
```

The two routes read opposite components of the satellite displacement: the
monochromatic period scales with cos γ, the Laue period with sin γ, where γ is the
angle between the scattering vector and the growth direction. A symmetric
reflection (γ = 0) would superimpose the satellites in the Laue case, so γ = 0
raises rather than returning NaN.

> **The Laue routes are not validated against experiment.** They agree with the
> monochromatic route to better than 1 % on the spots tested so far, but do not use
> them to revise reported values without an independent check. Two tests that would
> close this are skipped pending a measured spot centroid for a known calibration.

## Conventions the caller must respect

**The bright peak in the ROI is the bulk reflection, not SL₀.** In an MQW the
parent Bragg peak is typically ~70× stronger than the superlattice zero order,
which sits buried in its flank. `metrics.compute_metrics` reports the detected
order-zero peak as `bulk_pos` and deliberately does not emit `sl0_pos` — at that
layer there is no way to know where SL₀ is. The Laue routes locate SL₀ from the
ladder instead, without using the bright peak, and Λ is unaffected because the fit
never uses it.

**`chi_deg` is the monochromatic route's γ and must stay 0 there.** The Laue
routes ignore it and derive γ from `hkl` and `lattice`. Never pass LaueTools'
detector-space χ as γ; they are different quantities.

**A `.det` gives an orientation matrix, not a UB.** It carries no lattice metric, so
apply `B` before it — `geometry.lab_vectors_from_UB` does that and is the intended
entry point. On a hexagonal cell, skipping `B` tilts the direction as well as the
magnitude.

**The UB inside a `.det` belongs to the calibration crystal, not the sample.** Take
the geometry from it and the orientation from your own indexing;
`geometry.assert_ub_material` raises on the mismatch.

**A sample UB from indexing may arrive in a beam-∥-x frame** while LaueTools uses
beam ∥ +y. Wrap it with `geometry.ub_from_beam_x_frame`. Left uncorrected, |G| and γ
still come out right — both are frame-independent — while the predicted spot lands
tens of degrees away. `geometry.diagnose_ub_frame` identifies the case.

A LaueTools simulation overlaying the measured image does **not** clear this: the
simulator wants the raw UB and this module wants the converted one, so the overlay
checks the other convention. `parent_offset_deg` is the number that covers it — it
is the predicted-to-measured separation of the parent reflection, and it should be
a small fraction of a degree. The Laue routes refuse outright when the orientation
puts the reflection off the observable hemisphere.

**`(xcen, ycen)` is the detector reference point (PONI), not a beam centre.** In
reflection geometry the direct beam never reaches the detector, so nothing is
visible there.

**Δ(2θ) alone is not an angular separation**, and adding Δ(2θ) and Δχ in quadrature
overshoots. Use `geometry.angular_separation`, which applies the spherical law of
cosines. Both failure modes are pinned by tests.

**`axis_angle` is a descriptive output, not a calibrated quantity**, and must not
enter a period calculation. Use `train_delta_deg` for that comparison.

**The sign of a detected order is a detector-space convention.** The profile axis
is folded into a half-plane with no reference to the growth direction, so which
side of the parent counts as +n is luck, and the luck changes from reflection to
reflection. `period.resolve_order_sign` settles it against the model and the Laue
entry points apply it before anything reads the labels — the period would announce
an inversion by railing against a bound, but `delta_q` and the asymmetry
indicators would simply come out with the opposite sign. The verdict is reported
as `order_sign`, and it is refused rather than guessed when the reflection is not
confirmed (`parent_offset_deg`), the train is off the predicted line
(`train_delta_deg`), or the two trains are near perpendicular.

**A fit that rails against a period bound is not a measurement.** `fit_at_bound`
says so, `period_bounds_angstrom` says which bound, and the value is withheld
rather than printed. Railing against the *upper* bound is always a direction
problem, never a magnitude one: Λ sets the length of the predicted train and the
orientation sets its direction, so a train misaligned by δ gives
Λ_fitted = Λ_true / cos δ — inflating for δ < 90°, unbounded at 90°, and railing
beyond it. Opening the bounds does not help; it moves the rail.

## Detection parameters

Defaults that work on the reference data: `n_max=3`, `min_prominence=0.05`,
`strip_width=5.0`, `bg_sigma=20.0`, `peak_min_width=2.0`. Pass `verbose=True` for a
step-by-step trace, and `spacing_px` to override the auto-detected spacing.

Detection runs in two passes: a PCA axis estimate, detection, then an axis refined
from the 2-D satellite centroids and a second detection. Peaks are found on a
log-scaled profile so weak high orders keep prominence comparable to the parent,
and orders are assigned by rank from the centre outwards rather than by spacing,
which tolerates non-uniform spacing from axis misalignment.

## Tests

```
python -m pytest laue/satellite/tests -q
```

172 pass, 2 skipped. The two skips need a measured spot centroid to validate the
Laue reprojection.
