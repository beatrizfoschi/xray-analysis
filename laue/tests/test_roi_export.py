"""Regression tests for `export_roi_h5`.

Covers two layers: the HDF5 mechanics against a small synthetic stack (no
beamline data needed), and an end-to-end proof — using the same fixtures as
`satellite/tests/test_laue_period.py` — that `run_single_image` gives the
SAME angular result whether it reads from the original file or from the
export, with roi_center/boxsize completely unchanged either way. That last
point is the whole reason this module keeps data at its absolute detector
position instead of re-centring it: a re-centred version was tried first and
broke every angle the Laue routes compute (parent reprojection off by tens of
degrees), because those routes convert `crop_origin_px` to (2theta, chi)
through the detector calibration, which is only meaningful in absolute
coordinates.
"""

from __future__ import annotations

import math

import h5py
import numpy as np
import pytest

from laue.roi_export import export_roi_h5
from laue.satellite.geometry import (
    B_matrix_hexagonal, DetectorGeometry, G_magnitude, LAB_KI,
    direction_to_pixel, kf_hat, lab_vectors_from_UB, predict_satellite_pixels,
    theta_from_G,
)
from laue.satellite.run_single_image import run_single_image

KEY = 'entry/data'


# ── Part 1: HDF5 mechanics, small synthetic stack ────────────────────────────

N_FRAMES, H, W = 12, 60, 80
ROI_CENTER = (40, 30)   # (col, row)
BOXSIZE = 5


@pytest.fixture
def source_h5(tmp_path):
    rng = np.random.default_rng(0)
    # int32, and deliberately including a value beyond uint16 range, mirroring
    # the real bulk-peak amplitudes this export must not clip.
    stack = rng.integers(0, 1000, (N_FRAMES, H, W)).astype(np.int32)
    stack[3, 30, 40] = 436_261
    path = tmp_path / 'source.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset(KEY, data=stack)
    return path, stack


def _roi_slice(center, boxsize):
    col_c, row_c = center
    return (slice(row_c - boxsize, row_c + boxsize + 1),
            slice(col_c - boxsize, col_c + boxsize + 1))


def test_dataset_shape_matches_the_full_source_frame(source_h5, tmp_path):
    """The defining property of this design: shape == source (H, W), not the crop."""
    path, stack = source_h5
    out = tmp_path / 'roi.h5'
    export_roi_h5(path, KEY, out, ROI_CENTER, BOXSIZE, verbose=False)
    with h5py.File(out, 'r') as f:
        assert f[KEY].shape == (N_FRAMES, H, W)


def test_exported_roi_matches_source_at_the_same_absolute_position(source_h5, tmp_path):
    path, stack = source_h5
    out = tmp_path / 'roi.h5'
    export_roi_h5(path, KEY, out, ROI_CENTER, BOXSIZE, verbose=False)

    rs, cs = _roi_slice(ROI_CENTER, BOXSIZE)
    with h5py.File(out, 'r') as f:
        got = f[KEY][:, rs, cs]
    np.testing.assert_array_equal(got, stack[:, rs, cs])


def test_outside_the_roi_reads_back_as_zero(source_h5, tmp_path):
    """Sparsity guarantee: nothing outside the written window is stored."""
    path, stack = source_h5
    out = tmp_path / 'roi.h5'
    export_roi_h5(path, KEY, out, ROI_CENTER, BOXSIZE, verbose=False)
    with h5py.File(out, 'r') as f:
        ds = f[KEY]
        assert ds[0, 0, 0] == 0
        assert ds[0, -1, -1] == 0


def test_file_stays_small_despite_the_full_nominal_shape(source_h5, tmp_path):
    path, stack = source_h5
    out = tmp_path / 'roi.h5'
    export_roi_h5(path, KEY, out, ROI_CENTER, BOXSIZE, verbose=False)
    crop_bytes = N_FRAMES * (2 * BOXSIZE + 1) ** 2 * 4
    # generous bound: a few x the tight crop size, never anywhere near H*W*N*4
    assert out.stat().st_size < 20 * crop_bytes
    assert out.stat().st_size < N_FRAMES * H * W * 4


def test_dtype_defaults_to_source_and_does_not_clip(source_h5, tmp_path):
    """The bulk peak (436261 counts) must survive — this is why uint16 is unsafe."""
    path, stack = source_h5
    out = tmp_path / 'roi.h5'
    export_roi_h5(path, KEY, out, ROI_CENTER, BOXSIZE, verbose=False)
    with h5py.File(out, 'r') as f:
        ds = f[KEY]
        assert ds.dtype == np.int32
        assert int(ds[3, 30, 40]) == 436_261   # same absolute pixel as the source


def test_dataset_carries_provenance_and_warning_attrs(source_h5, tmp_path):
    path, stack = source_h5
    out = tmp_path / 'roi.h5'
    export_roi_h5(path, KEY, out, ROI_CENTER, BOXSIZE, verbose=False)
    with h5py.File(out, 'r') as f:
        attrs = f[KEY].attrs
        assert tuple(attrs['roi_center']) == ROI_CENTER
        assert int(attrs['boxsize']) == BOXSIZE
        assert 'WARNING_sparse_full_frame_shape' in attrs


def test_out_of_bounds_roi_raises_instead_of_silently_shrinking(source_h5, tmp_path):
    path, stack = source_h5
    out = tmp_path / 'roi.h5'
    with pytest.raises(ValueError, match='outside the'):
        export_roi_h5(path, KEY, out, (2, 2), BOXSIZE, verbose=False)


def test_frame_subset_preserves_source_indices_and_provenance(source_h5, tmp_path):
    path, stack = source_h5
    out = tmp_path / 'roi.h5'
    keep = np.array([1, 3, 7, 10])
    info = export_roi_h5(path, KEY, out, ROI_CENTER, BOXSIZE,
                         frame_indices=keep, verbose=False)
    assert info['n_frames'] == len(keep)

    rs, cs = _roi_slice(ROI_CENTER, BOXSIZE)
    with h5py.File(out, 'r') as f:
        got = f[KEY][:, rs, cs]
        src_idx = f[f'{KEY}_source_frame_idx'][:]
    np.testing.assert_array_equal(src_idx, keep)
    np.testing.assert_array_equal(got, stack[keep][:, rs, cs])


def test_boolean_mask_is_accepted_like_integer_indices(source_h5, tmp_path):
    path, stack = source_h5
    out = tmp_path / 'roi.h5'
    mask = np.zeros(N_FRAMES, dtype=bool)
    mask[[2, 5, 9]] = True
    info = export_roi_h5(path, KEY, out, ROI_CENTER, BOXSIZE,
                         frame_indices=mask, verbose=False)
    assert info['n_frames'] == 3
    with h5py.File(out, 'r') as f:
        src_idx = f[f'{KEY}_source_frame_idx'][:]
    np.testing.assert_array_equal(src_idx, np.array([2, 5, 9]))


# ── Part 2: end-to-end proof against run_single_image, unmodified ───────────
#
# Same physical scenario as satellite/tests/test_laue_period.py's fixtures —
# a GaN (105) parent placed at its Bragg-consistent pixel via a synthetic UB —
# reused here rather than duplicated in full, keeping only what's needed to
# paint a synthetic detector frame and drive run_single_image against it.

GAN_A, GAN_C = 3.189, 5.185
LAM = 0.727
HKL = (1, 0, 5)
LAMBDA_TRUE = 97.0
GEOM = DetectorGeometry(dd=69.984, xcen=1079.57, ycen=983.73,
                        xbet=0.173, xgam=0.382, pixelsize=0.075,
                        framedim=(2162, 2068))


def _rodrigues(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


def _rot_between(v1, v2):
    v1 = np.asarray(v1, float) / np.linalg.norm(v1)
    v2 = np.asarray(v2, float) / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    n = np.linalg.norm(axis)
    if n < 1e-14:
        return np.eye(3) if v1 @ v2 > 0 else -np.eye(3)
    return _rodrigues(axis, math.acos(np.clip(v1 @ v2, -1.0, 1.0)))


def _make_UB(spin_deg: float, azimuth_deg: float = 75.0):
    """A physically consistent UB for the (105) parent — see test_laue_period.py."""
    theta = theta_from_G(G_magnitude(*HKL, GAN_A, GAN_C), LAM)
    a = math.radians(azimuth_deg)
    G_dir = np.array([math.cos(theta) * math.cos(a),
                      -math.sin(theta),
                      math.cos(theta) * math.sin(a)])
    B = B_matrix_hexagonal(GAN_A, GAN_C)
    G_crystal_hat = (B @ np.array(HKL, dtype=float))
    G_crystal_hat /= np.linalg.norm(G_crystal_hat)
    return _rodrigues(G_dir, math.radians(spin_deg)) @ _rot_between(G_crystal_hat, G_dir)


def _gaussian_bump(frame, row_c, col_c, amplitude, sigma=2.0):
    r0, r1 = int(row_c) - 8, int(row_c) + 9
    c0, c1 = int(col_c) - 8, int(col_c) + 9
    rr, cc = np.mgrid[r0:r1, c0:c1]
    frame[r0:r1, c0:c1] += amplitude * np.exp(
        -0.5 * (((rr - row_c) / sigma) ** 2 + ((cc - col_c) / sigma) ** 2))


@pytest.fixture
def synthetic_scan(tmp_path):
    """A real GaN(105) parent + 3 satellite orders, painted at their true
    absolute pixel positions on a full-size synthetic detector frame."""
    UB = _make_UB(90.0)
    G_lab, z_lab = lab_vectors_from_UB(HKL, (GAN_A, GAN_C), UB)
    orders = [-3, -2, -1, 0]
    col, row = predict_satellite_pixels(G_lab, z_lab, LAMBDA_TRUE, orders, GEOM)

    H, W = GEOM.framedim
    frame = np.full((H, W), 5.0, dtype=np.float32)
    amps = {-3: 300.0, -2: 900.0, -1: 2500.0, 0: 50_000.0}
    for n, cc, rr in zip(orders, col, row):
        _gaussian_bump(frame, rr, cc, amps[n])

    parent_col, parent_row = direction_to_pixel(kf_hat(G_lab, LAB_KI), GEOM)
    roi_center = (int(round(parent_col)), int(round(parent_row)))
    boxsize = 60

    path = tmp_path / 'synthetic_full_detector.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset(KEY, data=frame[None, :, :].astype(np.int32))

    return {'path': path, 'UB': UB, 'roi_center': roi_center, 'boxsize': boxsize}


def _run(img_source, sc):
    return run_single_image(
        img_source=img_source, h5_img_key=KEY, frame_index=0,
        roi_center=sc['roi_center'], boxsize=sc['boxsize'],
        n_max=3, min_prominence=0.02, bg_sigma=0, axis_angle=None,
        hkl=HKL, lattice=(GAN_A, GAN_C), UB=sc['UB'], detector=GEOM,
        wavelength_angstrom=LAM, period_method='laue_forward',
        quiet=True, print_summary=False, show_plot=False,
    )


def test_exported_file_gives_the_same_parent_reprojection_as_the_source(
        synthetic_scan, tmp_path):
    """The end-to-end proof: unmodified run_single_image, same roi_center and
    boxsize, reading from the export instead of the source, must reproduce
    the same (tiny) parent reprojection error — not the ~86 deg garbage a
    re-centred export produced."""
    direct = _run(synthetic_scan['path'], synthetic_scan)
    assert direct['period'] is not None
    assert direct['period']['parent_offset_deg'] < 0.05

    export_path = tmp_path / 'roi.h5'
    export_roi_h5(synthetic_scan['path'], KEY, export_path,
                  synthetic_scan['roi_center'], synthetic_scan['boxsize'],
                  verbose=False)

    via_export = _run(export_path, synthetic_scan)
    assert via_export['period'] is not None
    assert via_export['period']['parent_offset_deg'] < 0.05
    assert via_export['period']['parent_offset_deg'] == pytest.approx(
        direct['period']['parent_offset_deg'], abs=1e-9)
    assert via_export['period']['period_angstrom'] == pytest.approx(
        direct['period']['period_angstrom'], rel=1e-9)
