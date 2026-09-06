"""
roi_peaklist.py — Peak positions from simulation-defined ROIs, measured by
centre of mass.

    ##########################################################################
    ##  POSITION CONVENTION — READ THIS BEFORE USING THE OUTPUT             ##
    ##                                                                      ##
    ##  The position reported for every spot in this module is the          ##
    ##  INTENSITY-WEIGHTED CENTRE OF MASS of the pixels inside a fixed      ##
    ##  ROI. It is NOT the centre of a fitted peak, and it is NOT the       ##
    ##  brightest pixel.                                                    ##
    ##                                                                      ##
    ##  When several sub-spots overlap inside one ROI — a GaN buffer and    ##
    ##  the layers above it, slightly misoriented — the COM is their        ##
    ##  intensity-weighted mean, not any one of them. That blending is      ##
    ##  deliberate: these sub-spots cannot be separated by segmentation,    ##
    ##  and the mean position is the quantity being tracked.                ##
    ##                                                                      ##
    ##  Consequence: the COM moves for two different reasons, and they are  ##
    ##  degenerate in the position alone —                                  ##
    ##    (a) the sub-spots themselves rotate / relax, or                   ##
    ##    (b) the relative intensity between them changes with the          ##
    ##        sub-spots where they were.                                    ##
    ##  Case (b) is common at the edge of a material, where thickness and   ##
    ##  absorption change. Use the shape columns to tell them apart: a      ##
    ##  rigid rotation translates the blend with little change in shape,    ##
    ##  while a change of weights moves `aspect_ratio`, `theta` and         ##
    ##  `skewness` along with the COM. See `plot_com_vs_boxsize` and the    ##
    ##  `total_counts` / `bg_level` diagnostics before reading any COM      ##
    ##  displacement as a lattice rotation.                                 ##
    ##########################################################################

Pipeline
--------
    simulate_pattern()        predicted spot positions, material + substrate
    forbidden_mask()          pixels excluded: detector gaps + substrate zones
    gaussian_background()     smooth fluorescence background over the frame
    build_peaklist()          one COM + moments per predicted material spot
    to_lauetools_dat()        (N, 9) peak file LaueTools can read

Everything the module needs is in `xray-analysis` itself plus LaueTools /
lauexplore. `gaussian_background` and the `.dat` writer are small enough that
they are reimplemented here rather than imported from `nrxrdct`, which is a
separate clone and is not installed in this environment.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft as sp_fft
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from laue._imaging import extract_crop
from laue.spot_metrics import inertia_tensor, streak_moments

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))

DAT_COLUMNS = [
    "peak_X", "peak_Y", "peak_I",
    "peak_fwaxmaj", "peak_fwaxmin", "peak_inclination",
    "Xdev", "Ydev", "peak_bkg",
]


# ── Calibration / UB from a LaueTools .fit ────────────────────────────────────

def load_ub_and_calibration(fit_path) -> dict:
    """UB matrix and detector calibration from a LaueTools `.fit` file.

    `lauexplore.FitFile` is tried first. It is the better reader when it
    works, but it assumes a refinement-style file: it requires the
    ``#File created at ... with ...`` header line, it computes ``UB @ B0`` and
    raises `AttributeError` when a file carries UB but no B0, and it parses the
    peak block with ``np.loadtxt(file)[:, 1:]``, which fails when there are no
    peak rows. A `.fit` written by the LaueTools *calibration* panel can miss
    any of those, so a regex scan of the blocks actually needed is used as a
    fallback.

    Returns
    -------
    dict with keys
        ``ub_matrix``               (3, 3) array
        ``calibration_parameters``  [distance, x_center, y_center, x_beta, x_gamma]
        ``pixel_size``              mm
        ``frame_shape``             (n_rows, n_cols)
        ``element``                 material name recorded in the file, or None
        ``lattice_parameters``      refined cell recorded in the file, or None
        ``source``                  ``"lauexplore"`` or ``"regex"``

    ``element`` and ``lattice_parameters`` are returned so the caller can check
    that the cell used to simulate matches the cell the file was indexed with —
    see `check_cell_consistency`.
    """
    fit_path = Path(fit_path)

    try:
        from lauexplore import FitFile

        # FitFile prints a line for every header line it does not recognise.
        # On a calibration-panel file that is hundreds of lines of noise before
        # it fails anyway, so the attempt is made quietly and only the reason
        # for the failure is reported.
        with contextlib.redirect_stdout(io.StringIO()):
            fit = FitFile(str(fit_path))
        framedim = fit.CCDdict["framedim"]
        return {
            "ub_matrix": np.asarray(fit.UB, dtype=float),
            "calibration_parameters": list(fit.CCDdict["DetectorParameters"]),
            "pixel_size": float(fit.CCDdict["pixelsize"]),
            "frame_shape": (int(float(framedim[0])), int(float(framedim[1]))),
            "element": getattr(fit, "element", None),
            "lattice_parameters": getattr(fit, "new_lattice_parameters", None),
            "source": "lauexplore",
        }
    except Exception as exc:                                   # noqa: BLE001
        print(f"FitFile could not read {fit_path.name} ({type(exc).__name__}: {exc});"
              f" falling back to a direct scan of the file.")

    return _scan_fit_file(fit_path)


def _scan_fit_file(fit_path: Path) -> dict:
    """Pull UB and the CCD block out of a `.fit` without parsing the rest."""
    text = fit_path.read_text()

    ub = _matrix_after(text, r"#UB matrix in q=\s*\(UB\)\s*B0\s*G\*")
    if ub is None:
        raise ValueError(f"No '#UB matrix' block found in {fit_path}")

    det = re.search(r"#DetectorParameters\s*\n#(.+)", text)
    pix = re.search(r"#pixelsize\s*\n#([0-9.eE+-]+)", text)
    dim = re.search(r"#Frame dimensions\s*\n#(.+)", text)
    if det is None or pix is None or dim is None:
        raise ValueError(
            f"{fit_path} has no complete CCD block "
            f"(#DetectorParameters / #pixelsize / #Frame dimensions)"
        )

    calib = _floats(det.group(1))
    framedim = _floats(dim.group(1))
    if len(calib) != 5 or len(framedim) != 2:
        raise ValueError(
            f"{fit_path}: expected 5 detector parameters and 2 frame dimensions, "
            f"read {len(calib)} and {len(framedim)} from\n"
            f"  {det.group(1).strip()}\n  {dim.group(1).strip()}"
        )

    element = re.search(r"#Element\s*\n#(.+)", text)
    cell = re.search(r"#new lattice parameters\s*\n#(.+)", text)

    return {
        "ub_matrix": ub,
        "calibration_parameters": calib,
        "pixel_size": float(pix.group(1)),
        "frame_shape": (int(framedim[0]), int(framedim[1])),
        "element": element.group(1).strip() if element else None,
        "lattice_parameters": np.array(_floats(cell.group(1))) if cell else None,
        "source": "regex",
    }


def _floats(text: str) -> list[float]:
    """Numbers on a line, tolerating the reprs LaueTools writes.

    `DetectorCalibration.py` writes `#DetectorParameters` as a list of
    ``np.float64(99.967)`` reprs and `#Frame dimensions` as a tuple in
    parentheses rather than a list in brackets. Both defeat a `float()` per
    whitespace-separated token, which is what `lauexplore`'s `FitFile` does —
    it raises `ValueError` on the first one.

    Stripping the constructor before scanning is not cosmetic: a bare number
    scan over ``np.float64(99.967)`` reads the ``64`` of ``float64`` as a
    value, silently shifting every parameter that follows.
    """
    cleaned = re.sub(r"np\.\w+\(", "(", text)
    return [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)]


def _matrix_after(text: str, header_pattern: str) -> np.ndarray | None:
    """The 3x3 matrix written on the three commented lines after a header."""
    m = re.search(header_pattern, text)
    if m is None:
        return None
    rows = []
    for line in text[m.end():].splitlines()[1:4]:
        vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if len(vals) != 3:
            return None
        rows.append([float(v) for v in vals])
    return np.array(rows)


def check_cell_consistency(calib: dict, material: str,
                           material_dictionary: dict | None = None,
                           tol: float = 5e-3) -> None:
    """Warn when the cell used to simulate is not the cell the `.fit` used.

    `Prepare_Grain` applies the B0 of *material* to the UB read from the file.
    The UB in a `.fit` is defined as ``q = UB B0 G*`` with the B0 of the
    **nominal** cell of its ``#Element`` — so that is what has to match. If the
    file was indexed with a different cell, every simulated position is
    systematically displaced: a silent failure that looks like a calibration
    error.

    The refined cell in ``#new lattice parameters`` is deliberately *not* the
    thing compared. It differs from the nominal one by the strain, which is
    already folded into UB; treating that difference as a mismatch would warn
    loudest exactly when the sample is most strained and the positions are in
    fact correct. It is printed as information.

    Prints rather than raises, because a deliberate change of cell is a
    legitimate thing to do.
    """
    from LaueTools.dict_LaueTools import dict_Materials

    mat_dict = material_dictionary if material_dictionary is not None else dict_Materials
    if material not in mat_dict:
        print(f"WARNING: '{material}' is not in the material dictionary given.")
        return

    sim_cell = np.asarray(mat_dict[material][1], dtype=float)
    element = calib.get("element")
    print(f"  simulating '{material}' with cell {np.round(sim_cell, 4)}")

    if element is None:
        print("  the .fit records no #Element — confirm by hand that it was "
              "indexed with this cell.")
    elif element == material:
        print(f"  .fit was indexed as {element!r} — same label, cells agree.")
    elif element not in mat_dict:
        print(f"  WARNING: the .fit was indexed as {element!r}, which is not in "
              f"the material dictionary given, so its cell cannot be checked "
              f"against {material!r}.")
    else:
        fit_cell = np.asarray(mat_dict[element][1], dtype=float)
        print(f"  .fit was indexed as {element!r} with cell {np.round(fit_cell, 4)}")
        rel = np.abs(sim_cell[:3] - fit_cell[:3]) / fit_cell[:3]
        if np.any(rel > tol):
            print(f"  WARNING: lattice constants differ by up to "
                  f"{100 * rel.max():.2f}% — simulated positions will be "
                  f"systematically displaced.")
        else:
            print("  cells agree.")

    refined = calib.get("lattice_parameters")
    if refined is not None:
        refined = np.asarray(refined, dtype=float)
        print(f"  refined cell in the .fit: {np.round(refined, 4)}")
        # Measured against the cell the file was *indexed* with, not against the
        # simulation cell: when those two disagree the difference is a mismatch,
        # not strain, and quoting it as strain would hide the mismatch.
        if element in mat_dict:
            nominal = np.asarray(mat_dict[element][1], dtype=float)
            strain = 100 * (refined[:3] - nominal[:3]) / nominal[:3]
            print(f"    ({', '.join(f'{s:+.2f}%' for s in strain)} on a, b, c vs the "
                  f"nominal {element!r} cell — this is the strain, already "
                  f"contained in UB; not a mismatch)")


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_pattern(
    material: str,
    ub_matrix: np.ndarray,
    calibration_parameters,
    pixel_size: float,
    frame_shape: tuple[int, int],
    *,
    Emin: float = 5.0,
    Emax: float = 29.0,
    material_dictionary: dict | None = None,
    remove_harmonics: bool = True,
    detector_diameter: float = 148.1212,
) -> pd.DataFrame:
    """Simulate a Laue pattern with the detector geometry given explicitly.

    Deliberately does not go through `lauexplore.peaks.simulate`, which takes
    the pixel size and frame shape from ``dict_CCD[camera_label]`` and defaults
    that label to ``"sCMOS"``. On an Eiger that wrong pixel size displaces
    simulated positions by 5-10+ px near the detector edges — enough for a ROI
    to miss its spot entirely. Here the geometry comes from the `.fit`.

    Parameters
    ----------
    material : key in *material_dictionary*, or in LaueTools' built-in one.
    ub_matrix : (3, 3) orientation matrix, as recorded in the `.fit`.
    calibration_parameters : [distance, x_center, y_center, x_beta, x_gamma].
    pixel_size : mm.
    frame_shape : (n_rows, n_cols).
    Emin, Emax : energy range in keV.
    remove_harmonics : drop (2h, 2k, 2l), (3h, 3k, 3l)… which land on the same
        pixel as (h, k, l) and would otherwise produce duplicate ROIs at one
        position. Keep True unless you specifically want the harmonic list.
    detector_diameter : simulation cutoff in mm before pixel conversion.
        Multiplied by 1.75 internally, as LaueTools' own callers do.

    Returns
    -------
    DataFrame with columns ``h, k, l, 2theta, chi, X, Y, Energy``, restricted
    to reflections landing inside the frame.
    """
    from LaueTools.CrystalParameters import Prepare_Grain
    from LaueTools.dict_LaueTools import dict_Materials
    from LaueTools.lauecore import SimulateLaue_full_np

    ub_matrix = np.asarray(ub_matrix, dtype=float)
    if ub_matrix.shape != (3, 3):
        raise ValueError(f"ub_matrix must be (3, 3), got {ub_matrix.shape}")

    mat_dict = material_dictionary if material_dictionary is not None else dict_Materials
    grain = Prepare_Grain(material, ub_matrix, dictmaterials=mat_dict)

    result = SimulateLaue_full_np(
        grain, Emin, Emax, list(calibration_parameters),
        detectordiameter=detector_diameter * 1.75,
        pixelsize=pixel_size,
        dim=tuple(frame_shape),
        dictmaterials=mat_dict,
        kf_direction="Z>0",
        removeharmonics=1 if remove_harmonics else 0,
    )

    two_theta, chi, hkl, x, y, energy = result[0], result[1], result[2], result[3], result[4], result[5]
    keep = (x > 0) & (x < frame_shape[1]) & (y > 0) & (y < frame_shape[0])

    return pd.DataFrame({
        "h": hkl[keep, 0], "k": hkl[keep, 1], "l": hkl[keep, 2],
        "2theta": two_theta[keep], "chi": chi[keep],
        "X": x[keep], "Y": y[keep],
        "Energy": energy[keep],
    }).reset_index(drop=True)


# ── Masks: detector gaps and substrate exclusion zones ────────────────────────

def detector_mask_from_stack(frames: np.ndarray) -> np.ndarray:
    """Valid-pixel mask from several frames: True where any frame ever counted.

    More robust than ``image != 0`` on a single frame, which also marks as dead
    every pixel that merely happened to record zero counts there. A pixel that
    counts in no frame at all is a gap or a dead pixel; one that counts in any
    frame is alive.
    """
    frames = np.asarray(frames)
    if frames.ndim == 2:
        return frames > 0
    return frames.max(axis=0) > 0


def exclusion_halves(exclusion_half) -> tuple[int, int]:
    """Normalise an exclusion size to ``(half_x, half_y)`` in pixels.

    Accepts a scalar for a square zone, or a ``(half_x, half_y)`` pair. The
    substrate reflections are streaked rather than round — an isotropic zone
    wide enough to cover the long axis eats the neighbouring material spot,
    while one sized for the short axis lets the streak leak into the ROI — so
    the two half-widths are independent.

    ``half_x`` is along columns and ``half_y`` along rows, matching the
    ``(X, Y) = (col, row)`` convention used for positions throughout.
    """
    if np.isscalar(exclusion_half):
        h = int(exclusion_half)
        return h, h
    hx, hy = exclusion_half
    return int(hx), int(hy)


def forbidden_mask(
    shape: tuple[int, int],
    blacklist_xy: np.ndarray,
    exclusion_half,
    *,
    detector_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Valid-pixel mask with a rectangular exclusion zone around each blacklisted spot.

    Returns ``True`` for pixels that may be used. Substrate spots are blanked
    over a ``(2*half_x + 1) x (2*half_y + 1)`` rectangle, so that a material ROI
    overlapping one of them contributes none of those pixels to its centre of
    mass — the exclusion is at pixel level, not at peak level.

    Parameters
    ----------
    shape : (n_rows, n_cols) of the detector frame.
    blacklist_xy : (N, 2) simulated substrate positions as (X, Y) = (col, row).
    exclusion_half : half-widths of the forbidden rectangle in pixels, either a
        scalar for a square or ``(half_x, half_y)``. See `exclusion_halves`.
    detector_mask : optional valid-pixel mask (gaps, dead pixels) to AND with.

    .. note:: The rectangle is axis-aligned. Where the substrate streaks run at
       an angle, ``half_x`` and ``half_y`` have to cover the streak's bounding
       box, which removes more than the streak itself.
    """
    mask = np.ones(shape, dtype=bool)

    blacklist_xy = np.asarray(blacklist_xy, dtype=float)
    if blacklist_xy.size:
        cols = np.rint(blacklist_xy[:, 0]).astype(int)
        rows = np.rint(blacklist_xy[:, 1]).astype(int)
        hx, hy = exclusion_halves(exclusion_half)
        for r, c in zip(rows, cols):
            r0, r1 = max(r - hy, 0), min(r + hy + 1, shape[0])
            c0, c1 = max(c - hx, 0), min(c + hx + 1, shape[1])
            if r0 < r1 and c0 < c1:
                mask[r0:r1, c0:c1] = False

    if detector_mask is not None:
        mask &= np.asarray(detector_mask, dtype=bool)

    return mask


def flag_close_neighbours(sim: pd.DataFrame, min_sep: float) -> pd.DataFrame:
    """Mark predicted spots that have another predicted spot within *min_sep*.

    With one average UB the buffer and the layers share each hkl, so they
    collapse onto a single predicted position and blend inside one ROI — which
    is the intended measurement. What this flags is the other case: two
    *different* hkl landing close enough that one ROI would mix unrelated
    reflections, and the COM would mean nothing.

    Adds ``n_neighbours`` and ``d_nearest`` columns and a boolean
    ``contaminated``. Nothing is dropped — filtering is left to the caller.
    """
    sim = sim.copy()
    xy = sim[["X", "Y"]].to_numpy(dtype=float)
    if len(xy) < 2:
        sim["n_neighbours"] = 0
        sim["d_nearest"] = np.inf
        sim["contaminated"] = False
        return sim

    tree = cKDTree(xy)
    dist, _ = tree.query(xy, k=2)            # k=1 is the point itself
    sim["d_nearest"] = dist[:, 1]
    sim["n_neighbours"] = [len(tree.query_ball_point(p, min_sep)) - 1 for p in xy]
    sim["contaminated"] = sim["d_nearest"] < min_sep
    return sim


# ── Background ────────────────────────────────────────────────────────────────

def gaussian_background(image: np.ndarray, valid_mask: np.ndarray,
                        sigma: float = 251.0) -> np.ndarray:
    """Smooth background estimate by FFT Gaussian filtering, gap-aware.

    ``background = (G_sigma * W·I) / (G_sigma * W)`` with ``W`` the valid-pixel
    mask, so detector gaps neither leak zeros into the estimate nor create a
    dip at their edges. FFT-based, so a large sigma costs no more than a small
    one.

    Reimplemented here rather than imported from `nrxrdct.laue.segmentation`,
    which is a separate clone not installed in this environment.
    """
    def _fft_gauss(arr: np.ndarray) -> np.ndarray:
        f = sp_fft.fft2(arr, workers=-1)
        ndi.fourier_gaussian(f, sigma=sigma, output=f)
        return sp_fft.ifft2(f, workers=-1).real.astype(np.float32)

    img = np.asarray(image, dtype=np.float32).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    img[~valid_mask] = 0.0

    smooth = _fft_gauss(img)
    norm = _fft_gauss(valid_mask.astype(np.float32))
    norm[norm < 1e-6] = 1.0
    return smooth / norm


# ── Core measurement ──────────────────────────────────────────────────────────

def _lower_tail_sigma(values: np.ndarray, pedestal: float,
                      bg_percentile: float) -> float:
    """Noise sigma from the lower tail only, without the half-distribution bias.

    Only pixels below the pedestal may be used: everything above it is where a
    spot would be, and a spot in the estimate inflates the noise and hides
    itself. But a MAD taken about the pedestal over that lower tail is *not*
    sigma — it measures the spread of half a distribution and comes out about
    0.65 sigma for Gaussian noise, which inflates every SNR by half again.
    That is not a small bookkeeping error: with the default box, pure noise
    then scores snr ~ 6, so a `min_snr` of 3 rejects nothing at all.

    Two lower-tail quantiles fix it exactly. For Gaussian noise the gap between
    the p-th and (p/4)-th percentiles is a known multiple of sigma, so dividing
    by that multiple recovers sigma itself. Both quantiles stay below the
    pedestal, so the estimator keeps the property that made the one-sided MAD
    attractive in the first place.
    """
    from scipy.stats import norm

    p_hi = float(np.clip(bg_percentile, 2.0, 90.0))
    p_lo = p_hi / 4.0
    span = float(norm.ppf(p_hi / 100.0) - norm.ppf(p_lo / 100.0))

    q_lo = float(np.percentile(values, p_lo))
    sigma = (pedestal - q_lo) / span if span > 0 else 0.0
    if sigma > 0:
        return sigma

    # Degenerate tail (quantised counts, a nearly empty ROI): fall back to the
    # one-sided MAD, biased but better than declaring the noise to be zero,
    # which would disable both the threshold and min_snr.
    low = values[values <= pedestal]
    return float(1.4826 * np.median(np.abs(low - pedestal))) if low.size else 0.0


def measure_roi(
    image: np.ndarray,
    valid_mask: np.ndarray,
    center_xy: tuple[float, float],
    boxsize: int,
    *,
    bg_percentile: float = 20.0,
    noise_nsigma: float = 1.5,
    min_counts: float = 50.0,
    min_snr: float = 3.0,
    min_valid_frac: float = 0.35,
    max_edge_weight_frac: float = 0.02,
) -> dict:
    """Centre of mass and shape moments of one ROI, excluding masked pixels.

    The ROI is a ``(2 * boxsize + 1)²`` box centred on *center_xy*, which is
    normally a *predicted* position. Masked pixels — detector gaps and the
    substrate exclusion zones — take no part in the background estimate, the
    centre of mass or the moments.

    Thresholding
    ------------
    A residual pedestal spread over the box pulls the COM towards the box
    centre: with signal ``S`` and pedestal total ``B`` the measured
    displacement is attenuated by ``S / (S + B)``. If the pedestal varies over
    a map, so does the attenuation, and the COM appears to move where nothing
    has. That is why the pedestal is removed per-ROI here even when a global
    background has already been subtracted.

    The cut applied afterwards is ``noise_nsigma`` times the *noise*, never a
    fraction of the peak. A fraction-of-peak threshold cuts a different part of
    the tail as a spot gets brighter or fainter, which at the edge of a
    material — where intensity changes fastest — manufactures exactly the COM
    drift one is trying to measure.

    Parameters
    ----------
    image : (Ny, Nx) frame, background-subtracted or raw.
    valid_mask : (Ny, Nx) bool, True where a pixel may be used.
    center_xy : (X, Y) = (column, row), 0-based, sub-pixel allowed.
    boxsize : ROI half-width in pixels.
    bg_percentile : percentile of the valid ROI pixels taken as the pedestal.
        A low percentile rather than the four corners, because in a crowded
        pattern a corner often contains a neighbouring spot.
    noise_nsigma : cut at this many robust standard deviations above the
        pedestal. 0 disables the cut and leaves only the clip at zero.
    min_counts, min_snr, min_valid_frac : acceptance thresholds. A ROI failing
        any of them is returned with ``accepted=False`` — never dropped
        silently, so the reason stays visible.
    max_edge_weight_frac : reject when more than this fraction of the
        thresholded signal sits on pixels touching the mask. See "Truncation"
        below. 0 disables the test.

    Truncation by the mask
    ----------------------
    A ROI overlapping a forbidden zone is not automatically bad: an exclusion
    square clipping an empty corner costs nothing. What is fatal is the mask
    reaching into the *spot*, because the COM is then the centre of whatever
    survived the cut rather than of the spot, and it moves as the cut moves.

    ``min_valid_frac`` cannot express that — it measures the box, and the box
    is not the spot. The same 34% of a 31x31 box is harmless in a corner and
    fatal through the core. So two spot-relative tests are applied instead:

    * the predicted position itself landing on a masked pixel, which means the
      ROI is centred inside the forbidden zone;
    * ``edge_weight_frac``, the share of the thresholded signal sitting on
      valid pixels that touch a masked one. A spot that stops well short of the
      mask contributes nothing to it; a spot cut through carries much of its
      weight right against the cut.

    Both cover detector gaps as well as substrate zones — a spot sliced by a
    module gap is untrustworthy for exactly the same reason.

    Returns
    -------
    dict with the COM (``X``, ``Y``), the displacement from *center_xy*
    (``dX``, ``dY``, ``dR``), the shape moments, and the diagnostics needed to
    decide whether a COM displacement is physical.
    """
    # Below this the pedestal is the single darkest pixel of the ROI, the noise
    # estimate collapses to zero, and both the n-sigma cut and min_snr stop
    # applying — silently, since an infinite SNR passes every test.
    if bg_percentile < 1.0:
        raise ValueError(
            f"bg_percentile={bg_percentile} is too low to estimate the noise; "
            f"the pedestal becomes the darkest pixel and both the n-sigma "
            f"threshold and min_snr stop having any effect. Use >= 1."
        )

    boxsize = int(boxsize)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    centre_px = (int(round(cx)), int(round(cy)))

    crop, (row0, col0) = extract_crop(image, centre_px, boxsize, coords="numpy")
    vmask_crop, _ = extract_crop(valid_mask.astype(np.float32), centre_px,
                                 boxsize, coords="numpy")
    vmask_crop = vmask_crop > 0.5            # zero padding off-frame = invalid

    out = {
        "X_pred": cx, "Y_pred": cy,
        "boxsize": boxsize,
        "n_valid": int(vmask_crop.sum()),
        "valid_frac": float(vmask_crop.mean()),
        "X": np.nan, "Y": np.nan, "dX": np.nan, "dY": np.nan, "dR": np.nan,
        "total_counts": 0.0, "peak_value": np.nan,
        "bg_level": np.nan, "bg_std": np.nan, "snr": np.nan,
        "lambda1": np.nan, "lambda2": np.nan, "aspect_ratio": np.nan,
        "theta": np.nan, "fwhm_maj": np.nan, "fwhm_min": np.nan,
        "skewness": np.nan, "kurtosis": np.nan,
        "centre_masked": False, "edge_weight_frac": np.nan, "d_mask": np.inf,
        "accepted": False, "reject_reason": "",
    }

    # The predicted position itself falling on a masked pixel means the ROI is
    # centred inside a forbidden zone. Recorded now, judged at the end, so the
    # measurement that led to the rejection stays visible.
    ry, rx = centre_px[1] - row0, centre_px[0] - col0
    if 0 <= ry < vmask_crop.shape[0] and 0 <= rx < vmask_crop.shape[1]:
        out["centre_masked"] = not bool(vmask_crop[ry, rx])

    if out["valid_frac"] < min_valid_frac:
        out["reject_reason"] = "too few valid pixels"
        return out

    valid_values = crop[vmask_crop].astype(np.float64)
    bg = float(np.percentile(valid_values, bg_percentile))
    bg_std = _lower_tail_sigma(valid_values, bg, bg_percentile)
    out["bg_level"], out["bg_std"] = bg, bg_std

    sig = crop.astype(np.float64) - bg
    sig[~vmask_crop] = 0.0
    if noise_nsigma > 0 and bg_std > 0:
        sig[sig < noise_nsigma * bg_std] = 0.0
    np.clip(sig, 0.0, None, out=sig)

    total = float(sig.sum())
    peak = float(sig.max())
    out["total_counts"], out["peak_value"] = total, peak
    out["snr"] = peak / bg_std if bg_std > 0 else np.inf

    if total <= 0:
        out["reject_reason"] = "no signal above threshold"
        return out

    h, w = sig.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x_com = float((sig * xx).sum() / total)
    y_com = float((sig * yy).sum() / total)

    out["X"] = col0 + x_com
    out["Y"] = row0 + y_com
    out["dX"] = out["X"] - cx
    out["dY"] = out["Y"] - cy
    out["dR"] = float(np.hypot(out["dX"], out["dY"]))

    lam1, lam2, aspect, theta = inertia_tensor(sig, x_com, y_com)
    out["lambda1"], out["lambda2"] = float(lam1), float(lam2)
    out["aspect_ratio"], out["theta"] = float(aspect), float(theta)
    out["fwhm_maj"] = float(_FWHM_FACTOR * np.sqrt(max(lam1, 0.0)))
    out["fwhm_min"] = float(_FWHM_FACTOR * np.sqrt(max(lam2, 0.0)))

    skew, kurt = streak_moments(sig, x_com, y_com, theta)
    out["skewness"], out["kurtosis"] = float(skew), float(kurt)

    # How much of the spot is leaning on the mask, and how far the mask is from
    # the centre of mass. Both are 0 / inf when the ROI holds no masked pixel.
    if not vmask_crop.all():
        touching = ndi.binary_dilation(~vmask_crop, np.ones((3, 3), bool)) & vmask_crop
        out["edge_weight_frac"] = float(sig[touching].sum() / total)
        dist = ndi.distance_transform_edt(vmask_crop)
        out["d_mask"] = float(dist[int(round(y_com)), int(round(x_com))])
    else:
        out["edge_weight_frac"] = 0.0

    if out["centre_masked"]:
        out["reject_reason"] = "prediction inside forbidden zone"
    elif max_edge_weight_frac > 0 and out["edge_weight_frac"] > max_edge_weight_frac:
        out["reject_reason"] = "spot truncated by mask"
    elif total < min_counts:
        out["reject_reason"] = "below min_counts"
    elif out["snr"] < min_snr:
        out["reject_reason"] = "below min_snr"
    else:
        out["accepted"] = True

    return out


def build_peaklist(
    image: np.ndarray,
    sim: pd.DataFrame,
    valid_mask: np.ndarray,
    boxsize: int,
    **measure_kwargs,
) -> pd.DataFrame:
    """Measure every predicted position and return one row per spot.

    Serial on purpose: a few hundred ROIs of a few thousand pixels each take
    well under a second, and a process pool would cost more to start than it
    saves. When this is scaled to a whole map, parallelise over *frames* with
    `laue.scan_pipeline._run_parallel` (joblib/loky) rather than over ROIs —
    `concurrent.futures.ProcessPoolExecutor` deadlocks under a Jupyter kernel.

    Returns the simulation columns (h, k, l, Energy, …) joined to the
    measurement columns, rejected rows included and marked.
    """
    rows = [
        measure_roi(image, valid_mask, (r.X, r.Y), boxsize, **measure_kwargs)
        for r in sim.itertuples(index=False)
    ]
    meas = pd.DataFrame(rows)
    keep = [c for c in ("h", "k", "l", "Energy", "2theta", "chi",
                        "d_nearest", "n_neighbours", "contaminated")
            if c in sim.columns]
    return pd.concat([sim[keep].reset_index(drop=True), meas], axis=1)


# ── Confirming the substrate prediction against the image ─────────────────────

def confirm_substrate_spots(
    image: np.ndarray,
    sim: pd.DataFrame,
    valid_mask: np.ndarray,
    *,
    boxsize: int = 8,
    min_snr: float = 5.0,
    max_shift: float = 6.0,
    material_xy: np.ndarray | None = None,
    **measure_kwargs,
) -> pd.DataFrame:
    """Mark which predicted substrate positions actually carry a spot.

    The substrate simulation is a *mask generator*, not a physical model, and
    its two errors do not cost the same. A predicted spot that is not there
    costs one small excluded region. A real spot that was not predicted drags a
    material centre of mass with nothing to show for it. The asymmetry argues
    for predicting generously.

    Generously means, in particular, past the experiment's own energy range: a
    thick perfect substrate diffracts strongly enough to appear in the weak
    high-energy tail of a white beam, at energies where a thin epilayer
    produces nothing measurable. The substrate's ``Emax`` is not the material's
    ``Emax``.

    Predicting generously *without* checking is the opposite failure. Taking
    sapphire from 29 to 50 keV grows the prediction from 233 to 1167 positions,
    and the forbidden zones built from all of them reject a third of the
    material spots — while most of those extra positions have no spot under
    them at all. So each candidate is measured, and only the ones carrying
    signal are kept.

    Parameters
    ----------
    image : frame to check against, background-subtracted.
    sim : predicted substrate positions, with ``X`` and ``Y`` columns.
    valid_mask : gaps and dead pixels. Deliberately *not* the forbidden mask —
        the forbidden zones are what this function is being used to build.
    boxsize : half-width of the check box. Small, so a neighbouring material
        spot is less likely to be counted as a confirmation.
    min_snr : peak over noise required to call a position occupied.
    max_shift : how far the local centre of mass may sit from the prediction
        and still be considered the same spot.
    material_xy : (N, 2) predicted positions of the *material*, if known.
        Strongly recommended — see below.

    Why the material positions matter
    ---------------------------------
    A candidate sitting a few pixels from a bright material reflection is
    confirmed by that reflection's own intensity, and the forbidden zone then
    lands on the material spot and destroys it. Tightening ``max_shift`` until
    that stops also stops real substrate spots being found, and the two cannot
    be separated by distance alone. Measured on one frame, sapphire coverage
    against material yield::

        max_shift=3, no material_xy    260 zones   103/165 covered   157 kept
        max_shift=6, no material_xy    362 zones   164/165 covered   130 kept
        max_shift=6, with material_xy  337 zones   164/165 covered   154 kept

    With the positions supplied, a candidate is rejected when the intensity
    confirming it is nearer to a material reflection than to the candidate
    itself — the signal is spoken for. That buys back the coverage without
    paying for it in material spots, so ``max_shift`` can be loose enough to
    absorb the residual error in the substrate prediction.

    Returns
    -------
    A copy of *sim* with ``confirmed`` (bool) plus the ``snr``, ``dR``,
    ``total_counts`` and ``d_material`` behind the decision, so a rejected
    candidate can be inspected rather than merely disappearing.
    """
    rows = [
        measure_roi(image, valid_mask, (r.X, r.Y), boxsize,
                    min_snr=0.0, min_counts=0.0, max_edge_weight_frac=0.0,
                    **measure_kwargs)
        for r in sim.itertuples(index=False)
    ]
    meas = pd.DataFrame(rows)

    out = sim.copy().reset_index(drop=True)
    out["snr"] = meas["snr"].to_numpy()
    out["total_counts"] = meas["total_counts"].to_numpy()
    out["dR"] = meas["dR"].to_numpy()

    dR = meas["dR"].to_numpy()
    has_spot = (meas["snr"].to_numpy() >= min_snr) & (dR <= max_shift) & np.isfinite(dR)

    # Which candidates were confirmed by intensity that belongs to the material.
    out["d_material"] = np.inf
    if material_xy is not None and len(material_xy):
        com = np.column_stack([meas["X"].to_numpy(), meas["Y"].to_numpy()])
        ok = np.isfinite(com).all(axis=1)
        d_mat = np.full(len(out), np.inf)
        if ok.any():
            d_mat[ok] = cKDTree(np.asarray(material_xy, float)).query(com[ok])[0]
        out["d_material"] = d_mat
        claimed = d_mat < dR
    else:
        claimed = np.zeros(len(out), bool)

    out["confirmed"] = has_spot & ~claimed

    n = int(out["confirmed"].sum())
    note = (f", {int((has_spot & claimed).sum())} dropped as material reflections"
            if material_xy is not None and len(material_xy) else
            "  [no material_xy given: candidates next to a material spot will be "
            "confirmed by it]")
    print(f"confirm_substrate_spots: {n} of {len(out)} predicted positions carry a "
          f"spot (snr >= {min_snr}, within {max_shift} px){note}")
    return out


# ── Simulated-vs-measured offset (coordinate convention check) ────────────────

def estimate_sim_offset(
    image: np.ndarray,
    sim: pd.DataFrame,
    valid_mask: np.ndarray,
    *,
    boxsize: int = 12,
    n_brightest: int = 40,
    max_shift: float = 4.0,
) -> tuple[float, float]:
    """Median (dX, dY) between predicted positions and the local COM.

    Settles the 0-based / 1-based question empirically instead of by
    assumption: LaueTools' own peak files follow the XMAS 1-based convention
    while numpy indexing is 0-based, and a systematic one-pixel offset is
    invisible in a full-detector plot but is large compared with the COM shifts
    this module measures.

    Uses only the brightest, well-isolated, uncontaminated spots, and only
    those whose COM lands within *max_shift* of the prediction, so that spots
    the average UB does not describe well do not drag the estimate.

    Returns ``(dX, dY)`` to be *added* to the simulated positions.
    """
    cand = sim
    if "contaminated" in cand.columns:
        cand = cand[~cand["contaminated"]]
    if not len(cand):
        return 0.0, 0.0

    meas = build_peaklist(image, cand, valid_mask, boxsize, min_snr=0.0,
                          min_counts=0.0)
    meas = meas[meas["accepted"] | (meas["total_counts"] > 0)]
    meas = meas[np.hypot(meas["dX"], meas["dY"]) < max_shift]
    if not len(meas):
        print("estimate_sim_offset: no spot matched within max_shift; offset left at 0.")
        return 0.0, 0.0

    meas = meas.nlargest(min(n_brightest, len(meas)), "total_counts")
    dx, dy = float(meas["dX"].median()), float(meas["dY"].median())
    print(f"estimate_sim_offset: {len(meas)} spots, median offset "
          f"dX={dx:+.2f} px, dY={dy:+.2f} px "
          f"(scatter {meas['dX'].std():.2f}, {meas['dY'].std():.2f})")
    return dx, dy


# ── LaueTools .dat output ─────────────────────────────────────────────────────

def to_lauetools_dat(peaks: pd.DataFrame, outpath, *, accepted_only: bool = True):
    """Write a LaueTools-compatible `.dat` peak file.

    Columns, in LaueTools' order::

        peak_X peak_Y peak_I peak_fwaxmaj peak_fwaxmin peak_inclination
        Xdev Ydev peak_bkg

    Filled from the COM measurement:

    * ``peak_X``, ``peak_Y`` — the centre of mass. **Not** a fitted centre; see
      the module header.
    * ``peak_I`` — background-subtracted peak pixel value, matching the
      convention of a peak-search `.dat`. Integrated intensity is kept in the
      DataFrame as ``total_counts`` but is not written here.
    * ``peak_fwaxmaj/min``, ``peak_inclination`` — from the inertia tensor, the
      equivalent Gaussian FWHM along the principal axes and their angle. These
      are moments of the actual intensity distribution, valid for a distorted
      or blended spot in a way a fitted Gaussian width would not be.
    * ``Xdev``, ``Ydev`` — written as 0. In LaueTools these are *fit residuals*,
      and no fit was done. The displacement from the predicted position lives
      in the DataFrame as ``dX``/``dY``; it is deliberately not written into
      these columns so nothing downstream mistakes it for a residual.
    * ``peak_bkg`` — the per-ROI pedestal that was subtracted.

    Rows are sorted by descending ``peak_I``, as LaueTools expects.
    """
    outpath = str(outpath)
    if not outpath.endswith(".dat"):
        outpath += ".dat"

    df = peaks[peaks["accepted"]] if accepted_only else peaks
    if not len(df):
        raise ValueError("No accepted peaks to write.")

    out = pd.DataFrame({
        "peak_X": df["X"].to_numpy(),
        "peak_Y": df["Y"].to_numpy(),
        "peak_I": df["peak_value"].to_numpy(),
        "peak_fwaxmaj": df["fwhm_maj"].to_numpy(),
        "peak_fwaxmin": df["fwhm_min"].to_numpy(),
        "peak_inclination": df["theta"].to_numpy(),
        "Xdev": np.zeros(len(df)),
        "Ydev": np.zeros(len(df)),
        "peak_bkg": df["bg_level"].to_numpy(),
    }).sort_values("peak_I", ascending=False).reset_index(drop=True)

    out.to_csv(outpath, sep=" ", index=False)
    print(f"Wrote {len(out)} peaks to {outpath}")
    return out
