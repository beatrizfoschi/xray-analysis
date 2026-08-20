"""Crystallographic and detector geometry for the satellite analysis.

Everything here derives from indexing (hkl, lattice, UB) or from the LaueTools
`.det` calibration — never from user-supplied angles. Split out of `metrics.py`,
which had grown to hold indicators, geometry and the period routes at once.

Landmines, each of which has cost a wrong result at least once:

- The matrix from a `.det` is an orientation only — a pure rotation, no lattice
  metric. Apply B before it: ``G_lab = UB @ B @ [h,k,l]``, via
  `lab_vectors_from_UB`. See `B_matrix_hexagonal` for why the name misleads and why
  omitting B tilts the direction on a hexagonal cell but not a cubic one.
- The UB inside a `.det` belongs to the **calibration crystal**, not the sample;
  `assert_ub_material` raises on the mismatch.
- A sample UB arrives in a beam-∥-x frame while LaueTools uses beam ∥ +y. Wrap it
  with `ub_from_beam_x_frame`; uncorrected, |G| and γ still come out right (both
  are frame-independent) while the predicted spot lands 25° away.
- ``(xcen, ycen)`` is the detector reference point (PONI), not a beam centre — in
  reflection geometry nothing is visible there.
- LaueTools' χ uses ``arctan``, not ``arctan2``, so it folds into (−90°, 90°].
  `assert_no_branch_crossing` enforces the condition that makes this safe.
"""

from __future__ import annotations

import math

import numpy as np
from typing import Any, Dict, List, Optional



# ══════════════════════════════════════════════════════════════════════════════
# Polychromatic (white-beam Laue) period extraction
#
# NOT VALIDATED AGAINST EXPERIMENT.  Selected explicitly via method=; the
# monochromatic path above remains the default.  See
# NOTES_laue_vs_mono_period.md and SPEC_polychromatic_period_implementation.md.
#
# Physics: for elastic scattering the Laue condition gives
#     û·Ĝ  = −sin θ                     (wavelength selection)
#     k̂_f = û − 2(û·Ĝ)Ĝ                 (outgoing direction)
# The second relation contains only the DIRECTION Ĝ, not |G|.  In white beam a
# change of |G| therefore selects a different wavelength but does not move the
# spot.  Satellites at G_n = G_0 + n·q·ẑ are observable only through the
# component of q·ẑ perpendicular to G_0, of size q·sin γ — the opposite
# component to the one the monochromatic route reads (q·cos γ).
# ══════════════════════════════════════════════════════════════════════════════


# ── Crystallography — all derived from indexing, never user-supplied ──────────

def gamma_from_hkl(h: int, k: int, l: int, a: float, c: float) -> float:
    """Angle (radians) between G_hkl and the c axis, for a hexagonal lattice.

    Purely crystallographic — this is the γ of the Laue treatment, and it must
    NOT be confused with the ``chi`` reported by LaueTools in (2θ, χ) detector
    space, which is a laboratory-frame azimuth around the beam.  Passing the
    latter where γ is expected has already produced a wrong result once.
    """
    G_par  = 2.0 * np.pi * l / c                                        # along c*
    G_perp = 2.0 * np.pi * np.sqrt((4.0 / 3.0) * (h * h + h * k + k * k) / a ** 2)
    return float(np.arctan2(G_perp, G_par))


def G_magnitude(h: int, k: int, l: int, a: float, c: float) -> float:
    """|G_hkl| in Å⁻¹ for a hexagonal lattice."""
    return float(2.0 * np.pi * np.sqrt(
        (4.0 / 3.0) * (h * h + h * k + k * k) / a ** 2 + l * l / c ** 2
    ))


def theta_from_G(G_mag: float, wavelength_angstrom: float) -> float:
    """Bragg angle θ (radians) selected by a node of magnitude |G| in white beam."""
    s = G_mag * wavelength_angstrom / (4.0 * np.pi)
    if not -1.0 <= s <= 1.0:
        raise ValueError(
            f"|G|·λ/4π = {s:.4f} is outside [-1, 1]: reflection {G_mag:.4f} Å⁻¹ "
            f"is not accessible at λ = {wavelength_angstrom} Å."
        )
    return float(np.arcsin(s))


def dihedral_phi(G0_lab: np.ndarray, z_lab: np.ndarray, u_lab: np.ndarray) -> float:
    """Dihedral angle φ (radians) between the (ẑ, G₀) plane and the scattering plane.

    Convention verified numerically against the reference values of the spec
    (§5.2/§5.3): the analytic route's S factor is φ-independent under this
    definition, which is what a sign or convention inversion would break.
    """
    G0h   = np.asarray(G0_lab, dtype=float)
    G0h   = G0h / np.linalg.norm(G0h)
    e_in  = np.asarray(z_lab, dtype=float) - (np.asarray(z_lab, dtype=float) @ G0h) * G0h
    n_in  = np.linalg.norm(e_in)
    if n_in == 0.0:
        raise ValueError(
            "symmetric reflection: ẑ ∥ G₀, the dihedral angle is undefined "
            "(satellites are degenerate in Laue geometry)"
        )
    e_in  = e_in / n_in
    e_out = np.cross(G0h, e_in)
    u     = np.asarray(u_lab, dtype=float)
    u_perp = u - (u @ G0h) * G0h
    return float(np.arctan2(u_perp @ e_out, u_perp @ e_in))


def build_canonical_frame(
    gamma_rad: float, theta_rad: float, phi_rad: float
) -> tuple:
    """Lab-frame (Ĝ₀, ẑ, û) reproducing a given (γ, θ, φ), for use without a UB.

    Ĝ₀ is placed along +z of an arbitrary frame and ẑ tilted by γ into +x; û is
    then fixed by the Bragg condition û·Ĝ₀ = −sin θ and by φ.  Any frame related
    to this one by a rigid rotation gives identical angular separations, so the
    choice is immaterial to the period — it only has to be self-consistent.

    Returns (G0_hat, z_hat, u_hat), each a unit 3-vector.
    """
    G0_hat = np.array([0.0, 0.0, 1.0])
    z_hat  = np.array([np.sin(gamma_rad), 0.0, np.cos(gamma_rad)])
    u_hat  = np.array([
        np.cos(theta_rad) * np.cos(phi_rad),
        np.cos(theta_rad) * np.sin(phi_rad),
        -np.sin(theta_rad),
    ])
    return G0_hat, z_hat, u_hat


def B_matrix_hexagonal(a: float, c: float) -> np.ndarray:
    """Reciprocal lattice matrix for a hexagonal cell, 2π convention.

    Columns are a*, b*, c* in the crystal Cartesian frame (z ∥ c, x ∥ a*), so
    ``|B @ [h,k,l]| == 2π/d_hkl`` and ``G_magnitude()`` agree by construction.

    A LaueTools ``.det`` does not carry this matrix, and does not claim to: it
    labels its own 3x3 "Orientation Matrix", and that is exactly what it holds — a
    pure rotation, orientation only (``refGe_22_mai.det``: det = 1.0000000,
    orthonormal to 4e-8, column norms 1). Nothing is mislabelled in the file.

    The friction is downstream. That value is conventionally read into a variable
    called ``UB``, and in the Busing & Levy convention "UB" means the *product*
    U.B — so the name invites skipping the lattice metric. Apply B explicitly::

        G_lab = UB @ B @ [h, k, l]

    Skipping it leaves the magnitude dimensionless. On a **hexagonal** cell it also
    tilts the direction, because a* != c* makes B a distortion rather than a uniform
    scale: GaN (105) comes out 9.3 deg off, (101) 17 deg, while (002) is exact
    because it lies along c. On a cubic cell B is proportional to the identity, so
    the same omission would only rescale the vector and would go unnoticed.
    """
    a_star = 4.0 * np.pi / (np.sqrt(3.0) * a)
    c_star = 2.0 * np.pi / c
    return np.array([
        [a_star, a_star / 2.0,               0.0],
        [0.0,    a_star * np.sqrt(3.0) / 2.0, 0.0],
        [0.0,    0.0,                        c_star],
    ])


def gamma_from_vectors(G: np.ndarray, z: np.ndarray) -> float:
    """Angle (radians) between a reciprocal vector and the growth axis.

    Frame-independent: gives the same value whether both vectors are expressed
    in the crystal frame or both in the laboratory frame, since UB is a rotation.
    """
    G = np.asarray(G, dtype=float)
    z = np.asarray(z, dtype=float)
    cos_g = (G @ z) / (np.linalg.norm(G) * np.linalg.norm(z))
    return float(np.arccos(np.clip(cos_g, -1.0, 1.0)))


# ── Detector geometry — LaueTools .det convention ────────────────────────────
#
# Mirrors LaueTools.LaueGeometry.calc_uflab / calc_xycam for kf_direction='Z>0'
# (read from the installed LaueTools source, not re-derived: the tilt sign
# convention differs between packages and both tilts here are well under 1°, so
# an inversion produces a small, visually invisible error).
#
# Laboratory frame: the incident beam runs along +y, hence LAB_KI below and
# 2θ = arccos(uf_y).  RECTPIX is 0 in LaueTools, so the pixel aspect correction
# is the identity and is omitted.

LAB_KI = np.array([0.0, 1.0, 0.0])


class DetectorGeometry:
    """The five LaueTools calibration parameters plus pixel size.

    ``(xcen, ycen)`` is the **detector reference point** — the orthogonal
    projection of the sample onto the detector plane, at distance ``dd``.  It is
    NOT a beam center: in reflection geometry the direct beam never reaches the
    detector, so nothing is visible there and it need not even fall inside the
    observable pattern.
    """

    __slots__ = ('dd', 'xcen', 'ycen', 'xbet', 'xgam', 'pixelsize',
                 'framedim', 'kf_direction', 'material', 'ccd_label')

    def __init__(self, dd, xcen, ycen, xbet, xgam, pixelsize,
                 framedim=None, kf_direction='Z>0',
                 material=None, ccd_label=None):
        if kf_direction != 'Z>0':
            raise NotImplementedError(
                f"kf_direction={kf_direction!r} is not implemented; only 'Z>0' "
                f"(reflection geometry) mirrors the LaueTools path used here."
            )
        self.dd           = float(dd)
        self.xcen         = float(xcen)
        self.ycen         = float(ycen)
        self.xbet         = float(xbet)
        self.xgam         = float(xgam)
        self.pixelsize    = float(pixelsize)
        self.framedim     = tuple(framedim) if framedim is not None else None
        self.kf_direction = kf_direction
        self.material     = material
        self.ccd_label    = ccd_label

    @property
    def calib(self) -> list:
        """The 5-parameter list in LaueTools order, for cross-checking."""
        return [self.dd, self.xcen, self.ycen, self.xbet, self.xgam]

    @classmethod
    def from_det_file(cls, path):
        """Read a LaueTools ``.det`` file.  Returns (geometry, UB_calib).

        The returned UB is reshaped to 3×3 — ``readfile_det`` hands back a flat
        9-element array.

        **The UB in a ``.det`` belongs to the crystal the detector was calibrated
        with**, which is usually a reference (Ge, Si), not the sample.  The
        geometry is what you want from this file; the orientation almost never
        is.  ``geom.material`` records which crystal it came from so the caller
        can check — see ``assert_ub_material``.
        """
        from LaueTools.IOLaueTools import readfile_det
        params, mat = readfile_det(str(path), nbCCDparameters=8, verbose=False)

        UB = np.asarray(mat, dtype=float)
        if UB.size != 9:
            raise ValueError(
                f'{path}: expected a 9-element orientation matrix on line 6, '
                f'got {UB.size} values.'
            )
        UB = UB.reshape(3, 3)

        dd, xcen, ycen, xbet, xgam = [float(v) for v in params[:5]]
        pixelsize = float(params[5]) if len(params) > 5 else 0.075
        framedim  = (int(params[6]), int(params[7])) if len(params) > 7 else None

        # The '# key : value' trailer is explicit, so prefer it where present.
        meta = {}
        with open(path, 'r') as f:
            for line in f:
                if line.lstrip().startswith('#') and ':' in line:
                    k, v = line.lstrip('# ').split(':', 1)
                    meta[k.strip().lower()] = v.strip()

        kf = meta.get('kf_direction', 'Z>0')
        return cls(dd, xcen, ycen, xbet, xgam, pixelsize, framedim,
                   kf_direction=kf,
                   material=meta.get('material'),
                   ccd_label=meta.get('ccdlabel')), UB

    def _tilt_terms(self):
        cosbeta = math.cos(math.pi / 2.0 - self.xbet * math.pi / 180.0)
        sinbeta = math.sin(math.pi / 2.0 - self.xbet * math.pi / 180.0)
        cosgam  = math.cos(-self.xgam * math.pi / 180.0)
        singam  = math.sin(-self.xgam * math.pi / 180.0)
        return cosbeta, sinbeta, cosgam, singam

    def __repr__(self):
        return (f'DetectorGeometry(dd={self.dd}, xcen={self.xcen}, '
                f'ycen={self.ycen}, xbet={self.xbet}, xgam={self.xgam}, '
                f'pixelsize={self.pixelsize})')


def assert_ub_material(geom: DetectorGeometry, expected_material: str) -> None:
    """Refuse a UB that belongs to the calibration crystal rather than the sample.

    A ``.det`` written by a Ge or Si calibration carries that reference crystal's
    orientation matrix.  Its **geometry** (dd, xcen, ycen, xbet, xgam, pixelsize)
    is exactly what the sample measurement needs; its **orientation** describes a
    different crystal, in a different lattice, at a different setting.  Feeding it
    to the Laue routes as the sample UB produces a confident, meaningless number.

    The sample UB comes from indexing the sample's own Laue pattern — a ``.fit``
    or ``.res`` from LaueTools indexation, not from the calibration file.
    """
    if geom.material is None:
        return
    if geom.material.strip().lower() != expected_material.strip().lower():
        raise ValueError(
            f"the .det was calibrated with {geom.material!r} but the analysis is "
            f"for {expected_material!r}.  Its geometry is fine to use, but its "
            f"orientation matrix is {geom.material}'s, not the sample's.  Load the "
            f"sample UB from the indexation of its own Laue pattern (.fit / .res) "
            f"and pass it explicitly as UB=."
        )


def pixel_to_direction(col, row, geom: DetectorGeometry) -> np.ndarray:
    """Pixel position → outgoing unit vector u_f in the laboratory frame.

    Mirrors ``LaueTools.LaueGeometry.calc_uflab`` for kf_direction='Z>0'.
    Accepts scalars or arrays; returns shape (3,) or (N, 3).
    """
    cosbeta, sinbeta, cosgam, singam = geom._tilt_terms()

    xcam1 = (np.asarray(col, dtype=float) - geom.xcen) * geom.pixelsize
    ycam1 = (np.asarray(row, dtype=float) - geom.ycen) * geom.pixelsize

    xca0 = cosgam * xcam1 - singam * ycam1
    yca0 = singam * xcam1 + cosgam * ycam1

    xO, yO, zO = geom.dd * np.array([0.0, cosbeta, sinbeta])

    xM = xO + xca0
    yM = yO + yca0 * sinbeta
    zM = zO - yca0 * cosbeta

    M = np.stack(np.broadcast_arrays(xM, yM, zM), axis=-1)
    return M / np.linalg.norm(M, axis=-1, keepdims=True)


def direction_to_pixel(uf, geom: DetectorGeometry):
    """Outgoing unit vector u_f → pixel position (col, row).

    Exact inverse of :func:`pixel_to_direction`; mirrors
    ``LaueTools.LaueGeometry.calc_xycam``.  Returns (col, row) as scalars for a
    single vector, or arrays for shape (N, 3) input.

    Raises ValueError if a direction points away from the detector plane, which
    would otherwise silently produce a mirrored position behind the sample.
    """
    cosbeta, sinbeta, cosgam, singam = geom._tilt_terms()

    uf = np.asarray(uf, dtype=float)
    single = uf.ndim == 1
    uf = np.atleast_2d(uf)
    uf = uf / np.linalg.norm(uf, axis=-1, keepdims=True)

    IOlab = geom.dd * np.array([0.0, cosbeta, sinbeta])
    unlab = IOlab / np.linalg.norm(IOlab)

    scal = uf @ unlab
    if np.any(scal <= 0.0):
        raise ValueError(
            "outgoing direction does not intersect the detector plane "
            "(u_f·n ≤ 0) — the reflection is not observable in this geometry."
        )

    IMlab = uf * (geom.dd / scal)[:, None]
    OMlab = IMlab - IOlab

    xca0 = OMlab[:, 0]
    if sinbeta != 0.0:
        yca0 = OMlab[:, 1] / sinbeta
    else:
        yca0 = -OMlab[:, 2] / cosbeta

    xcam1 = cosgam * xca0 + singam * yca0
    ycam1 = -singam * xca0 + cosgam * yca0

    col = geom.xcen + xcam1 / geom.pixelsize
    row = geom.ycen + ycam1 / geom.pixelsize

    if single:
        return float(col[0]), float(row[0])
    return col, row


def two_theta_from_direction(uf) -> np.ndarray:
    """2θ in degrees from an outgoing unit vector (ki ∥ +y, LaueTools frame)."""
    uf = np.asarray(uf, dtype=float)
    return np.degrees(np.arccos(np.clip(uf[..., 1], -1.0, 1.0)))


# ── Angular space (2θ, χ) — the space the Laue fit is closed in ──────────────
#
# ADDENDUM 2: the detector geometry is LaueTools' problem, not ours.  Measured
# peaks are converted to (2θ, χ) by LaueTools from the .det; the forward model
# predicts directions; the two are compared as angles.  This removes the
# projection factor S, the dihedral angle φ and the tilt conventions in one go.
#
# Convention, read from LaueTools.LaueGeometry.calc_uflab and verified to 1e-14°:
#     2θ = arccos(uf_y)                 polar angle from the incident beam (+y)
#     χ  = arctan(−uf_x / uf_z)         azimuth about the beam, +z toward −x
#
# χ uses arctan, not arctan2, so it is folded into (−90°, +90°].  For
# kf_direction='Z>0' the whole detector has uf_z > 0, so no pair of spots ever
# straddles the branch cut and the fold is harmless — but that is a property of
# the geometry, not a guarantee, so `assert_no_branch_crossing` checks it.


def direction_to_2theta_chi(uf) -> tuple:
    """Outgoing unit vector → (2θ, χ) in degrees, LaueTools convention.

    Identical to ``calc_uflab``'s angular output; kept here so the forward model
    can be expressed in the same convention as the measurement without a
    round trip through the detector.
    """
    uf = np.atleast_2d(np.asarray(uf, dtype=float))
    two_theta = np.degrees(np.arccos(np.clip(uf[:, 1], -1.0, 1.0)))
    chi = np.degrees(np.arctan(-uf[:, 0] / (uf[:, 2] + 1e-17)))
    if uf.shape[0] == 1:
        return float(two_theta[0]), float(chi[0])
    return two_theta, chi


def two_theta_chi_to_direction(two_theta_deg, chi_deg) -> np.ndarray:
    """(2θ, χ) → outgoing unit vector.  Assumes the uf_z > 0 hemisphere.

    Exact inverse of :func:`direction_to_2theta_chi` on that hemisphere, which is
    the whole detector for kf_direction='Z>0'.  Off it, χ's arctan fold makes the
    inverse ambiguous — see ``assert_no_branch_crossing``.
    """
    t = np.radians(np.asarray(two_theta_deg, dtype=float))
    c = np.radians(np.asarray(chi_deg, dtype=float))
    s = np.sin(t)
    return np.stack(np.broadcast_arrays(-np.sin(c) * s, np.cos(t), np.cos(c) * s),
                    axis=-1)


def assert_no_branch_crossing(uf, context: str = '') -> None:
    """Guard the arctan fold in χ: every direction must share the uf_z > 0 branch."""
    uf = np.atleast_2d(np.asarray(uf, dtype=float))
    if np.any(uf[:, 2] <= 0.0):
        raise ValueError(
            f"{context}some directions have uf_z <= 0, where LaueTools' χ = "
            f"arctan(-uf_x/uf_z) folds two distinct azimuths onto the same value. "
            f"Angular separations across that branch are wrong.  This does not "
            f"happen for kf_direction='Z>0'."
        )


def assert_reflection_observable(G_lab, u_hat, hkl, context: str = '') -> None:
    """Refuse an orientation that sends the parent reflection off the detector.

    ``uf_z <= 0`` means the model puts the reflection on the far side of the
    ``kf_direction='Z>0'`` hemisphere.  Nothing downstream detects this on its
    own: the forward fit compares *centred* directions, so a prediction pointing
    the wrong way does not blow up — it quietly drives Λ against a bound and
    reports a number.

    The dominant cause is a UB in the wrong laboratory frame, which is silent in
    every other check: |G| and γ are frame-independent, and LaueTools simulates a
    correct-looking pattern from the frame it expects.  Only the reprojection
    against a measured spot sees it.
    """
    kf = kf_hat(np.asarray(G_lab, dtype=float), np.asarray(u_hat, dtype=float))
    if kf[2] > 0.0:
        return
    h, k, l = hkl
    raise ValueError(
        f"{context}the orientation places ({h}{k}{l}) off the detector "
        f"(predicted uf_z = {kf[2]:+.4f} <= 0), so no prediction built on it is "
        f"meaningful.  The usual cause is a UB in the wrong laboratory frame: "
        f"indexing output often has the beam along +x while LaueTools uses +y. "
        f"Wrap it with ub_from_beam_x_frame, and use diagnose_ub_frame(UB, hkl, "
        f"lattice, two_theta_meas, chi_meas) to confirm which frame it is in. "
        f"Note that a UB in the wrong frame still simulates a correct-looking "
        f"pattern through LaueTools, so agreement there does not clear it."
    )


def angular_separation(two_theta_1, chi_1, two_theta_2, chi_2) -> np.ndarray:
    """Exact angular separation (radians) between two (2θ, χ) directions.

    Spherical law of cosines::

        cos Δψ = cos2θ₁·cos2θ₂ + sin2θ₁·sin2θ₂·cos(χ₁ − χ₂)

    Exact, and it needs no decision about projection.  Two things it replaces:

    * **Δ(2θ) alone is never acceptable.**  The satellite train is not generally
      radial, and the error depends on where it points: 0 % radial, −6.3 % at
      45°, −99.8 % azimuthal.  In the azimuthal case Δ(2θ) is 0.000281° against a
      true 0.149550° — the analysis would report no separation where a clear one
      exists.
    * **Naive quadrature √(Δ2θ² + Δχ²) overestimates by +44 %** azimuthally.
      (2θ, χ) are spherical, not Cartesian: the line element is
      ds² = d(2θ)² + sin²(2θ)·dχ², so Δχ must carry a sin(2θ) weight.  With it
      the error falls to 0.03 %, but there is no reason to approximate at all.
    """
    t1, c1 = np.radians(two_theta_1), np.radians(chi_1)
    t2, c2 = np.radians(two_theta_2), np.radians(chi_2)
    cos_d = (np.cos(t1) * np.cos(t2)
             + np.sin(t1) * np.sin(t2) * np.cos(c1 - c2))
    return np.arccos(np.clip(cos_d, -1.0, 1.0))


def pixels_to_2theta_chi(col, row, geom: DetectorGeometry) -> tuple:
    """Measured pixel positions → (2θ, χ) in degrees, via LaueTools.

    Delegates to ``LaueTools.LaueGeometry.calc_uflab`` so the ``.det`` tilt
    conventions stay LaueTools' responsibility.  Falls back to the in-module
    projection, which is verified bit-identical to it, only when LaueTools is
    unavailable.
    """
    try:
        from LaueTools.LaueGeometry import calc_uflab
    except ImportError:
        return direction_to_2theta_chi(pixel_to_direction(col, row, geom))

    col = np.atleast_1d(np.asarray(col, dtype=float))
    row = np.atleast_1d(np.asarray(row, dtype=float))
    two_theta, chi = calc_uflab(col, row, geom.calib, returnAngles=1,
                                pixelsize=geom.pixelsize,
                                kf_direction=geom.kf_direction)
    return np.asarray(two_theta, dtype=float), np.asarray(chi, dtype=float)


# ── Forward model ─────────────────────────────────────────────────────────────

def kf_hat(G: np.ndarray, u_hat: np.ndarray) -> np.ndarray:
    """Outgoing unit vector for a reciprocal node G under white-beam Laue."""
    G  = np.asarray(G, dtype=float)
    Gh = G / np.linalg.norm(G)
    u  = np.asarray(u_hat, dtype=float)
    return u - 2.0 * (u @ Gh) * Gh


def predict_satellite_directions(
    G0_lab: np.ndarray,
    z_lab: np.ndarray,
    period_angstrom: float,
    orders,
    u_hat: np.ndarray,
) -> List[np.ndarray]:
    """Outgoing directions of the satellite orders for a trial period Λ.

    Exact — no small-angle approximation, and no φ or S to compute: the
    geometry is carried by the vector algebra.
    """
    q  = 2.0 * np.pi / period_angstrom
    zh = np.asarray(z_lab, dtype=float)
    zh = zh / np.linalg.norm(zh)
    G0 = np.asarray(G0_lab, dtype=float)
    return [kf_hat(G0 + n * q * zh, u_hat) for n in orders]


def predict_angular_separations(
    G0_lab: np.ndarray,
    z_lab: np.ndarray,
    period_angstrom: float,
    orders,
    u_hat: np.ndarray,
) -> np.ndarray:
    """Angular separation (radians) between consecutive entries of ``orders``.

    Reproduces the non-uniform spacing of the exact model, which the analytic
    small-angle route cannot: the separations contract with increasing order.
    """
    dirs = predict_satellite_directions(G0_lab, z_lab, period_angstrom, orders, u_hat)
    return np.array([
        float(np.arccos(np.clip(dirs[i] @ dirs[i + 1], -1.0, 1.0)))
        for i in range(len(dirs) - 1)
    ])


# ── Detector: pixel separation → angular separation ───────────────────────────

def pixel_to_angle(
    delta_px: float,
    *,
    pixel_size_mm: float,
    detector_distance_mm: float,
    two_theta_deg: float,
    psi_deg: float,
) -> float:
    """Angular separation (radians) subtended by a displacement on a flat detector.

    Generalises the cos²(2θ) Jacobian of the monochromatic path to a displacement
    that is not purely radial.  With L = D/cos(2θ) the sample-to-spot distance and
    ψ the angle between the displacement and the local radial direction, only the
    component of the displacement perpendicular to the line of sight subtends an
    angle::

        Δψ_det = Δx · cos(2θ) · √(cos²ψ·cos²(2θ) + sin²ψ) / D

    ψ = 0 (radial) recovers Δx·cos²(2θ)/D exactly, i.e. the legacy Jacobian.
    ψ = 90° (azimuthal) gives Δx·cos(2θ)/D — no second obliquity, since the
    azimuthal direction is already perpendicular to the line of sight.
    """
    t2  = math.radians(two_theta_deg)
    psi = math.radians(psi_deg)
    obliquity = math.sqrt(
        math.cos(psi) ** 2 * math.cos(t2) ** 2 + math.sin(psi) ** 2
    )
    return (delta_px * pixel_size_mm) * math.cos(t2) * obliquity / detector_distance_mm


def psi_from_geometry(
    spot_row: float, spot_col: float,
    reference_row: float, reference_col: float,
    axis_angle_deg: float,
) -> float:
    """ψ in degrees: angle between the satellite axis and the local radial direction.

    ``axis_angle_deg`` is the refined satellite axis from ``detect_satellites``
    (degrees from +x, i.e. from +column).  The radial direction is the vector
    from the detector reference point ``(ycen, xcen)`` to the spot.

    ``(reference_row, reference_col)`` is ``(ycen, xcen)`` from the ``.det`` — the
    detector reference point, NOT a beam center: in reflection geometry the direct
    beam never reaches the detector, so nothing is visible there and it need not
    even fall inside the observable pattern.

    Result is folded into [0, 90] — only |cos ψ| and |sin ψ| enter
    ``pixel_to_angle``.  Needed by ``laue_analytic`` only; the forward route
    outputs the axis instead of consuming it.
    """
    radial_deg = math.degrees(math.atan2(spot_row - reference_row,
                                         spot_col - reference_col))
    d = abs((axis_angle_deg - radial_deg + 90.0) % 180.0 - 90.0)
    return d


# ── Laue geometry bundle ──────────────────────────────────────────────────────

def _laue_geometry(hkl, lattice, wavelength_angstrom: float) -> Dict[str, float]:
    """Derive γ, θ, |G₀| and 2θ from indexing.  Never defaulted, never user-supplied."""
    if hkl is None or lattice is None:
        raise ValueError(
            "Laue methods require both hkl=(h, k, l) and lattice=(a, c) in Angstrom. "
            "γ and 2θ are derived from indexing, not defaulted — passing the wrong "
            "value for either silently produces a wrong period."
        )
    h, k, l = hkl
    a, c    = lattice
    gamma   = gamma_from_hkl(h, k, l, a, c)
    G_mag   = G_magnitude(h, k, l, a, c)
    theta   = theta_from_G(G_mag, wavelength_angstrom)

    if abs(math.sin(gamma)) < 1e-12:
        raise ValueError(
            f"symmetric reflection ({h}{k}{l}): γ = 0, satellites are degenerate "
            f"in Laue geometry — they land on the same detector pixel and no "
            f"period can be extracted.  Use a reflection with l ≠ 0 and "
            f"(h, k) ≠ (0, 0)."
        )

    return {
        'gamma_rad':     gamma,
        'gamma_deg':     math.degrees(gamma),
        'theta_rad':     theta,
        'theta_deg':     math.degrees(theta),
        'two_theta_deg': 2.0 * math.degrees(theta),
        'G_magnitude':   G_mag,
    }


# ── Route B — exact forward model + 1-D fit (recommended) ─────────────────────

# Beam ∥ +x  →  beam ∥ +y.  Maps (x, y, z) → (−y, x, z); identical to the
# transpose of LaueTools' own from_ORlabframe_to_Lauetools permutation.
BEAM_X_TO_Y = np.array([[0.0, -1.0, 0.0],
                        [1.0,  0.0, 0.0],
                        [0.0,  0.0, 1.0]])


def ub_from_beam_x_frame(UB) -> np.ndarray:
    """Convert an orientation matrix from a beam-∥-x frame to the beam-∥-y one.

    This is a conversion, not a reader — the name says which way it goes.

    LaueTools carries two laboratory frames, and the orientation matrix means a
    different thing in each:

    * ``lauecore`` — the simulation path, and the frame a ``.fit`` UBmat is
      written in — is documented in its own source as ``x // ki``.
    * ``calc_uflab`` — the measurement path, pixels to (2θ, χ) — returns
      directions with the beam along +y.  That is the frame this module works
      in (``LAB_KI``), because the forward model is compared against measured
      angles.

    So the same UB goes **raw** into a LaueTools simulation and **converted**
    into anything here.  A UB in the wrong frame is still a perfect rotation,
    still gives the right |G| and the right γ — both are frame-independent —
    and still produces confident numbers.  It simply places every reflection
    somewhere else on the detector.

    Symptom: the reprojection check fails by tens of degrees and **no** indexed
    reflection lands near the measured spot.  ``diagnose_ub_frame`` tests the
    candidates for you.
    """
    return BEAM_X_TO_Y @ np.asarray(UB, dtype=float)


def diagnose_ub_frame(UB, hkl, lattice, two_theta_meas, chi_meas) -> Dict[str, Any]:
    """Test the usual laboratory-frame conventions against a measured spot.

    Returns ``{name: separation_deg}`` for each candidate plus ``'best'``.  A
    candidate agreeing to a fraction of a degree while the others are tens of
    degrees away identifies the frame — that pattern is not a coincidence.

    Diagnostic only: it reports, it does not silently reinterpret the UB.  Fix
    the frame at the point where the UB is loaded, so the correction is visible.
    """
    UB = np.asarray(UB, dtype=float)
    candidates = {
        'as given (beam ∥ y, LaueTools)': UB,
        'beam ∥ x → y': BEAM_X_TO_Y @ UB,
        'transposed':   UB.T,
        'transposed, beam ∥ x → y': BEAM_X_TO_Y @ UB.T,
    }
    out: Dict[str, Any] = {}
    for name, M in candidates.items():
        G_lab, _ = lab_vectors_from_UB(hkl, lattice, M)
        kf = kf_hat(G_lab, LAB_KI)
        if kf[2] <= 0.0:
            out[name] = float('inf')       # not on the detector at all
            continue
        tt, chi = direction_to_2theta_chi(kf)
        out[name] = float(np.degrees(angular_separation(
            tt, chi, two_theta_meas, chi_meas)))
    out['best'] = min((k for k in out if k != 'best'), key=lambda k: out[k])
    return out


def lab_vectors_from_UB(hkl, lattice, UB) -> tuple:
    """(G_lab, z_lab) in the laboratory frame from indexing and the stored UB.

    Applies B before UB — ``UB @ [h,k,l]`` alone gives the right direction but a
    meaningless magnitude, since the stored UB is a pure rotation carrying no
    lattice metric.  z_lab is the growth axis: for a hexagonal cell c* ∥ c, so
    [0,0,1] in the crystal Cartesian frame is already the growth direction.
    """
    B         = B_matrix_hexagonal(*lattice)
    UB_       = np.asarray(UB, dtype=float)
    G_crystal = B @ np.asarray(hkl, dtype=float)
    G_lab     = UB_ @ G_crystal
    z_lab     = UB_ @ np.array([0.0, 0.0, 1.0])
    return G_lab, z_lab / np.linalg.norm(z_lab)


def predict_satellite_pixels(
    G_lab, z_lab, period_angstrom: float, orders, geom: DetectorGeometry,
    u_hat=None,
):
    """Detector pixel positions of the satellite orders for a trial period Λ.

    The satellite axis on the detector is an OUTPUT of this model, never an
    input — which is what removes ``satellite_axis_psi_deg`` from the forward
    route.  Returns arrays (col, row).
    """
    u_vec = LAB_KI if u_hat is None else np.asarray(u_hat, dtype=float)
    u_vec = u_vec / np.linalg.norm(u_vec)
    dirs  = predict_satellite_directions(G_lab, z_lab, period_angstrom, orders, u_vec)
    return direction_to_pixel(np.array(dirs), geom)


def predict_satellite_angles(
    G_lab, z_lab, period_angstrom: float, orders, u_hat=None,
) -> tuple:
    """Satellite orders → (2θ, χ) in degrees, straight from the forward model.

    No detector involved: the model predicts directions and they are expressed in
    the same angular convention the measurement is converted into.  This is what
    removes S, φ and the tilt conventions from the fit.
    """
    u_vec = LAB_KI if u_hat is None else np.asarray(u_hat, dtype=float)
    u_vec = u_vec / np.linalg.norm(u_vec)
    dirs = np.array(predict_satellite_directions(
        G_lab, z_lab, period_angstrom, orders, u_vec))
    return direction_to_2theta_chi(dirs)


def predicted_satellite_axis_deg(
    G_lab, z_lab, period_angstrom: float, orders, geom: DetectorGeometry,
    u_hat=None,
) -> float:
    """Orientation of the predicted satellite row on the detector, degrees from +column.

    DIAGNOSTIC READOUT ONLY.  ADDENDUM 2 §4 withdraws the comparison against the
    measured ``axis_angle``: that quantity was introduced as an ad-hoc metric of
    the satellite-train inclination, not as a calibrated geometric observable, so
    it must not be used in any calculation or validation.  Use
    ``train_direction_delta_deg`` instead, which compares predicted and measured
    train directions in the calibrated angular space.
    """
    col, row = predict_satellite_pixels(
        G_lab, z_lab, period_angstrom, orders, geom, u_hat
    )
    d_col = float(col[-1] - col[0])
    d_row = float(row[-1] - row[0])
    ang = math.degrees(math.atan2(d_row, d_col))
    return (ang + 90.0) % 180.0 - 90.0


def _tangent_offsets(two_theta, chi, tt_ref, chi_ref) -> np.ndarray:
    """Local tangent-plane coordinates (radians) about a reference direction.

    Uses the correct spherical line element ds² = d(2θ)² + sin²(2θ)·dχ², so the
    azimuthal component carries its sin(2θ) weight.  Used for train DIRECTION
    only — magnitudes are always reported with the exact formula.
    """
    d_tt  = np.radians(np.asarray(two_theta, dtype=float) - tt_ref)
    d_chi = np.radians(np.asarray(chi, dtype=float) - chi_ref) * math.sin(
        math.radians(tt_ref))
    return np.stack([d_tt, d_chi], axis=-1)


def train_direction_delta_deg(
    tt_pred, chi_pred, tt_meas, chi_meas,
) -> float:
    """Angle (degrees) between the predicted and measured satellite train directions.

    Computed in the calibrated angular space, in the tangent plane at the
    measured parent, with the proper sin(2θ) metric.  This is the replacement for
    the withdrawn ``axis_angle`` comparison: it is the same idea — the train
    direction is predicted from the crystal orientation with no free parameter —
    but expressed in a quantity the geometry actually defines.

    Folded into [0, 90]: the train is a line, not an arrow.
    """
    tt_pred  = np.asarray(tt_pred, dtype=float)
    chi_pred = np.asarray(chi_pred, dtype=float)
    tt_meas  = np.asarray(tt_meas, dtype=float)
    chi_meas = np.asarray(chi_meas, dtype=float)

    ref_tt, ref_chi = float(tt_meas.mean()), float(chi_meas.mean())
    p = _tangent_offsets(tt_pred, chi_pred, ref_tt, ref_chi)
    m = _tangent_offsets(tt_meas, chi_meas, ref_tt, ref_chi)

    vp, vm = p[-1] - p[0], m[-1] - m[0]
    np_, nm = np.linalg.norm(vp), np.linalg.norm(vm)
    if np_ == 0.0 or nm == 0.0:
        return float('nan')
    cos_a = abs(float(vp @ vm) / (np_ * nm))
    return math.degrees(math.acos(min(cos_a, 1.0)))
