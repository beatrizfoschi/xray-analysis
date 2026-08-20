#!/usr/bin/env python3
"""
sapphire_peak_filter.py — Remove sapphire substrate peaks from GaN LaueTools
.dat peak files.

The sapphire substrate contributes its own Laue spots to every GaN frame.
The blacklist of sapphire positions is built by *simulating* the sapphire
Laue pattern from its material and UB matrix (via LaueTools) rather than by
reading the experimental positions out of an indexed .fit file: a .fit file
only lists spots that were found by the peak search AND survived
refinement, so weak or unindexed sapphire spots would leak straight through
a fit-based blacklist. Simulation predicts every sapphire reflection in the
chosen energy range, indexed or not.

Any GaN peak within --tol pixels of a blacklisted (simulated) position is
dropped.

Detector geometry (pixel size + frame shape) is resolved in this order:
explicit ``pixel_size``/``frame_shape`` > a ``fit_path``'s recorded CCD
calibration > LaueTools' ``camera_label`` lookup as a last resort. This
matters: silently defaulting to the wrong camera (e.g. LaueTools' "sCMOS"
default when the real detector is an Eiger) introduces a pixel-size
mismatch that grows with distance from the beam centre and can shift
simulated spot positions by 5-10+ px — enough to make most real sapphire
spots fall outside a normal matching tolerance.

Typical workflow (Jupyter notebook)
-----------------------------------
from laue.sapphire_peak_filter import simulate_sapphire_blacklist, filter_files

# Custom material (not in LaueTools' built-in dict_Materials): a dict with
# just this one entry is enough — Prepare_Grain/SimulateLaue_full_np only
# look up "Sapphire_epi" in whatever dict you pass, and the extinction rule
# string ("Al2O3") is resolved against LaueTools' global extinction table,
# not against this dict.
my_dict_mat = {"Sapphire_epi": ["Sapphire_epi", [4.758, 4.758, 12.991, 90, 90, 120], "Al2O3"]}

# UB matrix, calibration, pixel size AND frame shape are all read from the
# .fit file's own CCD calibration block — no camera_label guesswork needed.
blacklist_xy = simulate_sapphire_blacklist(
    "Sapphire_epi",
    fit_path = "blc16817_paper_data/sapphire/fitfiles_sub_new/frame_00000_g0.fit",
    material_dictionary = my_dict_mat,
    Emin = 5, Emax = 25,
)

# No .fit available: pass everything by hand instead.
blacklist_xy = simulate_sapphire_blacklist(
    "Sapphire_epi",
    ub_matrix              = my_ub_matrix,             # (3, 3) array
    calibration_parameters = [69.984, 1079.57, 983.73, 0.173, 0.382],
    pixel_size              = 0.075,                    # mm/pixel
    frame_shape             = (2162, 2068),              # (rows, cols)
    material_dictionary     = my_dict_mat,
    Emin = 5, Emax = 25,
)

summary = filter_files(
    datfiles     = "blc16817_paper_data/Pedro_SEG/datfiles_new/*.dat",
    blacklist_xy = blacklist_xy,
    outdir       = "blc16817_paper_data/Pedro_SEG/datfiles_no_sapphire",
    tol          = 3.0,
    workers      = 8,     # parallel over files; workers=1 to run sequentially
)
summary.sort_values("n_removed", ascending=False).head()

A CLI entry point (``python sapphire_peak_filter.py --datfiles ... --fitfile ...
--material ... --outdir ...``) is also available for SLURM/batch use — see
``main()`` below.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


# ── .dat I/O ────────────────────────────────────────────────────────────────

from laue.readers import load_calibration_from_fit, load_dat, load_fit_peaklist, write_dat




# ── Simulated blacklist ────────────────────────────────────────────────────────





def simulate_sapphire_blacklist(
    material: str,
    *,
    fit_path: str | Path | None = None,
    ub_matrix: np.ndarray | None = None,
    calibration_parameters: list | None = None,
    pixel_size: float | None = None,
    frame_shape: tuple[int, int] | None = None,
    camera_label: str | None = None,
    detector_diameter: float = 148.1212,
    material_dictionary: dict | None = None,
    Emin: float = 5.0,
    Emax: float = 25.0,
) -> np.ndarray:
    """Simulate the full sapphire Laue pattern to use as a matching blacklist.

    Unlike reading Xexp/Yexp from an indexed .fit file — which only lists
    spots that were found by the peak search AND survived refinement — this
    predicts every sapphire reflection in the given energy range, so weak or
    unindexed sapphire spots that would otherwise leak into the GaN peak
    list are also caught.

    Detector geometry (pixel_size + frame_shape) is resolved in this order:
    explicit ``pixel_size``/``frame_shape`` (if given) > ``fit_path``'s own
    CCD calibration block (if given and not overridden) > ``camera_label``
    looked up in LaueTools' own camera dictionary (last resort). This avoids
    silently defaulting to the wrong camera geometry (e.g. LaueTools'
    "sCMOS" default when the real detector is an Eiger), which shifts
    simulated spot positions by several to 10+ px near the detector edges —
    enough to make most real sapphire spots fall outside a normal matching
    tolerance.

    Parameters
    ----------
    material : material key, looked up in ``material_dictionary`` if given,
        otherwise in LaueTools' built-in materials dictionary.
    fit_path : optional .fit file to derive ub_matrix, calibration_parameters,
        pixel_size and frame_shape from (any of these can still be overridden
        by passing them explicitly).
    ub_matrix : (3, 3) orientation matrix of the sapphire substrate. Required
        unless derivable from ``fit_path``.
    calibration_parameters : [distance, x_center, y_center, x_beta, x_gamma].
        Required unless derivable from ``fit_path``.
    pixel_size : detector pixel size in mm.
    frame_shape : (n_rows, n_cols) detector frame shape in pixels.
    camera_label : LaueTools camera key (e.g. "EIGER_4M"), used only if
        pixel_size/frame_shape are still unknown after fit_path + overrides.
    detector_diameter : simulation cutoff diameter in mm, before pixel
        conversion (default 148.1212, LaueTools' usual value).
    material_dictionary : custom material dict, e.g. for a material not in
        LaueTools' built-in dict_Materials:
        ``{"Sapphire_epi": ["Sapphire_epi", [a,b,c,alpha,beta,gamma], "Al2O3"]}``.
        The extinction-rule string (3rd element) is resolved against
        LaueTools' own global extinction table, not against this dict.
    Emin, Emax : simulated energy range in keV.

    Returns
    -------
    (N, 2) array of simulated (X, Y) pixel positions, clipped to the detector.
    """
    if fit_path is not None:
        from_fit = load_calibration_from_fit(fit_path)
        if ub_matrix is None:
            ub_matrix = from_fit["ub_matrix"]
        if calibration_parameters is None:
            calibration_parameters = from_fit["calibration_parameters"]
        if pixel_size is None:
            pixel_size = from_fit["pixel_size"]
        if frame_shape is None:
            frame_shape = from_fit["frame_shape"]

    if ub_matrix is None or calibration_parameters is None:
        raise ValueError(
            "ub_matrix and calibration_parameters are required: pass them "
            "directly, or provide fit_path to derive them."
        )
    ub_matrix = np.asarray(ub_matrix, dtype=float)
    if ub_matrix.shape != (3, 3):
        raise ValueError(f"ub_matrix must be shape (3, 3), got {ub_matrix.shape}")

    if pixel_size is None or frame_shape is None:
        if camera_label is None:
            raise ValueError(
                "No detector geometry available: pass pixel_size + "
                "frame_shape, or camera_label, or fit_path."
            )
        from LaueTools.dict_LaueTools import dict_CCD
        if frame_shape is None:
            frame_shape = dict_CCD[camera_label][0]
        if pixel_size is None:
            pixel_size = dict_CCD[camera_label][1]

    from LaueTools.CrystalParameters import Prepare_Grain
    from LaueTools.dict_LaueTools import dict_Materials
    from LaueTools.lauecore import SimulateLaue_full_np

    mat_dict = material_dictionary if material_dictionary is not None else dict_Materials
    sim_params = Prepare_Grain(material, ub_matrix, dictmaterials=mat_dict)

    result = SimulateLaue_full_np(
        sim_params, Emin, Emax, calibration_parameters,
        detectordiameter = detector_diameter * 1.75,
        pixelsize        = pixel_size,
        dim              = frame_shape,
        dictmaterials    = mat_dict,
        kf_direction     = "Z>0",
        removeharmonics  = 0,
    )
    x, y = result[3], result[4]
    keep = (x > 0) & (x < frame_shape[1]) & (y > 0) & (y < frame_shape[0])
    return np.stack([x[keep], y[keep]], axis=1)


# ── Matching / filtering ──────────────────────────────────────────────────────

def find_sapphire_mask(
    peaks_xy: np.ndarray,
    blacklist_xy: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Boolean mask, True where a peak lies within `tol` px of a blacklist position."""
    mask = np.zeros(len(peaks_xy), dtype=bool)
    if len(peaks_xy) == 0 or len(blacklist_xy) == 0:
        return mask
    tree = cKDTree(blacklist_xy)
    dist, _ = tree.query(peaks_xy, k=1)
    return dist <= tol


def split_peaks(
    df: pd.DataFrame,
    blacklist_xy: np.ndarray,
    tol: float,
    *,
    x_col: str = "peak_X",
    y_col: str = "peak_Y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a peak DataFrame into (kept, removed) by proximity to blacklist_xy."""
    peaks_xy = df[[x_col, y_col]].to_numpy(dtype=float)
    mask = find_sapphire_mask(peaks_xy, blacklist_xy, tol)
    kept = df.loc[~mask].reset_index(drop=True)
    removed = df.loc[mask].reset_index(drop=True)
    return kept, removed


def _resolve_datfiles(patterns: list[str]) -> list[Path]:
    """Expand literal paths and/or glob patterns into a sorted, de-duplicated file list."""
    files: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            files.extend(Path(p) for p in glob(pattern))
        else:
            files.append(Path(pattern))
    return sorted(set(files))


# ── Batch filtering (parallel) ────────────────────────────────────────────────

def _filter_one(
    path: Path,
    blacklist_xy: np.ndarray,
    tol: float,
    x_col: str,
    y_col: str,
    outdir: Path,
) -> dict:
    """Filter a single .dat file and write the result to outdir. Runs in a worker process."""
    header_line, df = load_dat(path)
    kept, removed = split_peaks(df, blacklist_xy, tol, x_col=x_col, y_col=y_col)
    write_dat(kept, outdir / path.name, header_line)
    return {"file": path.name, "n_total": len(df), "n_kept": len(kept), "n_removed": len(removed)}


def filter_files(
    datfiles: str | Path | list[str | Path],
    blacklist_xy: np.ndarray,
    outdir: str | Path,
    *,
    tol: float = 3.0,
    x_col: str = "peak_X",
    y_col: str = "peak_Y",
    workers: int = 8,
) -> pd.DataFrame:
    """Remove sapphire peaks from many GaN .dat files, in parallel.

    Meant to be called directly from a notebook. Filtered files are written
    to `outdir` under their original name.

    Parameters
    ----------
    datfiles : a glob pattern, a single path, or a list of paths/patterns
        (e.g. ``"datfiles_new/*.dat"`` or a list of explicit frame paths).
    blacklist_xy : (N, 2) array of sapphire (x, y) pixel positions — typically
        from ``simulate_sapphire_blacklist``.
    outdir : output directory for the filtered .dat files (created if needed).
    tol : matching tolerance in pixels (default 3.0).
    workers : number of worker processes (default 8). Use workers=1 to run
        sequentially in the current process (e.g. while debugging).

    Returns
    -------
    pd.DataFrame with one row per file: file, n_total, n_kept, n_removed.
    """
    if isinstance(datfiles, (str, Path)):
        datfiles = [datfiles]
    files = _resolve_datfiles([str(p) for p in datfiles])
    if not files:
        raise FileNotFoundError(f"No .dat files matched: {datfiles}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    blacklist_xy = np.asarray(blacklist_xy, dtype=float)
    print(f"Sapphire blacklist: {len(blacklist_xy)} simulated peaks")

    rows: list[dict] = []
    if workers <= 1:
        for path in tqdm(files, desc="Filtering", unit="file"):
            rows.append(_filter_one(path, blacklist_xy, tol, x_col, y_col, outdir))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_filter_one, path, blacklist_xy, tol, x_col, y_col, outdir): path
                for path in files
            }
            with tqdm(total=len(files), desc="Filtering", unit="file") as pbar:
                for future in as_completed(futures):
                    rows.append(future.result())
                    pbar.update(1)

    summary = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    print(f"Total: {summary['n_kept'].sum()} kept, {summary['n_removed'].sum()} removed "
          f"across {len(summary)} file(s)  (tol={tol}px)")
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # I/O
    ap.add_argument("--datfiles", required=True, nargs="+",
                     help="One or more .dat files, and/or a glob pattern "
                          "(e.g. 'datfiles_new/*.dat').")
    ap.add_argument("--outdir", required=True,
                     help="Output directory for the filtered .dat files.")
    # Blacklist simulation
    ap.add_argument("--material", required=True,
                     help="Material key in LaueTools' built-in materials dictionary. "
                          "For a custom material (not in dict_Materials), use the Python "
                          "API instead: simulate_sapphire_blacklist(..., "
                          "material_dictionary={...}) — not exposed on this CLI.")
    ap.add_argument("--fitfile", default=None,
                     help="Sapphire .fit file to derive UB matrix, calibration, pixel size "
                          "and frame shape from (optional if all of --ub/--calib/"
                          "--pixel-size/--frame-shape are given explicitly).")
    ap.add_argument("--ub", type=float, nargs=9, default=None,
                     metavar=("UB11", "UB12", "UB13", "UB21", "UB22", "UB23", "UB31", "UB32", "UB33"),
                     help="3x3 UB matrix, row-major (overrides --fitfile).")
    ap.add_argument("--calib", type=float, nargs=5, default=None,
                     metavar=("DIST", "XCEN", "YCEN", "XBET", "XGAM"),
                     help="Detector calibration parameters (overrides --fitfile).")
    ap.add_argument("--pixel-size", type=float, default=None,
                     help="Detector pixel size in mm (overrides --fitfile / --camera-label).")
    ap.add_argument("--frame-shape", type=int, nargs=2, default=None,
                     metavar=("N_ROWS", "N_COLS"),
                     help="Detector frame shape in pixels (overrides --fitfile / --camera-label).")
    ap.add_argument("--camera-label", default=None,
                     help="LaueTools camera key (e.g. EIGER_4M), used only if pixel size / "
                          "frame shape are still unknown after --fitfile / explicit overrides.")
    ap.add_argument("--emin", type=float, default=5.0,
                     help="Minimum simulated energy in keV (default: 5.0).")
    ap.add_argument("--emax", type=float, default=25.0,
                     help="Maximum simulated energy in keV (default: 25.0).")
    # Matching
    ap.add_argument("--tol", type=float, default=3.0,
                     help="Matching tolerance in pixels (default: 3.0).")
    ap.add_argument("--x-col", default="peak_X",
                     help="Peak x-column name in the .dat files (default: peak_X).")
    ap.add_argument("--y-col", default="peak_Y",
                     help="Peak y-column name in the .dat files (default: peak_Y).")
    ap.add_argument("--workers", type=int, default=8,
                     help="Number of parallel worker processes (default: 8).")
    args = ap.parse_args()

    ub_matrix = np.array(args.ub).reshape(3, 3) if args.ub else None
    frame_shape = tuple(args.frame_shape) if args.frame_shape else None

    blacklist_xy = simulate_sapphire_blacklist(
        args.material,
        fit_path=args.fitfile,
        ub_matrix=ub_matrix,
        calibration_parameters=args.calib,
        pixel_size=args.pixel_size,
        frame_shape=frame_shape,
        camera_label=args.camera_label,
        Emin=args.emin, Emax=args.emax,
    )

    filter_files(
        args.datfiles, blacklist_xy, args.outdir,
        tol=args.tol, x_col=args.x_col, y_col=args.y_col,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
