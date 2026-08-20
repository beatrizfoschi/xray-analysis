#!/usr/bin/env python3
"""
nmf_sbatch_job.py

SLURM-ready NMF job for XEOL hyperspectral data.

Loads:
- HDF5 using lauexplore._parsers._h5.get_xeol(h5f, scan_number)
  + wavelength from f"{scan_number}.1/measurement/qepro_det1"][0]
OR
- .npy arrays: --spectra-npy + optionally --wl-npy

Runs:
- sklearn.decomposition.NMF with user options
- Uses SLURM_CPUS_PER_TASK for MKL/OpenMP/OpenBLAS threads

Outputs:
- nmf_out/W_maps.npy, H.npy, E_map.npy, wavelength.npy
- nmf_out/nmf_report_<stem>.pdf
"""

from __future__ import annotations

import os

# -------- set threads BEFORE numpy/sklearn import --------
cpus_env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
if cpus_env is not None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(cpus_env)

import matplotlib
matplotlib.use("Agg")

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from emission.NMF import run_nmf
except ImportError:
    # submit.sbatch runs `python nmf_sbatch_job.py` from inside pipelines/, so the
    # repository root — the parent of pipelines/ — is not on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from emission.NMF import run_nmf

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


# ── NMF ───────────────────────────────────────────────────────────────────────

def nmf_sklearn_hyperspectral(
    X,
    map_shape,
    n_components=3,
    wavelength=None,
    unit_name="Wavelength (nm)",
    loss="kullback-leibler",
    solver=None,
    init="nndsvda",
    max_iter=2000,
    random_state=0,
    l1_ratio=0.0,
    alpha_W=0.0,
    alpha_H=0.0,
    clip_negative=True,
    tol=1e-4,
):
    """Fit sklearn NMF on hyperspectral data X.

    Thin wrapper over `emission.NMF.run_nmf`; the defaults here are the batch-job
    ones (3 components, Kullback-Leibler, 2000 iterations) and deliberately differ
    from the interactive entry point's.

    Returns W_maps (nx,ny,K), H (K,ch), X_rec, E_map (nx,ny), model, wavelength.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_pixels, n_channels). Got {X.shape}")

    n_pixels, n_ch = X.shape
    nx, ny = map_shape
    if nx * ny != n_pixels:
        raise ValueError(f"map_shape product must match n_pixels ({nx*ny} != {n_pixels})")

    if wavelength is None:
        wavelength = np.arange(n_ch)

    W, H, X_rec, rmse, model = run_nmf(
        X, n_components,
        loss=loss, solver=solver, init=init, max_iter=max_iter,
        random_state=random_state, tol=tol, l1_ratio=l1_ratio,
        alpha_W=alpha_W, alpha_H=alpha_H, clip_negative=clip_negative,
    )

    W_maps = W.reshape(nx, ny, n_components)
    E_map  = rmse.reshape(nx, ny)

    return W_maps, H, X_rec, E_map, model, np.asarray(wavelength), unit_name


# ── Spectral helpers ──────────────────────────────────────────────────────────

def _wl_slice(wl: np.ndarray, w0: float, w1: float) -> slice:
    i0 = int(np.abs(wl - w0).argmin())
    i1 = int(np.abs(wl - w1).argmin())
    return slice(i0, i1 + 1)


def _auto_active_zone(spectra: np.ndarray, wl: np.ndarray) -> tuple[float, float]:
    mean_spec = spectra.mean(axis=0)
    peak_wl   = float(wl[mean_spec.argmax()])
    return max(float(wl[0]), peak_wl - 20.0), min(float(wl[-1]), peak_wl + 20.0)


def _intensity_maps(
    spectra: np.ndarray,
    wl: np.ndarray,
    monitor: np.ndarray,
    active_zone: tuple[float, float],
    norm_zone: tuple[float, float] | None,
) -> dict[str, np.ndarray]:
    mon          = np.where(monitor > 0, monitor, np.nan)
    panchromatic = spectra.sum(axis=1) / mon
    active       = spectra[:, _wl_slice(wl, *active_zone)].sum(axis=1) / mon
    if norm_zone is not None:
        dead   = spectra[:, _wl_slice(wl, *norm_zone)].sum(axis=1)
        dead   = np.where(dead > 0, dead, np.nan)
        active = active / dead
    return {"panchromatic": panchromatic, "active_zone": active}


# ── PDF report ────────────────────────────────────────────────────────────────

def _make_pdf_report(
    W_maps:       np.ndarray,
    H:            np.ndarray,
    E_map:        np.ndarray,
    wavelength:   np.ndarray,
    output_path:  Path,
    *,
    spectra:      np.ndarray | None = None,
    monitor:      np.ndarray | None = None,
    active_zone:  tuple[float, float] | None = None,
    norm_zone:    tuple[float, float] | None = None,
    extent:       list[float] | None = None,
    source_label: str = "",
    cmd_args:     str = "",
    model_info:   dict | None = None,
    invert_yaxis: bool = True,
) -> None:
    """Multi-page PDF report mirroring xeol_report.py layout."""
    K      = H.shape[0]
    nx, ny = W_maps.shape[:2]

    def _imshow(ax, data, **kw):
        lo = np.nanpercentile(data, 2)
        hi = np.nanpercentile(data, 98)
        im = ax.imshow(data, origin="upper", aspect="equal",
                       extent=extent, vmin=lo, vmax=hi, **kw)
        if invert_yaxis:
            ax.invert_yaxis()
        ax.set_xlabel("x" + (" (µm)" if extent else ""))
        ax.set_ylabel("y" + (" (µm)" if extent else ""))
        return im

    with PdfPages(output_path) as pdf:

        # ── Page 1: Metadata ──────────────────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        info = [
            "NMF XEOL Report",
            "─" * 50,
            "",
            f"Generated   : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
            f"Source      : {source_label}",
            "",
            f"Grid        : {nx} × {ny}  →  {nx*ny} pixels",
            f"λ range     : {wavelength[0]:.1f} – {wavelength[-1]:.1f} nm",
            f"NMF K       : {K} components",
        ]
        if active_zone:
            info += [f"Active zone : {active_zone[0]:.1f} – {active_zone[1]:.1f} nm"]
        if norm_zone:
            info += [f"Norm zone   : {norm_zone[0]:.1f} – {norm_zone[1]:.1f} nm"]
        if model_info:
            info += [""]
            for k, v in model_info.items():
                info.append(f"{k:<12}: {v}")
        if cmd_args:
            info += ["", f"Command     : {cmd_args}"]
        ax.text(0.08, 0.93, "\n".join(info),
                transform=ax.transAxes, fontsize=11,
                fontfamily="monospace", verticalalignment="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: Intensity maps (H5 only) ──────────────────────────────────
        if spectra is not None and monitor is not None and active_zone is not None:
            maps = _intensity_maps(spectra, wavelength, monitor, active_zone, norm_zone)
            panch_2d = maps["panchromatic"].reshape(nx, ny)
            az_2d    = maps["active_zone"].reshape(nx, ny)
            az_title = (f"Active zone {active_zone[0]:.0f}–{active_zone[1]:.0f} nm"
                        + (" / norm" if norm_zone else "") + " / monitor")

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Intensity Maps", fontsize=13)
            for ax, data, title in zip(axes,
                                       [panch_2d, az_2d],
                                       ["Panchromatic / monitor", az_title]):
                im = _imshow(ax, data, cmap="hot")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(title, fontsize=9)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── Page 3: Mean spectrum ─────────────────────────────────────────────
        src = spectra if spectra is not None else (W_maps.reshape(-1, K) @ H)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(wavelength, src.mean(axis=0), color="steelblue", lw=1.2)
        if active_zone:
            ax.axvspan(*active_zone, alpha=0.20, color="orange",
                       label=f"Active zone {active_zone[0]:.0f}–{active_zone[1]:.0f} nm")
        if norm_zone:
            ax.axvspan(*norm_zone, alpha=0.20, color="gray",
                       label=f"Norm zone {norm_zone[0]:.0f}–{norm_zone[1]:.0f} nm")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Mean spectrum (average over all scan points)")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 4: NMF component spectra ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in range(K):
            spec = H[k].copy()
            norm = np.linalg.norm(spec)
            if norm > 0:
                spec /= norm
            ax.plot(wavelength, spec, label=f"Component {k + 1}")
        if active_zone:
            ax.axvspan(*active_zone, alpha=0.10, color="orange")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u., normalised)")
        ax.set_title(f"NMF Component Spectra  ({K} components)")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 5: NMF abundance maps + residual ─────────────────────────────
        n_panels = K + 1
        ncols    = min(K + 1, 3)
        nrows    = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.5, nrows * 4.0),
                                 squeeze=False)
        fig.suptitle("NMF Abundance Maps", fontsize=13)
        flat = axes.flatten()

        for k in range(K):
            im = _imshow(flat[k], W_maps[:, :, k], cmap="viridis")
            flat[k].set_title(f"Component {k + 1}", fontsize=9)
            plt.colorbar(im, ax=flat[k], fraction=0.046, pad=0.04)

        im_r = _imshow(flat[K], E_map, cmap="Reds")
        flat[K].set_title("Residual (RMSE)", fontsize=9)
        plt.colorbar(im_r, ax=flat[K], fraction=0.046, pad=0.04)

        for ax in flat[K + 1:]:
            ax.set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run sklearn NMF on XEOL (SLURM batch).")

    # data sources
    ap.add_argument("--h5path",       help="Path to HDF5 file.")
    ap.add_argument("--scan-number",  type=int, help="Scan number for H5 reading.")
    ap.add_argument("--spectra-npy",  help="Path to spectra .npy (n_pixels, n_channels).")
    ap.add_argument("--wl-npy",       help="Path to wavelength .npy (n_channels,).")
    ap.add_argument("--monitor-npy",  help="Path to monitor .npy (n_pixels,).")

    # geometry
    ap.add_argument("--map-nx", type=int, required=True)
    ap.add_argument("--map-ny", type=int, required=True)

    # spectral zones
    ap.add_argument("--active-zone", type=float, nargs=2, metavar=("W0", "W1"),
                    default=None, help="Active wavelength range [nm] (default: auto).")
    ap.add_argument("--norm-zone",   type=float, nargs=2, metavar=("Z0", "Z1"),
                    default=None, help="Dead-zone range [nm] for normalisation.")

    # nmf params
    ap.add_argument("--n-components",  type=int,   default=3)
    ap.add_argument("--loss",          default="kullback-leibler",
                    choices=["frobenius", "kullback-leibler"])
    ap.add_argument("--solver",        default=None, choices=[None, "cd", "mu"],
                    nargs="?")
    ap.add_argument("--init",          default="nndsvda")
    ap.add_argument("--max-iter",      type=int,   default=2000)
    ap.add_argument("--tol",           type=float, default=1e-4)
    ap.add_argument("--random-state",  type=int,   default=0)
    ap.add_argument("--clip-negative", action="store_true", default=True)
    ap.add_argument("--no-clip-negative", dest="clip_negative", action="store_false")
    ap.add_argument("--extent", type=float, nargs=4,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), default=None)
    ap.add_argument("--no-invert-yaxis", action="store_true", default=False,
                    help="Do not invert y-axis on maps (default: y-axis is inverted).")

    # output
    ap.add_argument("--outdir", default="nmf_out")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    cpus = max(1, cpus)

    # ── load data ──────────────────────────────────────────────────────────────
    monitor = wl = None
    source_label = ""

    if args.spectra_npy:
        X = np.load(args.spectra_npy)
        wl = np.load(args.wl_npy) if args.wl_npy else None
        if args.monitor_npy:
            monitor = np.load(args.monitor_npy)
        source_label = args.spectra_npy
        print(f"Loaded spectra from npy: {args.spectra_npy} | shape={X.shape}")
    else:
        if not (args.h5path and args.scan_number is not None):
            raise SystemExit("Provide either --spectra-npy OR (--h5path and --scan-number).")
        from lauexplore._parsers import _h5
        with h5py.File(args.h5path, "r") as h5f:
            X       = np.array(_h5.get_xeol(h5f, args.scan_number))
            wl      = h5f[f"{args.scan_number}.1/measurement/qepro_det1"][0]
            monitor = h5f[f"{args.scan_number}.1/measurement/mon"][()]
        source_label = f"{args.h5path}  (scan {args.scan_number})"
        print(f"Loaded spectra from H5: {args.h5path} | scan={args.scan_number} | shape={X.shape}")

    spectra = X  # always pass raw spectra for mean-spectrum page

    active_zone = tuple(args.active_zone) if args.active_zone else (
        _auto_active_zone(X, wl) if wl is not None else None
    )
    norm_zone = tuple(args.norm_zone) if args.norm_zone else None

    # ── run NMF ───────────────────────────────────────────────────────────────
    t0 = time.time()
    ctx = threadpool_limits(limits=cpus) if threadpool_limits else _dummy_ctx()
    with ctx:
        W_maps, H, _, E_map, model, wl_used, _ = nmf_sklearn_hyperspectral(
            X,
            map_shape=(args.map_nx, args.map_ny),
            n_components=args.n_components,
            wavelength=wl,
            loss=args.loss,
            solver=args.solver,
            init=args.init,
            max_iter=args.max_iter,
            random_state=args.random_state,
            clip_negative=args.clip_negative,
            tol=args.tol,
        )

    elapsed = time.time() - t0
    print(f"NMF done in {elapsed:.2f} s | n_iter_={getattr(model, 'n_iter_', 'NA')}")

    # ── save arrays ───────────────────────────────────────────────────────────
    np.save(os.path.join(args.outdir, "W_maps.npy"),    W_maps)
    np.save(os.path.join(args.outdir, "H.npy"),         H)
    np.save(os.path.join(args.outdir, "E_map.npy"),     E_map)
    np.save(os.path.join(args.outdir, "wavelength.npy"), wl_used)

    # ── PDF report ────────────────────────────────────────────────────────────
    stem        = Path(args.h5path).stem if args.h5path else "npy"
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.outdir) / f"nmf_report_{stem}_{timestamp}.pdf"

    model_info = {
        "loss":    args.loss,
        "init":    args.init,
        "max_iter": args.max_iter,
        "tol":     args.tol,
        "n_iter":  getattr(model, "n_iter_", "NA"),
        "rec_err": f"{getattr(model, 'reconstruction_err_', float('nan')):.4g}",
        "elapsed": f"{elapsed:.1f} s",
        "cpus":    cpus,
    }

    try:
        _make_pdf_report(
            W_maps, H, E_map, wl_used,
            output_path,
            spectra=spectra,
            monitor=monitor,
            active_zone=active_zone,
            norm_zone=norm_zone,
            extent=args.extent,
            source_label=source_label,
            cmd_args=" ".join(sys.argv),
            model_info=model_info,
            invert_yaxis=not args.no_invert_yaxis,
        )
        print(f"Saved PDF report: {output_path}")
    except Exception as exc:
        import traceback
        print(f"ERROR generating PDF: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    print(f"Saved arrays in:  {args.outdir}/")


class _dummy_ctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
