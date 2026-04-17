#!/usr/bin/env python3
"""
nmf_xeol_job.py

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
- nmf_out/nmf_panel.png, nmf_panel.pdf
"""

import os

# -------- set threads BEFORE numpy/sklearn import --------
cpus_env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
if cpus_env is not None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(cpus_env)

import matplotlib
matplotlib.use("Agg")

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import NMF

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


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
    """
    Fit sklearn NMF on hyperspectral data X and return component maps and spectra.

    Parameters
    ----------
    X : array, shape (n_pixels, n_channels)
        Input hyperspectral matrix (non-negative).
    map_shape : tuple (nx, ny)
        Spatial dimensions of the scan; must satisfy nx * ny == n_pixels.
    n_components : int
        Number of NMF components.
    wavelength : array, optional
        Wavelength axis for plotting. Defaults to channel indices.
    loss : str
        Beta loss: "frobenius" or "kullback-leibler".

    Returns
    -------
    W_maps : ndarray, shape (nx, ny, n_components)
    H : ndarray, shape (n_components, n_channels)
    X_rec : ndarray, shape (n_pixels, n_channels)
    E_map : ndarray, shape (nx, ny) — per-pixel RMSE
    model : fitted sklearn NMF object
    wavelength : ndarray
    unit_name : str
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_pixels, n_channels). Got {X.shape}")

    n_pixels, n_ch = X.shape
    nx, ny = map_shape
    if nx * ny != n_pixels:
        raise ValueError(f"map_shape product must match n_pixels ({nx*ny} != {n_pixels})")

    if clip_negative and X.min() < 0:
        X = X.copy()
        X[X < 0] = 0.0
    elif X.min() < 0:
        raise ValueError("NMF requires non-negative data. Use clip_negative.")

    if wavelength is None:
        wavelength = np.arange(n_ch)

    if solver is None:
        solver = "mu" if loss == "kullback-leibler" else "cd"

    model = NMF(
        n_components=n_components,
        init=init,
        solver=solver,
        beta_loss=loss,
        tol = tol,
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
    )

    W = model.fit_transform(X)
    H = model.components_

    X_rec = W @ H
    rmse = np.sqrt(np.mean((X - X_rec) ** 2, axis=1))

    W_maps = W.reshape(nx, ny, n_components)
    E_map = rmse.reshape(nx, ny)

    return W_maps, H, X_rec, E_map, model, np.asarray(wavelength), unit_name


def plot_nmf_panel(
    W_maps,
    H,
    E_map,
    wavelength,
    unit_name="Wavelength (nm)",
    normalize_spectra=False,
    titles=None,
    figsize=(10, 10),
    extent=None,
):
    """
    Render a paper-style NMF panel: component spectra on top, spatial maps below.

    Top row: all K component spectra on a single axes.
    Bottom row: one imshow per component map, plus a residual (RMSE) map.

    Returns the matplotlib Figure object.
    """
    K = H.shape[0]
    if titles is None:
        titles = [f"Component {k+1}" for k in range(K)]

    fig = plt.figure(figsize=figsize)

    ax0 = plt.subplot2grid((2, K + 1), (0, 0), colspan=K + 1)
    for k in range(K):
        y = H[k].astype(np.float64, copy=True)
        if normalize_spectra:
            y /= (np.linalg.norm(y) + 1e-16)
        ax0.plot(wavelength, y, label=str(k + 1))
    ax0.set_xlabel(unit_name)
    ax0.set_ylabel("Intensity (a.u.)" if normalize_spectra else "Intensity")
    ax0.legend(ncol=min(K, 6), frameon=False)
    ax0.grid(True, alpha=0.25)

    for k in range(K):
        ax = plt.subplot2grid((2, K + 1), (1, k))
        im = ax.imshow(W_maps[:, :, k], origin="upper", extent=extent)
        ax.set_title(titles[k])
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axR = plt.subplot2grid((2, K + 1), (1, K))
    imR = axR.imshow(E_map, origin="upper", extent=extent)
    axR.set_title("Residual (RMSE)")
    axR.invert_yaxis()
    plt.colorbar(imR, ax=axR, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Run sklearn NMF on XEOL (SLURM batch).")

    # data sources
    ap.add_argument("--h5path", help="Path to HDF5 file.")
    ap.add_argument("--scan-number", type=int, help="Scan number for H5 reading.")
    ap.add_argument("--spectra-npy", help="Path to spectra .npy (n_pixels, n_channels).")
    ap.add_argument("--wl-npy", help="Path to wavelength .npy (n_channels,).")

    # geometry
    ap.add_argument("--map-nx", type=int, required=True)
    ap.add_argument("--map-ny", type=int, required=True)

    # nmf params (NOW SUPPORTED)
    ap.add_argument("--n-components", type=int, default=3)
    ap.add_argument("--loss", default="kullback-leibler", choices=["frobenius", "kullback-leibler"])
    ap.add_argument("--solver", default=None, choices=[None, "cd", "mu"], nargs="?", help="Override solver.")
    ap.add_argument("--init", default="nndsvda")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--clip-negative", action="store_true", default=True)
    ap.add_argument("--no-clip-negative", dest="clip_negative", action="store_false")
    ap.add_argument("--extent", type=float, nargs=4,metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="Extent for imshow: xmin xmax ymin ymax", default=None,)

    # outputs / plot
    ap.add_argument("--outdir", default="nmf_out")
    ap.add_argument("--normalize-spectra", action="store_true", default=False)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    cpus = max(1, cpus)

    # ---- load data ----
    if args.spectra_npy:
        X = np.load(args.spectra_npy)
        wl = np.load(args.wl_npy) if args.wl_npy else None
        print(f"Loaded spectra from npy: {args.spectra_npy} | shape={X.shape}")
    else:
        if not (args.h5path and args.scan_number is not None):
            raise SystemExit("Provide either --spectra-npy OR (--h5path and --scan-number).")

        from lauexplore._parsers import _h5
        with h5py.File(args.h5path, "r") as h5f:
            X = np.array(_h5.get_xeol(h5f, args.scan_number))
            wl = h5f[f"{args.scan_number}.1/measurement/qepro_det1"][0]
        print(f"Loaded spectra from H5: {args.h5path} | scan={args.scan_number} | shape={X.shape}")

    # ---- run nmf ----
    t0 = time.time()
    if threadpool_limits is not None:
        ctx = threadpool_limits(limits=cpus)
    else:
        class _Dummy:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        ctx = _Dummy()

    with ctx:
        W_maps, H, X_rec, E_map, model, wl_used, unit = nmf_sklearn_hyperspectral(
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
            tol = args.tol,
        )

    print(f"NMF done in {time.time()-t0:.2f} s | n_iter_={getattr(model, 'n_iter_', 'NA')} | tol = {args.tol}")
    print(f"init ={args.init}")

    # ---- save arrays ----
    np.save(os.path.join(args.outdir, "W_maps.npy"), W_maps)
    np.save(os.path.join(args.outdir, "H.npy"), H)
    np.save(os.path.join(args.outdir, "E_map.npy"), E_map)
    np.save(os.path.join(args.outdir, "wavelength.npy"), wl_used)

    # ---- plot ----
    extent = args.extent if args.extent is not None else None

    fig = plot_nmf_panel(
        W_maps, H, E_map, wl_used,
        unit_name=unit,
        normalize_spectra=args.normalize_spectra,
        extent=extent,  # add if you want with x/y mm
        figsize=(10, 10),
    )
    fig.savefig(os.path.join(args.outdir, "nmf_panel.png"), dpi=200, bbox_inches="tight")
    # fig.savefig(os.path.join(args.outdir, "nmf_panel.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved outputs in: {args.outdir}")


if __name__ == "__main__":
    main()