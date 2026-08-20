"""
nmf_per_seg.py

Run NMF independently per segmented microLED device, enabling spectral
comparison between high- and low-emission groups.

Designed to be imported in a Jupyter notebook OR run as a standalone script.

Typical notebook usage
----------------------
    from emission.nmf_per_seg import run_segmented_nmf

    results = run_segmented_nmf(
        spectra     = xeol_corr,      # (n_pixels, n_channels) array
        labels      = labels,         # (ny, nx) int array from segment_leds
        emission_map= new_blue,       # (ny, nx) float – used to rank LEDs
        wavelength  = xeol.wl_array, # (n_channels,) array
        n_components= 3,
        outdir      = "nmf_per_led_out",
    )

Outputs
-------
Per LED  →  outdir/led_{id:02d}/
    W.npy            (n_led_pixels, K)
    H.npy            (K, n_channels)
    W_fullmap.npy    (ny, nx, K)   – NaN outside the LED
    E_fullmap.npy    (ny, nx)      – NaN outside the LED
    wavelength.npy   (n_channels,)

Per group → outdir/group_high/  and  outdir/group_low/
    W_maps.npy, H.npy, E_map.npy, wavelength.npy   (same convention as nmf_sbatch_job)

Summary figure → outdir/comparison_spectra.png
"""

import os
import numpy as np
import matplotlib

if __name__ == "__main__":
    # Only force a non-interactive backend for standalone/headless runs
    # (e.g. SLURM). Forcing it on import breaks Jupyter's inline display.
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from emission.NMF import run_nmf


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _run_nmf(X_pixels, n_components, loss="kullback-leibler",
             init="nndsvda", max_iter=2000, random_state=0):
    """
    Fit NMF on X_pixels (n_pixels, n_channels).

    Returns
    -------
    W     : (n_pixels, K)
    H     : (K, n_channels)
    rmse  : (n_pixels,)  per-pixel reconstruction RMSE
    model : fitted sklearn NMF object
    """
    W, H, _, rmse, model = run_nmf(
        X_pixels, n_components,
        loss=loss, init=init, max_iter=max_iter, random_state=random_state,
    )
    return W, H, rmse, model


def _build_fullmap(values_flat, mask_flat, full_shape, n_components=None):
    """
    Scatter a flat array of values (n_masked_pixels [, K]) into a full 2D map.

    Parameters
    ----------
    values_flat  : (n_masked,) or (n_masked, K)
    mask_flat    : (ny*nx,) bool
    full_shape   : (ny, nx)
    n_components : int or None  –  set to K for weight maps, None for RMSE maps

    Returns NaN-filled array of shape (ny, nx [, K]).
    """
    ny, nx = full_shape
    idx = np.where(mask_flat)[0]
    iy = idx // nx
    ix = idx % nx

    if n_components is not None:
        out = np.full((ny, nx, n_components), np.nan)
        out[iy, ix, :] = values_flat
    else:
        out = np.full((ny, nx), np.nan)
        out[iy, ix] = values_flat

    return out


# ---------------------------------------------------------------------------
# Segment-level NMF
# ---------------------------------------------------------------------------

def run_nmf_on_segment(X_flat, mask_flat, full_shape, n_components,
                       loss="kullback-leibler", init="nndsvda",
                       max_iter=2000, random_state=0):
    """
    Run NMF on the pixels selected by *mask_flat* and return full-map arrays.

    Parameters
    ----------
    X_flat     : (n_total_pixels, n_channels)
    mask_flat  : (n_total_pixels,) bool
    full_shape : (ny, nx)

    Returns
    -------
    W          : (n_segment_pixels, K)
    H          : (K, n_channels)
    rmse       : (n_segment_pixels,)
    W_fullmap  : (ny, nx, K)  NaN outside segment
    E_fullmap  : (ny, nx)     NaN outside segment
    model      : fitted NMF object
    """
    X_seg = X_flat[mask_flat]
    n_pixels = X_seg.shape[0]

    if n_pixels < n_components:
        raise ValueError(
            f"Segment has {n_pixels} pixels but n_components={n_components}. "
            "Lower n_components or skip this segment."
        )

    W, H, rmse, model = _run_nmf(
        X_seg, n_components, loss=loss, init=init,
        max_iter=max_iter, random_state=random_state,
    )
    W_fullmap = _build_fullmap(W,    mask_flat, full_shape, n_components)
    E_fullmap = _build_fullmap(rmse, mask_flat, full_shape)

    return W, H, rmse, W_fullmap, E_fullmap, model


# ---------------------------------------------------------------------------
# LED classification
# ---------------------------------------------------------------------------

def classify_leds(labels, emission_map, high_percentile=75):
    """
    Split LED labels into 'high' and 'low' emission groups.

    The mean emission of each LED (within its segmented pixels) is computed.
    LEDs whose mean exceeds the *high_percentile* of all LED means are 'high'.

    Parameters
    ----------
    labels          : (ny, nx) int array  –  0 = background
    emission_map    : (ny, nx) float array
    high_percentile : float  –  percentile threshold (default 50 = median split)

    Returns
    -------
    led_means : dict {label_id: mean_emission}
    high_leds : list of int
    low_leds  : list of int
    """
    unique_leds = [int(l) for l in np.unique(labels) if l != 0]

    led_means = {}
    for led_id in unique_leds:
        mask = labels == led_id
        led_means[led_id] = float(np.mean(emission_map[mask]))

    means = np.array(list(led_means.values()))
    threshold = np.percentile(means, high_percentile)

    high_leds = [lid for lid, m in led_means.items() if m >= threshold]
    low_leds  = [lid for lid, m in led_means.items() if m <  threshold]

    return led_means, high_leds, low_leds


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_comparison(H_high, H_low, wavelength, outdir="."):
    """Save a figure comparing NMF component spectra for high vs low groups."""
    K = H_high.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4), sharey=False)
    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        h_hi = H_high[k] / (np.linalg.norm(H_high[k]) + 1e-16)
        h_lo = H_low[k]  / (np.linalg.norm(H_low[k])  + 1e-16)
        ax.plot(wavelength, h_hi, label="high emission", color="C0")
        ax.plot(wavelength, h_lo, label="low emission",  color="C1", linestyle="--")
        ax.set_title(f"Component {k + 1}")
        ax.set_xlabel("Wavelength (nm)")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)

    plt.suptitle("NMF spectra: high vs low emission LEDs", y=1.01)
    plt.tight_layout()
    outpath = os.path.join(outdir, "comparison_spectra.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_per_led_spectra(per_led_results, wavelength, ncols=4,
                         normalize=True, outdir=None):
    """
    Plot NMF component spectra for each LED individually (notebook-friendly).

    Blue = high-emission LED, orange = low-emission LED.
    One figure per NMF component.

    Parameters
    ----------
    per_led_results : dict returned by run_segmented_nmf["per_led"]
    wavelength      : (n_channels,) array
    ncols           : int  –  number of columns in the subplot grid
    normalize       : bool  –  L2-normalise spectra before plotting
    outdir          : str or None  –  if given, saves .png files there
    """
    led_ids = sorted(lid for lid, r in per_led_results.items() if "H" in r)
    if not led_ids:
        print("No successful per-LED results to plot.")
        return

    K = per_led_results[led_ids[0]]["H"].shape[0]
    nrows = int(np.ceil(len(led_ids) / ncols))

    for k in range(K):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharey=True
        )
        axes = np.array(axes).ravel()

        for ax_idx, led_id in enumerate(led_ids):
            r = per_led_results[led_id]
            h = r["H"][k].copy()
            if normalize:
                h /= (np.linalg.norm(h) + 1e-16)
            color = "C0" if r["group"] == "high" else "C1"
            axes[ax_idx].plot(wavelength, h, color=color, lw=1)
            axes[ax_idx].set_title(
                f"LED {led_id} ({r['group']},\n"
                f"RMSE={r['rmse_mean']:.3f})",
                fontsize=8,
            )
            axes[ax_idx].grid(True, alpha=0.2)

        for ax in axes[len(led_ids):]:
            ax.set_visible(False)

        plt.suptitle(
            f"Component {k + 1}  —  blue=high emission, orange=low emission",
            y=1.01,
        )
        plt.tight_layout()

        if outdir:
            outpath = os.path.join(outdir, f"per_led_component_{k + 1}.png")
            plt.savefig(outpath, dpi=150, bbox_inches="tight")
            print(f"Saved: {outpath}")

        plt.show()


# ---------------------------------------------------------------------------
# Single/selected-label NMF (interactive inspection)
# ---------------------------------------------------------------------------

def plot_led_nmf(led_ids, H, W_fullmap, E_fullmap, wavelength,
                 percentile_clip=(2, 98), cmap="viridis", ncols=2, outdir=None):
    """
    Plot abundance maps, component spectra, and residual map for one NMF fit
    (one LED, or several LEDs pooled together upstream into a single fit).

    Maps share x/y axes (pan/zoom together) and are laid out two per row.

    Parameters
    ----------
    led_ids         : int or list of int  –  used only for the title / filename
    H               : (K, n_channels)  –  NMF component spectra
    W_fullmap       : (ny, nx, K)      –  abundance maps, NaN outside the group
    E_fullmap       : (ny, nx)         –  per-pixel RMSE, NaN outside the group
    wavelength      : (n_channels,)
    percentile_clip : (lo, hi)  –  colour scale percentiles for the maps
    ncols           : int  –  maps per row (default 2)
    outdir          : str or None  –  if given, saves led_{ids}_nmf.png there
    """
    if isinstance(led_ids, (int, np.integer)):
        led_ids = [int(led_ids)]
    label_str = ", ".join(str(l) for l in led_ids)
    file_tag  = "_".join(f"{l:02d}" for l in led_ids)

    K = H.shape[0]
    maps = [(f"Abundance k={k + 1}", W_fullmap[:, :, k], cmap) for k in range(K)]
    maps.append(("Residual (RMSE)", E_fullmap, "magma"))

    nrows_maps = int(np.ceil(len(maps) / ncols))
    fig = plt.figure(figsize=(4.5 * ncols, 3.5 * nrows_maps + 3))
    gs = fig.add_gridspec(nrows_maps + 1, ncols)

    axes = []
    for idx, (title, m, mcmap) in enumerate(maps):
        r, c = divmod(idx, ncols)
        share_kwargs = {"sharex": axes[0], "sharey": axes[0]} if axes else {}
        ax = fig.add_subplot(gs[r, c], **share_kwargs)
        vmin, vmax = np.nanpercentile(m, percentile_clip)
        im = ax.imshow(m, origin="lower", aspect="equal",
                       cmap=mcmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes.append(ax)

    ax_spec = fig.add_subplot(gs[nrows_maps, :])
    for k in range(K):
        ax_spec.plot(wavelength, H[k], label=f"Component {k + 1}")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Intensity (a.u.)")
    ax_spec.legend(frameon=False)
    ax_spec.grid(True, alpha=0.25)

    fig.suptitle(f"LED {label_str}  —  NMF (K={K})", y=1.01)
    plt.tight_layout()

    if outdir:
        outpath = os.path.join(outdir, f"led_{file_tag}_nmf.png")
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {outpath}")

    plt.show()


def run_nmf_for_labels(
    spectra,
    labels,
    led_ids,
    wavelength=None,
    n_components=3,
    loss="kullback-leibler",
    init="nndsvda",
    max_iter=2000,
    random_state=0,
    percentile_clip=(2, 98),
    ncols=2,
    outdir=None,
    verbose=True,
):
    """
    Run NMF on one or more chosen LED labels and plot the results
    (abundance maps, component spectra, residual).

    If more than one label is given, their pixels are pooled together into
    a single NMF fit (not one run per label) — use this to analyse a group
    of LEDs jointly.

    Meant for interactive/notebook use, to inspect chosen LEDs without
    running the full `run_segmented_nmf` sweep over every label.

    Parameters
    ----------
    spectra      : (n_pixels, n_channels) array or path to .npy
    labels       : (ny, nx) int array or path to .npy  –  0 = background
    led_ids      : int or list of int  –  label id(s) to pool and run NMF on
    wavelength   : (n_channels,) array or path to .npy, optional
    n_components : int  –  number of NMF components (K)
    loss         : 'kullback-leibler' or 'frobenius'
    init         : NMF initialisation (default 'nndsvda')
    max_iter     : int
    random_state : int
    percentile_clip : (lo, hi)  –  colour scale percentiles for the maps
    ncols        : int  –  maps per row in the plot (default 2)
    outdir       : str or None  –  if given, saves one PNG for the group there
    verbose      : bool

    Returns
    -------
    result : dict with 'H', 'W', 'W_fullmap', 'E_fullmap', 'rmse_mean',
             'n_pixels', 'led_ids'
    """
    if isinstance(spectra, (str, os.PathLike)):
        spectra = np.load(spectra)
    if isinstance(labels, (str, os.PathLike)):
        labels = np.load(labels)
    if isinstance(wavelength, (str, os.PathLike)):
        wavelength = np.load(wavelength)

    spectra = np.asarray(spectra, dtype=np.float64)
    labels  = np.asarray(labels, dtype=int)

    full_shape = labels.shape
    ny, nx = full_shape
    if spectra.shape[0] != ny * nx:
        raise ValueError(
            f"spectra.shape[0]={spectra.shape[0]} must equal ny*nx={ny * nx}. "
            "Check that labels and spectra share the same pixel ordering."
        )

    if wavelength is None:
        wavelength = np.arange(spectra.shape[1], dtype=float)

    if isinstance(led_ids, (int, np.integer)):
        led_ids = [int(led_ids)]
    else:
        led_ids = [int(l) for l in led_ids]

    labels_flat = labels.ravel()
    mask_flat   = np.isin(labels_flat, led_ids)

    missing = [l for l in led_ids if not np.any(labels_flat == l)]
    if missing:
        raise ValueError(f"Label(s) {missing} not found in `labels`.")

    n_px = int(mask_flat.sum())
    if verbose:
        print(f"LEDs {sorted(led_ids)}: {n_px} px pooled, "
             f"running NMF (K={n_components}) ... ", end="", flush=True)

    W, H, rmse, W_fullmap, E_fullmap, model = run_nmf_on_segment(
        spectra, mask_flat, full_shape, n_components,
        loss=loss, init=init, max_iter=max_iter, random_state=random_state,
    )

    if verbose:
        print(f"ok (mean RMSE={np.mean(rmse):.4f})")

    result = {
        "H":         H,
        "W":         W,
        "W_fullmap": W_fullmap,
        "E_fullmap": E_fullmap,
        "rmse_mean": float(np.mean(rmse)),
        "n_pixels":  n_px,
        "led_ids":   sorted(led_ids),
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    plot_led_nmf(
        led_ids, H, W_fullmap, E_fullmap, wavelength,
        percentile_clip=percentile_clip, ncols=ncols, outdir=outdir,
    )

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_segmented_nmf(
    spectra,
    labels,
    emission_map,
    wavelength=None,
    n_components=3,
    high_percentile=75,
    loss="kullback-leibler",
    init="nndsvda",
    max_iter=2000,
    random_state=0,
    outdir="nmf_per_led_out",
    verbose=True,
):
    """
    Run NMF per LED and per emission group (high / low).

    Parameters
    ----------
    spectra          : (n_pixels, n_channels) array or path to .npy
    labels           : (ny, nx) int array or path to .npy  –  0 = background
    emission_map     : (ny, nx) float array or path to .npy
                       2D image used to rank LEDs (e.g. new_blue = blue/sap*4)
    wavelength       : (n_channels,) array or path to .npy, optional
    n_components     : int  –  number of NMF components
    high_percentile  : float  –  LEDs above this percentile of mean emission
                       are considered 'high' (default 50 = median split)
    loss             : 'kullback-leibler' or 'frobenius'
    init             : NMF initialisation (default 'nndsvda')
    max_iter         : int
    random_state     : int
    outdir           : str  –  root output directory
    save_figures     : bool  –  save comparison PNG
    verbose          : bool

    Returns
    -------
    results : dict
        'per_led'   : dict {led_id: {'H', 'W', 'W_fullmap', 'E_fullmap',
                                      'rmse_mean', 'n_pixels', 'group', 'mean_emission'}}
        'led_means' : dict {led_id: mean_emission}  — classificação por emissão
        'high_leds' : list of int                   — apenas informação
        'low_leds'  : list of int                   — apenas informação
    """
    # --- load arrays if paths were passed ---
    if isinstance(spectra, (str, os.PathLike)):
        spectra = np.load(spectra)
    if isinstance(labels, (str, os.PathLike)):
        labels = np.load(labels)
    if isinstance(emission_map, (str, os.PathLike)):
        emission_map = np.load(emission_map)
    if isinstance(wavelength, (str, os.PathLike)):
        wavelength = np.load(wavelength)

    spectra      = np.asarray(spectra,      dtype=np.float64)
    labels       = np.asarray(labels,       dtype=int)
    emission_map = np.asarray(emission_map, dtype=float)

    full_shape = labels.shape       # (ny, nx)
    ny, nx     = full_shape
    n_px_total = ny * nx

    if spectra.shape[0] != n_px_total:
        raise ValueError(
            f"spectra.shape[0]={spectra.shape[0]} must equal ny*nx={n_px_total}. "
            "Check that labels and spectra share the same pixel ordering."
        )

    if wavelength is None:
        wavelength = np.arange(spectra.shape[1], dtype=float)

    os.makedirs(outdir, exist_ok=True)

    # --- classify LEDs ---
    led_means, high_leds, low_leds = classify_leds(
        labels, emission_map, high_percentile
    )
    if verbose:
        print(f"LED classification  (high_percentile={high_percentile}%)")
        print(f"  High-emission LEDs: {sorted(high_leds)}")
        print(f"  Low-emission  LEDs: {sorted(low_leds)}")

    labels_flat = labels.ravel()
    unique_leds = sorted(int(l) for l in np.unique(labels) if l != 0)

    # --- per-LED NMF ---
    per_led_results = {}
    if verbose:
        print(f"\nRunning per-LED NMF (n_components={n_components}):")

    for led_id in unique_leds:
        mask_flat = labels_flat == led_id
        n_px      = int(mask_flat.sum())
        led_dir   = os.path.join(outdir, f"led_{led_id:02d}")
        os.makedirs(led_dir, exist_ok=True)

        if verbose:
            group = "HIGH" if led_id in high_leds else "LOW "
            print(f"  LED {led_id:3d} [{group}] {n_px:5d} px ... ", end="", flush=True)

        try:
            W, H, rmse, W_fullmap, E_fullmap, model = run_nmf_on_segment(
                spectra, mask_flat, full_shape, n_components,
                loss=loss, init=init, max_iter=max_iter, random_state=random_state,
            )
            np.save(os.path.join(led_dir, "W.npy"),         W)
            np.save(os.path.join(led_dir, "H.npy"),         H)
            np.save(os.path.join(led_dir, "W_fullmap.npy"), W_fullmap)
            np.save(os.path.join(led_dir, "E_fullmap.npy"), E_fullmap)
            np.save(os.path.join(led_dir, "wavelength.npy"),wavelength)

            per_led_results[led_id] = {
                "H":             H,            # (K, n_channels)
                "W":             W,            # (n_led_pixels, K)  — flat
                "W_fullmap":     W_fullmap,    # (ny, nx, K)        — NaN outside LED
                "E_fullmap":     E_fullmap,    # (ny, nx)           — NaN outside LED
                "rmse_mean":     float(np.mean(rmse)),
                "n_pixels":      n_px,
                "group":         "high" if led_id in high_leds else "low",
                "mean_emission": led_means[led_id],
            }
            if verbose:
                print(f"ok  (mean RMSE={np.mean(rmse):.4f})")

        except Exception as exc:
            per_led_results[led_id] = {"error": str(exc)}
            if verbose:
                print(f"FAILED: {exc}")

    if verbose:
        print(f"\nAll outputs saved in: {outdir}")

    return {
        "per_led":   per_led_results,
        "led_means": led_means,   # {led_id: mean_emission}
        "high_leds": high_leds,   # classification info only
        "low_leds":  low_leds,
    }


# ---------------------------------------------------------------------------
# CLI (no SLURM)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Segmented NMF per microLED device (local, no SLURM)."
    )
    ap.add_argument("--spectra-npy",      required=True,  help="Path to spectra .npy (n_pixels, n_channels)")
    ap.add_argument("--labels-npy",       required=True,  help="Path to labels .npy (ny, nx)")
    ap.add_argument("--emission-npy",     required=True,  help="Path to emission map .npy (ny, nx) for high/low classification")
    ap.add_argument("--wl-npy",           default=None,   help="Path to wavelength .npy (n_channels,)")
    ap.add_argument("--n-components",     type=int,   default=3)
    ap.add_argument("--high-percentile",  type=float, default=50,
                    help="LEDs above this percentile of mean emission are 'high' (default: 50)")
    ap.add_argument("--loss",    default="kullback-leibler",
                    choices=["frobenius", "kullback-leibler"])
    ap.add_argument("--max-iter",  type=int, default=2000)
    ap.add_argument("--outdir",    default="nmf_per_led_out")
    args = ap.parse_args()

    run_segmented_nmf(
        spectra      = args.spectra_npy,
        labels       = args.labels_npy,
        emission_map = args.emission_npy,
        wavelength   = args.wl_npy,
        n_components = args.n_components,
        high_percentile = args.high_percentile,
        loss         = args.loss,
        max_iter     = args.max_iter,
        outdir       = args.outdir,
    )
