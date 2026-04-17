import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

def nmf_sklearn_hyperspectral(
    X,                 # (n_pixels, n_channels)
    map_shape,         # (nx, ny) with nx*ny == n_pixels
    n_components=4,
    wavelength=None,   # (n_channels,)
    unit_name="Wavelength (nm)",
    loss="frobenius",  # "frobenius" or "kullback-leibler"
    solver=None,       # None -> pick default based on loss
    init="nndsvda",
    max_iter=1000,
    random_state=0,
    l1_ratio=0.0,
    alpha_W=0.0,
    alpha_H=0.0,
    clip_negative=True,
    show_progress=True,
):
    """
    Fits NMF with scikit-learn on hyperspectral data X and returns:
      - W_maps: (nx, ny, K) component intensity maps
      - H:      (K, n_channels) component spectra
      - X_rec:  (n_pixels, n_channels) reconstructed data
      - E_map:  (nx, ny) RMSE per pixel
      - model:  fitted sklearn NMF object
    """
    if show_progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=4, desc="NMF (sklearn)", unit="step")
        def step(msg):
            pbar.set_postfix_str(msg)
            pbar.update(1)
    else:
        pbar = None
        def step(msg):  # no-op
            return

    try:
        step("prepare input")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_pixels, n_channels). Got {X.shape}.")

        n_pixels, n_ch = X.shape
        nx, ny = map_shape
        if nx * ny != n_pixels:
            raise ValueError(f"map_shape product must match n_pixels. Got {nx*ny} vs {n_pixels}.")

        # NMF requires non-negative data
        if clip_negative and X.min() < 0:
            X = X.copy()
            X[X < 0] = 0.0
        elif X.min() < 0:
            raise ValueError("NMF requires non-negative X. Clip negatives or change preprocessing.")

        if wavelength is None:
            wavelength = np.arange(n_ch)

        # Choose solver default
        if solver is None:
            solver = "mu" if loss == "kullback-leibler" else "cd"

        model = NMF(
            n_components=n_components,
            init=init,
            solver=solver,
            beta_loss=loss,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
        )

        # Fit
        step("fit_transform (this is the long part)")
        W = model.fit_transform(X)      # (n_pixels, K)
        H = model.components_           # (K, n_channels)

        # Reconstruction
        step("reconstruct")
        X_rec = W @ H

        # Residuals / reshape
        step("rmse + reshape")
        rmse = np.sqrt(np.mean((X - X_rec) ** 2, axis=1))
        E_map = rmse.reshape(nx, ny)
        W_maps = W.reshape(nx, ny, n_components)

        return W_maps, H, X_rec, E_map, model, wavelength, unit_name

    finally:
        if pbar is not None:
            pbar.close()


def plot_nmf_panel(
    W_maps,
    H,
    E_map,
    wavelength,
    unit_name="Wavelength (nm)",
    normalize_spectra=True,
    titles=None,
    figsize=(14, 7),
    extent = None, 
):
    """
    Paper-like panel:
      - Top: spectra of components
      - Bottom: component maps + residual
    """
    K = H.shape[0]
    if titles is None:
        titles = [f"Component {k+1}" for k in range(K)]

    fig = plt.figure(figsize=figsize)

    # --- spectra (top, full width) ---
    ax0 = plt.subplot2grid((2, K+1), (0, 0), colspan=K+1)
    for k in range(K):
        y = H[k].copy()
        if normalize_spectra:
            y /= (np.linalg.norm(y) + 1e-16)
        ax0.plot(wavelength, y, label=str(k+1))
    ax0.set_xlabel(unit_name)
    ax0.set_ylabel("Intensity (a.u.)" if normalize_spectra else "Intensity")
    ax0.legend(ncol=min(K, 6), frameon=False)
    ax0.grid(True, alpha=0.25)

    # --- component maps (bottom row) ---
    for k in range(K):
        ax = plt.subplot2grid((2, K+1), (1, k))
        im = ax.imshow(W_maps[:, :, k], origin="upper", extent=extent)
        ax.set_title(titles[k])
        ax.invert_yaxis()
        # ax.set_xticks([])
        # ax.set_yticks([])
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- residual map (last column) ---
    axR = plt.subplot2grid((2, K+1), (1, K))
    imR = axR.imshow(E_map, origin="upper", extent=extent)
    axR.set_title("Residual (RMSE)")
    axR.invert_yaxis()
    # axR.set_xticks([])
    # axR.set_yticks([])
    plt.colorbar(imR, ax=axR, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()