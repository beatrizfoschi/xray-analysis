import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.optimize import curve_fit

from utils.fitting import gaussian


# paleta categórica (ordem fixa — nunca ciclar/reordenar)
CATEGORICAL_COLORS = [
    # "#2a78d6",  # blue
    # "#1baf7a",  # aqua
    # "#eda100",  # yellow
    # "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]


def plot_histograms(dfs, columns, labels=None, bins=30, ncols=3,
                     figsize=None, alpha=0.55, percentile_clip=None,
                     density=False, fit_gaussian=False, cmap=None,
                     cmap_center=None, fit_color=None):
    """
    Histogramas de uma ou mais colunas, para um ou vários DataFrames sobrepostos.

    dfs : pd.DataFrame ou list[pd.DataFrame]
    columns : str ou list[str] — uma coluna por subplot
    labels : list[str], legenda por DataFrame (default: 'df 0', 'df 1', ...)
    percentile_clip : (low, high), opcional — recorta o range do eixo x
        pelos percentis calculados no pool de todos os dfs (exclui outliers da vista
        e, se fit_gaussian=True, também da amostra usada no ajuste)
    fit_gaussian : bool
        Se True, ajusta uma gaussiana (mínimos quadrados nos bins do histograma,
        não nos dados brutos — o pico central domina o ajuste) a cada série,
        sobrepõe a curva e anota μ ± desvio padrão da média (SEM = σ/√n) e FWHM.
    cmap : str, list[str] ou dict[str, str], opcional
        Colormap para combinar com o mapa 2D do mesmo metric (ex.: 'viridis'
        para 'period_nm' em plot_satellite_maps). Uma única string aplica-se a todas
        as colunas; lista/dict associam um cmap a cada coluna (mesma ordem/keys
        que `columns`). Se None (default), usa a paleta categórica fixa.
        Com um único DataFrame, cada barra é colorida conforme o seu próprio
        valor de x (gradiente, igual à colorbar do mapa 2D); com vários
        DataFrames sobrepostos, cada série usa um tom sólido diferente do
        cmap (a coloração por bin perderia a distinção entre séries).
    cmap_center : float, dict[str, float] ou None
        Valor (ex.: a média) que deve cair no centro do cmap, usando
        TwoSlopeNorm — replica o centering usado para 'axis_angle' em
        plot_satellite_maps. Só tem efeito com `cmap` definido e um único DataFrame.
        Se None, usa Normalize linear simples (lo, hi).
    fit_color : str, list[str] ou None
        Cor da curva gaussiana (fit_gaussian=True), para se destacar das barras
        (que com `cmap` seguem o gradiente, não uma cor sólida). Uma única
        string aplica-se a todas as séries; lista associa uma cor por DataFrame
        (mesma ordem que `dfs`). Se None (default), usa a cor da série (tom do
        cmap ou paleta categórica), como antes.
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    if isinstance(columns, str):
        columns = [columns]
    if labels is None:
        labels = [f"df {i}" for i in range(len(dfs))]
    if len(labels) != len(dfs):
        raise ValueError("labels deve ter o mesmo tamanho que dfs")
    if cmap is None and len(dfs) > len(CATEGORICAL_COLORS):
        raise ValueError(f"máximo de {len(CATEGORICAL_COLORS)} dataframes suportados")

    if cmap is not None:
        if isinstance(cmap, str):
            cmap = {col: cmap for col in columns}
        elif not isinstance(cmap, dict):
            if len(cmap) != len(columns):
                raise ValueError("cmap deve ter o mesmo tamanho que columns")
            cmap = dict(zip(columns, cmap))
    if cmap_center is not None and not isinstance(cmap_center, dict):
        cmap_center = {col: cmap_center for col in columns}
    if fit_color is not None:
        if isinstance(fit_color, str):
            fit_colors = [fit_color] * len(dfs)
        else:
            if len(fit_color) != len(dfs):
                raise ValueError("fit_color deve ter o mesmo tamanho que dfs")
            fit_colors = list(fit_color)
    else:
        fit_colors = None

    n = len(columns)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, col in zip(axes, columns):
        values_per_df = [df[col].dropna().to_numpy() for df in dfs]
        pooled = np.concatenate(values_per_df) if len(values_per_df) > 1 else values_per_df[0]

        hist_range = None
        if percentile_clip is not None:
            hist_range = tuple(np.percentile(pooled, percentile_clip))

        fit_lines = []

        if cmap is not None:
            cm = plt.get_cmap(cmap[col])
            n_df = len(dfs)
            positions = [0.6] if n_df == 1 else np.linspace(0.25, 0.85, n_df)
            colors = [cm(p) for p in positions]
        else:
            colors = CATEGORICAL_COLORS

        for k, (values, label, color) in enumerate(zip(values_per_df, labels, colors)):
            counts, edges, patches = ax.hist(values, bins=bins, range=hist_range, density=density,
                     alpha=alpha, linewidth=1.0, label=label)

            if cmap is not None and len(dfs) == 1:
                cm = plt.get_cmap(cmap[col])
                lo, hi = hist_range if hist_range is not None else (float(values.min()), float(values.max()))
                if cmap_center is not None and col in cmap_center:
                    norm = TwoSlopeNorm(vmin=lo, vcenter=cmap_center[col], vmax=hi)
                else:
                    norm = Normalize(vmin=lo, vmax=hi)
                bin_centers = 0.5 * (edges[:-1] + edges[1:])
                for patch, bc in zip(patches, bin_centers):
                    bin_color = cm(norm(bc))
                    patch.set_facecolor(bin_color)
                    patch.set_edgecolor(bin_color)
            else:
                for patch in patches:
                    patch.set_facecolor(color)
                    patch.set_edgecolor(color)

            if fit_gaussian and values.size > 1:
                fit_values = values
                if hist_range is not None:
                    fit_values = values[(values >= hist_range[0]) & (values <= hist_range[1])]
                if fit_values.size < 2:
                    continue

                centers = 0.5 * (edges[:-1] + edges[1:])
                p0 = [counts.max(), centers[np.argmax(counts)], np.std(fit_values)]
                try:
                    popt, _ = curve_fit(gaussian, centers, counts, p0=p0, maxfev=5000)
                except RuntimeError:
                    continue
                A, mu, sigma = popt
                sigma = abs(sigma)
                sem = sigma / np.sqrt(fit_values.size)
                fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

                line_color = fit_colors[k] if fit_colors is not None else color
                x_fit = np.linspace(edges[0], edges[-1], 300)
                ax.plot(x_fit, gaussian(x_fit, A, mu, sigma), color=line_color, linewidth=1.8)
                fit_lines.append(f"{label}: μ={mu:.3g}±{sem:.2g}  FWHM={fwhm:.3g}")

        if fit_lines:
            ax.text(0.98, 0.98, "\n".join(fit_lines), transform=ax.transAxes,
                     ha="right", va="top", fontsize=7,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               alpha=0.8, edgecolor="#cccccc"))

        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("density" if density else "count")
        for spine in ax.spines.values():
            spine.set_visible(True)

    for ax in axes[n:]:
        ax.set_visible(False)

    if len(dfs) > 1:
        handles, labs = axes[0].get_legend_handles_labels()
        fig.legend(handles, labs, loc="upper center", ncol=len(dfs),
                   bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.tight_layout()
    return fig, axes[:n]
