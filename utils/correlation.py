import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, theilslopes


def _valid_xy(a, b, mask, percentile_clip=None):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shapes diferentes: {a.shape} vs {b.shape}")

    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    x = a[valid].ravel()
    y = b[valid].ravel()
    if x.size < 2:
        raise ValueError("menos de 2 pontos válidos para calcular a correlação")

    if percentile_clip is not None:
        lo, hi = percentile_clip
        x_lo, x_hi = np.percentile(x, [lo, hi])
        y_lo, y_hi = np.percentile(y, [lo, hi])
        keep = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
        x, y = x[keep], y[keep]
        if x.size < 2:
            raise ValueError("menos de 2 pontos válidos após percentile_clip")

    return x, y


def _add_fit_line(ax, x, y, fit, color):
    """
    fit : {"ols", "theilsen", "median", None}
        "ols"      — mínimos quadrados (np.polyfit); sensível a outliers.
        "theilsen" — reta robusta de Theil-Sen (mediana das inclinações par a par);
                     ignora outliers. Subamostra se n > 5000 (O(n²) em memória).
        "median"   — tendência central: mediana de y em faixas de x (o diagnóstico
                     mais honesto de monotonicidade numa nuvem ruidosa).
        None/False — não desenha reta.
    """
    if not fit:
        return

    if fit == "ols":
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.array([x.min(), x.max()])
        ax.plot(x_fit, slope * x_fit + intercept, color=color, linewidth=1.5)

    elif fit == "theilsen":
        xs, ys = x, y
        if x.size > 5000:
            idx = np.random.default_rng(0).choice(x.size, 5000, replace=False)
            xs, ys = x[idx], y[idx]
        slope, intercept, _, _ = theilslopes(ys, xs)
        x_fit = np.array([x.min(), x.max()])
        ax.plot(x_fit, slope * x_fit + intercept, color=color, linewidth=1.5)

    elif fit == "median":
        edges = np.linspace(x.min(), x.max(), 21)
        idx = np.digitize(x, edges)
        centers, meds = [], []
        for b in range(1, len(edges)):
            m = idx == b
            if m.sum() >= 5:
                centers.append(0.5 * (edges[b - 1] + edges[b]))
                meds.append(np.median(y[m]))
        ax.plot(centers, meds, color=color, linewidth=1.8, marker="o", markersize=3)

    else:
        raise ValueError(f"fit inválido: {fit!r} (use 'ols', 'theilsen', 'median' ou None)")


def _plot_correlation_scatter(x, y, stat_symbol, stat, p, xlabel, ylabel,
                               figsize, color, fit_color, s, alpha, annotate, fit):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=s, alpha=alpha, color=color, edgecolor="none")

    _add_fit_line(ax, x, y, fit, fit_color)

    if annotate:
        ax.text(0.02, 0.98, f"{stat_symbol} = {stat:.3f}\np = {p:.2e}\nn = {x.size}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8, edgecolor="#cccccc"))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig, ax


def plot_pearson_correlation(a, b, mask=None, percentile_clip=None, fit="ols",
                              xlabel="array A", ylabel="array B",
                              figsize=(5, 5), color="#4a3aa7", fit_color="#e34948",
                              s=8, alpha=0.4, annotate=True):
    """
    Correlação de Pearson entre dois arrays 2D, pixel a pixel (relação linear).

    a, b : arrays 2D do mesmo shape
    mask : array bool 2D, opcional — True mantém o pixel (default: usa todos os
        pixels finitos, removendo NaN/Inf de a e/ou b)
    percentile_clip : (low, high), opcional — remove outliers antes de calcular a
        correlação, descartando pontos fora dos percentis [low, high] calculados
        independentemente em x e em y (ex.: (2, 98))
    fit : {"ols", "theilsen", "median", None} — reta/tendência sobreposta
        (ver _add_fit_line). "theilsen" e "median" são robustas a outliers.

    Retorna (fig, ax, r, p):
        r : coeficiente de correlação de Pearson (-1 a 1)
        p : p-valor do teste (scipy.stats.pearsonr)
    """
    x, y = _valid_xy(a, b, mask, percentile_clip)
    r, p = pearsonr(x, y)
    fig, ax = _plot_correlation_scatter(x, y, "r", r, p, xlabel, ylabel,
                                         figsize, color, fit_color, s, alpha, annotate, fit)
    return fig, ax, r, p


def plot_spearman_correlation(a, b, mask=None, percentile_clip=None, fit="theilsen",
                               xlabel="array A", ylabel="array B",
                               figsize=(5, 5), color="#4a3aa7", fit_color="#e34948",
                               s=8, alpha=0.4, annotate=True):
    """
    Correlação de Spearman entre dois arrays 2D, pixel a pixel (relação monotônica,
    baseada nos postos/ranks — mais robusta a outliers e a relações não lineares
    que a de Pearson).

    a, b : arrays 2D do mesmo shape
    mask : array bool 2D, opcional — True mantém o pixel (default: usa todos os
        pixels finitos, removendo NaN/Inf de a e/ou b)
    percentile_clip : (low, high), opcional — remove outliers antes de calcular a
        correlação, descartando pontos fora dos percentis [low, high] calculados
        independentemente em x e em y (ex.: (2, 98))
    fit : {"ols", "theilsen", "median", None} — reta/tendência sobreposta
        (ver _add_fit_line). Default "theilsen": Spearman não tem inclinação, então
        uma reta OLS (sensível a outliers) não é o anchor visual certo.

    Retorna (fig, ax, rho, p):
        rho : coeficiente de correlação de Spearman (-1 a 1)
        p   : p-valor do teste (scipy.stats.spearmanr)
    """
    x, y = _valid_xy(a, b, mask, percentile_clip)
    rho, p = spearmanr(x, y)
    fig, ax = _plot_correlation_scatter(x, y, "ρ", rho, p, xlabel, ylabel,
                                         figsize, color, fit_color, s, alpha, annotate, fit)
    return fig, ax, rho, p
