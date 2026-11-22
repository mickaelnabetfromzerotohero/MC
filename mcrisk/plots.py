"""Charts in the style of the explainer: dark background, green for gains, red for losses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .estimate import TRADING_DAYS, MarketParams  # noqa: E402
from .risk import RiskSummary  # noqa: E402
from .simulate import SimulationResult  # noqa: E402

BG = "#1c1c1e"
PANEL = "#2a2a2d"
TEXT = "#f2f2f2"
MUTED = "#9a9a9f"
GREEN = "#3fae7a"
RED = "#e05a5a"
DARK_RED = "#a83232"

SERIF = {"family": "serif"}
MONO = {"family": "monospace"}


def _money(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def _density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian kernel density estimate (Silverman bandwidth), numpy only."""
    v = np.asarray(values, dtype=float)
    spread = min(v.std(ddof=1), (np.quantile(v, 0.75) - np.quantile(v, 0.25)) / 1.34)
    bw = 0.9 * spread * v.size ** (-0.2)
    if bw <= 0:
        bw = 1.0
    sample = v if v.size <= 4000 else np.random.default_rng(0).choice(v, 4000, replace=False)
    z = (grid[:, None] - sample[None, :]) / bw
    return np.exp(-0.5 * z**2).sum(axis=1) / (sample.size * bw * np.sqrt(2 * np.pi))


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _figure(width: float, height: float) -> plt.Figure:
    fig = plt.figure(figsize=(width, height), facecolor=BG)
    return fig


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- #
# 1. Inputs: growth, swing, correlations
# ---------------------------------------------------------------------- #
def plot_inputs(
    params: MarketParams,
    path: Path,
    prices: pd.DataFrame | None = None,
    names: dict[str, str] | None = None,
    label: str = "last 5 years",
) -> Path:
    from matplotlib.colors import LinearSegmentedColormap

    names = names or {}
    k = params.n_assets
    fig = _figure(12, 1.4 + 1.5 * k)
    grid = fig.add_gridspec(k, 3, width_ratios=[2.2, 1.1, 1.5], wspace=0.2, hspace=0.6,
                            top=0.8, bottom=0.06)
    fig.suptitle("estimate growth, swing and correlations", color=TEXT, fontsize=16, y=0.97, **SERIF)
    fig.text(0.125, 0.86, label, color=MUTED, fontsize=9, **MONO)

    for i, ticker in enumerate(params.tickers):
        ax = fig.add_subplot(grid[i, 0])
        _style(ax)
        title = f"{names[ticker]} · {ticker}" if ticker in names else ticker
        ax.set_title(title, loc="left", color=TEXT, fontsize=12, **SERIF)
        ax.set_yticks([])
        if prices is not None and ticker in prices:
            series = prices[ticker] / prices[ticker].iloc[0]
            ax.plot(series.index, series.to_numpy(), color=TEXT, lw=0.9)
            ax.plot(series.index, np.exp(params.mu[i] * np.arange(len(series))), color=GREEN,
                    lw=1, ls="--")
            ax.set_yscale("log")
            ax.set_yticks([])
            ax.minorticks_off()
        else:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            ax.text(0.0, 0.3, "no price history (demo inputs)", color=MUTED, fontsize=9,
                    transform=ax.transAxes, **MONO)

        num = fig.add_subplot(grid[i, 1])
        num.axis("off")
        growth = params.annual_growth[i]
        num.text(0.05, 0.35, f"{growth:+.1%}", color=GREEN if growth >= 0 else RED,
                 fontsize=14, transform=num.transAxes, **SERIF)
        num.text(0.6, 0.35, f"±{params.annual_swing[i]:.0%}", color=TEXT, fontsize=14,
                 transform=num.transAxes, **SERIF)
        if i == 0:
            num.text(0.05, 1.05, "growth\na year", color=GREEN, fontsize=8, transform=num.transAxes, **MONO)
            num.text(0.6, 1.05, "swing\na year", color=TEXT, fontsize=8, transform=num.transAxes, **MONO)

    ax = fig.add_subplot(grid[:, 2])
    corr = params.corr
    cmap = LinearSegmentedColormap.from_list("bg_green", [PANEL, GREEN])
    ax.imshow(np.clip(corr, 0, 1), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(k), params.tickers, color=TEXT)
    ax.set_yticks(range(k), params.tickers, color=TEXT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for r in range(k):
        for c in range(k):
            ax.text(c, r, f"{corr[r, c]:.2f}", ha="center", va="center", color=TEXT, fontsize=11, **MONO)
    ax.set_title("correlations", color=TEXT, fontsize=12, **SERIF)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
# 2. Fan of simulated futures + where they ended
# ---------------------------------------------------------------------- #
def plot_fan(result: SimulationResult, path: Path, n_paths: int = 400, seed: int = 0) -> Path:
    paths = result.paths
    capital = result.capital
    final = result.final
    up = final > capital

    fig = _figure(10, 5.6)
    grid = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.02)
    ax = fig.add_subplot(grid[0])
    side = fig.add_subplot(grid[1], sharey=ax)
    _style(ax)
    _style(side)

    rng = np.random.default_rng(seed)
    idx = rng.choice(result.n_sims, size=min(n_paths, result.n_sims), replace=False)
    days = np.arange(result.days + 1)
    for i in idx:
        ax.plot(days, paths[i], color=GREEN if up[i] else RED, lw=0.6, alpha=0.18)
    ax.axhline(capital, color=MUTED, lw=0.8, ls=":")
    ax.text(0.3, capital, f"start: {_money(capital)}", color=TEXT, fontsize=11,
            va="bottom", **SERIF)
    ax.set_xlim(0, result.days)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(f"{result.days} trading days", color=MUTED, **MONO)
    ax.set_yticks([])
    lo, hi = np.quantile(final, [0.002, 0.998])
    pad = 0.05 * (hi - lo)
    ax.set_ylim(min(lo, paths[idx].min()) - pad, max(hi, paths[idx].max()) + pad)

    bins = np.linspace(*ax.get_ylim(), 50)
    counts, edges = np.histogram(final, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    side.barh(centers, counts, height=np.diff(edges) * 0.9,
              color=np.where(centers > capital, GREEN, RED))
    side.axis("off")
    side.text(0.98, 0.98, f"ended up\n{up.sum():,}", color=GREEN, transform=side.transAxes,
              va="top", ha="right", fontsize=10, **MONO)
    side.text(0.98, 0.02, f"ended down\n{(~up).sum():,}", color=RED, transform=side.transAxes,
              va="bottom", ha="right", fontsize=10, **MONO)

    fig.suptitle(f"then simulate {result.n_sims:,} futures", color=TEXT, fontsize=16, **SERIF)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
# 3. Profit and loss distribution with VaR and expected shortfall
# ---------------------------------------------------------------------- #
def plot_distribution(result: SimulationResult, summary: RiskSummary, path: Path, bins: int = 70) -> Path:
    pnl = result.pnl
    var = summary.value_at_risk
    es = summary.expected_shortfall

    fig = _figure(10, 6)
    ax = fig.add_axes([0.06, 0.32, 0.9, 0.58])
    _style(ax)
    lo, hi = np.quantile(pnl, [0.001, 0.999])
    counts, edges = np.histogram(pnl, bins=np.linspace(lo, hi, bins))
    centers = (edges[:-1] + edges[1:]) / 2
    colors = np.where(centers <= -var, DARK_RED, np.where(centers < 0, RED, GREEN))
    ax.bar(centers, counts, width=np.diff(edges) * 0.92, color=colors)
    ax.axvline(-var, color=RED, ls="--", lw=1.2)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda x, _: _money(x))
    ax.text(0.01, 0.02, "← lose", color=RED, transform=ax.transAxes, **MONO)
    ax.text(0.99, 0.02, "gain →", color=GREEN, transform=ax.transAxes, ha="right", **MONO)
    ax.text(0.99, 0.95, "each bar: how many\nfutures ended there", color=TEXT,
            transform=ax.transAxes, ha="right", va="top", fontsize=10, **SERIF)
    ax.set_title(f"{summary.days} trading days, {summary.n_sims:,} futures, start {_money(summary.capital)}",
                 color=MUTED, fontsize=10, **MONO)

    for x0, title, body in (
        (0.06, "value at risk",
         f"1 month in {summary.one_in}: lose more than {_money(var)}"),
        (0.53, "expected shortfall",
         f"those worst months: {_money(es)} on average"),
    ):
        box = fig.add_axes([x0, 0.05, 0.43, 0.17])
        box.set_facecolor(PANEL)
        box.set_xticks([])
        box.set_yticks([])
        for spine in box.spines.values():
            spine.set_color(MUTED)
        box.text(0.04, 0.72, title, color=MUTED, fontsize=10, transform=box.transAxes, **MONO)
        box.text(0.04, 0.25, body, color=TEXT, fontsize=12, transform=box.transAxes, **SERIF)

    fig.suptitle("what a bad month costs", color=TEXT, fontsize=16, y=0.98, **SERIF)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
# 4. Moving together vs each on its own
# ---------------------------------------------------------------------- #
def plot_correlation_effect(
    together: SimulationResult,
    alone: SimulationResult,
    s_together: RiskSummary,
    s_alone: RiskSummary,
    path: Path,
) -> Path:
    fig = _figure(10, 6.4)
    ax = fig.add_axes([0.06, 0.36, 0.9, 0.52])
    _style(ax)
    both = np.concatenate([together.pnl, alone.pnl])
    lo, hi = np.quantile(both, [0.001, 0.999])
    x = np.linspace(lo, hi, 400)
    d_t = _density(together.pnl, x)
    d_a = _density(alone.pnl, x)
    ax.fill_between(x, d_t, color=PANEL)
    ax.plot(x, d_t, color=TEXT, lw=1.6, label="moving together, as they do")
    ax.plot(x, d_a, color=TEXT, lw=1.3, ls="--", label="if each moved on its own")
    ax.set_ylim(0, max(d_t.max(), d_a.max()) * 1.25)
    for loss, dens, color in ((s_together.value_at_risk, d_t, RED), (s_alone.value_at_risk, d_a, MUTED)):
        ax.vlines(-loss, 0, np.interp(-loss, x, dens), color=color, ls="--", lw=1.2)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(lambda x, _: _money(x))
    legend = ax.legend(loc="upper left", frameon=False, prop=SERIF | {"size": 11})
    for text in legend.get_texts():
        text.set_color(TEXT)

    ratio = s_together.value_at_risk / s_alone.value_at_risk
    bar = fig.add_axes([0.06, 0.06, 0.9, 0.2])
    bar.set_facecolor(BG)
    bar.axis("off")
    bar.set_xlim(0, 1)
    bar.set_ylim(0, 1)
    scale = 0.45 / max(s_together.value_at_risk, s_alone.value_at_risk)
    bar.text(0, 0.88, f"bad-month loss (value at risk, {s_together.confidence:.0%})",
             color=TEXT, fontsize=12, **SERIF)
    bar.add_patch(plt.Rectangle((0, 0.48), s_together.value_at_risk * scale, 0.18, color=TEXT))
    bar.text(s_together.value_at_risk * scale + 0.01, 0.57,
             f"{_money(s_together.value_at_risk)} · moving together", color=TEXT, va="center",
             fontsize=11, **SERIF)
    bar.add_patch(plt.Rectangle((0, 0.12), s_alone.value_at_risk * scale, 0.18, fill=False,
                                ec=TEXT, ls="--"))
    bar.text(s_alone.value_at_risk * scale + 0.01, 0.21,
             f"{_money(s_alone.value_at_risk)} · each on its own", color=TEXT, va="center",
             fontsize=11, **SERIF)
    bar.text(0.86, 0.38, f"×{ratio:.1f}", color=RED, fontsize=30, va="center", ha="right", **SERIF)
    bar.text(0.875, 0.38, "bigger\nbad-month loss", color=TEXT, fontsize=10, va="center", **SERIF)

    fig.suptitle("correlation, not the individual stocks, drives portfolio risk",
                 color=TEXT, fontsize=15, y=0.97, **SERIF)
    return _save(fig, path)


def annual_label(prices: pd.DataFrame | None) -> str:
    if prices is None or prices.empty:
        return "demo inputs (from the video)"
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    return f"last {years:.1f} years · {prices.index[0]:%Y-%m-%d} → {prices.index[-1]:%Y-%m-%d}"


__all__ = [
    "TRADING_DAYS",
    "annual_label",
    "plot_correlation_effect",
    "plot_distribution",
    "plot_fan",
    "plot_inputs",
]
