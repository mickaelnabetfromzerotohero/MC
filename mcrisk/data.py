"""Price data: download from Yahoo Finance, read a CSV, or use the demo inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .estimate import MarketParams

# Demo inputs: growth and swing as printed in the "Monte Carlo Portfolio Risk"
# video (Nvidia, Apple, Tesla, last 5 years). The video does not show the
# correlations, so DEMO_CORRELATION is an assumption: one value for every pair,
# chosen so that the demo lands close to the video's figures (VaR about $1,435
# moving together vs about $1,021 each on its own).
DEMO_TICKERS = ("NVDA", "AAPL", "TSLA")
DEMO_NAMES = {"NVDA": "Nvidia", "AAPL": "Apple", "TSLA": "Tesla"}
DEMO_GROWTH = (0.40, 0.20, 0.025)
DEMO_SWING = (0.52, 0.28, 0.59)
DEMO_CORRELATION = 0.40


def demo_params() -> MarketParams:
    return MarketParams.from_annual(DEMO_TICKERS, DEMO_GROWTH, DEMO_SWING, DEMO_CORRELATION)


def _validate(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if prices.shape[1] == 0:
        raise ValueError("No price columns found.")
    if len(prices) < 30:
        raise ValueError(f"Need at least 30 rows of prices, got {len(prices)}.")
    if (prices <= 0).to_numpy().any():
        raise ValueError("Prices must be strictly positive.")
    return prices


def load_prices_csv(path: str | Path, tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """Read daily prices: first column is the date, then one column per ticker."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.columns = [str(c).strip() for c in prices.columns]
    if tickers:
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"Tickers not in {path}: {missing}")
        prices = prices[list(tickers)]
    return _validate(prices)


def download_prices(
    tickers: Sequence[str], years: float = 5, end: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    """Daily adjusted close prices from Yahoo Finance for the last ``years`` years."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("Downloading prices needs yfinance: pip install yfinance") from exc

    end_ts = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    start_ts = end_ts - pd.DateOffset(months=int(round(years * 12)))
    raw = yf.download(
        list(tickers),
        start=start_ts.strftime("%Y-%m-%d"),
        end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"No prices downloaded for {list(tickers)}. Check the tickers and your internet "
            "connection, or pass a CSV with --csv."
        )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.columns = [str(c) for c in close.columns]
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        raise RuntimeError(f"No prices for: {missing}")
    return _validate(close[list(tickers)])


def synthetic_prices(
    params: MarketParams, days: int = 5 * 252, seed: int | None = 0, start: str = "2021-01-04"
) -> pd.DataFrame:
    """Price history drawn from ``params``; used by the tests."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((days, params.n_assets))
    from .simulate import shock_factor

    log_r = params.mu + z @ shock_factor(params.cov).T
    prices = 100 * np.exp(np.vstack([np.zeros(params.n_assets), np.cumsum(log_r, axis=0)]))
    index = pd.bdate_range(start, periods=days + 1)
    return pd.DataFrame(prices, index=index, columns=list(params.tickers))
