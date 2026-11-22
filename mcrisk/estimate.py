"""Estimate growth, swing and correlations from daily prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class MarketParams:
    """Daily log-return model for a set of assets.

    Log returns are assumed jointly normal and independent from one day to the
    next: ``r_t ~ N(mu, cov)``. Everything is stored per trading day; the
    ``annual_*`` properties translate it into the "growth a year" and
    "swing a year" figures shown to a reader.
    """

    tickers: tuple[str, ...]
    mu: np.ndarray  # mean daily log return, shape (k,)
    cov: np.ndarray  # covariance of daily log returns, shape (k, k)

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=float).reshape(-1)
        cov = np.asarray(self.cov, dtype=float)
        k = len(self.tickers)
        if k == 0:
            raise ValueError("At least one ticker is required.")
        if len(set(self.tickers)) != k:
            raise ValueError(f"Duplicate tickers: {self.tickers}")
        if mu.shape != (k,):
            raise ValueError(f"mu must have {k} values, got shape {mu.shape}.")
        if cov.shape != (k, k):
            raise ValueError(f"cov must be {k}x{k}, got shape {cov.shape}.")
        if not np.allclose(cov, cov.T, atol=1e-12):
            raise ValueError("cov must be symmetric.")
        if np.any(np.diag(cov) <= 0):
            raise ValueError("Every asset needs a positive variance.")
        if np.linalg.eigvalsh(cov).min() < -1e-10 * np.abs(cov).max():
            raise ValueError("cov is not positive semi-definite.")
        object.__setattr__(self, "tickers", tuple(self.tickers))
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "cov", cov)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_prices(cls, prices: pd.DataFrame) -> "MarketParams":
        """Estimate the model from a table of daily prices (one column per ticker)."""
        if prices.shape[1] == 0:
            raise ValueError("The price table has no columns.")
        clean = prices.sort_index().dropna(how="any")
        if len(clean) < 30:
            raise ValueError(f"Need at least 30 days of prices, got {len(clean)}.")
        if (clean <= 0).to_numpy().any():
            raise ValueError("Prices must be strictly positive.")
        log_returns = np.log(clean).diff().dropna(how="any")
        return cls(
            tickers=tuple(str(c) for c in clean.columns),
            mu=log_returns.mean().to_numpy(),
            cov=log_returns.cov().to_numpy(),
        )

    @classmethod
    def from_annual(
        cls,
        tickers: Sequence[str],
        growth: Sequence[float],
        swing: Sequence[float],
        corr: np.ndarray | float,
    ) -> "MarketParams":
        """Build the model from yearly figures.

        growth: yearly growth, e.g. 0.40 for +40% a year.
        swing: yearly volatility, e.g. 0.52 for ±52% a year.
        corr: correlation matrix, or a single number used for every pair.
        """
        g = np.asarray(growth, dtype=float)
        s = np.asarray(swing, dtype=float)
        k = len(tickers)
        if g.shape != (k,) or s.shape != (k,):
            raise ValueError("growth and swing need one value per ticker.")
        if np.any(g <= -1):
            raise ValueError("growth must be above -100%.")
        if np.isscalar(corr):
            c = np.full((k, k), float(corr))
            np.fill_diagonal(c, 1.0)
        else:
            c = np.asarray(corr, dtype=float)
        if c.shape != (k, k) or not np.allclose(np.diag(c), 1.0):
            raise ValueError("corr must be a k x k matrix with ones on the diagonal.")
        daily_sigma = s / np.sqrt(TRADING_DAYS)
        return cls(
            tickers=tuple(tickers),
            mu=np.log1p(g) / TRADING_DAYS,
            cov=np.outer(daily_sigma, daily_sigma) * c,
        )

    # ------------------------------------------------------------------ #
    # Derived figures
    # ------------------------------------------------------------------ #
    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def sigma(self) -> np.ndarray:
        """Daily volatility of log returns."""
        return np.sqrt(np.diag(self.cov))

    @property
    def corr(self) -> np.ndarray:
        s = self.sigma
        c = self.cov / np.outer(s, s)
        np.fill_diagonal(c, 1.0)
        return c

    @property
    def annual_growth(self) -> np.ndarray:
        """Typical yearly growth: exp(252 * mean daily log return) - 1."""
        return np.expm1(self.mu * TRADING_DAYS)

    @property
    def annual_swing(self) -> np.ndarray:
        """Yearly volatility: daily volatility * sqrt(252)."""
        return self.sigma * np.sqrt(TRADING_DAYS)

    def independent(self) -> "MarketParams":
        """Same assets, same growth and swing, but every correlation set to zero."""
        return MarketParams(self.tickers, self.mu, np.diag(np.diag(self.cov)))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"growth_a_year": self.annual_growth, "swing_a_year": self.annual_swing},
            index=list(self.tickers),
        )

    def corr_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.corr, index=list(self.tickers), columns=list(self.tickers))
