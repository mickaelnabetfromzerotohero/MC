"""Simulate portfolio futures with correlated daily shocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .estimate import MarketParams


@dataclass(frozen=True)
class SimulationResult:
    """Simulated portfolio values, one row per future, one column per day (day 0 = start)."""

    paths: np.ndarray  # shape (n_sims, days + 1)
    capital: float
    weights: np.ndarray
    tickers: tuple[str, ...]
    correlated: bool

    @property
    def n_sims(self) -> int:
        return self.paths.shape[0]

    @property
    def days(self) -> int:
        return self.paths.shape[1] - 1

    @property
    def final(self) -> np.ndarray:
        return self.paths[:, -1]

    @property
    def pnl(self) -> np.ndarray:
        """Profit (positive) or loss (negative) in dollars at the end of the horizon."""
        return self.final - self.capital


def normalize_weights(weights: Sequence[float] | None, n_assets: int) -> np.ndarray:
    """Equal weights when none are given; otherwise check and rescale to sum to 1."""
    if weights is None:
        return np.full(n_assets, 1.0 / n_assets)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape != (n_assets,):
        raise ValueError(f"Expected {n_assets} weights, got {w.size}.")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative (long-only portfolio).")
    total = w.sum()
    if total <= 0:
        raise ValueError("Weights must not all be zero.")
    return w / total


def shock_factor(cov: np.ndarray) -> np.ndarray:
    """Matrix L with L @ L.T == cov, used to turn independent normals into correlated ones.

    Cholesky when cov is positive definite; an eigen-decomposition otherwise
    (for example two perfectly correlated assets).
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        return vecs * np.sqrt(np.clip(vals, 0.0, None))


def draw_shocks(n_sims: int, days: int, n_assets: int, seed: int | None = None) -> np.ndarray:
    """Independent standard normal draws, shape (n_sims, days, n_assets)."""
    if n_sims < 1 or days < 1:
        raise ValueError("n_sims and days must be at least 1.")
    return np.random.default_rng(seed).standard_normal((n_sims, days, n_assets))


def simulate(
    params: MarketParams,
    weights: Sequence[float] | None = None,
    capital: float = 10_000.0,
    days: int = 21,
    n_sims: int = 10_000,
    seed: int | None = None,
    independent: bool = False,
    shocks: np.ndarray | None = None,
) -> SimulationResult:
    """Simulate ``n_sims`` futures of a buy-and-hold portfolio over ``days`` trading days.

    Each day every stock gets a log return ``mu + L @ z`` where ``z`` is a vector
    of independent standard normals and ``L @ L.T`` is the covariance. With
    ``independent=True`` the covariance is replaced by its diagonal: each stock
    keeps its own growth and swing but moves on its own.

    Pass the same ``shocks`` to two calls to compare scenarios on identical
    random draws (common random numbers), which removes simulation noise from
    the comparison.
    """
    if capital <= 0:
        raise ValueError("capital must be positive.")
    k = params.n_assets
    w = normalize_weights(weights, k)
    if shocks is None:
        shocks = draw_shocks(n_sims, days, k, seed)
    if shocks.ndim != 3 or shocks.shape[2] != k:
        raise ValueError(f"shocks must have shape (n_sims, days, {k}).")

    model = params.independent() if independent else params
    factor = shock_factor(model.cov)
    log_returns = model.mu + shocks @ factor.T  # (n_sims, days, k)
    growth = np.exp(np.cumsum(log_returns, axis=1))  # value of $1 in each stock
    values = growth @ (capital * w)  # (n_sims, days)

    start = np.full((values.shape[0], 1), float(capital))
    return SimulationResult(
        paths=np.hstack([start, values]),
        capital=float(capital),
        weights=w,
        tickers=params.tickers,
        correlated=not independent,
    )


def simulate_pair(
    params: MarketParams,
    weights: Sequence[float] | None = None,
    capital: float = 10_000.0,
    days: int = 21,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> tuple[SimulationResult, SimulationResult]:
    """Run the portfolio twice on the same random draws: as the stocks really move
    together, and as if each moved on its own."""
    shocks = draw_shocks(n_sims, days, params.n_assets, seed)
    together = simulate(params, weights, capital, shocks=shocks)
    alone = simulate(params, weights, capital, shocks=shocks, independent=True)
    return together, alone
