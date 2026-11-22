"""Risk measures read off simulated profit and loss."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .simulate import SimulationResult


def _check(pnl: np.ndarray, confidence: float) -> np.ndarray:
    if not 0.5 <= confidence < 1:
        raise ValueError("confidence must be in [0.5, 1).")
    arr = np.asarray(pnl, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("pnl is empty.")
    return arr


def value_at_risk(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Loss exceeded in only (1 - confidence) of futures, as a positive dollar amount.

    At 95%: one month in 20 loses more than this.
    """
    arr = _check(pnl, confidence)
    return float(-np.quantile(arr, 1 - confidence))


def expected_shortfall(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Average loss over the worst (1 - confidence) of futures, as a positive dollar amount."""
    arr = _check(pnl, confidence)
    cutoff = np.quantile(arr, 1 - confidence)
    return float(-arr[arr <= cutoff].mean())


@dataclass(frozen=True)
class RiskSummary:
    confidence: float
    n_sims: int
    days: int
    capital: float
    n_up: int
    n_down: int
    mean_pnl: float
    median_pnl: float
    value_at_risk: float
    expected_shortfall: float
    best: float
    worst: float

    @property
    def prob_loss(self) -> float:
        return self.n_down / self.n_sims

    @property
    def one_in(self) -> int:
        """95% -> 20, as in "1 month in 20"."""
        return round(1 / (1 - self.confidence))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["prob_loss"] = self.prob_loss
        return d


def summarize(result: SimulationResult, confidence: float = 0.95) -> RiskSummary:
    pnl = result.pnl
    return RiskSummary(
        confidence=confidence,
        n_sims=result.n_sims,
        days=result.days,
        capital=result.capital,
        n_up=int((pnl > 0).sum()),
        n_down=int((pnl <= 0).sum()),
        mean_pnl=float(pnl.mean()),
        median_pnl=float(np.median(pnl)),
        value_at_risk=value_at_risk(pnl, confidence),
        expected_shortfall=expected_shortfall(pnl, confidence),
        best=float(pnl.max()),
        worst=float(pnl.min()),
    )
