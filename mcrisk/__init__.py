"""Monte Carlo Portfolio Risk.

Estimate each stock's growth, swing (volatility) and the correlations between
stocks from price history, simulate thousands of one-month futures for the
portfolio, and read the risk off the simulated outcomes.
"""

from .estimate import TRADING_DAYS, MarketParams
from .risk import RiskSummary, expected_shortfall, summarize, value_at_risk
from .simulate import SimulationResult, simulate, simulate_pair

__all__ = [
    "TRADING_DAYS",
    "MarketParams",
    "RiskSummary",
    "SimulationResult",
    "expected_shortfall",
    "simulate",
    "simulate_pair",
    "summarize",
    "value_at_risk",
]

__version__ = "0.1.0"
