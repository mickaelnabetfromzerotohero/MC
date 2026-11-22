import numpy as np
import pandas as pd
import pytest

from mcrisk import TRADING_DAYS, MarketParams
from mcrisk.data import demo_params, synthetic_prices


def test_from_annual_round_trip():
    p = MarketParams.from_annual(["A", "B"], [0.40, 0.025], [0.52, 0.59], 0.3)
    np.testing.assert_allclose(p.annual_growth, [0.40, 0.025])
    np.testing.assert_allclose(p.annual_swing, [0.52, 0.59])
    np.testing.assert_allclose(p.corr, [[1, 0.3], [0.3, 1]])


def test_from_prices_recovers_the_model():
    truth = MarketParams.from_annual(["X", "Y", "Z"], [0.15, 0.05, 0.30], [0.20, 0.30, 0.45],
                                     np.array([[1, 0.6, 0.2], [0.6, 1, -0.3], [0.2, -0.3, 1]]))
    prices = synthetic_prices(truth, days=40 * TRADING_DAYS, seed=1)
    est = MarketParams.from_prices(prices)
    assert est.tickers == ("X", "Y", "Z")
    np.testing.assert_allclose(est.annual_swing, truth.annual_swing, rtol=0.03)
    np.testing.assert_allclose(est.corr, truth.corr, atol=0.03)
    # Drift is noisy by nature: its standard error is swing / sqrt(years).
    years = 40
    drift_error = np.abs(est.mu - truth.mu) * TRADING_DAYS
    assert np.all(drift_error < 3 * truth.annual_swing / np.sqrt(years))


def test_from_prices_ignores_unsorted_index_and_gaps():
    prices = synthetic_prices(demo_params(), days=300, seed=2)
    shuffled = prices.sample(frac=1, random_state=0)
    shuffled.iloc[5, 0] = np.nan
    a = MarketParams.from_prices(shuffled)
    assert a.n_assets == 3
    assert np.all(np.isfinite(a.cov))


def test_independent_keeps_variances_and_drops_correlations():
    p = demo_params()
    ind = p.independent()
    np.testing.assert_allclose(np.diag(ind.cov), np.diag(p.cov))
    np.testing.assert_allclose(ind.corr, np.eye(3))
    np.testing.assert_allclose(ind.mu, p.mu)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tickers=("A", "A"), mu=[0, 0], cov=np.eye(2)),
        dict(tickers=("A",), mu=[0, 0], cov=np.eye(1)),
        dict(tickers=("A", "B"), mu=[0, 0], cov=[[1, 2], [2, 1]]),  # not PSD
        dict(tickers=("A", "B"), mu=[0, 0], cov=[[1, 0.1], [0.2, 1]]),  # not symmetric
        dict(tickers=("A", "B"), mu=[0, 0], cov=[[0, 0], [0, 1]]),  # zero variance
    ],
)
def test_invalid_models_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MarketParams(**kwargs)


def test_short_or_non_positive_prices_are_rejected():
    idx = pd.bdate_range("2024-01-01", periods=10)
    with pytest.raises(ValueError):
        MarketParams.from_prices(pd.DataFrame({"A": np.linspace(10, 11, 10)}, index=idx))
    idx = pd.bdate_range("2024-01-01", periods=50)
    bad = pd.DataFrame({"A": np.linspace(-1, 10, 50)}, index=idx)
    with pytest.raises(ValueError):
        MarketParams.from_prices(bad)
