import numpy as np
import pytest

from mcrisk import MarketParams, simulate, simulate_pair, summarize, value_at_risk
from mcrisk.data import demo_params
from mcrisk.simulate import normalize_weights, shock_factor

Z_05 = -1.6448536269514722  # 5% quantile of the standard normal


def test_shapes_start_value_and_reproducibility():
    p = demo_params()
    a = simulate(p, capital=10_000, days=21, n_sims=500, seed=7)
    b = simulate(p, capital=10_000, days=21, n_sims=500, seed=7)
    assert a.paths.shape == (500, 22)
    assert np.all(a.paths[:, 0] == 10_000)
    assert np.all(a.paths > 0)
    np.testing.assert_array_equal(a.paths, b.paths)
    assert a.pnl.shape == (500,)


def test_single_asset_matches_the_lognormal_formula():
    p = MarketParams.from_annual(["A"], [0.10], [0.30], 1.0)
    days = 21
    res = simulate(p, capital=10_000, days=days, n_sims=200_000, seed=0)
    m, s = p.mu[0] * days, p.sigma[0] * np.sqrt(days)
    exact_var = 10_000 * (1 - np.exp(m + s * Z_05))
    assert value_at_risk(res.pnl) == pytest.approx(exact_var, rel=0.01)
    assert res.final.mean() == pytest.approx(10_000 * np.exp(m + s**2 / 2), rel=0.002)


def test_zero_correlation_gives_identical_scenarios():
    p = MarketParams.from_annual(["A", "B"], [0.1, 0.2], [0.3, 0.4], 0.0)
    together, alone = simulate_pair(p, n_sims=1_000, seed=3)
    np.testing.assert_allclose(together.paths, alone.paths)


def test_positive_correlation_raises_portfolio_risk():
    p = demo_params()
    together, alone = simulate_pair(p, n_sims=20_000, seed=11)
    s_t, s_a = summarize(together), summarize(alone)
    assert s_t.value_at_risk > s_a.value_at_risk
    assert s_t.expected_shortfall > s_a.expected_shortfall
    # Same growth and swing per stock: the average outcome barely moves.
    assert abs(s_t.mean_pnl - s_a.mean_pnl) < 0.1 * s_a.value_at_risk


def test_negative_correlation_lowers_portfolio_risk():
    p = MarketParams.from_annual(["A", "B"], [0.1, 0.1], [0.3, 0.3], -0.6)
    together, alone = simulate_pair(p, n_sims=20_000, seed=5)
    assert summarize(together).value_at_risk < summarize(alone).value_at_risk


def test_demo_reproduces_the_video_figures():
    """Video: VaR $1,435, ES $1,776, each on its own $1,021, x1.4. Demo correlations are assumed."""
    together, alone = simulate_pair(demo_params(), capital=10_000, days=21, n_sims=10_000, seed=42)
    s_t, s_a = summarize(together), summarize(alone)
    assert s_t.value_at_risk == pytest.approx(1_435, rel=0.05)
    assert s_t.expected_shortfall == pytest.approx(1_776, rel=0.05)
    assert s_a.value_at_risk == pytest.approx(1_021, rel=0.05)
    assert round(s_t.value_at_risk / s_a.value_at_risk, 1) == 1.4


def test_weights_are_normalized_and_validated():
    np.testing.assert_allclose(normalize_weights(None, 4), [0.25] * 4)
    np.testing.assert_allclose(normalize_weights([2, 1, 1], 3), [0.5, 0.25, 0.25])
    for bad in ([1, 1], [1, -1, 1], [0, 0, 0]):
        with pytest.raises(ValueError):
            normalize_weights(bad, 3)


def test_all_weight_on_one_stock_ignores_the_others():
    p = demo_params()
    full = simulate(p, weights=[1, 0, 0], n_sims=2_000, seed=1)
    only = simulate(MarketParams(("NVDA",), p.mu[:1], p.cov[:1, :1]), n_sims=2_000, seed=1)
    # Cholesky's first row only uses the first shock, so the NVDA paths are identical.
    shocks_first_asset_only = np.allclose(shock_factor(p.cov)[0, 1:], 0)
    assert shocks_first_asset_only
    assert value_at_risk(full.pnl) == pytest.approx(value_at_risk(only.pnl), rel=0.05)


def test_shock_factor_handles_singular_covariance():
    cov = np.array([[0.04, 0.04], [0.04, 0.04]])  # two perfectly correlated assets
    f = shock_factor(cov)
    np.testing.assert_allclose(f @ f.T, cov, atol=1e-12)
