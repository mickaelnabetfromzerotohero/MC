import numpy as np
import pytest

from mcrisk import expected_shortfall, simulate, summarize, value_at_risk
from mcrisk.data import demo_params


def test_var_and_es_on_a_known_sample():
    pnl = np.arange(-100, 100, dtype=float)  # 200 equally likely outcomes
    # 5% quantile of -100..99 (linear interpolation) is -90.05
    assert value_at_risk(pnl, 0.95) == pytest.approx(90.05)
    # The 10 worst outcomes are -100..-91, average -95.5
    assert expected_shortfall(pnl, 0.95) == pytest.approx(95.5)


def test_es_is_never_below_var():
    pnl = np.random.default_rng(0).standard_t(df=3, size=50_000)
    for c in (0.9, 0.95, 0.99):
        assert expected_shortfall(pnl, c) >= value_at_risk(pnl, c)


def test_higher_confidence_means_bigger_var():
    pnl = np.random.default_rng(1).normal(size=50_000)
    assert value_at_risk(pnl, 0.99) > value_at_risk(pnl, 0.95) > value_at_risk(pnl, 0.9)


@pytest.mark.parametrize("c", [0.2, 1.0, 1.5])
def test_confidence_bounds(c):
    with pytest.raises(ValueError):
        value_at_risk(np.ones(10), c)


def test_empty_pnl_is_rejected():
    with pytest.raises(ValueError):
        expected_shortfall(np.array([]))


def test_summary_counts_add_up():
    res = simulate(demo_params(), n_sims=3_000, seed=4)
    s = summarize(res)
    assert s.n_up + s.n_down == 3_000
    assert s.one_in == 20
    assert s.worst <= -s.value_at_risk <= s.median_pnl <= s.best
    assert 0 < s.prob_loss < 1
    assert set(s.to_dict()) >= {"value_at_risk", "expected_shortfall", "prob_loss"}
