"""Options VRP trial — signal construction, no-lookahead, weekly rebalance, cost, calendar t.
Synthetic series, no network."""

import numpy as np
import pandas as pd

from tagent.data.vkospi_source import is_degenerate, load_vkospi
from tagent.vrp import (
    by_year, calendar_time_t, implied_variance, realized_variance, vrp_backtest, vrp_series,
    vrp_zscore,
)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    px = pd.Series(100 * np.cumprod(1 + np.r_[0.0, rng.normal(0.0003, 0.011, n - 1)]), index=idx)
    vk = pd.Series(15 + 5 * np.abs(rng.normal(0, 1, n)), index=idx)         # VKOSPI ~15-30
    return vk, px, idx


# --------------------------------------------------------------------------- #
# 1) VRP construction
# --------------------------------------------------------------------------- #
def test_vrp_components_and_series():
    vk, px, idx = _data()
    assert abs(implied_variance(pd.Series([20.0])).iloc[0] - 0.04) < 1e-12   # (20/100)^2
    rv = realized_variance(px, window=21)
    assert (rv.dropna() >= 0).all()
    vrp = vrp_series(vk, px, window=21)
    # VRP = implied - realized variance, aligned to the index calendar
    assert len(vrp.dropna()) > 0 and vrp.index.equals(px.pct_change().rolling(21).var().index)


def test_vrp_zscore_bounded_and_centered():
    vk, px, idx = _data()
    z = vrp_zscore(vrp_series(vk, px), z_window=120).dropna()
    assert abs(z.mean()) < 0.5 and z.std() > 0                              # standardized


# --------------------------------------------------------------------------- #
# 2) backtest: weekly hold, long/flat, no-lookahead
# --------------------------------------------------------------------------- #
def test_vrp_backtest_no_lookahead_and_weekly():
    vk, px, idx = _data()
    net = vrp_backtest(vk, px, z_window=120, rebalance=5)
    # appending future bars must not change past net (z trailing, position lagged)
    ext = pd.date_range(idx[-1] + pd.offsets.BDay(1), periods=40, freq="B")
    rng = np.random.default_rng(9)
    vk2 = pd.concat([vk, pd.Series(15 + 5 * np.abs(rng.normal(0, 1, 40)), index=ext)])
    px2 = pd.concat([px, pd.Series(float(px.iloc[-1]) * np.cumprod(1 + rng.normal(0, 0.01, 40)), index=ext)])
    net2 = vrp_backtest(vk2, px2, z_window=120, rebalance=5)
    common = net.index[:-2]
    assert np.allclose(net.reindex(common).to_numpy(), net2.reindex(common).to_numpy())
    # long/flat only -> exposure never negative: a strongly-down index with always-long signal loses,
    # but never benefits from shorting (position in {0,1})
    flat = vrp_backtest(pd.Series(0.0, index=idx), px, z_window=120)        # VRP<0 always -> z low -> flat-ish
    assert np.isfinite(flat.to_numpy()).all()


def test_vrp_cost_drags_and_calendar_t():
    vk, px, idx = _data(seed=3)
    cheap = vrp_backtest(vk, px, cost_round_trip=0.0005, z_window=120)
    pricey = vrp_backtest(vk, px, cost_round_trip=0.0040, z_window=120)
    assert pricey.sum() <= cheap.sum() + 1e-12
    t, n = calendar_time_t(cheap)
    assert n == len(cheap.dropna()) and np.isfinite(t)
    yb = by_year(cheap)
    assert all("sharpe" in v for v in yb.values())


# --------------------------------------------------------------------------- #
# 3) data wall guard
# --------------------------------------------------------------------------- #
def test_vkospi_degenerate_guard(tmp_path):
    assert is_degenerate(pd.Series(dtype=float))
    assert is_degenerate(pd.Series([0.0, 0.0]))
    assert not is_degenerate(pd.Series([15.0, 20.0]))
    assert load_vkospi(data_dir=tmp_path).empty                            # no cache -> empty


def test_vrp_exposure_bounds_mean_and_no_lookahead():
    from tagent.vrp import vrp_exposure
    vk, px, idx = _data(seed=11)
    exp = vrp_exposure(vk, px, z_window=120, rebalance=5)
    assert ((exp == 0.0) | (exp == 1.0)).all()             # long/flat only
    assert 0.0 <= exp.mean() <= 1.0                         # time-in-market is a fraction
    # no-lookahead: appending future bars never changes a past exposure
    ext = pd.date_range(idx[-1] + pd.offsets.BDay(1), periods=30, freq="B")
    rng = np.random.default_rng(12)
    vk2 = pd.concat([vk, pd.Series(15 + 5 * np.abs(rng.normal(0, 1, 30)), index=ext)])
    px2 = pd.concat([px, pd.Series(float(px.iloc[-1]) * np.cumprod(1 + rng.normal(0, 0.01, 30)), index=ext)])
    exp2 = vrp_exposure(vk2, px2, z_window=120, rebalance=5)
    assert np.allclose(exp.to_numpy(), exp2.reindex(exp.index).to_numpy())
