"""Track C multi-market trend — same rule per market, risk-parity weights, no-lookahead."""

import numpy as np
import pandas as pd

from tagent.multi_market_trend import (
    CRYPTO_ROLL, INDEX_ROLL, combined_return, market_sleeve, risk_parity_weights,
)
from tagent.trend_core import binary_trend_net


# --------------------------------------------------------------------------- #
# 1) same locked rule per market; only the cost/roll differs by kind
# --------------------------------------------------------------------------- #
def test_market_sleeve_same_rule_different_cost():
    idx = pd.date_range("2010-01-01", periods=400, freq="B")
    close = pd.Series(100 * np.cumprod(1 + np.full(400, 0.001)), index=idx)   # steady uptrend, always in
    s_index = market_sleeve(close, "index")
    s_crypto = market_sleeve(close, "crypto")
    # identical rule -> identical exposure; crypto just pays more roll/funding each in-market bar
    diff = (s_index - s_crypto).dropna()
    assert (diff > 0).mean() > 0.95                                          # crypto drags below index
    assert abs(diff.mean() - (CRYPTO_ROLL - INDEX_ROLL) / 252) < 5e-4        # ~ the funding gap/day
    # and it IS the locked binary rule (matches binary_trend_net with index costs)
    assert np.allclose(s_index.to_numpy(),
                       binary_trend_net(close, regime_ma=200, mode="next_bar",
                                        roll_annual=INDEX_ROLL, switch_cost=0.00025).to_numpy())


# --------------------------------------------------------------------------- #
# 2) risk-parity weights: inverse-vol, capped, sum to 1, no-lookahead
# --------------------------------------------------------------------------- #
def _two_sleeves(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    lo = pd.Series(rng.normal(0.0003, 0.005, n), index=idx)                  # low vol
    hi = pd.Series(rng.normal(0.0003, 0.030, n), index=idx)                  # high vol
    return pd.DataFrame({"LO": lo, "HI": hi})


def test_risk_parity_inverse_vol_capped_and_sums_to_one():
    s = _two_sleeves()
    w = risk_parity_weights(s, vol_window=30, rebalance="ME", cap=0.60).dropna()
    assert np.allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-9)             # fully invested
    assert (w["LO"] >= w["HI"] - 1e-9).all()                                 # low-vol gets more weight
    assert (w.to_numpy() <= 0.60 + 1e-9).all()                              # cap holds


def test_risk_parity_weights_no_lookahead():
    s = _two_sleeves()
    w = risk_parity_weights(s, vol_window=30, rebalance="ME", cap=0.60)
    ext_idx = pd.date_range(s.index[-1] + pd.offsets.BDay(1), periods=40, freq="B")
    rng = np.random.default_rng(9)
    s2 = pd.concat([s, pd.DataFrame({"LO": rng.normal(0, 0.05, 40),
                                     "HI": rng.normal(0, 0.05, 40)}, index=ext_idx)])
    w2 = risk_parity_weights(s2, vol_window=30, rebalance="ME", cap=0.60)
    common = w.dropna().index
    assert np.allclose(w.loc[common].to_numpy(), w2.loc[common].to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# 3) combined-portfolio accounting
# --------------------------------------------------------------------------- #
def test_combined_return_is_weighted_sum_over_common_sample():
    s = _two_sleeves()
    comb = combined_return(s, ["LO", "HI"], vol_window=30, rebalance="ME", cap=0.60)
    w = risk_parity_weights(s, vol_window=30, rebalance="ME", cap=0.60)
    expect = (w * s).sum(axis=1).reindex(comb.index)
    assert np.allclose(comb.to_numpy(), expect.dropna().to_numpy())
    # common-sample restriction: a market present only on a subset shrinks the sample
    s3 = s.copy()
    s3.loc[s3.index[:50], "HI"] = np.nan
    comb3 = combined_return(s3, ["LO", "HI"], vol_window=30)
    assert len(comb3) < len(s3)
