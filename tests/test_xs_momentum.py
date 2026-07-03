"""Cross-sectional crypto momentum — synthetic panels, no network."""

import numpy as np
import pandas as pd

from tagent.xs_momentum import (
    XSMomConfig, align_close, backtest, momentum_signal, sweep_lookbacks, _weights,
)


def _coin(prices, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    return pd.DataFrame({"close": np.asarray(prices, float)}, index=idx)


def _panel(price_dict):
    return {c: _coin(p) for c, p in price_dict.items()}


# --------------------------------------------------------------------------- #
# momentum signal + ranking
# --------------------------------------------------------------------------- #
def test_momentum_signal_is_trailing_return():
    close = align_close(_panel({"A": [100, 110, 121, 133.1]}))
    mom = momentum_signal(close, lookback=1)
    assert abs(mom["A"].iloc[1] - 0.10) < 1e-9            # 110/100 - 1
    assert pd.isna(mom["A"].iloc[0])                      # no history at bar 0


def test_weights_long_top_short_bottom_dollar_neutral():
    # A strongest up, D weakest -> A longed, D shorted at the last bar
    close = align_close(_panel({
        "A": [100, 104, 109, 115], "B": [100, 102, 104, 106],
        "C": [100, 101, 101, 101], "D": [100, 99, 98, 97]}))
    w = _weights(close, XSMomConfig(lookback=1, top_q=0.25, hysteresis=0.0))
    last = w.iloc[-1]
    assert last["A"] > 0 and last["D"] < 0                # long winner, short loser
    assert abs(last.sum()) < 1e-9                         # dollar-neutral
    assert abs(last[last > 0].sum() - 1.0) < 1e-9         # long leg sums to +1


def test_long_only_has_no_shorts():
    close = align_close(_panel({"A": [100, 104, 109, 115], "B": [100, 99, 98, 97]}))
    w = _weights(close, XSMomConfig(lookback=1, top_q=0.5, allow_short=False))
    assert (w.iloc[-1] >= 0).all() and abs(w.iloc[-1].sum() - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# economics: momentum persists -> positive; mean-reverts -> negative
# --------------------------------------------------------------------------- #
def _trending_panel(n=120, k=8, seed=0, persist=True):
    """k coins; each has a persistent drift (winners keep winning) if persist,
    else drifts reverse each bar (momentum should lose)."""
    rng = np.random.default_rng(seed)
    drifts = np.linspace(-0.01, 0.01, k)                  # spread of trends
    out = {}
    for i in range(k):
        steps = rng.normal(drifts[i], 0.005, n)
        if not persist:                                   # flip the sign each bar -> mean-revert
            steps = steps * (np.where(np.arange(n) % 2 == 0, 1.0, -1.0))
        out[f"C{i}"] = _coin(100 * np.cumprod(1 + np.r_[0.0, steps[:-1]]))
    return out


def test_momentum_pays_when_trends_persist():
    r = backtest(_trending_panel(persist=True), XSMomConfig(lookback=3, cost_bps=0, slippage_bps=0))
    assert r["stats"]["total_return"] > 0                 # long winners / short losers pays
    assert r["stats"]["sharpe"] > 0


def test_momentum_loses_when_mean_reverting():
    r = backtest(_trending_panel(persist=False), XSMomConfig(lookback=1, cost_bps=0, slippage_bps=0))
    assert r["stats"]["total_return"] < 0                 # chasing momentum into reversals loses


# --------------------------------------------------------------------------- #
# costs + hysteresis
# --------------------------------------------------------------------------- #
def test_costs_reduce_net_and_count_turnover():
    panel = _trending_panel(persist=True, seed=1)
    free = backtest(panel, XSMomConfig(lookback=2, cost_bps=0, slippage_bps=0))
    paid = backtest(panel, XSMomConfig(lookback=2, cost_bps=20, slippage_bps=10))
    assert paid["net"].sum() < free["net"].sum()
    assert free["avg_turnover"] > 0


def test_hysteresis_reduces_turnover():
    panel = _trending_panel(persist=True, seed=2)
    none = backtest(panel, XSMomConfig(lookback=1, hysteresis=0.0))
    hyst = backtest(panel, XSMomConfig(lookback=1, hysteresis=0.3))
    assert hyst["avg_turnover"] <= none["avg_turnover"]   # banding holds names -> less churn


# --------------------------------------------------------------------------- #
# no-lookahead
# --------------------------------------------------------------------------- #
def test_no_lookahead_future_bars_do_not_change_earlier_weights():
    base = {"A": [100, 104, 109, 115, 120], "B": [100, 99, 98, 97, 96],
            "C": [100, 101, 100, 101, 100]}
    w1 = _weights(align_close(_panel(base)), XSMomConfig(lookback=2))
    extended = {k: v + [v[-1] * 1.5] for k, v in base.items()}      # wild FUTURE bar appended
    w2 = _weights(align_close(_panel(extended)), XSMomConfig(lookback=2))
    # weights for the overlapping earlier bars are unchanged by the future
    assert np.allclose(w1.to_numpy(), w2.iloc[: len(w1)].to_numpy(), equal_nan=True)


def test_position_earns_next_bar_return_not_current():
    # a coin flat then a one-bar SPIKE: momentum at the spike bar is high, but the
    # P&L of holding into it must come from the PRIOR bar's decision, not hindsight.
    close = align_close(_panel({"A": [100, 100, 100, 130], "B": [100, 100, 100, 100],
                                "C": [100, 100, 100, 70]}))
    r = backtest(_panel({"A": [100, 100, 100, 130], "B": [100, 100, 100, 100],
                         "C": [100, 100, 100, 70]}),
                 XSMomConfig(lookback=1, cost_bps=0, slippage_bps=0))
    # before the spike everything is flat -> no momentum dispersion -> ~0 P&L,
    # and the final bar is dropped (no forward return) so the spike isn't "earned"
    assert abs(r["net"].sum()) < 1e-6


# --------------------------------------------------------------------------- #
# backtest output + sweep
# --------------------------------------------------------------------------- #
def test_backtest_reports_basket_benchmark():
    r = backtest(_trending_panel(persist=True), XSMomConfig(lookback=3))
    assert "basket" in r and "basket_stats" in r and r["n_bars"] > 0
    assert set(r["stats"]) >= {"sharpe", "max_drawdown", "cagr", "total_return"}


def test_sweep_lookbacks_covers_one_to_seven():
    rows = sweep_lookbacks(_trending_panel(persist=True), lookbacks=range(1, 8))
    assert [row["lookback"] for row in rows] == list(range(1, 8))
    assert all("sharpe" in row and "avg_turnover" in row for row in rows)


# =========================================================================== #
# classic long-horizon momentum: skip-recent, low-freq rebalance, reversal
# =========================================================================== #
def test_momentum_signal_skips_recent_week():
    close = align_close(_panel({"A": [100, 110, 121, 133.1, 146.41]}))  # +10%/bar
    sig = momentum_signal(close, lookback=2, skip_recent=1)
    # at t=4: uses close[3]/close[1]-1 = 133.1/110-1; the most recent bar is excluded
    assert abs(sig["A"].iloc[4] - (133.1 / 110.0 - 1.0)) < 1e-9
    plain = momentum_signal(close, lookback=2, skip_recent=0)
    assert abs(sig["A"].iloc[4] - plain["A"].iloc[3]) < 1e-9   # skip = the lagged plain signal


def test_low_frequency_rebalance_cuts_turnover():
    panel = _trending_panel(persist=True, seed=4, n=140)
    daily = backtest(panel, XSMomConfig(lookback=10, rebalance=1))
    weekly = backtest(panel, XSMomConfig(lookback=10, rebalance=7))
    assert weekly["avg_turnover"] < daily["avg_turnover"]      # far less churn
    assert weekly["avg_turnover"] <= daily["avg_turnover"] * 0.5


def test_rebalance_holds_weights_between_rebalances():
    panel = _trending_panel(persist=True, seed=5, n=60)
    w = _weights(align_close(panel), XSMomConfig(lookback=5, rebalance=5))
    # rows strictly between rebalance bars equal the prior row (held, not re-traded)
    for t in range(1, len(w)):
        if t % 5 != 0:
            assert np.allclose(w.iloc[t].to_numpy(), w.iloc[t - 1].to_numpy())


def test_reversal_long_losers_pays_on_mean_reversion():
    panel = _trending_panel(persist=False, seed=6)               # mean-reverting
    mom = backtest(panel, XSMomConfig(lookback=1, reverse=False, cost_bps=0, slippage_bps=0))
    rev = backtest(panel, XSMomConfig(lookback=1, reverse=True, cost_bps=0, slippage_bps=0))
    assert rev["stats"]["total_return"] > 0 > mom["stats"]["total_return"]   # opposite signs
    assert rev["gross"].sum() > 0 > mom["gross"].sum()                       # reversal has gross edge


def test_reversal_loses_on_persistent_trend():
    panel = _trending_panel(persist=True, seed=7)
    rev = backtest(panel, XSMomConfig(lookback=3, reverse=True, cost_bps=0, slippage_bps=0))
    assert rev["stats"]["total_return"] < 0   # longing losers in a trending market loses


def test_no_lookahead_with_skip_recent():
    base = {"A": [100, 102, 105, 109, 114, 120], "B": [100, 99, 98, 97, 96, 95],
            "C": [100, 101, 100, 101, 100, 101]}
    w1 = _weights(align_close(_panel(base)), XSMomConfig(lookback=2, skip_recent=1))
    ext = {k: v + [v[-1] * 1.5] for k, v in base.items()}        # wild future bar
    w2 = _weights(align_close(_panel(ext)), XSMomConfig(lookback=2, skip_recent=1))
    assert np.allclose(w1.to_numpy(), w2.iloc[: len(w1)].to_numpy(), equal_nan=True)
