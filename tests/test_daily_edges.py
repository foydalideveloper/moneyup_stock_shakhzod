"""Short-horizon daily KR edges — mock daily bars, no network."""

import numpy as np
import pandas as pd

from tagent.intraday_backtest import CostModel
from tagent.strategies.overnight_effect import overnight_backtest, overnight_gross
from tagent.strategies.short_term_reversal import reversal_backtest, reversal_signal
from tagent.strategies.us_leadlag_daily import (
    event_day_returns, event_stock_returns, leadlag_backtest, summarize,
)
from tagent.xs_momentum import align_close


def _panel(rows):
    """rows: {sym: [(date, open, close), ...]} -> {sym: OHLCV df}."""
    out = {}
    for sym, recs in rows.items():
        idx = pd.to_datetime([r[0] for r in recs])
        out[sym] = pd.DataFrame(
            {"open": [r[1] for r in recs], "high": [max(r[1], r[2]) for r in recs],
             "low": [min(r[1], r[2]) for r in recs], "close": [r[2] for r in recs],
             "volume": [100] * len(recs)}, index=idx)
    return out


# --------------------------------------------------------------------------- #
# overnight effect: close[t] -> open[t+1]
# --------------------------------------------------------------------------- #
def test_overnight_gross_close_to_next_open():
    p = _panel({
        "A": [("2020-01-02", 100, 100), ("2020-01-03", 102, 100), ("2020-01-06", 100, 100)],
        "B": [("2020-01-02", 100, 100), ("2020-01-03", 98, 100), ("2020-01-06", 100, 100)],
    })
    g = overnight_gross(p)
    # day0: A open[1]/close[0]=102/100=+2%, B 98/100=-2% -> EW 0%
    assert abs(g.iloc[0]) < 1e-12
    # day1: A 100/100=0, B 100/100=0 -> 0; last day dropped (no next open)
    assert len(g) == 2


def test_overnight_no_lookahead_and_cost():
    base = {"A": [("2020-01-02", 100, 100), ("2020-01-03", 101, 100), ("2020-01-06", 100, 100)]}
    g3 = overnight_gross(_panel(base))
    ext = {"A": base["A"] + [("2020-01-07", 200, 100)]}      # arbitrary future day appended
    g4 = overnight_gross(_panel(ext))
    assert np.allclose(g3.iloc[:1].to_numpy(), g4.iloc[:1].to_numpy())   # earlier day unchanged
    bt = overnight_backtest(_panel(base), cost=CostModel())
    assert np.allclose((bt["gross"] - bt["net"]).to_numpy(), bt["cost_frac"])   # cost each day


# --------------------------------------------------------------------------- #
# short-term reversal: long the losers
# --------------------------------------------------------------------------- #
def _trend_panel(n=20):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return {
        "L": pd.DataFrame({"open": 100 * 0.98 ** np.arange(n), "high": 1, "low": 1,
                           "close": 100 * 0.98 ** np.arange(n), "volume": 1}, index=idx),
        "M": pd.DataFrame({"open": [100.0] * n, "high": 1, "low": 1,
                           "close": [100.0] * n, "volume": 1}, index=idx),
        "W": pd.DataFrame({"open": 100 * 1.02 ** np.arange(n), "high": 1, "low": 1,
                           "close": 100 * 1.02 ** np.arange(n), "volume": 1}, index=idx),
    }


def test_reversal_signal_ranks_loser_highest():
    panel = _trend_panel()
    sig = reversal_signal(align_close(panel), lookback=2)
    last = sig.dropna().iloc[-1]
    assert last["L"] > last["M"] > last["W"]                 # biggest loser -> highest signal


def test_reversal_backtest_longs_the_loser():
    panel = _trend_panel(n=30)
    from tagent.stock_momentum import classic_config
    from dataclasses import replace
    cfg = replace(classic_config("kr", allow_short=False), cost_bps=0.0, slippage_bps=0.0)
    r = reversal_backtest(panel, lookback=2, hold=2, top_q=0.4, cfg=cfg)
    w = r["weights"]
    invested = w[(w != 0).any(axis=1)]
    assert invested["L"].mean() > 0 and np.allclose(invested["W"].to_numpy(), 0.0)


# --------------------------------------------------------------------------- #
# US lead-lag: trade KR only after a sharp US overnight drop
# --------------------------------------------------------------------------- #
def test_leadlag_event_filtering_and_no_lookahead():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    panel = _panel({
        "A": [(d, 100, c) for d, c in zip(dates, [100, 101, 100, 100])],   # +1% on 01-03
        "B": [(d, 100, c) for d, c in zip(dates, [100, 99, 100, 100])],    # -1% on 01-03
    })
    # US down -2% on 01-02 -> KR 01-03 is the event day (uses the prior US session)
    spy = pd.Series([-0.02, 0.005, 0.004],
                    index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    ev = event_day_returns(panel, spy, threshold=-0.01)
    assert list(ev.index) == [pd.Timestamp("2020-01-03")]    # only the post-US-drop day
    assert abs(ev.iloc[0]) < 1e-12                            # EW(+1%, -1%) intraday = 0
    # appending a future US session must not change the earlier event set
    spy2 = pd.concat([spy, pd.Series([-0.05], index=pd.to_datetime(["2020-01-07"]))])
    ev2 = event_day_returns(panel, spy2, threshold=-0.01)
    assert list(ev2.index) == [pd.Timestamp("2020-01-03")]


def test_leadlag_backtest_cost_and_count():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    panel = _panel({"A": [(d, 100, 102) for d in dates]})    # +2% intraday each day
    spy = pd.Series([-0.03, -0.03], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    bt = leadlag_backtest(panel, spy, threshold=-0.01, cost=CostModel())
    assert bt["n_events"] >= 1
    assert np.allclose((bt["gross"] - bt["net"]).to_numpy(), bt["cost_frac"])


def test_event_stock_returns_more_obs_than_ew_and_no_lookahead():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    panel = _panel({
        "A": [(d, 100, c) for d, c in zip(dates, [100, 102, 100])],   # +2% intraday on 01-03
        "B": [(d, 100, c) for d, c in zip(dates, [100, 98, 100])],    # -2% intraday on 01-03
    })
    spy = pd.Series([-0.03, -0.03], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    ps = event_stock_returns(panel, spy, threshold=-0.01)
    ew = event_day_returns(panel, spy, threshold=-0.01)
    assert len(ps) == 2 * len(ew)                            # per-stock = N_stocks x EW
    on_0103 = ps.xs(pd.Timestamp("2020-01-03"), level=0)
    assert sorted(round(v, 4) for v in on_0103.values) == [-0.02, 0.02]
    # appending a FUTURE US session must not change the event observations
    spy2 = pd.concat([spy, pd.Series([-0.09], index=pd.to_datetime(["2020-01-07"]))])
    assert len(event_stock_returns(panel, spy2, threshold=-0.01)) == len(ps)


def test_summarize_cost_winrate_and_tstat():
    r = pd.Series([0.02, 0.00, -0.01, 0.03])
    s = summarize(r, cost_frac=0.004)
    assert s["n"] == 4
    assert abs(s["gross_exp"] - float(r.mean())) < 1e-12
    assert abs(s["net_exp"] - float(r.mean() - 0.004)) < 1e-12
    assert s["win"] == float(((r - 0.004) > 0).mean())       # net wins
    assert summarize(pd.Series([], dtype=float))["n"] == 0
