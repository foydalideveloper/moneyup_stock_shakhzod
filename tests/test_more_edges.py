"""vol-spike / flows-following / earnings-drift daily edges — mock data, no network."""

import numpy as np
import pandas as pd

from tagent.strategies.earnings_drift import earnings_events, pead_event_returns
from tagent.strategies.flows_following import flows_signal
from tagent.strategies.us_leadlag_daily import summarize
from tagent.strategies.vol_spike_bounce import (
    event_dates, event_stock_returns, vol_spike_mask,
)


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
# 1) vol-spike bounce
# --------------------------------------------------------------------------- #
def test_vol_spike_mask_asof_threshold():
    kr = pd.to_datetime(["2020-01-03", "2020-01-06"])
    vix_change = pd.Series([0.20, 0.05], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    m = vol_spike_mask(kr, vix_change, 0.15)
    assert bool(m.iloc[0]) is True and bool(m.iloc[1]) is False     # 01-03 sees +20%, 01-06 sees +5%


def test_vol_spike_event_returns_and_independence():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    panel = _panel({"A": [(d, 100, c) for d, c in zip(dates, [100, 102, 100])],
                    "B": [(d, 100, c) for d, c in zip(dates, [100, 98, 100])]})
    vix = pd.Series([0.30, 0.0], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    ps = event_stock_returns(panel, vix, threshold=0.15, hold=1)
    assert sorted(round(v, 4) for v in ps.xs(pd.Timestamp("2020-01-03"), level=0).values) == [-0.02, 0.02]
    assert list(event_dates(panel, vix, 0.15)) == [pd.Timestamp("2020-01-03")]
    # appending a future vol session does not change the past events (no-lookahead)
    vix2 = pd.concat([vix, pd.Series([0.9], index=pd.to_datetime(["2020-01-06"]))])
    assert len(event_stock_returns(panel, vix2, 0.15, 1)) == len(ps)


def test_vol_spike_hold_n_days():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    # spike before 01-03; enter open[01-03]=100, exit close[01-06] (hold=2) = 110
    panel = _panel({"A": [("2020-01-02", 100, 100), ("2020-01-03", 100, 105),
                          ("2020-01-06", 106, 110), ("2020-01-07", 110, 111)]})
    vix = pd.Series([0.30], index=pd.to_datetime(["2020-01-02"]))
    ps = event_stock_returns(panel, vix, threshold=0.15, hold=2)   # open[01-03]->close[01-06]
    assert abs(ps.iloc[0] - (110 / 100 - 1)) < 1e-9


# --------------------------------------------------------------------------- #
# 2) flows-following (수급)
# --------------------------------------------------------------------------- #
def test_flows_signal_ranks_buyers_and_no_lookahead():
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    nb = pd.DataFrame({"A": [1e9] * 80, "B": [-1e9] * 80, "C": [0.0] * 80}, index=idx)
    sig = flows_signal(nb, lookback=5, lag=1)
    last = sig.dropna(how="all").iloc[-1]
    assert last["A"] > last["B"]                                    # strongest buyer ranks highest
    assert pd.isna(sig.iloc[-1]["C"])                              # no flows -> no signal
    # last-day flows must NOT change any in-sample signal value (lag=1 -> uses past only)
    nb2 = nb.copy()
    nb2.iloc[-1] = [-9e9, 9e9, 5e9]
    assert flows_signal(nb2, lookback=5, lag=1).equals(sig)


# --------------------------------------------------------------------------- #
# 3) earnings drift (PEAD)
# --------------------------------------------------------------------------- #
def test_earnings_events_filters_and_groups():
    disc = [{"type": "earnings", "symbol": "005930", "time": "2020-01-03"},
            {"type": "other", "symbol": "005930", "time": "2020-01-05"},
            {"type": "earnings", "symbol": "000660", "time": "2020-01-06"}]
    ev = earnings_events(disc)
    assert ev == {"005930": [pd.Timestamp("2020-01-03")], "000660": [pd.Timestamp("2020-01-06")]}


def _pead_panel():
    idx = pd.bdate_range("2020-01-02", periods=12)         # 01-02,03,06,07,08,09,...
    o = {d: 100.0 for d in idx}
    c = {d: 100.0 for d in idx}
    c[pd.Timestamp("2020-01-03")] = 100.0                  # prev close (filing day)
    o[pd.Timestamp("2020-01-06")] = 102.0                  # entry open (gap up -> r0=+2%)
    c[pd.Timestamp("2020-01-09")] = 108.0                  # exit close (entry+hold=3)
    df = pd.DataFrame({"open": [o[d] for d in idx], "high": 1, "low": 1,
                       "close": [c[d] for d in idx], "volume": 1}, index=idx)
    return {"005930": df}


def test_pead_entry_after_filing_and_positive_condition():
    panel = _pead_panel()
    ev = {"005930": [pd.Timestamp("2020-01-03")]}
    ret = pead_event_returns(panel, ev, hold=3, conditional=True)
    assert len(ret) == 1
    assert ret.index[0] == pd.Timestamp("2020-01-06")      # entry is the NEXT trading day after filing
    assert abs(ret.iloc[0] - (108.0 / 102.0 - 1.0)) < 1e-9  # drift = exit_close/entry_open - 1


def test_pead_skips_negative_reaction_when_conditional():
    panel = _pead_panel()
    panel["005930"].loc[pd.Timestamp("2020-01-06"), "open"] = 98.0   # gap DOWN -> r0<0
    ret = pead_event_returns(panel, {"005930": [pd.Timestamp("2020-01-03")]},
                             hold=3, conditional=True)
    assert len(ret) == 0                                   # positive-only filter drops it
    ret2 = pead_event_returns(panel, {"005930": [pd.Timestamp("2020-01-03")]},
                              hold=3, conditional=False)
    assert len(ret2) == 1                                  # unconditional keeps it


def test_summarize_reused():
    s = summarize(pd.Series([0.02, -0.01, 0.03]), cost_frac=0.004)
    assert s["n"] == 3 and abs(s["net_exp"] - (float(pd.Series([0.02, -0.01, 0.03]).mean()) - 0.004)) < 1e-12
