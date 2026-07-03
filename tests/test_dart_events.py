"""DART event studies — parsing, size-matched control, calendar-time t, gap slippage,
no-lookahead. Synthetic panels, no network."""

import numpy as np
import pandas as pd

from tagent.dart_events import (
    add_size_matched, calendar_time_excess, calendar_time_tstat, classify_event,
    event_trades, events_by_symbol, gap_decomposition, net_with_gap_slippage,
    quintile_daily_returns, size_quintiles,
)
from tagent.kr_universe import membership_panel


# --------------------------------------------------------------------------- #
# 1) per-type disclosure parsing
# --------------------------------------------------------------------------- #
def test_classify_event_locked_rules():
    assert classify_event("주식소각결정") == "cancellation"
    assert classify_event("[기재정정]주식소각결정") is None              # 정정 excluded
    assert classify_event("주식소각결정(자회사의 주요경영사항)") is None  # 자회사 excluded
    assert classify_event("주요사항보고서(자기주식취득결정)") == "acquisition"
    assert classify_event("자기주식취득신탁계약체결결정") is None         # 신탁 excluded
    assert classify_event("주요사항보고서(자기주식처분결정)") is None     # 처분 excluded
    assert classify_event("단일판매ㆍ공급계약체결") == "contract"
    assert classify_event("[정정]단일판매ㆍ공급계약체결") is None
    assert classify_event("분기보고서") is None


def test_events_by_symbol():
    df = pd.DataFrame({"time": ["2020-01-02", "2020-03-02", "2020-02-02"],
                       "symbol": ["000660", "000660", "005930"],
                       "type": ["cancellation", "cancellation", "contract"]})
    ev = events_by_symbol(df, "cancellation")
    assert set(ev) == {"000660"} and len(ev["000660"]) == 2
    assert ev["000660"][0] == pd.Timestamp("2020-01-02")


# --------------------------------------------------------------------------- #
# fixture: deterministic-ish panel with ordered caps
# --------------------------------------------------------------------------- #
def _panel(n_names=10, n_days=220, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    panel, shares = {}, {}
    for i in range(n_names):
        code = f"{i:06d}"
        r = 0.002 * i + rng.normal(0.0, 0.0003, n_days)        # higher i -> higher drift
        close = 100 * np.cumprod(1 + np.r_[0.0, r[:-1]])
        panel[code] = pd.DataFrame({"open": close, "close": close}, index=idx)
        shares[code] = float(1000 * (i + 1))                   # higher i -> bigger cap
    memb = membership_panel({"2018-01-01": list(panel)}, idx, symbols=list(panel))
    return panel, memb, shares, idx


# --------------------------------------------------------------------------- #
# 2) size quintiles + matched control
# --------------------------------------------------------------------------- #
def test_size_quintiles_order_and_quintile_returns():
    panel, memb, shares, idx = _panel()
    q = size_quintiles(panel, memb, shares, n_q=5)
    last = q.iloc[-1]
    assert last["000009"] == 5 and last["000000"] == 1          # biggest cap -> Q5, smallest -> Q1
    assert set(last.dropna().unique()) <= {1, 2, 3, 4, 5}
    qret = quintile_daily_returns(panel, q, n_q=5)
    # Q5 daily return == mean of its two members' (000008, 000009) daily returns
    close = pd.DataFrame({c: d["close"] for c, d in panel.items()})
    ret = close.pct_change(fill_method=None)
    exp = ret[["000008", "000009"]].mean(axis=1)
    assert abs(qret[5].iloc[-1] - exp.iloc[-1]) < 1e-12


def test_add_size_matched_abnormal_equals_ret_minus_control():
    panel, memb, shares, idx = _panel()
    q = size_quintiles(panel, memb, shares, n_q=5)
    qret = quintile_daily_returns(panel, q, n_q=5)
    ev = {"000009": [idx[80]]}                                  # event on a Q5 name
    tr = event_trades(panel, ev, hold=20, membership=memb)
    sm = add_size_matched(tr, panel, q, qret, hold=20)
    row = sm.iloc[0]
    assert int(row["quintile"]) == 5
    # control_ret = compounded Q5 daily return over the held window [entry+1 .. entry+hold]
    ei = idx.get_loc(tr["entry"].iloc[0])
    exp_ctrl = float((1 + qret[5].iloc[ei + 1:ei + 21].fillna(0)).prod() - 1)
    assert abs(row["control_ret"] - exp_ctrl) < 1e-9
    assert abs(row["abnormal"] - (row["ret"] - row["control_ret"])) < 1e-12


# --------------------------------------------------------------------------- #
# 3) calendar-time portfolio + NW t
# --------------------------------------------------------------------------- #
def test_calendar_time_excess_and_tstat():
    panel, memb, shares, idx = _panel()
    q = size_quintiles(panel, memb, shares, n_q=5)
    qret = quintile_daily_returns(panel, q, n_q=5)
    ev = {"000009": [idx[80]]}                                  # Q5 name out-drifts its quintile peer
    tr = event_trades(panel, ev, hold=30, membership=memb)
    sm = add_size_matched(tr, panel, q, qret, hold=30)
    ex = calendar_time_excess(sm, panel, q, qret, hold=30)
    assert len(ex) > 0 and np.isfinite(ex.to_numpy()).all()
    assert ex.mean() > 0                                        # 000009 drifts above its quintile mean
    t, n = calendar_time_tstat(ex, hold=30)
    assert n == len(ex.dropna()) and np.isfinite(t)


# --------------------------------------------------------------------------- #
# 4) no-lookahead timing
# --------------------------------------------------------------------------- #
def test_event_trades_enter_after_filing_no_lookahead():
    panel, memb, shares, idx = _panel()
    ev = {"000005": [idx[100]]}                                 # filing on idx[100]
    tr = event_trades(panel, ev, hold=20, membership=memb)
    assert tr["entry"].iloc[0] == idx[101]                      # first open strictly AFTER filing
    # a filing with < hold bars left yields no trade
    assert len(event_trades(panel, {"000005": [idx[-3]]}, hold=20, membership=memb)) == 0


# --------------------------------------------------------------------------- #
# 5) conservative-slippage gap entry + gap/post-open decomposition
# --------------------------------------------------------------------------- #
def test_gap_slippage_and_decomposition():
    tr = pd.DataFrame({
        "entry": pd.to_datetime(["2020-01-02", "2020-02-03"]),
        "symbol": ["000001", "000002"],
        "ret": [0.05, 0.03], "r0": [0.04, 0.005],              # first is a gap-up entry (r0>1%)
        "abnormal": [0.02, 0.01]})
    net = net_with_gap_slippage(tr, cost=0.0071, gap_threshold=0.01, gap_extra=0.0030)
    # gap-up trade pays cost + extra slippage; non-gap trade pays only cost
    assert abs(net.iloc[0] - (0.02 - 0.0071 - 0.0030)) < 1e-12
    assert abs(net.iloc[1] - (0.01 - 0.0071)) < 1e-12
    gd = gap_decomposition(tr)
    assert abs(gd["mean_gap_r0"] - 0.0225) < 1e-9 and abs(gd["mean_post_open"] - 0.04) < 1e-9
    assert gd["frac_gap_up"] == 0.5                             # one of two entries gapped up
