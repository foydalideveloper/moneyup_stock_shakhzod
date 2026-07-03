"""Index-rebalance fade — change parsing, fade windows + no-lookahead, cost split, t-stats.
Synthetic series, no network."""

import numpy as np
import pandas as pd

from tagent.index_rebalance import (
    CASH_ROUND_TRIP, SSF_ROUND_TRIP, calendar_time_excess, calendar_time_tstat,
    fade_event_returns, fade_net, plain_tstat, reconstruct_changes, review_effective_dates,
)


# --------------------------------------------------------------------------- #
# 1) membership-change parsing + review calendar
# --------------------------------------------------------------------------- #
def test_review_effective_dates_jun_dec_after_2nd_thursday():
    ti = pd.bdate_range("2024-01-01", "2024-12-31")
    effs = review_effective_dates(2024, 2024, ti)
    assert len(effs) == 2 and {d.month for d in effs} == {6, 12}
    assert effs[0] == pd.Timestamp("2024-06-14")           # first trading day after 2nd Thu (06-13)
    assert effs[1] == pd.Timestamp("2024-12-13")           # after 2nd Thu (12-12)


def test_reconstruct_changes_diff():
    eff = pd.Timestamp("2024-06-14")
    snaps = {eff: {"before": {"000001", "000002", "000003"}, "after": {"000002", "000003", "000009"}}}
    ch = reconstruct_changes(snaps)
    by = dict(zip(ch["ticker"], ch["action"]))
    assert by["000009"] == "add" and by["000001"] == "delete" and len(ch) == 2
    # empty/partial snapshots are skipped (no fake events)
    assert reconstruct_changes({eff: {"before": set(), "after": {"000001"}}}).empty


# --------------------------------------------------------------------------- #
# fixture: an add that underperforms + a delete that outperforms the benchmark
# --------------------------------------------------------------------------- #
def _setup():
    idx = pd.bdate_range("2024-01-01", periods=140)
    bench = pd.Series(100 * np.cumprod(1 + np.r_[0.0, np.full(139, 0.001)]), index=idx)   # +0.1%/day
    add = pd.Series(100.0, index=idx)                                                     # flat -> underperforms
    dele = pd.Series(100 * np.cumprod(1 + np.r_[0.0, np.full(139, 0.003)]), index=idx)    # +0.3%/day -> outperforms
    panel = {"000001": pd.DataFrame({"close": add}), "000002": pd.DataFrame({"close": dele})}
    eff = idx[60]
    changes = pd.DataFrame({"effective": [eff, eff], "ticker": ["000001", "000002"],
                            "action": ["add", "delete"]})
    return panel, bench, changes, idx, eff


# --------------------------------------------------------------------------- #
# 2) fade windows + sign + no-lookahead
# --------------------------------------------------------------------------- #
def test_fade_event_returns_sign_and_no_lookahead():
    panel, bench, changes, idx, eff = _setup()
    ev = fade_event_returns(changes, panel, bench, entry_lag=1, hold=20)
    assert len(ev) == 2
    a = ev[ev["action"] == "add"].iloc[0]
    d = ev[ev["action"] == "delete"].iloc[0]
    assert a["entry"] == idx[idx.get_loc(eff) + 1]         # enter the day AFTER effective
    assert a["abnormal"] < 0 and a["strategy_ret"] > 0     # add underperforms -> SHORT wins
    assert d["abnormal"] > 0 and d["strategy_ret"] > 0     # delete outperforms -> LONG wins
    # no-lookahead: appending future bars doesn't change the already-complete windows
    ext_idx = pd.date_range(idx[-1] + pd.offsets.BDay(1), periods=30, freq="B")
    panel2 = {c: pd.concat([panel[c], pd.DataFrame({"close": [float(panel[c]["close"].iloc[-1])] * 30},
                                                   index=ext_idx)]) for c in panel}
    bench2 = pd.concat([bench, pd.Series([float(bench.iloc[-1])] * 30, index=ext_idx)])
    ev2 = fade_event_returns(changes, panel2, bench2, entry_lag=1, hold=20)
    assert np.allclose(ev["strategy_ret"].to_numpy(), ev2["strategy_ret"].to_numpy())
    # events whose window falls off the data are dropped (no-lookahead)
    late = pd.DataFrame({"effective": [idx[-3]], "ticker": ["000001"], "action": ["add"]})
    assert fade_event_returns(late, panel, bench, hold=20).empty


# --------------------------------------------------------------------------- #
# 3) SSF / cash cost split
# --------------------------------------------------------------------------- #
def test_fade_net_cost_split():
    panel, bench, changes, idx, eff = _setup()
    ev = fade_event_returns(changes, panel, bench, entry_lag=1, hold=20)
    net = fade_net(ev, ssf_rt=SSF_ROUND_TRIP, cash_rt=CASH_ROUND_TRIP)
    a = ev[ev["action"] == "add"].iloc[0]
    d = ev[ev["action"] == "delete"].iloc[0]
    # both events share the effective date -> compare the value set (add pays SSF, delete cash)
    expect = sorted([a["strategy_ret"] - SSF_ROUND_TRIP, d["strategy_ret"] - CASH_ROUND_TRIP])
    assert np.allclose(sorted(net.to_numpy()), expect)


# --------------------------------------------------------------------------- #
# 4) calendar-time excess + t-stats
# --------------------------------------------------------------------------- #
def test_calendar_time_and_plain_t():
    panel, bench, changes, idx, eff = _setup()
    ev = fade_event_returns(changes, panel, bench, entry_lag=1, hold=20)
    ex = calendar_time_excess(ev, panel, bench, entry_lag=1, hold=20)
    assert len(ex) > 0 and ex.mean() > 0                   # both legs positive -> positive excess
    t, n = calendar_time_tstat(ex, lag=20)
    assert n == len(ex.dropna()) and np.isfinite(t)
    assert plain_tstat([0.01, 0.012, 0.009, 0.011]) > 3    # tight positive sample -> significant
    assert plain_tstat([1.0]) == 0.0                       # too few obs -> 0
