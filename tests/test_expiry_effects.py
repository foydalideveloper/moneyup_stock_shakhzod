"""KR expiry-calendar effects — calendar construction, event windows, no-lookahead, costs,
non-overlapping t-stat. Synthetic series, no network."""

import numpy as np
import pandas as pd

from tagent.expiry_effects import (
    HOLD, by_year, event_returns, expiry_dates, nth_weekday, plain_tstat, reversal_net,
    reversal_regression,
)


# --------------------------------------------------------------------------- #
# 1) expiry-calendar construction (2nd Thursday, snap to trading day, quarterly)
# --------------------------------------------------------------------------- #
def test_nth_weekday_second_thursday():
    assert nth_weekday(2024, 6, weekday=3, n=2) == pd.Timestamp("2024-06-13")
    assert nth_weekday(2024, 3, weekday=3, n=2) == pd.Timestamp("2024-03-14")
    assert nth_weekday(2024, 1, weekday=3, n=2).weekday() == 3      # is a Thursday


def test_expiry_dates_snap_and_quarterly_filter():
    ti = pd.bdate_range("2024-01-01", "2024-12-31")
    monthly = expiry_dates("2024-01-01", "2024-12-31", ti)
    assert len(monthly) == 12 and all(d in ti for d in monthly)     # one per month, trading days
    # quarterly = witching months only
    q = expiry_dates("2024-01-01", "2024-12-31", ti, quarterly_only=True)
    assert {d.month for d in q} == {3, 6, 9, 12} and len(q) == 4
    # holiday snap: drop the 2nd Thursday from the calendar -> expiry moves to the day before
    thu = nth_weekday(2024, 6, 3, 2)
    ti2 = ti[ti != thu]
    e = expiry_dates("2024-06-01", "2024-06-30", ti2)
    assert e[0] < thu and e[0] == ti2[ti2.searchsorted(thu, side="right") - 1]


# --------------------------------------------------------------------------- #
# 2) event-window extraction + no-lookahead
# --------------------------------------------------------------------------- #
def _close(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.cumprod(1 + np.r_[0.0, rng.normal(0.0003, 0.01, n - 1)]), index=idx)


def test_event_returns_values_and_no_lookahead():
    c = _close()
    exps = expiry_dates(c.index.min(), c.index.max(), c.index)
    ev = event_returns(c, exps, pre_window=5, hold=2)
    assert len(ev) > 0
    E = ev["expiry"].iloc[3]
    ei = c.index.get_loc(E)
    assert abs(ev["pre_drift"].iloc[3] - (c.iloc[ei] / c.iloc[ei - 5] - 1)) < 1e-12
    assert abs(ev["fwd_ret"].iloc[3] - (c.iloc[ei + 2] / c.iloc[ei] - 1)) < 1e-12
    # no-lookahead: appending future bars leaves already-complete events unchanged
    ext = pd.concat([c, pd.Series(float(c.iloc[-1]) * np.cumprod(1 + np.full(40, 0.001)),
                                  index=pd.bdate_range(c.index[-1] + pd.offsets.BDay(1), periods=40))])
    exps2 = expiry_dates(ext.index.min(), ext.index.max(), ext.index)
    ev2 = event_returns(ext, exps2, pre_window=5, hold=2)
    merged = ev.merge(ev2, on="expiry", suffixes=("", "_2"))
    assert np.allclose(merged["fwd_ret"], merged["fwd_ret_2"])
    # an expiry too close to the end (incomplete fwd window) is dropped
    assert ev["expiry"].max() <= c.index[-1 - 2]


# --------------------------------------------------------------------------- #
# 3) reversal trade + cost/slippage + regression sign
# --------------------------------------------------------------------------- #
def test_reversal_net_formula_and_costs():
    ev = pd.DataFrame({"expiry": pd.to_datetime(["2020-03-12", "2020-06-11"]),
                       "pre_drift": [0.02, -0.03], "expiry_ret": [0.0, 0.0],
                       "fwd_ret": [0.01, -0.015]})
    cheap = reversal_net(ev, cost_round_trip=0.0005)
    # -sign(+0.02)*0.01 - 0.0005 = -0.0105 ; -sign(-0.03)*(-0.015) - 0.0005 = -0.0155
    assert abs(cheap.iloc[0] - (-0.0105)) < 1e-12 and abs(cheap.iloc[1] - (-0.0155)) < 1e-12
    pricey = reversal_net(ev, cost_round_trip=0.0040)
    assert (pricey < cheap).all()                                  # higher cost -> lower net


def test_reversal_regression_detects_reversal():
    rng = np.random.default_rng(1)
    pre = rng.normal(0, 0.02, 200)
    fwd = -0.5 * pre + rng.normal(0, 0.005, 200)                   # built-in reversal (b<0)
    ev = pd.DataFrame({"pre_drift": pre, "fwd_ret": fwd})
    reg = reversal_regression(ev)
    assert reg["beta"] < 0 and reg["t_beta"] < -2 and reg["n"] == 200


# --------------------------------------------------------------------------- #
# 4) non-overlapping events + plain t-stat
# --------------------------------------------------------------------------- #
def test_events_non_overlapping_and_plain_t():
    ti = pd.bdate_range("2018-01-01", "2024-12-31")
    exps = expiry_dates(ti.min(), ti.max(), ti)
    pos = [ti.get_loc(e) for e in exps]
    gaps = np.diff(pos)
    assert gaps.min() > HOLD + 1                                   # consecutive expiries don't overlap a 2d hold
    # plain t-stat: clear positive mean + small real variance -> significant
    rng = np.random.default_rng(2)
    x = pd.Series(0.01 + rng.normal(0.0, 0.002, 100))
    assert plain_tstat(x) > 5
    y = rng.normal(0.0, 1.0, 400)
    assert abs(plain_tstat(y)) < 3                                 # zero-mean noise -> small t
    assert plain_tstat(pd.Series([0.02] * 6)) == 0.0              # true sd=0 -> guarded to 0
    yb = by_year(pd.Series(y, index=pd.bdate_range("2020-01-01", periods=400)))
    assert all("mean_pct" in v for v in yb.values())
