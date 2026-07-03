"""KR derivatives expiry-calendar effects — ONE pre-registered trial (Stoll-Whaley).

Externally anchored (Stoll-Whaley expiration price-reversal effects + KR program/index-
arbitrage literature). Pre-scheduled, NON-OVERLAPPING events => a clean plain t-stat (no
overlap problem). LOCKED spec, no variant search:

  * Calendar: KOSPI200 monthly OPTION expiry = 2nd Thursday of every month; quarterly
    FUTURES expiry ("witching") = 2nd Thursday of Mar/Jun/Sep/Dec. Each known in advance and
    snapped to the actual trading day on/before it (KRX moves a holiday expiry earlier).
  * Tradable pattern (PRIMARY): reversal of the pre-expiry arbitrage drift. At the expiry
    close, take the position OPPOSITE to the trailing ``PRE_WINDOW``-day index drift (the
    arbitrage build-up) and hold ``HOLD`` days. net per event = -sign(pre_drift) x
    (close[E+HOLD]/close[E] - 1) - cost. No-lookahead: the drift is known at the entry close.
  * Descriptive (SECONDARY): unconditional expiry-day mean return + the reversal regression
    (post-expiry return on pre-expiry drift; beta<0 = reversal).
  * Program-trading/arbitrage volumes: NOT available in this environment (not in pykrx;
    KRX MDC aggregates are zero here) -> calendar-only, as flagged by the runner.

Costs: futures ~0.05% round trip + a conservative 2x slippage stress. Pure numpy/pandas;
unit-tested on synthetic series (no network).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# pre-registered constants (LOCKED)
PRE_WINDOW = 5                   # trailing trading-day drift into expiry (one week)
HOLD = 2                         # trading days held after the expiry close
FUT_ROUND_TRIP = 0.0005           # futures ~0.05% round trip
WITCHING_MONTHS = (3, 6, 9, 12)  # quarterly futures expiry months


def nth_weekday(year: int, month: int, weekday: int = 3, n: int = 2) -> pd.Timestamp:
    """The ``n``-th ``weekday`` (Mon=0..Sun=6; Thu=3) of a month. KR expiry = 2nd Thursday."""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def expiry_dates(start, end, trading_index, quarterly_only: bool = False) -> List[pd.Timestamp]:
    """KOSPI200 expiry trading-days over [start, end]: the 2nd Thursday of each month
    (quarterly witching months only if ``quarterly_only``), snapped to the actual trading
    day on/before it (holiday-shifted earlier, per KRX)."""
    ti = pd.DatetimeIndex(trading_index)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for d in pd.date_range(start, end, freq="MS"):
        if quarterly_only and d.month not in WITCHING_MONTHS:
            continue
        thu = nth_weekday(d.year, d.month, weekday=3, n=2)
        pos = ti.searchsorted(thu, side="right") - 1       # last trading day <= 2nd Thursday
        if 0 <= pos < len(ti):
            day = ti[pos]
            if start <= day <= end:
                out.append(day)
    return sorted(set(out))


def event_returns(close, expiries, pre_window: int = PRE_WINDOW, hold: int = HOLD) -> pd.DataFrame:
    """Per-event windows around each expiry. Columns: [expiry, pre_drift, expiry_ret,
    fwd_ret], where pre_drift = close[E]/close[E-pre_window]-1 (known at the expiry close),
    expiry_ret = close[E]/close[E-1]-1, fwd_ret = close[E+hold]/close[E]-1 (the reversal
    window). Drops events whose windows fall off the data — strictly no-lookahead."""
    c = pd.Series(close, dtype=float).dropna()
    idx = c.index
    rows = []
    for E in expiries:
        if E not in idx:
            continue
        ei = idx.get_loc(E)
        if ei - pre_window < 0 or ei + hold >= len(idx):
            continue
        rows.append({
            "expiry": E,
            "pre_drift": float(c.iloc[ei] / c.iloc[ei - pre_window] - 1.0),
            "expiry_ret": float(c.iloc[ei] / c.iloc[ei - 1] - 1.0),
            "fwd_ret": float(c.iloc[ei + hold] / c.iloc[ei] - 1.0),
        })
    return pd.DataFrame(rows, columns=["expiry", "pre_drift", "expiry_ret", "fwd_ret"])


def reversal_net(events: pd.DataFrame, cost_round_trip: float = FUT_ROUND_TRIP) -> pd.Series:
    """Net per-event reversal return: -sign(pre_drift) x fwd_ret - one round trip, indexed
    by expiry date. Non-overlapping by construction (monthly events, short hold)."""
    if events.empty:
        return pd.Series(dtype=float)
    pos = -np.sign(events["pre_drift"].to_numpy())
    net = pos * events["fwd_ret"].to_numpy() - cost_round_trip
    return pd.Series(net, index=pd.DatetimeIndex(events["expiry"]))


def plain_tstat(x) -> float:
    """t-stat of the mean (mean / standard error). Valid because events are
    NON-OVERLAPPING (independent), so no HAC correction is needed."""
    a = np.asarray(pd.Series(x).dropna(), float)
    n = len(a)
    if n < 3:
        return 0.0
    sd = a.std(ddof=1)
    return float(a.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0


def reversal_regression(events: pd.DataFrame) -> dict:
    """OLS fwd_ret ~ a + b*pre_drift. b<0 (t<0) = post-expiry reversal of the pre-expiry
    drift; the descriptive Stoll-Whaley signature."""
    df = events[["pre_drift", "fwd_ret"]].dropna()
    if len(df) < 5:
        return {"beta": 0.0, "t_beta": 0.0, "n": len(df)}
    x = df["pre_drift"].to_numpy()
    y = df["fwd_ret"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    s2 = float(resid @ resid) / (len(x) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(np.diag(cov))[1])
    return {"beta": float(b[1]), "t_beta": (float(b[1] / se) if se > 0 else 0.0), "n": len(df)}


def by_year(net: pd.Series) -> dict:
    net = pd.Series(net).dropna()
    out = {}
    if net.empty or not isinstance(net.index, pd.DatetimeIndex):
        return out
    for y, seg in net.groupby(net.index.year):
        out[str(int(y))] = {"n": int(len(seg)), "mean_pct": float(seg.mean() * 100),
                            "total_pct": float(seg.sum() * 100)}
    return out
