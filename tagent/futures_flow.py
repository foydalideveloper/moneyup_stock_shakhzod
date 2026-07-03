"""Foreign-futures-flow timing on KOSPI200 futures — ONE pre-registered trial.

Externally anchored (foreign derivatives flow is a watched KR signal). LOCKED design, no
variant search:

  * Signal: z-score of trailing 5-day FOREIGN net KOSPI200-futures flow (net-buy value),
    standardized over a 252-day window. Position NEXT day: LONG if z >= +Z_ENTER, SHORT if
    z <= -Z_ENTER, else FLAT. Institutional flow is a pre-registered SECONDARY.
  * No-lookahead: flow is end-of-day; the position is set for the next session and earns
    that session's futures return (position lagged one bar).
  * Costs: futures ~0.05% round trip + roll; signal flips often (~5bps/day break-even), so
    a CONSERVATIVE 2x slippage stress is the decider. Reported at 0.05% and 2x.
  * Metrics: net Sharpe/CAGR/maxDD, calendar-time Newey-West t, by-year, and a DECAY check
    (early-half vs late-half — the strongest published results are old and it's watched).

Pure numpy/pandas; unit-tested on synthetic series (no network). The data source is
tagent.data.krx_deriv_investor (KRX MDC, futures investor net flow).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from tagent.decomposition import PPY, _series_stats
from tagent.strategies.earnings_drift import newey_west_tstat

# pre-registered constants (LOCKED)
FLOW_LOOKBACK = 5                 # trailing 5-day flow sum
Z_WINDOW = 252                    # standardization window (~1yr)
Z_ENTER = 1.0                     # |z| >= 1 std -> take the position, else flat
FUT_ROUND_TRIP = 0.0005           # futures ~0.05% round trip
ROLL_ANNUAL = 0.0012             # ~12 bps/yr roll


def flow_zscore(flow, lookback: int = FLOW_LOOKBACK, z_window: int = Z_WINDOW) -> pd.Series:
    """z-score of the trailing ``lookback``-day flow sum, standardized over ``z_window``.
    Uses only data through t (no-lookahead)."""
    f = pd.Series(flow, dtype=float)
    s = f.rolling(lookback, min_periods=lookback).sum()
    mu = s.rolling(z_window, min_periods=max(20, z_window // 4)).mean()
    sd = s.rolling(z_window, min_periods=max(20, z_window // 4)).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def flow_signal(flow, z_enter: float = Z_ENTER, lookback: int = FLOW_LOOKBACK,
                z_window: int = Z_WINDOW) -> pd.Series:
    """Long(+1)/flat(0)/short(-1) from the flow z-score, known at end-of-day t."""
    z = flow_zscore(flow, lookback, z_window)
    pos = pd.Series(0.0, index=z.index)
    pos[z >= z_enter] = 1.0
    pos[z <= -z_enter] = -1.0
    return pos


def flow_backtest(flow, fut_close, z_enter: float = Z_ENTER, cost_round_trip: float = FUT_ROUND_TRIP,
                  roll_annual: float = ROLL_ANNUAL, lookback: int = FLOW_LOOKBACK,
                  z_window: int = Z_WINDOW, ppy: int = PPY) -> pd.Series:
    """Net daily return of the flow-timing book: position (lagged one bar, no-lookahead)
    earns the futures daily return, minus turnover cost (|Δpos| x one-way) + roll while held."""
    z = flow_zscore(flow, lookback, z_window)
    sig = pd.Series(0.0, index=z.index)
    sig[z >= z_enter] = 1.0
    sig[z <= -z_enter] = -1.0
    ret = pd.Series(fut_close, dtype=float).pct_change(fill_method=None)
    idx = ret.index.union(sig.index)
    sig = sig.reindex(idx).fillna(0.0)
    ret = ret.reindex(idx)
    applied = sig.shift(1).fillna(0.0)                    # EOD signal -> next session (no-lookahead)
    turn = applied.diff().abs().fillna(applied.abs())
    cost = turn * (cost_round_trip / 2.0) + applied.abs() * (roll_annual / ppy)
    return (applied * ret - cost).reindex(ret.dropna().index).dropna()


def calendar_time_t(net, lag: int = 5) -> Tuple[float, int]:
    e = pd.Series(net).dropna()
    return newey_west_tstat(e, lag=lag), int(len(e))


def by_year(net, ppy: int = PPY) -> dict:
    net = pd.Series(net).dropna()
    out = {}
    if net.empty or not isinstance(net.index, pd.DatetimeIndex):
        return out
    for y, seg in net.groupby(net.index.year):
        st = _series_stats(seg, ppy)
        out[str(int(y))] = {"sharpe": st["sharpe"], "total_return": st["total_return"], "n": st["n"]}
    return out


def decay_split(net, ppy: int = PPY) -> dict:
    """Early-half vs late-half Sharpe (is the edge concentrated in the old sample?)."""
    net = pd.Series(net).dropna()
    n = len(net)
    if n < 60:
        return {"early": _series_stats(net, ppy), "late": _series_stats(net.iloc[:0], ppy), "mid": None}
    mid = net.index[n // 2]
    return {"mid": str(pd.Timestamp(mid).date()),
            "early": _series_stats(net.iloc[:n // 2], ppy),
            "late": _series_stats(net.iloc[n // 2:], ppy)}


def is_degenerate(flow) -> bool:
    """True if the flow series is empty or all-zero/all-NaN (no tradable signal) — used to
    report a data wall honestly instead of a fake verdict."""
    f = pd.Series(flow, dtype=float).dropna()
    return len(f) == 0 or bool((f == 0).all())
