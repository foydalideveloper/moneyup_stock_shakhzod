"""Volatility-spike bounce — buy KR large-caps the day after a VOL SPIKE.

Risk-off bounce triggered by a volatility jump (e.g. VIX up >= X% in a US session)
rather than only a US price drop. On the next KR day, buy the equal-weight universe at
the open and sell at the close (or after ``hold`` days). Sweeps spike thresholds; the
runner reports event count, win rate, NET expectancy/trade and a t-stat on the
point-in-time, survivorship-corrected universe under conservative slippage, plus
whether it adds INDEPENDENT events beyond the US-price-drop signal.

No-lookahead: the vol change is the last US session that closed BEFORE the KR date
(reuses :func:`overnight_for_dates`); the trade opens at that KR open. Pure; the VIX
series is injected in tests.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tagent.data.intraday_history import overnight_for_dates
from tagent.intraday_backtest import CostModel
from tagent.strategies.us_leadlag_daily import _wide, summarize  # noqa: F401  (summarize re-exported)


def vol_spike_mask(kr_index, vol_change: pd.Series, threshold: float) -> pd.Series:
    """Boolean per KR date: did the last US session's vol measure JUMP by >= threshold
    (e.g. VIX +0.15 = +15%)? As-of, so strictly no-lookahead."""
    asof = overnight_for_dates(vol_change, kr_index)
    s = pd.Series([asof.get(pd.Timestamp(d).date(), float("nan")) for d in kr_index],
                  index=kr_index)
    return s >= threshold


def _hold_return(panel, hold: int) -> pd.DataFrame:
    """Per-name return from the entry open to the exit close: open[t] -> close[t+hold-1]
    (hold=1 = same-day intraday). Non-positive prices guarded."""
    opens, closes = _wide(panel, "open"), _wide(panel, "close")
    opens, closes = opens.where(opens > 0), closes.where(closes > 0)
    if hold <= 1:
        return closes / opens - 1.0
    return closes.shift(-(hold - 1)) / opens - 1.0


def event_stock_returns(panel, vol_change: pd.Series, threshold: float = 0.15,
                        hold: int = 1, membership: Optional[pd.DataFrame] = None) -> pd.Series:
    """PER-STOCK returns on vol-spike event days (one obs per stock-event)."""
    ret = _hold_return(panel, hold)
    if membership is not None:
        m = membership.reindex(index=ret.index, columns=ret.columns).fillna(False)
        ret = ret.where(m.to_numpy(dtype=bool))
    sub = ret[vol_spike_mask(ret.index, vol_change, threshold)]
    return sub.stack().dropna()


def event_dates(panel, vol_change: pd.Series, threshold: float = 0.15) -> pd.Index:
    """The KR dates that qualify as vol-spike events (for the independence check)."""
    idx = _wide(panel, "close").index
    mask = vol_spike_mask(idx, vol_change, threshold)
    return idx[mask.fillna(False)]


def leadlag_backtest(panel, vol_change: pd.Series, threshold: float = 0.15, hold: int = 1,
                     membership: Optional[pd.DataFrame] = None,
                     cost: Optional[CostModel] = None) -> dict:
    """Net-of-cost EW vol-spike bounce. Each event is one KR round trip."""
    cost = cost or CostModel()
    cf = cost.round_trip_frac()
    ret = _hold_return(panel, hold)
    if membership is not None:
        m = membership.reindex(index=ret.index, columns=ret.columns).fillna(False)
        ret = ret.where(m.to_numpy(dtype=bool))
    ew = ret.mean(axis=1)[vol_spike_mask(ret.index, vol_change, threshold)].dropna()
    return {"gross": ew, "net": ew - cf, "cost_frac": cf, "n_events": int(len(ew)),
            "threshold": threshold, "hold": hold}
