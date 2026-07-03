"""US lead-lag (daily) — trade KR the day after a sharp US overnight drop.

When the US (SPY/QQQ) falls >= X% overnight, KR typically gaps down at the open. On
those event days we take an INTRADAY long in the equal-weight (point-in-time) KR
universe — buy at the open (after the US move is known), sell at the close — betting
on a partial bounce. The edge is measured per event day, net of one KR round trip.

No-lookahead: the US overnight return is the last US session that closed STRICTLY
before the KR date (reuses :func:`tagent.data.intraday_history.overnight_for_dates`),
so it is known at the KR open; the position is opened at that open and closed the same
day. Pure; unit-tested on mock daily bars.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tagent.data.intraday_history import overnight_for_dates
from tagent.intraday_backtest import CostModel


def _wide(panel, field: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(df[field], errors="coerce")
                         for c, df in panel.items()}).sort_index()


def event_day_returns(panel, spy_returns: pd.Series, threshold: float = -0.01,
                      membership: Optional[pd.DataFrame] = None) -> pd.Series:
    """Equal-weight KR intraday (open->close) return on days whose US overnight return
    is <= ``threshold`` (gross). Index = the qualifying event dates."""
    opens, closes = _wide(panel, "open"), _wide(panel, "close")
    opens, closes = opens.where(opens > 0), closes.where(closes > 0)   # drop bad (0) prices
    day_ret = closes / opens - 1.0                   # buy open, sell close (single day)
    if membership is not None:
        m = membership.reindex(index=day_ret.index, columns=day_ret.columns).fillna(False)
        day_ret = day_ret.where(m.to_numpy(dtype=bool))
    ew = day_ret.mean(axis=1)                          # EW across members
    overnight = overnight_for_dates(spy_returns, ew.index)   # {date: US overnight ret}
    onv = pd.Series([overnight.get(pd.Timestamp(d).date(), float("nan")) for d in ew.index],
                    index=ew.index)
    return ew[(onv <= threshold)].dropna()


def _event_mask(index, spy_returns, threshold) -> pd.Series:
    overnight = overnight_for_dates(spy_returns, index)
    onv = pd.Series([overnight.get(pd.Timestamp(d).date(), float("nan")) for d in index],
                    index=index)
    return onv <= threshold


def event_stock_returns(panel, spy_returns: pd.Series, threshold: float = -0.01,
                        membership: Optional[pd.DataFrame] = None) -> pd.Series:
    """PER-STOCK intraday (open->close) returns on event days — one observation per
    (stock, event-day), so the sample is N_stocks x larger than the EW series. Index
    is the event date (repeated per stock)."""
    opens, closes = _wide(panel, "open"), _wide(panel, "close")
    opens, closes = opens.where(opens > 0), closes.where(closes > 0)
    day_ret = closes / opens - 1.0
    if membership is not None:
        m = membership.reindex(index=day_ret.index, columns=day_ret.columns).fillna(False)
        day_ret = day_ret.where(m.to_numpy(dtype=bool))
    sub = day_ret[_event_mask(day_ret.index, spy_returns, threshold)]
    return sub.stack().dropna()                       # (date, stock) -> return


def summarize(returns: pd.Series, cost_frac: float = 0.0) -> dict:
    """n, win rate, gross + NET expectancy/observation, and a t-stat on the net mean
    (mean / standard error) — so 'is the sample meaningful?' is explicit."""
    r = pd.Series(returns).dropna()
    n = int(len(r))
    if n == 0:
        return {"n": 0, "win": 0.0, "gross_exp": 0.0, "net_exp": 0.0, "tstat": 0.0}
    net = r - cost_frac
    sd = float(net.std())
    tstat = float(net.mean() / (sd / (n ** 0.5))) if sd > 0 else 0.0
    return {"n": n, "win": float((net > 0).mean()), "gross_exp": float(r.mean()),
            "net_exp": float(net.mean()), "tstat": tstat}


def leadlag_backtest(panel, spy_returns: pd.Series, threshold: float = -0.01,
                     membership: Optional[pd.DataFrame] = None,
                     cost: Optional[CostModel] = None) -> dict:
    """Net-of-cost US-lead-lag event book. Each event day is one KR round trip."""
    cost = cost or CostModel()
    cf = cost.round_trip_frac()
    gross = event_day_returns(panel, spy_returns, threshold, membership)
    net = gross - cf
    return {"gross": gross, "net": net, "cost_frac": cf, "n_events": int(len(net)),
            "threshold": threshold}
