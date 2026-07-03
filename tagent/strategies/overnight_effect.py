"""Overnight effect — buy at the close, sell at the next open (close->open return).

The documented 'overnight anomaly' is that close->open returns differ systematically
from intraday. Here we hold the equal-weight (point-in-time) universe overnight EVERY
day: buy all members at the close, sell at the next open. That is a full round trip
each day, so the daily close->open return must clear the round-trip KR cost to be a
net edge.

Pure + no-lookahead: the position is committed at the close of bar t (membership known
at t); the return ``open[t+1]/close[t]-1`` is realised at the next open. ``requests``-
free; unit-tested on mock daily bars.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tagent.intraday_backtest import CostModel


def _wide(panel, field: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(df[field], errors="coerce")
                         for c, df in panel.items()}).sort_index()


def overnight_gross(panel, membership: Optional[pd.DataFrame] = None) -> pd.Series:
    """Equal-weight daily close->open return across eligible members (gross)."""
    opens, closes = _wide(panel, "open"), _wide(panel, "close")
    opens, closes = opens.where(opens > 0), closes.where(closes > 0)   # drop bad (0) prices
    onr = opens.shift(-1) / closes - 1.0            # [t] = close[t] -> open[t+1]
    if membership is not None:
        m = membership.reindex(index=onr.index, columns=onr.columns).fillna(False)
        onr = onr.where(m.to_numpy(dtype=bool))
    gross = onr.mean(axis=1)                          # EW across members (NaN-safe)
    return gross.iloc[:-1].dropna()                  # last day has no next open


def overnight_backtest(panel, membership: Optional[pd.DataFrame] = None,
                       cost: Optional[CostModel] = None) -> dict:
    """Net-of-cost overnight book. One round trip per day, so ``cost.round_trip_frac``
    is charged on every held day."""
    cost = cost or CostModel()
    cf = cost.round_trip_frac()
    gross = overnight_gross(panel, membership)
    net = gross - cf
    return {"gross": gross, "net": net, "cost_frac": cf, "n_days": int(len(net))}
