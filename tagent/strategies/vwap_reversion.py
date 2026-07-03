"""Hypothesis C — VWAP reversion (long).

Compute the intraday VWAP cumulatively from the day's bars. When price deviates
BELOW VWAP by ``dev_pct`` (oversold), BUY expecting a reversion back toward VWAP.
Exit on a VWAP touch, a +Y take, a -Z stop, or the close.

The VWAP-touch exit is expressed through the backtester's take-profit: the target
is set to the distance from the entry up to the current VWAP (capped at Y), so the
position closes when price reverts to VWAP (or hits the Y cap / Z stop / close
first) — no engine change needed.

Pure + no-lookahead: VWAP at bar t uses only bars 0..t (cumulative within the day);
the entry is taken at the first oversold bar and the exit is simulated forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from tagent.intraday_backtest import DayPlan, Strategy


@dataclass(frozen=True)
class VwapReversionParams:
    dev_pct: float = 0.01            # D: how far below VWAP triggers a BUY
    take_pct: float = 0.015          # Y: cap on the take-profit (VWAP target usually nearer)
    stop_pct: float = 0.01           # Z: stop-loss
    warmup: int = 5                  # require this many bars before trusting VWAP


def intraday_vwap(day_df) -> np.ndarray:
    """Cumulative intraday VWAP per bar (typical price = (H+L+C)/3, volume-weighted).
    Value at bar t uses only bars 0..t — strictly no-lookahead."""
    h = day_df["high"].to_numpy(float)
    l = day_df["low"].to_numpy(float)
    c = day_df["close"].to_numpy(float)
    v = day_df["volume"].to_numpy(float)
    tp = (h + l + c) / 3.0
    cum_pv = np.cumsum(tp * v)
    cum_v = np.cumsum(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cum_v > 0, cum_pv / cum_v, c)     # fall back to close if no volume yet
    return vwap


def make_strategy(params: VwapReversionParams) -> Strategy:
    def strategy(day_df, ctx):
        n = len(day_df)
        if n <= params.warmup + 1:
            return None
        c = day_df["close"].to_numpy(float)
        vwap = intraday_vwap(day_df)
        for t in range(params.warmup, n):
            if vwap[t] > 0 and (c[t] / vwap[t] - 1.0) <= -params.dev_pct:
                entry = float(c[t])
                to_vwap = vwap[t] / entry - 1.0           # reversion target (>0 since below VWAP)
                take = min(params.take_pct, to_vwap) if to_vwap > 0 else params.take_pct
                return DayPlan(entry_index=t, entry_price=entry,
                               take_pct=take, stop_pct=params.stop_pct)
        return None
    return strategy


def default_grid() -> List[VwapReversionParams]:
    grid: List[VwapReversionParams] = []
    for d in (0.005, 0.01, 0.015):
        for y in (0.01, 0.015, 0.02):
            for z in (0.005, 0.01):
                grid.append(VwapReversionParams(dev_pct=d, take_pct=y, stop_pct=z))
    return grid
