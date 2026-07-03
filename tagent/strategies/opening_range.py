"""Hypothesis B — opening-range breakout (long).

Define the opening range as the high/low of the first ``open_minutes`` bars. After
that window, BUY the first bar whose HIGH breaks above the opening-range high
(optionally requiring the breakout bar's volume to exceed a multiple of the
opening-range average volume). Exit on +Y take / -Z stop / market close.

Pure + no-lookahead: the opening range uses only the first R bars, the breakout is
detected bar-by-bar, the fill is at the breakout level (a realistic stop-buy), and
the exit is simulated forward by the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from tagent.intraday_backtest import DayPlan, Strategy


@dataclass(frozen=True)
class OpeningRangeParams:
    open_minutes: int = 30           # R: opening-range window length (bars)
    take_pct: float = 0.01           # Y take-profit
    stop_pct: float = 0.01           # Z stop-loss
    vol_mult: float = 0.0            # 0 = no volume filter; else breakout vol >= mult * OR avg vol
    max_bars: int = 0                # 0 = scan to close; else stop looking this many bars after R


def opening_range(day_df, open_minutes: int) -> Tuple[float, float]:
    """(high, low) of the first ``open_minutes`` bars."""
    window = day_df.iloc[:open_minutes]
    return float(window["high"].max()), float(window["low"].min())


def make_strategy(params: OpeningRangeParams) -> Strategy:
    def strategy(day_df, ctx):
        n = len(day_df)
        r = params.open_minutes
        if n <= r + 1:
            return None
        highs = day_df["high"].to_numpy(float)
        opens = day_df["open"].to_numpy(float)
        vols = day_df["volume"].to_numpy(float)
        or_hi = highs[:r].max()
        or_vol = vols[:r].mean() if params.vol_mult else 0.0
        last = n if params.max_bars <= 0 else min(n, r + params.max_bars)
        for j in range(r, last):
            if highs[j] > or_hi:                          # breakout above the opening range
                if params.vol_mult and not (vols[j] >= params.vol_mult * or_vol):
                    continue                              # volume confirmation failed
                entry_price = max(or_hi, opens[j])        # fill at the breakout level (stop-buy)
                return DayPlan(entry_index=j, entry_price=entry_price,
                               take_pct=params.take_pct, stop_pct=params.stop_pct)
        return None
    return strategy


def default_grid() -> List[OpeningRangeParams]:
    grid: List[OpeningRangeParams] = []
    for r in (15, 30, 60):
        for y in (0.005, 0.01, 0.015):
            for z in (0.005, 0.01):
                grid.append(OpeningRangeParams(open_minutes=r, take_pct=y, stop_pct=z))
    return grid
