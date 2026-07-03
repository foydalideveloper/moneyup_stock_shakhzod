"""Hypothesis A — overnight-overreaction reversal (gap-down BUY).

Idea: when a stock GAPS DOWN at the open beyond X% AND that move is driven by a
negative US overnight return (systemic, not idiosyncratic), the open is an
overreaction that tends to partially reverse intraday. So BUY shortly after the
open, then exit on a +Y% take-profit, a -Z% stop, or the market close.

The "driven by the overnight" condition (require the US overnight return <=
``overnight_thresh``) is what separates a systemic gap from idiosyncratic bad news
— we only fade gaps the whole market gapped into.

Pure + no-lookahead: the decision uses bar 0's open, the prior close, and the
(pre-open) overnight return only; entry executes at a later bar's open and the exit
is simulated forward by the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tagent.intraday_backtest import DayPlan, Strategy


@dataclass(frozen=True)
class GapReversalParams:
    gap_pct: float = 0.02            # X: require open <= prev_close * (1 - X)  (gap DOWN)
    take_pct: float = 0.01           # Y: take-profit above entry
    stop_pct: float = 0.01           # Z: stop-loss below entry
    entry_minute: int = 5            # buy this many bars after the open
    overnight_thresh: float = -0.005  # require US overnight return <= this (systemic down)


def make_strategy(params: GapReversalParams) -> Strategy:
    """Build a (day_df, ctx) -> Optional[DayPlan] callable for these parameters."""
    def strategy(day_df, ctx):
        if len(day_df) <= params.entry_minute + 1:
            return None
        prev_close = ctx.get("prev_close")
        if not prev_close or prev_close <= 0:
            return None                                  # first day / no prior close
        overnight = ctx.get("overnight_return")
        if overnight is None or overnight > params.overnight_thresh:
            return None                                  # not a systemic overnight-down day
        open_px = float(day_df["open"].iloc[0])
        gap = open_px / prev_close - 1.0
        if gap > -params.gap_pct:
            return None                                  # not a big enough gap DOWN
        entry_price = float(day_df["open"].iloc[params.entry_minute])
        return DayPlan(entry_index=params.entry_minute, entry_price=entry_price,
                       take_pct=params.take_pct, stop_pct=params.stop_pct)
    return strategy


def default_grid() -> List[GapReversalParams]:
    """A small X/Y/Z sweep for the walk-forward parameter selection."""
    grid: List[GapReversalParams] = []
    for x in (0.015, 0.02, 0.03):
        for y in (0.005, 0.01, 0.015):
            for z in (0.005, 0.01):
                grid.append(GapReversalParams(gap_pct=x, take_pct=y, stop_pct=z))
    return grid
