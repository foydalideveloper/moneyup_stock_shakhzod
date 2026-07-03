"""Rule-based "best opportunity" conditions.

This is the file you edit most. Each rule reads the current SymbolState and
returns a Signal when it triggers. These are EXAMPLE rules, not proven winners:
a signal means "look now", never "this trade will win". Validate on paper first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from tagent.state import SymbolState


@dataclass
class StrategyParams:
    tight_spread_pct: float = 0.05    # spread below this % = liquid/cheap to enter
    near_low_pct: float = 0.30        # within this % of session low = "near support"
    breakout_pct: float = 0.20        # this % above the rolling high = breakout
    drop_from_high_pct: float = 1.00  # this % below session high = pullback


@dataclass
class Signal:
    symbol: str
    kind: str
    reason: str
    price: float
    side: str  # "buy" or "sell"


def evaluate(s: SymbolState, params: Optional[StrategyParams] = None,
             stale_after: float = 30.0, now: Optional[datetime] = None) -> List[Signal]:
    """Return all signals currently true for this symbol."""
    p = params or StrategyParams()
    out: List[Signal] = []

    # Need price + quote, and the feed must be fresh.
    if s.last_price <= 0 or not s.has_quote() or s.is_stale(stale_after, now):
        return out

    # Rule 1: possible entry — near session low AND easy/cheap to trade.
    if s.session_low > 0:
        dist_from_low = (s.last_price - s.session_low) / s.session_low * 100.0
        if 0 <= dist_from_low <= p.near_low_pct and s.spread_pct <= p.tight_spread_pct:
            out.append(Signal(
                s.symbol, "near_low_tight_spread",
                f"near session low ({dist_from_low:.2f}% above {s.session_low:.2f}), "
                f"tight spread {s.spread_pct:.3f}%",
                s.last_price, "buy"))

    # Rule 2: momentum — breaking above the PRIOR window high (excludes the
    # current price, so the comparison is meaningful).
    rhigh = s.prev_window_high
    if rhigh > 0 and s.last_price >= rhigh * (1 + p.breakout_pct / 100.0):
        out.append(Signal(
            s.symbol, "breakout",
            f"broke above {s.window_seconds}s high {rhigh:.2f} -> {s.last_price:.2f}",
            s.last_price, "buy"))

    # Rule 3: pullback — dropped meaningfully from the session high.
    if s.session_high > 0:
        drop = (s.session_high - s.last_price) / s.session_high * 100.0
        if drop >= p.drop_from_high_pct:
            out.append(Signal(
                s.symbol, "pullback_from_high",
                f"down {drop:.2f}% from session high {s.session_high:.2f}",
                s.last_price, "sell"))

    return out
