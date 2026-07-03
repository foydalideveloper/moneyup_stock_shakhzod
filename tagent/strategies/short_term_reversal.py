"""Short-term reversal — long the recent LOSERS (oversold), hold N days.

Rank the universe by trailing ``lookback``-day return and LONG the biggest losers
(the signal is the NEGATIVE trailing return, so the worst performer ranks highest),
equal-weight the top quantile, hold ``hold`` days. Reuses the xs_momentum engine via
its ``signal=`` injection, so point-in-time membership, KR costs, and the rebalance
hold are all handled identically to the momentum study.

No-lookahead: the trailing return at t uses only past closes; the position earns the
next bar; the engine rebalances only every ``hold`` bars.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import pandas as pd

from tagent.stock_momentum import classic_config
from tagent.xs_momentum import XSMomConfig, align_close, backtest, momentum_signal


def reversal_signal(close_wide: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Higher = bigger loser = more oversold. Signal = -(trailing lookback return)."""
    return -momentum_signal(close_wide, lookback, 0)


def reversal_backtest(panel, lookback: int = 3, hold: int = 3, top_q: float = 0.2,
                      membership: Optional[pd.DataFrame] = None,
                      cfg: Optional[XSMomConfig] = None, periods_per_year: int = 252) -> dict:
    """Backtest long-only short-term reversal (long losers) net of KR costs."""
    close = align_close(panel)
    sig = reversal_signal(close, lookback)
    cfg = cfg or classic_config("kr", allow_short=False)
    cfg = replace(cfg, rebalance=max(1, int(hold)), top_q=top_q)
    return backtest(panel, cfg, periods_per_year, membership=membership, signal=sig)
