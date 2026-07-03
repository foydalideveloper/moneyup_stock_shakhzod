"""Flows-following (수급) — long names with the strongest recent foreign+institutional
net buying, rebalanced every few days.

Signal = trailing net-buy INTENSITY: the trailing ``lookback``-day sum of (foreign +
institutional) net buys, normalized by the name's own typical net-buy magnitude (so
it is "unusually strong buying for THIS name", not just "biggest KRW flow = megacap"),
then cross-sectionally z-scored. Higher = more bought = long the top quantile. Reuses
the xs_momentum engine via ``signal=`` (point-in-time membership, KR costs, hold).

NO-LOOKAHEAD: 수급 for day t is only known at/after t's close (and KRX can report with
a lag), so the signal is shifted by ``lag`` days — the rank at t uses net-buy through
t-lag only. Pure; flows are loaded from cached CSVs (no network here).
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from tagent.data.flows_source import load_flows
from tagent.strategies.factor_utils import xs_zscore


def net_buy_panel(symbols: Sequence[str], data_dir=None) -> pd.DataFrame:
    """Wide [date x symbol] daily (foreign_net + inst_net) net-buy value, for the
    symbols whose ``<sym>_flows.csv`` exists."""
    cols = {}
    for s in symbols:
        try:
            df = load_flows(s, data_dir=data_dir)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        f = pd.to_numeric(df.get("foreign_net", 0.0), errors="coerce").fillna(0.0)
        i = pd.to_numeric(df.get("inst_net", 0.0), errors="coerce").fillna(0.0)
        cols[str(s)] = f + i
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


def flows_signal(net_buy: pd.DataFrame, lookback: int = 5, lag: int = 1,
                 norm_window: int = 60) -> pd.DataFrame:
    """Cross-sectional net-buy intensity signal (higher = stronger recent buying),
    lagged ``lag`` days for no-lookahead."""
    trail = net_buy.rolling(lookback, min_periods=1).sum().shift(lag)
    scale = net_buy.abs().rolling(norm_window, min_periods=10).mean().shift(lag)
    intensity = trail / scale.where(scale > 0)            # net buy relative to own normal
    return xs_zscore(intensity)


def flows_backtest(panel, symbols=None, lookback: int = 5, hold: int = 5, top_q: float = 0.2,
                   lag: int = 1, membership: Optional[pd.DataFrame] = None,
                   cfg=None, periods_per_year: int = 252, data_dir=None) -> dict:
    """Backtest long-only flows-following net of KR costs (reuses the engine)."""
    from dataclasses import replace
    from tagent.stock_momentum import classic_config
    from tagent.xs_momentum import align_close, backtest
    syms = list(symbols) if symbols is not None else list(panel)
    nb = net_buy_panel(syms, data_dir=data_dir)
    if nb.empty:
        return backtest(panel, cfg or classic_config("kr", allow_short=False), periods_per_year)
    sig = flows_signal(nb, lookback=lookback, lag=lag).reindex(columns=list(align_close(panel).columns))
    cfg = cfg or classic_config("kr", allow_short=False)
    cfg = replace(cfg, rebalance=max(1, int(hold)), top_q=top_q)
    return backtest(panel, cfg, periods_per_year, membership=membership, signal=sig)
