"""Vectorized backtest with realistic costs.

A backtest that ignores costs is fiction, so spread/fees/slippage are subtracted
on every position change. Positions are acted on the NEXT bar (position.shift(1))
to avoid lookahead. Pure pandas/numpy — fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    net_returns: pd.Series
    stats: dict = field(default_factory=dict)


def position_from_proba(proba: pd.Series, threshold: float = 0.55) -> pd.Series:
    """Go long (1) when the model's probability exceeds the threshold, else flat."""
    return (proba >= threshold).astype(int)


def run_backtest(close: pd.Series, position: pd.Series, cost_bps: float = 5.0,
                 slippage_bps: float = 2.0, periods_per_year: int = 252) -> BacktestResult:
    close = close.astype(float)
    position = position.reindex(close.index).fillna(0.0).astype(float)

    ret = close.pct_change().fillna(0.0)
    acted = position.shift(1).fillna(0.0)          # enter on the next bar
    gross = acted * ret

    # Cost charged whenever the position changes (entering/exiting/flipping).
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps + slippage_bps) / 10_000.0

    net = gross - cost
    equity = (1.0 + net).cumprod()

    stats = _compute_stats(net, equity, position, acted, ret, periods_per_year)
    return BacktestResult(equity=equity, net_returns=net, stats=stats)


def _compute_stats(net: pd.Series, equity: pd.Series, position: pd.Series,
                   acted: pd.Series, ret: pd.Series, ppy: int) -> dict:
    n = len(net)
    final_equity = float(equity.iloc[-1]) if n else 1.0
    total_return = final_equity - 1.0

    ann_return = (final_equity ** (ppy / n) - 1.0) if n > 0 and final_equity > 0 else float("nan")
    std = float(net.std())
    sharpe = float(net.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if n else 0.0

    turnover = position.diff().abs().fillna(position.abs())
    n_trades = int((turnover > 0).sum())  # counts the initial entry too
    in_market = acted != 0
    bars_in_market = int(in_market.sum())
    hit_rate = float((net[in_market] > 0).mean()) if bars_in_market > 0 else float("nan")

    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_vol": std * np.sqrt(ppy),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "bars_in_market": bars_in_market,
        "hit_rate": hit_rate,
        "final_equity": final_equity,
    }
