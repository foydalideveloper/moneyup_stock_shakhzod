"""Options variance-risk-premium (VRP) timing on KOSPI200 — ONE pre-registered trial.

Externally anchored (the variance risk premium predicts equity returns; high VRP =
volatility is richly priced -> higher expected index returns). LOCKED spec, ONE signal
only (VRP — NOT skew or put/call, which would be extra trials):

  * VRP = implied variance (VKOSPI/100)^2 minus the trailing REALIZED_WINDOW-day realized
    variance of KOSPI200 (annualized), matching VKOSPI's ~1-month implied horizon.
  * Standardize VRP over a 252-day window; exposure to KOSPI200 futures = LONG when the VRP
    z-score >= 0 (rich premium), else FLAT (long/flat, no shorting). WEEKLY rebalance (low
    turnover). No-lookahead: VRP known end-of-day t, position held over the following week.
  * Costs: futures ~0.05% round trip + conservative 2x stress (weekly = low turnover).
  * Metrics: net Sharpe/CAGR/maxDD, calendar-time Newey-West t, by-year. Verdict: real only
    if calendar NW t >= ~2.5-3 AND survives slippage AND stable by-year.

Pure numpy/pandas; unit-tested on synthetic series (no network). Data: tagent.data.vkospi_source.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from tagent.decomposition import PPY, _series_stats
from tagent.strategies.earnings_drift import newey_west_tstat

# pre-registered constants (LOCKED)
REALIZED_WINDOW = 21             # ~1-month realized variance (matches VKOSPI 30-day implied)
Z_WINDOW = 252                   # VRP standardization window
REBALANCE = 5                    # weekly (5 trading days)
Z_ENTER = 0.0                    # LONG when VRP z-score >= 0, else FLAT
FUT_ROUND_TRIP = 0.0005           # futures ~0.05% round trip


def implied_variance(vkospi) -> pd.Series:
    """Annualized implied variance from the VKOSPI level (percent vol)."""
    return (pd.Series(vkospi, dtype=float) / 100.0) ** 2


def realized_variance(idx_close, window: int = REALIZED_WINDOW, ppy: int = PPY) -> pd.Series:
    """Trailing annualized realized variance of the index (uses only past returns)."""
    r = pd.Series(idx_close, dtype=float).pct_change(fill_method=None)
    return r.rolling(window, min_periods=max(5, window // 2)).var() * ppy


def vrp_series(vkospi, idx_close, window: int = REALIZED_WINDOW) -> pd.Series:
    """VRP = implied variance - trailing realized variance, on the index's calendar."""
    iv = implied_variance(vkospi)
    rv = realized_variance(idx_close, window)
    iv = iv.reindex(rv.index).ffill()
    return iv - rv


def vrp_zscore(vrp, z_window: int = Z_WINDOW) -> pd.Series:
    v = pd.Series(vrp, dtype=float)
    mu = v.rolling(z_window, min_periods=max(20, z_window // 4)).mean()
    sd = v.rolling(z_window, min_periods=max(20, z_window // 4)).std()
    return (v - mu) / sd.replace(0.0, np.nan)


def _weekly_hold(sig: pd.Series, rebalance: int = REBALANCE) -> pd.Series:
    """Hold the signal constant between weekly rebalance bars (every ``rebalance`` days)."""
    held = sig.astype(float).copy()
    keep = (np.arange(len(held)) % rebalance) == 0
    held[~keep] = np.nan
    return held.ffill()


def vrp_exposure(vkospi, idx_close, z_enter: float = Z_ENTER, window: int = REALIZED_WINDOW,
                 z_window: int = Z_WINDOW, rebalance: int = REBALANCE) -> pd.Series:
    """The LAGGED long/flat position actually applied each day (0 or 1) — no-lookahead.
    Its mean is the strategy's TIME-IN-MARKET (long fraction)."""
    idx_close = pd.Series(idx_close, dtype=float)
    z = vrp_zscore(vrp_series(vkospi, idx_close, window), z_window)
    sig = (z >= z_enter).astype(float)
    return _weekly_hold(sig, rebalance).shift(1).fillna(0.0)


def vrp_backtest(vkospi, idx_close, z_enter: float = Z_ENTER, cost_round_trip: float = FUT_ROUND_TRIP,
                 window: int = REALIZED_WINDOW, z_window: int = Z_WINDOW,
                 rebalance: int = REBALANCE, ppy: int = PPY) -> pd.Series:
    """Net daily return of the VRP long/flat book. LONG when the (weekly-rebalanced) VRP
    z-score >= z_enter, else flat; position lagged one bar (no-lookahead); futures cost on
    turnover."""
    idx_close = pd.Series(idx_close, dtype=float)
    applied = vrp_exposure(vkospi, idx_close, z_enter, window, z_window, rebalance)
    ret = idx_close.pct_change(fill_method=None).reindex(applied.index)
    turn = applied.diff().abs().fillna(applied.abs())
    cost = turn * (cost_round_trip / 2.0)
    return (applied * ret - cost).dropna()


def calendar_time_t(net, lag: int = REBALANCE) -> Tuple[float, int]:
    e = pd.Series(net).dropna()
    return newey_west_tstat(e, lag=lag), int(len(e))


def by_year(net, ppy: int = PPY) -> dict:
    net = pd.Series(net).dropna()
    out = {}
    if net.empty or not isinstance(net.index, pd.DatetimeIndex):
        return out
    for y, seg in net.groupby(net.index.year):
        st = _series_stats(seg, ppy)
        out[str(int(y))] = {"sharpe": st["sharpe"], "total_return": st["total_return"], "n": st["n"]}
    return out
