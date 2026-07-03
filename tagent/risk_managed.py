"""Risk-managed momentum overlay — cut the bear-regime crash tail.

Plain KR long-only 12-1 momentum has a real but crash-prone edge (survivorship-
corrected Sharpe ~0.6, but -54% in bear regimes, 20% Monte-Carlo bust). Two
classic, causal overlays scale the book's exposure DOWN before the worst of it:

1. **Market-regime filter** — when the overall market (the point-in-time basket
   NAV) is below its long-term moving average (a confirmed downtrend), cut
   exposure toward cash; full exposure in uptrends. (Faber / time-series trend.)
2. **Volatility targeting** — scale exposure inversely to the strategy's own
   recent realized volatility, targeting a fixed annualized vol, **capped at 1x**
   (long-only, no leverage). Momentum crashes are preceded by vol spikes, so this
   de-risks into turbulence (Barroso & Santa-Clara, "Momentum has its moments").

NO-LOOKAHEAD is the whole game. The engine indexes returns as *forward* returns:
``net[t]`` and ``basket[t]`` are the t->t+1 return. So the exposure decision for
``net[t]`` may use information only through price-time t — which means the signal
series (basket NAV, realized vol) must be **lagged one bar** before they scale the
return. Every signal here is ``.shift(1)``-ed; appending future bars never changes
a past exposure (unit-tested).

Pure pandas/numpy; the runner (scripts/validate_kr_risk_managed.py) drives the
survivorship-corrected walk-forward OOS through it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tagent.report import perf_stats


@dataclass
class RiskOverlayConfig:
    regime_ma: int = 200            # market SMA window in trading days (~10 months)
    regime_off: float = 0.0         # exposure when market is below its MA (0 = to cash)
    vol_target: float = 0.15        # annualized target volatility (Barroso ~0.12-0.15)
    vol_window: int = 63            # realized-vol lookback (~3 months)
    vol_cap: float = 1.0            # max exposure — long-only, no leverage
    min_exposure: float = 0.0       # floor on exposure
    periods_per_year: int = 252


def regime_exposure(market_nav, cfg: RiskOverlayConfig | None = None) -> pd.Series:
    """1.0 in uptrends, ``regime_off`` when the market NAV is below its ``regime_ma``
    moving average. The NAV is lagged one bar (``shift(1)``) so the exposure for the
    forward return at t uses only the trend confirmed through price-time t — strictly
    no-lookahead. During the initial MA warmup (no trend yet) exposure is 1.0.
    """
    cfg = cfg or RiskOverlayConfig()
    nav = pd.Series(market_nav, dtype=float)
    nav_lag = nav.shift(1)                                   # info known at the bar's start
    sma = nav_lag.rolling(cfg.regime_ma, min_periods=cfg.regime_ma).mean()
    exp = pd.Series(1.0, index=nav.index)
    below = (nav_lag < sma) & sma.notna()
    exp[below] = cfg.regime_off
    return exp


def vol_target_exposure(net, cfg: RiskOverlayConfig | None = None) -> pd.Series:
    """Exposure = target_vol / recent_realized_vol, clipped to [min_exposure, vol_cap].

    Realized vol uses the strategy's own returns through the PREVIOUS bar
    (``.shift(1)``), so the scale for the forward return at t is known at t — no
    lookahead. Warmup (insufficient history) defaults to full (capped) exposure.
    """
    cfg = cfg or RiskOverlayConfig()
    net = pd.Series(net, dtype=float)
    ann = np.sqrt(cfg.periods_per_year)
    realized = net.rolling(cfg.vol_window, min_periods=max(5, cfg.vol_window // 3)).std() * ann
    realized_lag = realized.shift(1)                         # vol known before bar t
    raw = cfg.vol_target / realized_lag
    exp = raw.clip(lower=cfg.min_exposure, upper=cfg.vol_cap)
    return exp.fillna(cfg.vol_cap)


def combined_exposure(net, market_nav, cfg: RiskOverlayConfig | None = None) -> pd.Series:
    """Regime filter x vol target, aligned to ``net``'s index, clipped to the cap.
    Both legs are causal, so the product is causal."""
    cfg = cfg or RiskOverlayConfig()
    net = pd.Series(net, dtype=float)
    r = regime_exposure(market_nav, cfg).reindex(net.index).fillna(1.0)
    v = vol_target_exposure(net, cfg).reindex(net.index).fillna(cfg.vol_cap)
    return (r * v).clip(lower=cfg.min_exposure, upper=cfg.vol_cap)


def apply_overlay(net, market_nav, cfg: RiskOverlayConfig | None = None,
                  mode: str = "combined") -> dict:
    """Scale a (forward-indexed) net return series by the chosen causal overlay.

    ``mode`` is 'regime', 'vol', or 'combined'. Returns the exposure series, the
    scaled net returns, and perf stats for both scaled and raw — for a head-to-head.
    """
    cfg = cfg or RiskOverlayConfig()
    net = pd.Series(net, dtype=float)
    if mode == "regime":
        exp = regime_exposure(market_nav, cfg).reindex(net.index).fillna(1.0)
    elif mode == "vol":
        exp = vol_target_exposure(net, cfg).reindex(net.index).fillna(cfg.vol_cap)
    else:
        exp = combined_exposure(net, market_nav, cfg)
    scaled = (exp * net).reindex(net.index)
    return {
        "mode": mode, "exposure": exp, "net": scaled, "raw_net": net,
        "stats": perf_stats(scaled.dropna(), cfg.periods_per_year),
        "raw_stats": perf_stats(net.dropna(), cfg.periods_per_year),
        "avg_exposure": float(exp.mean()),
        "frac_in_cash": float((exp <= cfg.min_exposure + 1e-9).mean()),
    }
