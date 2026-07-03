"""Standard CTA-style managed-futures trend ensemble — ONE pre-registered trial.

Every element is literature-standard (Moskowitz-Ooi-Pedersen time-series momentum + the
classic managed-futures vol-targeting stack); NOTHING is fitted, and the SAME rule/params
apply to every market (no per-market or per-lookback tuning):

  * Markets (all futures/perps — no cash-tax trap): KOSPI-200, S&P 500, crypto (BTC+ETH
    as one sleeve), USD/KRW.
  * Signal: blend THREE pre-specified trend lookbacks — 1/3/12-month time-series momentum
    (sign of the trailing 21/63/252-day return), equal-weighted -> a position in [-1,1].
    Next-bar lagged (decide at close t, earn t->t+1). Long AND short (it's a futures CTA).
  * Vol-target each sleeve to equal risk (position = signal x target_vol/realized_vol);
    combine via inverse-trailing-vol risk parity; scale the whole book to a fixed
    portfolio vol target. Monthly rebalance. Strictly no-lookahead.

Reported as ONE trial vs the single-lookback binary core (multi_market_trend), with
with/without-crypto and with/without-USD/KRW robustness so no single sleeve carries it.

Pure numpy/pandas; unit-tested on synthetic series (no network).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tagent.decomposition import PPY, _series_stats
from tagent.index_calibration import load_index_close
from tagent.multi_market_trend import risk_parity_weights
from tagent.trend_core import by_decade

# pre-registered constants (LOCKED — literature-standard, nothing fitted)
LOOKBACKS = (21, 63, 252)        # 1/3/12-month time-series momentum (MOP 2012)
VOL_WINDOW = 63                  # ~3-month realized-vol lookback
SLEEVE_TARGET_VOL = 0.15         # per-sleeve risk target (cancels in the final scaling)
PORT_TARGET_VOL = 0.10           # fixed portfolio vol target (standard CTA ~10%)
SLEEVE_POS_CAP = 2.0             # cap per-sleeve vol-target leverage (low-vol blowup guard)
SCALAR_CAP = 3.0                 # cap portfolio vol-scaling leverage
WEIGHT_CAP = 0.60                # max single-sleeve risk-parity weight

# realistic per-kind costs: (one-way switch cost, annual roll/funding drag on |position|)
COSTS = {"index": (0.00025, 0.0012), "crypto": (0.0005, 0.10), "fx": (0.0001, 0.0)}

MARKETS: Dict[str, dict] = {
    "KOSPI200": {"files": ["kospi200_long"], "kind": "index"},
    "S&P500": {"files": ["gspc"], "kind": "index"},
    "crypto": {"files": ["btc", "eth"], "kind": "crypto"},
    "USDKRW": {"files": ["usdkrw"], "kind": "fx"},
}


def tsmom_signal(close, lookbacks=LOOKBACKS) -> pd.Series:
    """Blended time-series-momentum signal in [-1,1]: equal-weight mean of sign(trailing
    L-day return) over ``lookbacks``. Uses only data through t (no-lookahead)."""
    close = pd.Series(close, dtype=float).dropna()
    sigs = [np.sign(close / close.shift(L) - 1.0) for L in lookbacks]
    return pd.concat(sigs, axis=1).mean(axis=1)


def vol_targeted_sleeve(close, switch_cost: float, roll_annual: float,
                        sleeve_target: float = SLEEVE_TARGET_VOL, vol_window: int = VOL_WINDOW,
                        pos_cap: float = SLEEVE_POS_CAP, ppy: int = PPY) -> pd.Series:
    """One market's vol-targeted TSMOM sleeve return (net), forward-indexed (t->t+1).

    position[t] = blended_signal[t] x clip(target_vol / realized_vol[t], -cap, cap), both
    known at close t; earns the t->t+1 return. Cost = |Δposition| x switch + |position| x
    roll/ppy (turnover + carry). Strictly no-lookahead."""
    close = pd.Series(close, dtype=float).dropna()
    sig = tsmom_signal(close)
    ret = close.pct_change(fill_method=None)
    rvol = ret.rolling(vol_window, min_periods=max(5, vol_window // 2)).std() * np.sqrt(ppy)
    scale = (sleeve_target / rvol).clip(upper=pos_cap)
    pos = (sig * scale)
    fwd = ret.shift(-1)
    turn = pos.diff()
    if len(turn):
        turn.iloc[0] = pos.iloc[0]
    cost = turn.abs() * switch_cost + pos.abs() * (roll_annual / ppy)
    return (pos * fwd - cost).dropna()


def market_sleeve(cfg: dict) -> pd.Series:
    """Sleeve return for a market config (crypto = equal-weight of its BTC+ETH sub-sleeves)."""
    sw, roll = COSTS[cfg["kind"]]
    subs = []
    for f in cfg["files"]:
        c = load_index_close(f).dropna()
        if not c.empty:
            subs.append(vol_targeted_sleeve(c, sw, roll))
    if not subs:
        return pd.Series(dtype=float)
    return pd.concat(subs, axis=1).mean(axis=1).dropna() if len(subs) > 1 else subs[0]


def load_sleeves(markets: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    markets = markets or MARKETS
    return pd.DataFrame({name: market_sleeve(cfg) for name, cfg in markets.items()}).sort_index()


def portfolio_vol_scale(combined_raw, port_target: float = PORT_TARGET_VOL,
                        window: int = VOL_WINDOW, cap: float = SCALAR_CAP, ppy: int = PPY) -> pd.Series:
    """Scale the book to a fixed portfolio vol target: scalar[t] = port_target /
    trailing_realized_vol, LAGGED one bar (uses only returns realised by t), capped."""
    r = pd.Series(combined_raw, dtype=float)
    rvol = r.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(ppy)
    return (port_target / rvol).shift(1).clip(upper=cap).fillna(0.0)


def cta_combined(sleeves: pd.DataFrame, markets: List[str], vol_window: int = VOL_WINDOW,
                 rebalance: str = "ME", cap: float = WEIGHT_CAP,
                 port_target: float = PORT_TARGET_VOL, scalar_cap: float = SCALAR_CAP) -> pd.Series:
    """Risk-parity-combined, portfolio-vol-targeted CTA book over the common sample."""
    sub = sleeves[markets].dropna(how="any")
    if sub.empty:
        return pd.Series(dtype=float)
    w = risk_parity_weights(sub, vol_window, rebalance, cap)
    raw = (w * sub).sum(axis=1).reindex(sub.index)
    scale = portfolio_vol_scale(raw, port_target, vol_window, scalar_cap)
    return (scale * raw).dropna()


def ensemble_report(sleeves: pd.DataFrame, markets: List[str], ppy: int = PPY, **kw) -> dict:
    comb = cta_combined(sleeves, markets, **kw)
    sub = sleeves[markets].dropna(how="any")
    return {"markets": markets, "n_common": int(len(sub)),
            "start": (str(sub.index.min().date()) if len(sub) else None),
            "end": (str(sub.index.max().date()) if len(sub) else None),
            "combined": _series_stats(comb, ppy), "by_decade": by_decade(comb, ppy),
            "combined_net": comb,
            "per_sleeve": {m: _series_stats(sleeves[m].dropna(), ppy) for m in markets}}
