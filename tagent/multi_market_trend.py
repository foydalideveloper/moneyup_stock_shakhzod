"""Track C — multi-market trend extension: the pre-registered Boxpi insurance.

Apply the IDENTICAL locked trend rule from Track A (binary 200d-MA, next-bar lagged
execution — `trend_core.binary_trend_net`) to several free-daily-data markets with NO
per-market re-optimisation: KOSPI-200, S&P 500, BTC, ETH. Same rule everywhere (that is
the point — no new research risk; only realistic per-market COSTS differ). Then combine
into a vol-targeted risk-parity book: inverse-trailing-vol weights (capped, rebalanced
monthly on trailing VOL, never trailing return), strictly no-lookahead.

The thesis under test: diversified trend lifts Sharpe toward 0.7-1.0 and rides through
the 2011-16 KOSPI "Boxpi" range (because S&P/crypto trended then) — or it does not.
Reported with AND without crypto so a short-history high-vol sleeve can't carry the claim.

Pure numpy/pandas; unit-tested on synthetic series (no network).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tagent.decomposition import _series_stats, PPY
from tagent.index_calibration import load_index_close
from tagent.trend_core import binary_trend_net, by_decade

# pre-registered constants (LOCKED — no per-market tuning)
REGIME_MA = 200
VOL_WINDOW = 63                  # trailing realized-vol lookback for risk parity
WEIGHT_CAP = 0.60                # max single-sleeve weight (diversification floor)
# realistic per-market costs (one-way switch + annual roll/funding) — NOT optimized
INDEX_SWITCH, INDEX_ROLL = 0.00025, 0.0012      # futures ~0.05% round trip + ~12bps/yr roll
CRYPTO_SWITCH, CRYPTO_ROLL = 0.0005, 0.10       # ~0.1% taker round trip + ~10%/yr perp funding

MARKETS: Dict[str, dict] = {
    "KOSPI200": {"file": "kospi200_long", "kind": "index"},
    "S&P500": {"file": "gspc", "kind": "index"},
    "BTC": {"file": "btc", "kind": "crypto"},
    "ETH": {"file": "eth", "kind": "crypto"},
}


def market_sleeve(close, kind: str) -> pd.Series:
    """The IDENTICAL locked trend rule applied to one market's close, with that market's
    realistic costs. Same code path (`binary_trend_net`, 200d-MA, next-bar) for all."""
    if kind == "crypto":
        sw, roll = CRYPTO_SWITCH, CRYPTO_ROLL
    else:
        sw, roll = INDEX_SWITCH, INDEX_ROLL
    return binary_trend_net(close, regime_ma=REGIME_MA, mode="next_bar",
                            roll_annual=roll, switch_cost=sw)


def load_sleeves(markets: Optional[Dict[str, dict]] = None, data_dir=None) -> pd.DataFrame:
    """[date x market] daily net trend-sleeve returns (each on its own history, outer-joined)."""
    markets = markets or MARKETS
    cols = {}
    for name, cfg in markets.items():
        close = load_index_close(cfg["file"], data_dir=data_dir).dropna()
        if close.empty:
            continue
        cols[name] = market_sleeve(close, cfg["kind"])
    return pd.DataFrame(cols).sort_index()


def risk_parity_weights(sleeves: pd.DataFrame, vol_window: int = VOL_WINDOW,
                        rebalance: str = "ME", cap: float = WEIGHT_CAP) -> pd.DataFrame:
    """Inverse-trailing-vol (equal-risk) weights, capped + renormalised, held between
    monthly/quarterly rebalances. NO-LOOKAHEAD: vol is trailing and LAGGED one bar, and
    weights are recomputed only at each ``rebalance`` boundary from data through it."""
    vol = sleeves.rolling(vol_window, min_periods=max(5, vol_window // 2)).std().shift(1)
    inv = (1.0 / vol.replace(0.0, np.nan))
    w = inv.div(inv.sum(axis=1), axis=0)
    if cap and cap < 1.0:
        for _ in range(sleeves.shape[1]):                    # hold over-weights at cap, give
            over = w > cap                                   # the remainder to under-cap sleeves
            if not bool(over.to_numpy().any()):
                break
            w = w.mask(over, cap)
            capped = w.where(over, 0.0).sum(axis=1)
            under = (~over) & w.notna()
            under_sum = w.where(under, 0.0).sum(axis=1)
            scale = ((1.0 - capped).clip(lower=0.0) / under_sum.replace(0.0, np.nan)).fillna(0.0)
            w = w.where(over, w.mul(scale, axis=0))
    # hold weights constant between rebalance boundaries (first trading day of each period)
    period = sleeves.index.to_period("Q" if rebalance.upper().startswith("Q") else "M")
    is_reb = pd.Series(period, index=sleeves.index) != pd.Series(period, index=sleeves.index).shift(1)
    w_held = w.copy()
    w_held[~is_reb.to_numpy()] = np.nan                      # keep only rebalance-day targets
    return w_held.ffill()


def combined_return(sleeves: pd.DataFrame, markets: List[str], vol_window: int = VOL_WINDOW,
                    rebalance: str = "ME", cap: float = WEIGHT_CAP) -> pd.Series:
    """Risk-parity combined daily return over the COMMON sample of ``markets`` (dates
    where every included sleeve has a return)."""
    sub = sleeves[markets].dropna(how="any")
    if sub.empty:
        return pd.Series(dtype=float)
    w = risk_parity_weights(sub, vol_window, rebalance, cap)
    return (w * sub).sum(axis=1).reindex(sub.index).dropna()


def portfolio_report(sleeves: pd.DataFrame, markets: List[str], vol_window: int = VOL_WINDOW,
                     rebalance: str = "ME", cap: float = WEIGHT_CAP, ppy: int = PPY) -> dict:
    """Per-market + combined stats over the common sample of ``markets``, plus by-decade."""
    sub = sleeves[markets].dropna(how="any")
    comb = combined_return(sleeves, markets, vol_window, rebalance, cap)
    per = {m: _series_stats(sub[m], ppy) for m in markets}
    return {
        "markets": markets, "n_common": int(len(sub)),
        "start": (str(sub.index.min().date()) if len(sub) else None),
        "end": (str(sub.index.max().date()) if len(sub) else None),
        "per_market": per, "combined": _series_stats(comb, ppy),
        "combined_by_decade": by_decade(comb, ppy), "combined_net": comb,
        "best_single_sharpe": max((per[m]["sharpe"] for m in markets), default=0.0),
    }


def subperiod_stats(net: pd.Series, start: str, end: str, ppy: int = PPY) -> dict:
    net = pd.Series(net).dropna()
    seg = net.loc[(net.index >= pd.Timestamp(start)) & (net.index <= pd.Timestamp(end))]
    return _series_stats(seg, ppy)
