"""Lock down the deployable trend-futures core: drawdown diagnostic + continuous v2.

PART 1 — DRAWDOWN DIAGNOSTIC. Identify the episode behind the long-history filtered
core's worst drawdown (the ~-49.7% that SIZING MUST SURVIVE, not a benign -23%) and
quantify how much the realistic signal-at-close / trade-next-bar EXECUTION LAG worsens
it vs an idealised same-bar fill.

PART 2 — CONTINUOUS-EXPOSURE v2 (PRE-REGISTERED, not Boxpi-tuned). Replace the binary
200d on/off with a continuous exposure:

    exposure = clip( logistic(k * trend_z) * (target_vol / realized_vol), 0, 1 )

trend_z = (close - SMA_ma) / rolling_std(close - SMA_ma, std_window); realized_vol =
annualised rolling stdev of index returns. All signals use only data through the
decision bar and are applied to the NEXT bar's return (no-lookahead). Constants are
LOCKED in :class:`ContinuousV2Spec` (k=1, target_vol=0.15, ma=200, std_window=200,
vol_window=63) before any v2 result is seen.

Pure numpy/pandas; unit-tested on synthetic series (no network).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tagent.decomposition import FUT_ROUND_TRIP, PPY, _one_way, _series_stats
from tagent.index_calibration import ROLL_FRICTION_ANNUAL


# --------------------------------------------------------------------------- #
# drawdown utilities
# --------------------------------------------------------------------------- #
def equity_curve(net) -> pd.Series:
    return (1.0 + pd.Series(net, dtype=float).dropna()).cumprod()


def drawdown_episode(net) -> dict:
    """Worst peak-to-trough drawdown of a return series: depth, peak/trough dates, and
    the recovery date (or None if never recovered within the sample)."""
    eq = equity_curve(net)
    if eq.empty:
        return {"depth": 0.0, "peak_date": None, "trough_date": None, "recovery_date": None,
                "peak_to_trough_days": 0}
    peak = eq.cummax()
    dd = eq / peak - 1.0
    trough_date = dd.idxmin()
    depth = float(dd.loc[trough_date])
    peak_val = float(peak.loc[trough_date])
    pre = eq.loc[:trough_date]
    peak_date = pre[pre >= peak_val - 1e-12].index[-1] if (pre >= peak_val - 1e-12).any() else eq.index[0]
    post = eq.loc[trough_date:]
    rec = post[post >= peak_val]
    recovery_date = rec.index[0] if len(rec) else None
    return {"depth": depth, "peak_date": peak_date, "trough_date": trough_date,
            "recovery_date": recovery_date,
            "peak_to_trough_days": int(eq.index.get_loc(trough_date) - eq.index.get_loc(peak_date))}


# --------------------------------------------------------------------------- #
# PART 1 — binary filter with explicit execution lag (same-bar vs next-bar)
# --------------------------------------------------------------------------- #
def binary_trend_net(close, regime_ma: int = 200, mode: str = "next_bar",
                     roll_annual: float = ROLL_FRICTION_ANNUAL,
                     switch_cost: Optional[float] = None, ppy: int = PPY) -> pd.Series:
    """Binary 200d-MA trend filter, net, with an explicit execution-lag ``mode``.

    The signal ``sig[t] = close[t] >= SMA_ma[t]`` is known AT close t. Then:
      * ``next_bar`` (REALISTIC, no-lookahead): hold over t->t+1, i.e. earn
        ``close[t+1]/close[t]-1`` — you act at/after the close that produced the signal.
      * ``same_bar`` (IDEALISED / lookahead): credit the bar that ENDS at t
        (``close[t]/close[t-1]-1``) — impossible to trade, used only to size the lag cost.
    Costs: futures one-way ``switch_cost`` on each flip + prorated ``roll_annual``.
    """
    switch_cost = _one_way(FUT_ROUND_TRIP) if switch_cost is None else switch_cost
    close = pd.Series(close, dtype=float).dropna()
    sma = close.rolling(regime_ma, min_periods=regime_ma).mean()
    sig = (close >= sma).astype(float)
    sig[sma.isna()] = 1.0                                  # warmup -> in-market (matches binary v1)
    if mode == "same_bar":
        ret = close.pct_change(fill_method=None)
    elif mode == "next_bar":
        ret = close.pct_change(fill_method=None).shift(-1)
    else:
        raise ValueError("mode must be 'same_bar' or 'next_bar'")
    switch = sig.diff()
    if len(switch):
        switch.iloc[0] = sig.iloc[0]
    roll = sig * (roll_annual / ppy)
    return (sig * ret - switch.abs() * switch_cost - roll).dropna()


def execution_lag_compare(close, regime_ma: int = 200, ppy: int = PPY) -> dict:
    """Same-bar (idealised) vs next-bar (realistic) binary filter: stats + worst DD for
    each, so the cost of the trade-next-bar execution lag is explicit."""
    out = {}
    for mode in ("same_bar", "next_bar"):
        net = binary_trend_net(close, regime_ma, mode=mode, ppy=ppy)
        out[mode] = {"stats": _series_stats(net, ppy), "episode": drawdown_episode(net), "net": net}
    return out


# --------------------------------------------------------------------------- #
# PART 2 — continuous-exposure v2 (LOCKED constants)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContinuousV2Spec:
    """PRE-REGISTERED continuous-exposure constants (locked before any v2 result)."""
    regime_ma: int = 200             # trend MA window
    std_window: int = 200            # window for the trend z-score denominator
    logistic_k: float = 1.0          # logistic steepness on the trend z-score
    target_vol: float = 0.15         # annualised vol target (Barroso-ish, matches overlay default)
    vol_window: int = 63             # realized-vol lookback (~3 months)
    roll_annual: float = ROLL_FRICTION_ANNUAL
    ppy: int = PPY


def _logistic(x, k: float = 1.0) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-k * pd.Series(x, dtype=float)))


def continuous_exposure(close, spec: Optional[ContinuousV2Spec] = None) -> pd.Series:
    """Continuous exposure in [0,1] from the LOCKED v2 rule, known at close t (applied to
    the next bar's return by the caller). Warmup (signals undefined) -> 0 (flat)."""
    spec = spec or ContinuousV2Spec()
    close = pd.Series(close, dtype=float).dropna()
    sma = close.rolling(spec.regime_ma, min_periods=spec.regime_ma).mean()
    dist = close - sma
    z = dist / dist.rolling(spec.std_window, min_periods=spec.std_window).std()
    rvol = close.pct_change(fill_method=None).rolling(
        spec.vol_window, min_periods=spec.vol_window).std() * np.sqrt(spec.ppy)
    exp = (_logistic(z, spec.logistic_k) * (spec.target_vol / rvol)).clip(0.0, 1.0)
    return exp.fillna(0.0)


def continuous_trend_net(close, spec: Optional[ContinuousV2Spec] = None,
                         switch_cost: Optional[float] = None) -> pd.Series:
    """v2 net forward return: exposure[t] (known at close t) earns t->t+1, net of
    continuous turnover cost (|d exposure| x futures one-way) + prorated roll friction."""
    spec = spec or ContinuousV2Spec()
    switch_cost = _one_way(FUT_ROUND_TRIP) if switch_cost is None else switch_cost
    close = pd.Series(close, dtype=float).dropna()
    exp = continuous_exposure(close, spec)
    ret = close.pct_change(fill_method=None).shift(-1)
    turn = exp.diff()
    if len(turn):
        turn.iloc[0] = exp.iloc[0]
    roll = exp * (spec.roll_annual / spec.ppy)
    return (exp * ret - turn.abs() * switch_cost - roll).dropna()


def by_decade(net, ppy: int = PPY) -> Dict[str, dict]:
    net = pd.Series(net).dropna()
    out: Dict[str, dict] = {}
    if net.empty or not isinstance(net.index, pd.DatetimeIndex):
        return out
    for dec, seg in net.groupby((net.index.year // 10) * 10):
        out[f"{int(dec)}s"] = _series_stats(seg, ppy)
    return out


# --------------------------------------------------------------------------- #
# sizing off the REAL drawdown (deploy_spec pre-commitment A)
# --------------------------------------------------------------------------- #
def size_from_maxdd(strategy_maxdd: float, tolerable_account_dd: float = -0.15) -> float:
    """Exposure multiplier so the account's tolerable drawdown is not breached if the
    strategy repeats its worst historical DD: ``|tolerable| / |strategy_maxdd|`` (capped
    at 1.0 — never lever the core)."""
    s = abs(float(strategy_maxdd))
    if s <= 0:
        return 1.0
    return float(min(1.0, abs(tolerable_account_dd) / s))
