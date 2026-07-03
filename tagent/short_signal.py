"""YouTuber 공매도 signal — ONE pre-registered trial (honest framework, free pykrx data).

HYPOTHESIS (systematized from the video [09:59-13:24]): a stock's short-selling PROPORTION
(공매도 비중 = short_volume / volume) and its 1-3 day CHANGE predict the next 1-5 day
cross-sectional return — rising short read as OVERHANG (the name underperforms), while a LOW
short-% on a big down day is "not structural -> mean-revert / buy the dip".

LOCKED spec (no variant search):
  * Signal per stock/day: the 3-day CHANGE in short ratio, known at end of day t (short data is
    EOD + a 1-day release delay -> shifted 1). The LEVEL is a pre-registered SECONDARY.
  * Portfolio: each day cross-sectionally demean the signal and gross-normalize to sum|w|=1
    (dollar-neutral L/S). OVERHANG sign = LONG the low-short names, SHORT the high-short (so a
    POSITIVE book return means high short underperformed). Held HOLD days as overlapping
    calendar-time cohorts; entered the NEXT session (lag 1) -> no-lookahead. The opposite sign is
    the SQUEEZE / mean-reversion alternative — both are pre-registered and tested.
  * Short-BAN windows (2020-03..2021-05, 2023-11..2025-03) are EXCLUDED: shorting is suppressed to
    ~0 by regulation there, so the signal is a regulatory artifact, not information.
  * Cost: KR cash round trip 0.20% on daily turnover; reported at 1x and a CONSERVATIVE 2x slippage.
  * Metrics: net mean/Sharpe, calendar-time Newey-West t (lag=HOLD), by-year.
  * 대차잔고 (short_balance) level/change is a pre-registered SECONDARY but is NOT in the free
    pykrx cache (all-NaN) -> flagged honestly, not faked.
  * VERDICT: real only if |calendar t| >= ~2.5-3 AND the book survives 2x cost AND is stable
    by-year. Flat / negative is fine.

Pure numpy/pandas; unit-tested on synthetic wide frames (no network).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from tagent.decomposition import CASH_ROUND_TRIP, PPY, _series_stats
from tagent.strategies.earnings_drift import newey_west_tstat

CHG_LOOKBACK = 3            # 1-3 day change in short ratio (the primary, pre-registered)
HOLD = 5                   # next 1-5 day horizon
ENTER_LAG = 1              # enter the NEXT session after the signal day (no-lookahead)
RELEASE_DELAY = 1          # short data for trading day D is published after D
T_BAR = 2.5                # calendar-time NW |t| bar to call an edge real
SLIPPAGE_STRESS = 2.0


# --------------------------------------------------------------------------- #
# signal construction
# --------------------------------------------------------------------------- #
def short_ratio_wide(short_by_sym: Dict[str, pd.DataFrame],
                     close_wide: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """{sym: canonical short_df} -> wide short_ratio [date x sym]. If a source lacks total
    volume, complete the ratio from ``close_wide`` volume (when supplied)."""
    cols = {}
    for sym, df in short_by_sym.items():
        sr = pd.to_numeric(df["short_ratio"], errors="coerce") if "short_ratio" in df else None
        if sr is None or sr.isna().all():
            vol = pd.to_numeric(df.get("volume"), errors="coerce") if "volume" in df else None
            sv = pd.to_numeric(df.get("short_volume"), errors="coerce") if "short_volume" in df else None
            if sv is not None and vol is not None:
                sr = sv / vol.replace(0.0, np.nan)
            else:
                continue
        cols[str(sym)] = sr
    wide = pd.DataFrame(cols).sort_index()
    if len(wide):
        wide.index = pd.to_datetime(wide.index)
    return wide


def signal_change(sr_wide: pd.DataFrame, lookback: int = CHG_LOOKBACK,
                  release_delay: int = RELEASE_DELAY) -> pd.DataFrame:
    """The 1-3 day CHANGE in short ratio, delayed by the release lag (known at end of day t)."""
    sr = sr_wide.shift(release_delay)                       # short data for D known only after D
    return sr - sr.shift(lookback)


def signal_level(sr_wide: pd.DataFrame, release_delay: int = RELEASE_DELAY) -> pd.DataFrame:
    """The short-ratio LEVEL (secondary), release-delayed."""
    return sr_wide.shift(release_delay)


def _demean_norm(sig: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally demean each row and gross-normalize to sum|w| = 1 (dollar-neutral)."""
    d = sig.sub(sig.mean(axis=1), axis=0)
    denom = d.abs().sum(axis=1).replace(0.0, np.nan)
    return d.div(denom, axis=0)


def long_short_weights(signal_wide: pd.DataFrame, *, overhang: bool = True,
                       ban_mask: Optional[pd.Series] = None) -> pd.DataFrame:
    """Target L/S weights formed at day t. OVERHANG sign LONGS low-short / SHORTS high-short, so a
    positive book return = high short underperforms. Ban days are zeroed (regulatory artifact)."""
    w = _demean_norm(signal_wide)
    if overhang:
        w = -w                                             # long low-signal, short high-signal
    w = w.fillna(0.0)
    if ban_mask is not None:
        ban = ban_mask.reindex(w.index).fillna(False).to_numpy(dtype=bool)
        if ban.any():
            w.loc[ban] = 0.0
    return w


# --------------------------------------------------------------------------- #
# no-lookahead, costed, overlapping calendar-time backtest
# --------------------------------------------------------------------------- #
def backtest(signal_wide: pd.DataFrame, close_wide: pd.DataFrame, *, hold: int = HOLD,
             overhang: bool = True, cost_round_trip: float = CASH_ROUND_TRIP,
             enter_lag: int = ENTER_LAG, ban_mask: Optional[pd.Series] = None) -> pd.Series:
    """Net daily return of the cross-sectional short-signal L/S book. Weights formed at t are
    applied from t+enter_lag and held ``hold`` days (overlapping cohorts averaged), earning each
    day's stock return minus turnover cost. No-lookahead: returns on day d use only weights from
    signal days <= d-enter_lag (and the signal itself is already release-delayed)."""
    ret = close_wide.pct_change(fill_method=None)
    w_target = long_short_weights(signal_wide, overhang=overhang, ban_mask=ban_mask)
    w_target = w_target.reindex(index=ret.index, columns=ret.columns).fillna(0.0)
    applied = w_target.shift(enter_lag).fillna(0.0)        # signal at t -> position from t+lag
    active = applied.rolling(hold, min_periods=1).mean()   # overlapping calendar-time cohorts
    gross = (active * ret).sum(axis=1)
    turn = active.diff().abs().sum(axis=1)
    turn = turn.fillna(active.abs().sum(axis=1))
    net = gross - turn * (cost_round_trip / 2.0)
    held = active.abs().sum(axis=1) > 0                     # only days with a real position
    return net[held].dropna()


def calendar_time_t(net, lag: int = HOLD) -> Tuple[float, int]:
    e = pd.Series(net).dropna()
    return newey_west_tstat(e, lag=lag), int(len(e))


def by_year(net, ppy: int = PPY) -> dict:
    net = pd.Series(net).dropna()
    out: dict = {}
    if net.empty or not isinstance(net.index, pd.DatetimeIndex):
        return out
    for y, seg in net.groupby(net.index.year):
        st = _series_stats(seg, ppy)
        out[str(int(y))] = {"sharpe": st["sharpe"], "total_return": st["total_return"], "n": st["n"]}
    return out


def is_degenerate(signal_wide: pd.DataFrame) -> bool:
    """True if the signal has no cross-sectional variation to trade (empty / all-NaN / all-equal)
    — report a data wall honestly instead of a fake verdict."""
    s = signal_wide.dropna(how="all")
    if s.empty or s.shape[1] < 2:
        return True
    spread = (s.max(axis=1) - s.min(axis=1)).dropna()
    return spread.empty or bool((spread.abs() < 1e-12).all())


def evaluate(net_1x: pd.Series, net_2x: pd.Series, *, t_bar: float = T_BAR, ppy: int = PPY) -> dict:
    """Verdict for ONE book: real only if the 2x-costed book makes money with calendar-time
    |t| >= t_bar AND is stable by-year. ``net`` is signed so positive = the book's direction works."""
    s1, s2 = _series_stats(net_1x, ppy), _series_stats(net_2x, ppy)
    t1, n = calendar_time_t(net_1x)
    t2, _ = calendar_time_t(net_2x)
    yr = by_year(net_2x, ppy)
    signs = [v["total_return"] > 0 for v in yr.values()]
    stable = len(signs) >= 2 and (sum(signs) / len(signs) >= 0.6)
    survives = bool(s2["total_return"] > 0 and s2["sharpe"] > 0)     # positive net of 2x cost
    clears = bool(t2 >= t_bar and survives and stable)
    return {
        "n": int(n), "t_1x": round(float(t1), 2), "t_2x": round(float(t2), 2),
        "sharpe_1x": round(float(s1["sharpe"]), 2), "sharpe_2x": round(float(s2["sharpe"]), 2),
        "ret_1x_pct": round(float(s1["total_return"]) * 100, 2),
        "ret_2x_pct": round(float(s2["total_return"]) * 100, 2),
        "survives_cost": survives, "stable_by_year": stable, "by_year": yr, "clears": clears,
    }
