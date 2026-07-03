"""Beta-neutral low-volatility (Frazzini-Pedersen Betting-Against-Beta) — ONE trial.

Externally-anchored (low-vol/BAB is unusually strong in Korea), pre-registered, judged at
calendar-time NW t >= ~2.5-3. Design is locked (dart-style spec in the runner docstring):

  * Universe: ONLY SSF-liquid names (real tradable size), BOTH legs from the SAME universe.
  * Signal: trailing realized beta vs KOSPI-200. Build beta-neutral BAB — LONG low-beta
    LEVERED to beta 1, SHORT high-beta DE-LEVERED to beta 1 => net beta ~0 by construction
    (NOT dollar-neutral, which is secretly short-beta). Monthly rebalance (low-vol doesn't
    decay weekly; extra turnover is pure cost).
  * SECTOR-NEUTRAL: legs are built WITHIN each sector (identical sector weights both legs),
    so it is not a banks-vs-semis sector bet.
  * Costs: all-SSF both legs + single-name rolls + borrow premium on hard-to-borrow shorts.
  * Verdict: real edge only if calendar-time NW t >= ~2.5-3 AND realized beta ~0 AND
    survives all-in SSF costs AND not carried by one sector. Flat/negative is fine.

Pure numpy/pandas over cached CSVs; unit-tested on synthetic panels (no network).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tagent.strategies.earnings_drift import newey_west_tstat
from tagent.xs_momentum import align_close

# pre-registered constants (LOCKED)
BETA_WINDOW = 252                 # 1-year trailing realized beta
WITHIN_SECTOR_Q = 0.33            # within each sector: bottom third long / top third short
MIN_SECTOR_NAMES = 2              # need >=2 SSF names in a sector to form a within-sector pair
LEV_CAP = 3.0                     # cap leg leverage (low-beta blowup guard)
# all-in costs (conservative; both legs all-SSF)
TX_PER_SIDE = 0.0008              # ~8 bps/side SSF transaction+slippage
ROLL_ANNUAL = 0.012              # ~12 single-name rolls/yr x ~10 bps, on gross exposure
BORROW_ANNUAL = 0.02             # ~2%/yr borrow premium on the (hard-to-borrow) short notional
PPY = 252


def ssf_liquid_universe(members_symbols: List[str], ssf: Dict[str, bool],
                        sectors: Dict[str, str], have: List[str]) -> Dict[str, str]:
    """{symbol: sector} for names that are SSF-liquid AND have price data. Both legs draw
    from this single set (real tradable size)."""
    return {s: sectors.get(s, "기타") for s in members_symbols
            if ssf.get(s, False) and s in set(have)}


def rolling_beta(panel, market_close, window: int = BETA_WINDOW) -> pd.DataFrame:
    """[date x name] trailing realized beta vs the market over ``window`` daily returns
    (cov/var). Each value at t uses only returns through t (no-lookahead)."""
    close = align_close(panel)
    r = close.pct_change(fill_method=None)
    rm = pd.Series(market_close, dtype=float).reindex(close.index).ffill().pct_change(fill_method=None)
    rm = rm.reindex(r.index)
    mp = max(window // 2, 20)
    var = rm.rolling(window, min_periods=mp).var()
    cov = r.mul(rm, axis=0).rolling(window, min_periods=mp).mean() \
        - r.rolling(window, min_periods=mp).mean().mul(rm.rolling(window, min_periods=mp).mean(), axis=0)
    return cov.div(var, axis=0)


def _rebalance_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    period = index.to_period("M")
    first = pd.Series(period, index=index) != pd.Series(period, index=index).shift(1)
    return list(index[first.to_numpy()])


def bab_book(panel, beta: pd.DataFrame, universe: Dict[str, str], membership=None,
             beta_window: int = BETA_WINDOW, q: float = WITHIN_SECTOR_Q,
             lev_cap: float = LEV_CAP) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-name daily BAB weights (sector-neutral, beta-neutral leverage), held monthly.

    At each rebalance: within each sector, rank SSF names by trailing beta; LONG the bottom
    ``q`` LEVERED by 1/beta_L, SHORT the top ``q`` DE-LEVERED by 1/beta_H (net beta ~0).
    Sectors are weighted by name count and IDENTICAL across legs (sector-neutral). Returns
    (W [date x name], legmeta [rebalance x beta_L,beta_H,lev_L,lev_H,n_sectors])."""
    close = align_close(panel)
    idx = close.index
    names = [n for n in universe if n in close.columns]
    W = pd.DataFrame(0.0, index=idx, columns=names)
    memb = (membership.reindex(index=idx, columns=close.columns).fillna(False)
            if membership is not None else None)
    sectors = {n: universe[n] for n in names}
    by_sector: Dict[str, List[str]] = {}
    for n in names:
        by_sector.setdefault(sectors[n], []).append(n)

    reb_set = set(_rebalance_dates(idx))
    meta_rows = []
    prev = pd.Series(0.0, index=names)
    for ti, t in enumerate(idx):
        if t in reb_set:
            if ti < beta_window:
                W.iloc[ti] = prev.to_numpy()
                continue
            b_t = beta.loc[t]
            elig = {}
            for sec, members in by_sector.items():
                avail = [n for n in members
                         if np.isfinite(b_t.get(n, np.nan))
                         and (memb is None or bool(memb.at[t, n]))]
                if len(avail) >= MIN_SECTOR_NAMES:
                    elig[sec] = sorted(avail, key=lambda n: b_t[n])
            if elig:
                total = sum(len(v) for v in elig.values())
                long_w = pd.Series(0.0, index=names)
                short_w = pd.Series(0.0, index=names)
                bl_num = bh_num = 0.0
                for sec, ranked in elig.items():
                    ws = len(ranked) / total
                    k = max(1, int(round(q * len(ranked))))
                    lows, highs = ranked[:k], ranked[-k:]
                    for n in lows:
                        long_w[n] += ws / len(lows)
                    for n in highs:
                        short_w[n] += ws / len(highs)
                    bl_num += ws * float(np.mean([b_t[n] for n in lows]))
                    bh_num += ws * float(np.mean([b_t[n] for n in highs]))
                beta_l, beta_h = bl_num, bh_num
                lev_l = min(lev_cap, 1.0 / beta_l) if beta_l > 0 else 0.0
                lev_h = min(lev_cap, 1.0 / beta_h) if beta_h > 0 else 0.0
                prev = long_w * lev_l - short_w * lev_h
                meta_rows.append({"date": t, "beta_L": beta_l, "beta_H": beta_h,
                                  "lev_L": lev_l, "lev_H": lev_h, "n_sectors": len(elig)})
        W.iloc[ti] = prev.to_numpy()
    meta = pd.DataFrame(meta_rows).set_index("date") if meta_rows else pd.DataFrame()
    return W, meta


def bab_net(panel, W: pd.DataFrame, tx_per_side: float = TX_PER_SIDE,
            roll_annual: float = ROLL_ANNUAL, borrow_annual: float = BORROW_ANNUAL,
            ppy: int = PPY) -> pd.Series:
    """Net daily BAB return: sum_i W[t,i]*fwd[t,i] minus all-in costs (turnover x SSF tx +
    gross x roll + short-notional x borrow), prorated daily."""
    close = align_close(panel)[W.columns]
    fwd = close.pct_change(fill_method=None).shift(-1)
    Wn = W.to_numpy(float)
    gross_ret = pd.Series((Wn * np.nan_to_num(fwd.to_numpy(float), nan=0.0)).sum(axis=1), index=W.index)
    turn = pd.Series(np.abs(np.diff(Wn, axis=0, prepend=np.zeros((1, Wn.shape[1])))).sum(axis=1), index=W.index)
    gross = pd.Series(np.abs(Wn).sum(axis=1), index=W.index)
    short_notional = pd.Series(np.where(Wn < 0, -Wn, 0.0).sum(axis=1), index=W.index)
    cost = turn * tx_per_side + gross * (roll_annual / ppy) + short_notional * (borrow_annual / ppy)
    valid = fwd.notna().any(axis=1)
    return (gross_ret - cost)[valid]


def realized_beta(bab_ret: pd.Series, market_close) -> float:
    """OLS beta of the (forward-indexed) BAB return on the market's forward return — the
    construction targets ~0."""
    rm = pd.Series(market_close, dtype=float).reindex(
        pd.Index(bab_ret.index).union(pd.Index(bab_ret.index))).ffill()
    rmf = rm.pct_change(fill_method=None).shift(-1).reindex(bab_ret.index)
    df = pd.concat([bab_ret, rmf], axis=1).dropna()
    if len(df) < 30:
        return 0.0
    x = df.iloc[:, 1].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    v = float(((x - x.mean()) ** 2).sum())
    return float(((x - x.mean()) * (y - y.mean())).sum() / v) if v > 0 else 0.0


def calendar_time_t(bab_ret: pd.Series, lag: int = 21) -> Tuple[float, int]:
    e = pd.Series(bab_ret).dropna()
    return newey_west_tstat(e, lag=lag), int(len(e))


def sector_contributions(panel, W: pd.DataFrame, universe: Dict[str, str]) -> Dict[str, float]:
    """Cumulative gross P&L attributed to each sector (long+short legs) — to check no
    single sector carries the spread."""
    close = align_close(panel)[W.columns]
    fwd = close.pct_change(fill_method=None).shift(-1)
    contrib = (W * fwd).sum(axis=0)
    out: Dict[str, float] = {}
    for n in W.columns:
        out[universe.get(n, "기타")] = out.get(universe.get(n, "기타"), 0.0) + float(contrib.get(n, 0.0))
    return out
