"""Index-rebalance FADE at short horizon — ONE pre-registered trial.

Externally anchored: KR evidence shows post-inclusion momentum loss + post-deletion rebound
(the crowd front-runs the effective date, so the move FADES afterward). LOCKED spec, no
variant search:

  * Events: KOSPI200 semi-annual reviews (effective the first trading day AFTER the June/
    December 2nd-Thursday expiry) — adds + deletes, reconstructed by diffing historical
    constituent snapshots. (MSCI Korea is a separate provider, not in KRX/pykrx -> KOSPI200
    only here; flagged.)
  * The FADE (tradable): enter the day AFTER the effective date (the effective day itself is
    crowded with index-fund flow) and hold ``HOLD`` days. SHORT adds (post-inclusion loss) /
    LONG deletes (post-deletion rebound). Strategy return per event = -abnormal for adds,
    +abnormal for deletes, where abnormal = name return - size-matched benchmark over the
    same window. No-lookahead: entry is strictly after the effective date.
  * Costs: SSF ~0.05-0.1% + roll for short adds (large/liquid -> favorable); cash 0.20% for
    long deletes; conservative 2x slippage stress.
  * Verdict: real only if size-matched abnormal positive AND calendar-time t >= ~2.5-3 AND
    survives costs. Low frequency -> a slow sleeve at most.

Pure numpy/pandas; unit-tested on synthetic series (no network).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from tagent.expiry_effects import nth_weekday
from tagent.strategies.earnings_drift import newey_west_tstat

# pre-registered constants (LOCKED)
ENTRY_LAG = 1                    # enter the day AFTER the effective date
HOLD = 20                        # fade window: ~1 month
SSF_ROUND_TRIP = 0.0008           # ~0.08% SSF round trip for short adds
CASH_ROUND_TRIP = 0.0020          # 0.20% cash round trip for long deletes
REVIEW_MONTHS = (6, 12)          # KOSPI200 semi-annual reviews


def review_effective_dates(start_year: int, end_year: int, trading_index) -> List[pd.Timestamp]:
    """First trading day strictly AFTER the 2nd Thursday of each June/December (the KOSPI200
    periodic-change effective date), snapped to the trading calendar."""
    ti = pd.DatetimeIndex(trading_index)
    out = []
    for y in range(start_year, end_year + 1):
        for m in REVIEW_MONTHS:
            thu = nth_weekday(y, m, weekday=3, n=2)
            pos = ti.searchsorted(thu, side="right")        # first trading day after the 2nd Thursday
            if 0 <= pos < len(ti):
                out.append(ti[pos])
    return sorted(set(out))


def reconstruct_changes(snapshots: Dict[pd.Timestamp, dict]) -> pd.DataFrame:
    """Adds/deletes per review from before/after constituent snapshots. ``snapshots`` =
    {effective_date: {'before': set, 'after': set}}. Returns [effective, ticker, action]."""
    rows = []
    for eff, snap in snapshots.items():
        before, after = set(snap.get("before", set())), set(snap.get("after", set()))
        if not before or not after:
            continue
        for t in sorted(after - before):
            rows.append({"effective": pd.Timestamp(eff), "ticker": str(t).zfill(6), "action": "add"})
        for t in sorted(before - after):
            rows.append({"effective": pd.Timestamp(eff), "ticker": str(t).zfill(6), "action": "delete"})
    return pd.DataFrame(rows, columns=["effective", "ticker", "action"]).sort_values(
        ["effective", "action"]).reset_index(drop=True)


def fade_event_returns(changes: pd.DataFrame, panel, bench_close, entry_lag: int = ENTRY_LAG,
                       hold: int = HOLD) -> pd.DataFrame:
    """Per-event abnormal + strategy return over the post-effective fade window.

    For each change: entry = effective + ``entry_lag`` trading days; abnormal = name return
    minus the size-matched benchmark return over [entry, entry+hold]; strategy_ret =
    -abnormal for adds (short), +abnormal for deletes (long). No-lookahead (entry strictly
    after the effective date). Columns: [effective, ticker, action, entry, abnormal,
    strategy_ret]."""
    from tagent.xs_momentum import align_close
    close = align_close(panel)
    idx = close.index
    bench = pd.Series(bench_close, dtype=float).reindex(idx).ffill()
    rows = []
    for r in changes.itertuples():
        sym = str(r.ticker).zfill(6)
        if sym not in close.columns or pd.Timestamp(r.effective) not in idx:
            continue
        ei = idx.get_loc(pd.Timestamp(r.effective)) + entry_lag
        xi = ei + hold
        if ei < 1 or xi >= len(idx):
            continue
        s = close[sym]
        if pd.isna(s.iloc[ei]) or pd.isna(s.iloc[xi]) or s.iloc[ei] <= 0:
            continue
        name_ret = float(s.iloc[xi] / s.iloc[ei] - 1.0)
        bench_ret = float(bench.iloc[xi] / bench.iloc[ei] - 1.0)
        abnormal = name_ret - bench_ret
        strat = -abnormal if r.action == "add" else abnormal      # SHORT adds / LONG deletes
        rows.append({"effective": pd.Timestamp(r.effective), "ticker": sym, "action": r.action,
                     "entry": idx[ei], "abnormal": abnormal, "strategy_ret": strat})
    return pd.DataFrame(rows, columns=["effective", "ticker", "action", "entry", "abnormal",
                                       "strategy_ret"])


def fade_net(events: pd.DataFrame, ssf_rt: float = SSF_ROUND_TRIP,
             cash_rt: float = CASH_ROUND_TRIP) -> pd.Series:
    """Net strategy return per event: adds pay the SSF round trip, deletes the cash round
    trip. Indexed by entry date."""
    if events.empty:
        return pd.Series(dtype=float)
    cost = np.where(events["action"].to_numpy() == "add", ssf_rt, cash_rt)
    net = events["strategy_ret"].to_numpy() - cost
    return pd.Series(net, index=pd.DatetimeIndex(events["entry"]))


def calendar_time_excess(events: pd.DataFrame, panel, bench_close, entry_lag: int = ENTRY_LAG,
                         hold: int = HOLD) -> pd.Series:
    """Daily calendar-time excess: each day, EW abnormal daily return of all fade positions
    active that day (sign +1 for deletes / -1 for adds), vs the benchmark. NW t of this is
    the rigorous significance test under event clustering."""
    from collections import defaultdict
    from tagent.xs_momentum import align_close
    close = align_close(panel)
    idx = close.index
    ret = close.pct_change(fill_method=None)
    bret = pd.Series(bench_close, dtype=float).reindex(idx).ffill().pct_change(fill_method=None)
    by_day: Dict[pd.Timestamp, list] = defaultdict(list)
    for r in events.itertuples():
        sym = r.ticker
        if sym not in ret.columns or pd.Timestamp(r.entry) not in idx:
            continue
        ei = idx.get_loc(pd.Timestamp(r.entry))
        sign = -1.0 if r.action == "add" else 1.0
        for j in range(ei, min(ei + hold, len(idx))):
            by_day[idx[j]].append((sym, sign))
    rows = {}
    for d, held in by_day.items():
        vals = [sign * (ret.at[d, s] - bret.at[d]) for s, sign in held
                if pd.notna(ret.at[d, s]) and pd.notna(bret.at[d])]
        if vals:
            rows[d] = float(np.mean(vals))
    return pd.Series(rows).sort_index()


def plain_tstat(x) -> float:
    a = np.asarray(pd.Series(x).dropna(), float)
    n = len(a)
    if n < 3:
        return 0.0
    sd = a.std(ddof=1)
    return float(a.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0


def calendar_time_tstat(excess: pd.Series, lag: int = HOLD):
    e = pd.Series(excess).dropna()
    return newey_west_tstat(e, lag=lag), int(len(e))
