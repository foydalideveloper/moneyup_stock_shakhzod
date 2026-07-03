"""DART corporate-event studies — size-matched abnormal returns + calendar-time NW t.

Track B: do treasury-cancellation (소각), acquisition (취득), and supply-contract
(단일판매·공급계약) filings predict drift, or is it just bull-market beta (the trap that
killed PEAD)? Methodology is locked in dart_event_studies_spec.md:

  * NO raw-return headline. Primary = SIZE-MATCHED abnormal return (event return minus the
    same-period return of the EW portfolio of the stock's market-cap QUINTILE within the
    PIT universe). Secondary = CALENDAR-TIME portfolio Newey-West t (daily EW of active
    events minus matched size-quintile controls). Raw is a footnote.
  * Beta autopsy: event return minus beta x KOSPI-200 over the held window.
  * No-lookahead: enter the first tradable open AFTER the filing. Non-overlapping per name.
  * Holds 20/40/60d (pre-registered). Costs 0.20% base, 0.51%/0.71% stress, +0.30% extra
    gap-entry slippage when the entry open gaps up (r0 > +1%). Gap (r0) vs post-open share
    of the drift is reported.

Pure numpy/pandas over cached CSVs; unit-tested on synthetic panels (no network).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tagent.strategies.earnings_drift import newey_west_tstat, pead_trades
from tagent.xs_momentum import align_close

# pre-registered constants
HOLDS = (20, 40, 60)
BASE_COST = 0.0020
STRESS_COSTS = (0.0051, 0.0071)
GAP_THRESHOLD = 0.01            # entry open gaps up > +1% -> hard-to-fill
GAP_EXTRA_SLIP = 0.0030         # +0.30% extra slippage on a gap-up entry
N_QUINTILES = 5


# --------------------------------------------------------------------------- #
# event loading + classification (mirror of the downloader's locked rules)
# --------------------------------------------------------------------------- #
def classify_event(title: str) -> Optional[str]:
    t = str(title or "")
    if "정정" in t:
        return None
    if "소각" in t and "자회사" not in t:
        return "cancellation"
    if "자기주식취득" in t and "결정" in t and "신탁" not in t and "처분" not in t:
        return "acquisition"
    if "단일판매" in t or "공급계약" in t:
        return "contract"
    return None


def events_by_symbol(df: pd.DataFrame, event_type: str) -> Dict[str, List[pd.Timestamp]]:
    """{symbol: sorted unique filing dates} for one event type from a disclosures frame
    (columns time, symbol, type[, title])."""
    out: Dict[str, set] = {}
    sub = df[df["type"] == event_type]
    for r in sub.itertuples():
        out.setdefault(str(r.symbol).zfill(6), set()).add(pd.Timestamp(r.time))
    return {s: sorted(ds) for s, ds in out.items()}


# --------------------------------------------------------------------------- #
# size quintiles within the PIT universe (cap = shares x close, members only)
# --------------------------------------------------------------------------- #
def size_quintiles(panel, membership, shares: Dict[str, float],
                   n_q: int = N_QUINTILES) -> pd.DataFrame:
    """[date x symbol] integer size quintile (1=smallest .. n_q=largest) among ACTIVE
    members each day; NaN for non-members. Cap = cached shares x daily close."""
    close = align_close(panel)
    sh = pd.Series({c: float(shares.get(c, np.nan)) for c in close.columns})
    cap = close.mul(sh, axis=1)
    if membership is not None:
        m = membership.reindex(index=close.index, columns=close.columns).fillna(False)
        cap = cap.where(m.to_numpy(dtype=bool))
    pct = cap.rank(axis=1, pct=True)                       # 0..1 within active members
    q = np.ceil(pct * n_q).clip(1, n_q)
    return q


def quintile_daily_returns(panel, quintiles: pd.DataFrame,
                           n_q: int = N_QUINTILES) -> pd.DataFrame:
    """[date x quintile] EW close-to-close daily return of active members in each
    quintile (the size-matched control portfolios)."""
    close = align_close(panel)
    ret = close.pct_change(fill_method=None)
    out = {}
    for qi in range(1, n_q + 1):
        mask = (quintiles == qi).reindex(index=ret.index, columns=ret.columns).fillna(False)
        out[qi] = ret.where(mask.to_numpy(dtype=bool)).mean(axis=1)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# per-event trades + size-matched abnormal + gap decomposition
# --------------------------------------------------------------------------- #
def event_trades(panel, events_by_sym, hold: int, membership=None,
                 non_overlapping: bool = True) -> pd.DataFrame:
    """Unconditional event trades (enter first open after filing, hold ``hold`` days),
    via the shared no-lookahead engine. Columns [entry, symbol, ret, r0]."""
    return pead_trades(panel, events_by_sym, hold=hold, conditional=False,
                       membership=membership, non_overlapping=non_overlapping)


def add_size_matched(trades: pd.DataFrame, panel, quintiles: pd.DataFrame,
                     qret: pd.DataFrame, hold: int) -> pd.DataFrame:
    """Add ``quintile``, ``control_ret`` (same-period buy&hold of the firm's size-quintile
    portfolio) and ``abnormal = ret - control_ret`` to each trade. No-lookahead: the
    quintile is taken at entry; the control return spans the same held bars."""
    close = align_close(panel)
    idx = close.index
    out = []
    for t in trades.itertuples():
        if t.symbol not in quintiles.columns or t.entry not in idx:
            continue
        ei = idx.get_loc(t.entry)
        xi = ei + hold
        if xi >= len(idx):
            continue
        q = quintiles.iat[ei, quintiles.columns.get_loc(t.symbol)]
        if not np.isfinite(q):
            continue
        q = int(q)
        ctrl = float((1.0 + qret[q].iloc[ei + 1:xi + 1].fillna(0.0)).prod() - 1.0)  # hold window
        rec = t._asdict()
        rec.pop("Index", None)
        rec["quintile"] = q
        rec["control_ret"] = ctrl
        rec["abnormal"] = float(t.ret) - ctrl
        out.append(rec)
    cols = list(trades.columns) + ["quintile", "control_ret", "abnormal"]
    return pd.DataFrame(out, columns=cols)


# --------------------------------------------------------------------------- #
# calendar-time portfolio: daily EW(active events) - EW(matched quintile controls)
# --------------------------------------------------------------------------- #
def calendar_time_excess(trades: pd.DataFrame, panel, quintiles: pd.DataFrame,
                         qret: pd.DataFrame, hold: int) -> pd.Series:
    """Daily calendar-time excess return: on each day, EW close-to-close return of all
    events active that day MINUS the EW of their entry-date size-quintile controls. The
    NW t-stat of this series (lag=hold) is the headline significance test."""
    close = align_close(panel)
    ret = close.pct_change(fill_method=None)
    idx = close.index
    from collections import defaultdict
    ev_by_day: Dict[pd.Timestamp, list] = defaultdict(list)   # day -> [(sym, q_entry)]
    for t in trades.itertuples():
        if t.symbol not in ret.columns or t.entry not in idx:
            continue
        ei = idx.get_loc(t.entry)
        q = quintiles.iat[ei, quintiles.columns.get_loc(t.symbol)]
        if not np.isfinite(q):
            continue
        q = int(q)
        for j in range(ei + 1, min(ei + hold, len(idx) - 1) + 1):   # held days (close-to-close)
            ev_by_day[idx[j]].append((t.symbol, q))
    rows = {}
    for d, holds in ev_by_day.items():
        er = [ret.at[d, s] for s, _ in holds if pd.notna(ret.at[d, s])]
        cr = [qret[q].at[d] for _, q in holds if pd.notna(qret[q].at[d])]
        if er and cr:
            rows[d] = float(np.mean(er) - np.mean(cr))
    return pd.Series(rows).sort_index()


# --------------------------------------------------------------------------- #
# costs incl. gap-entry slippage + gap/post-open decomposition
# --------------------------------------------------------------------------- #
def net_with_gap_slippage(trades: pd.DataFrame, cost: float,
                          gap_threshold: float = GAP_THRESHOLD,
                          gap_extra: float = GAP_EXTRA_SLIP, column: str = "abnormal") -> pd.Series:
    """Net of ``cost`` plus ``gap_extra`` extra slippage on gap-up entries (r0 >
    ``gap_threshold``). Nets the ``column`` (abnormal by default; 'ret' for the raw footnote)."""
    base = trades[column].astype(float)
    extra = np.where(trades["r0"].astype(float) > gap_threshold, gap_extra, 0.0)
    return base - cost - pd.Series(extra, index=trades.index)


def gap_decomposition(trades: pd.DataFrame) -> dict:
    """Split the event move into the entry GAP (r0, prev_close->entry_open, which a
    next-open entry MISSES) and the POST-OPEN drift (ret, entry_open->exit, what we earn)."""
    r0 = trades["r0"].astype(float)
    ret = trades["ret"].astype(float)
    total = (1.0 + r0) * (1.0 + ret) - 1.0
    return {"mean_gap_r0": float(r0.mean()), "mean_post_open": float(ret.mean()),
            "mean_total": float(total.mean()),
            "gap_share": float(r0.mean() / total.mean()) if abs(total.mean()) > 1e-9 else float("nan"),
            "frac_gap_up": float((r0 > GAP_THRESHOLD).mean())}


def calendar_time_tstat(excess: pd.Series, hold: int) -> Tuple[float, int]:
    """Newey-West t-stat (lag=hold) of the daily calendar-time excess series + n days."""
    e = pd.Series(excess).dropna()
    return (newey_west_tstat(e, lag=hold), int(len(e)))
