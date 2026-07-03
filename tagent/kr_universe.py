"""Point-in-time Korean equity universe — survivorship-bias correction.

A momentum study on *today's* liquid 50 names is biased: it silently excludes
companies that were large then but later shrank, merged, or **delisted** (the
losers), and it can't include names not yet listed at the start. That inflates
both the absolute return and the apparent edge.

This module reconstructs the tradable universe **as it was at each rebalance
date** using pykrx's as-of-date snapshots:

* :func:`top_caps_on` — the top-N names by market cap *on a given date*
  (``get_market_cap_by_ticker``). A name that was top-100 in 2017 but delisted in
  2020 appears in the 2017 snapshot and not the 2024 one — exactly the point-in-
  time membership we need. KOSPI-200 deposit files are unavailable in this pykrx,
  so market-cap rank is the broadest as-of-date universe it exposes.
* :func:`pit_members` — reconstruct membership at a list of as-of dates (monthly).
* :func:`membership_panel` — expand monthly snapshots to a daily [date x name]
  boolean mask using an **as-of (backward) fill**: day t's membership comes from
  the most recent snapshot whose date is <= t, so it is strictly no-lookahead.

The engine (:func:`tagent.xs_momentum.backtest`) takes this mask via its optional
``membership=`` argument: only members are ranked/held at each bar and the basket
benchmark is taken over members too.

pykrx is imported lazily (and injectable) so the package + test suite import with
no pykrx and run with no network — tests pass synthetic snapshots directly.

Honest limitation: pykrx OHLCV for a delisted name ends at its last trading day,
so the strategy stops earning it rather than booking a final delisting crash; the
correction captures the dominant effect (weak names being *excluded* from today's
set are now *included* while they were members) but is mildly optimistic on the
delisting tail.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from tagent.config import DATA_DIR

MEMBERS_CSV = "kr_pit_members.csv"          # cached long-format membership snapshots
SSF_CSV = "kr_ssf_available.csv"            # per-name single-stock-futures availability
_CAP_COL_HINT = "시가총액"                   # market-cap column in get_market_cap_by_ticker


def _import_pykrx():
    """Lazy import of pykrx's ``stock`` module (patched out in tests)."""
    try:
        from pykrx import stock
    except ImportError as e:  # pragma: no cover
        raise ImportError("pykrx not installed. Run: pip install -r requirements.txt") from e
    return stock


def _yyyymmdd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _daystr(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _code(t) -> str:
    """Normalise a ticker: zero-pad numeric KRX codes to 6 digits, leave other
    strings (e.g. synthetic test names) untouched."""
    s = str(t)
    return s.zfill(6) if s.isdigit() else s


def monthly_asof_dates(start, end) -> List[pd.Timestamp]:
    """First calendar day of each month in [start, end] — the as-of dates we
    reconstruct membership on (snapshots roll forward to daily bars via an as-of
    fill, so the exact day need not be a trading day)."""
    return list(pd.date_range(pd.Timestamp(start).normalize(),
                              pd.Timestamp(end).normalize(), freq="MS"))


def top_caps_on(date, top_n: int = 100, market: str = "KOSPI", stock=None) -> List[str]:
    """Top-``top_n`` tickers by market cap **as of ``date``** (point-in-time).

    Returns 6-digit codes ordered by descending market cap. Empty list if KRX has
    no snapshot for that date (e.g. pre-2014 or a holiday with no data)."""
    stock = stock or _import_pykrx()
    df = stock.get_market_cap_by_ticker(_yyyymmdd(date), market)
    if df is None or len(df) == 0:
        return []
    cap_col = next((c for c in df.columns if _CAP_COL_HINT in str(c)), None)
    if cap_col is None:
        return []
    # On a NON-TRADING day KRX returns every market cap as 0 -> the sort degenerates
    # to ticker order and the "top-N" is garbage. Drop non-positive caps so a holiday
    # snapshot comes back EMPTY (and is skipped / as-of-filled) rather than wrong.
    s = pd.to_numeric(df[cap_col], errors="coerce")
    s = s[s > 0].sort_values(ascending=False)
    return [_code(t) for t in s.head(top_n).index]


def top_caps_asof(date, top_n: int = 100, markets: Sequence[str] = ("KOSPI",),
                  stock=None, max_forward: int = 6) -> List[str]:
    """Union of per-market top-N caps for the first TRADING day on/after ``date``.

    Month-start as-of dates are often holidays/weekends (all-zero caps); this walks
    forward up to ``max_forward`` days to the next day with real market-cap data, so
    every month gets a valid snapshot instead of a garbage or skipped one."""
    stock = stock or _import_pykrx()
    for k in range(max_forward + 1):
        d = pd.Timestamp(date) + pd.Timedelta(days=k)
        members: List[str] = []
        for mk in markets:
            members += top_caps_on(d, top_n=top_n, market=mk, stock=stock)
        members = sorted(set(members))
        if members:
            return members
    return []


def pit_members(dates: Sequence, top_n: int = 100, markets: Sequence[str] = ("KOSPI",),
                stock=None) -> Dict[str, List[str]]:
    """Reconstruct point-in-time membership: {as-of-date 'YYYY-MM-DD': [tickers]}.

    Union of the per-market top-N caps on each date. Snapshots that come back empty
    (KRX gap) are skipped rather than recorded as an empty universe."""
    stock = stock or _import_pykrx()
    out: Dict[str, List[str]] = {}
    for d in dates:
        members: List[str] = []
        for mk in markets:
            members += top_caps_on(d, top_n=top_n, market=mk, stock=stock)
        members = sorted(set(members))
        if members:
            out[_daystr(d)] = members
    return out


def universe_symbols(members_by_date: Dict[str, Iterable[str]]) -> List[str]:
    """Sorted union of every ticker that was a member on any snapshot date — the set
    of OHLCV series to download (includes since-delisted names)."""
    return sorted({_code(t) for m in members_by_date.values() for t in m})


def membership_panel(members_by_date: Dict[str, Iterable[str]], daily_index,
                     symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Daily [date x symbol] boolean eligibility mask from monthly snapshots.

    Each daily date inherits the membership of the most recent snapshot whose as-of
    date is <= it (an as-of / backward fill). Dates before the first snapshot are
    all-False. Strictly no-lookahead: a name appears only once its snapshot has
    occurred, and disappears the first snapshot after it leaves the top-N / delists.
    """
    snap_dates = sorted(pd.Timestamp(d) for d in members_by_date)
    by_ts = {pd.Timestamp(d): {_code(t) for t in m} for d, m in members_by_date.items()}
    if symbols is None:
        symbols = universe_symbols(members_by_date)
    symbols = [_code(s) for s in symbols]
    daily_index = pd.DatetimeIndex(daily_index)

    if not snap_dates:
        return pd.DataFrame(False, index=daily_index, columns=symbols)

    import numpy as np
    sym_pos = {s: i for i, s in enumerate(symbols)}
    snap_idx = pd.DatetimeIndex(snap_dates)
    # pos[i] = index of the latest snapshot date <= daily_index[i]  (-1 if none yet)
    pos = snap_idx.searchsorted(daily_index, side="right") - 1
    snap_vecs = {}
    for d in snap_dates:
        v = np.zeros(len(symbols), dtype=bool)
        for t in by_ts[d]:
            if t in sym_pos:
                v[sym_pos[t]] = True
        snap_vecs[d] = v
    arr = np.zeros((len(daily_index), len(symbols)), dtype=bool)
    for i, p in enumerate(pos):
        if p >= 0:
            arr[i] = snap_vecs[snap_dates[p]]
    return pd.DataFrame(arr, index=daily_index, columns=symbols)


# --------------------------------------------------------------------------- #
# cache (long-format CSV: date,ticker), so reruns skip the network
# --------------------------------------------------------------------------- #
def members_csv_path(data_dir=None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / MEMBERS_CSV


def save_members(members_by_date: Dict[str, Iterable[str]], data_dir=None) -> Path:
    rows = [(d, _code(t)) for d, m in members_by_date.items() for t in m]
    path = members_csv_path(data_dir)
    pd.DataFrame(rows, columns=["date", "ticker"]).to_csv(path, index=False)
    return path


def load_members(data_dir=None) -> Dict[str, List[str]]:
    """Load cached membership snapshots, or {} if not present."""
    path = members_csv_path(data_dir)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={"ticker": str})
    out: Dict[str, List[str]] = {}
    for d, grp in df.groupby("date"):
        out[str(d)] = sorted({_code(t) for t in grp["ticker"]})
    return out


# --------------------------------------------------------------------------- #
# single-stock-futures (SSF) availability — which names are hedgeable / tradable
# via a listed single-stock future. KRX lists SSFs for the liquid large/mid caps;
# KOSPI-200 membership is the sourced proxy (every K200 name carries an SSF, plus
# a handful of large KOSDAQ names). Used to flag where a PEAD/momentum book can be
# delta-hedged or expressed in futures (0.05% cost) rather than cash (0.20%).
# --------------------------------------------------------------------------- #
def ssf_csv_path(data_dir=None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / SSF_CSV


def mark_ssf_available(symbols: Sequence[str], ssf_members: Iterable[str]) -> "pd.DataFrame":
    """[ticker, ssf_available] for each ``symbols`` name, True iff it is in
    ``ssf_members`` (the SSF-eligible set, e.g. KOSPI-200 constituents). Pure."""
    ssf = {_code(t) for t in ssf_members}
    rows = [(_code(s), _code(s) in ssf) for s in sorted({_code(x) for x in symbols})]
    return pd.DataFrame(rows, columns=["ticker", "ssf_available"])


def save_ssf_available(symbols: Sequence[str], ssf_members: Iterable[str],
                       data_dir=None) -> Path:
    df = mark_ssf_available(symbols, ssf_members)
    path = ssf_csv_path(data_dir)
    df.to_csv(path, index=False)
    return path


def load_ssf_available(data_dir=None) -> Dict[str, bool]:
    """Load the {ticker: ssf_available} map, or {} if not cached."""
    path = ssf_csv_path(data_dir)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={"ticker": str})
    return {_code(t): bool(v) for t, v in zip(df["ticker"], df["ssf_available"])}
