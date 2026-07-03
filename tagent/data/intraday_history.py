"""Minute-bar history for KR symbols + the US overnight return feature.

Intraday research needs minute bars (open/high/low/close/volume) per symbol and,
as a systemic driver, the **US overnight return** for each KR trading day (the
overreaction-reversal hypothesis is about gaps DRIVEN by the overnight US move).

Sources, in order of availability:
  * cached CSV  ``data/<SYMBOL>_1m.csv``  (the portable fallback; columns
    timestamp,open,high,low,close,volume) — primary in this environment,
  * an injected ``fetch(symbol) -> DataFrame`` (Kiwoom 분봉 / a data vendor) when a
    live source exists; its result is cached to the same CSV.

pykrx has no public minute endpoint and Kiwoom 분봉 is terminal/IP-gated, so the
loader is built around the CSV cache + an injectable fetcher and is fully testable
offline. The US overnight return is derived from the cached daily SPY series.

No-lookahead: the overnight return for KR day D uses only the LAST US session that
closed strictly before D (so it is known at D's open).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from tagent.config import DATA_DIR

MINUTE_SUFFIX = "_1m.csv"
_OHLCV = ["open", "high", "low", "close", "volume"]


def minute_csv_path(symbol: str, data_dir=None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / f"{str(symbol).upper()}{MINUTE_SUFFIX}"


def available_minute_symbols(data_dir=None) -> List[str]:
    """Symbols with a cached ``*_1m.csv`` (excludes the SPY benchmark)."""
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    out = []
    for p in sorted(glob.glob(str(base / f"*{MINUTE_SUFFIX}"))):
        name = os.path.basename(p)[:-len(MINUTE_SUFFIX)]
        if name.upper() != "SPY":
            out.append(name)
    return out


def load_intraday(symbol: str, data_dir=None,
                  fetch: Optional[Callable[[str], pd.DataFrame]] = None,
                  cache: bool = True) -> pd.DataFrame:
    """Minute OHLCV for ``symbol`` indexed by timestamp. Reads the cached CSV; if
    absent and ``fetch`` is given, fetches + caches. Empty DataFrame if neither."""
    path = minute_csv_path(symbol, data_dir)
    if path.exists():
        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    elif fetch is not None:
        df = fetch(symbol)
        if df is not None and not df.empty and cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index_label="timestamp")
    else:
        return pd.DataFrame(columns=_OHLCV)
    if df is None or df.empty:
        return pd.DataFrame(columns=_OHLCV)
    df = df.rename(columns=str.lower)
    keep = [c for c in _OHLCV if c in df.columns]
    df = df[keep].astype(float).sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    return df


def load_spy_daily_returns(data_dir=None, symbol: str = "SPY") -> pd.Series:
    """Daily close-to-close returns of the US index proxy (cached SPY_1d.csv), or an
    empty Series if not present."""
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    path = base / f"{symbol}_1d.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True).sort_index()
    return df["close"].astype(float).pct_change().dropna()


def overnight_for_dates(spy_returns: pd.Series, kr_dates) -> Dict:
    """Map each KR trading date to the US overnight return = the daily SPY return of
    the LAST US session that closed strictly before that KR date (no-lookahead).

    Returns ``{date: float}`` keyed by ``datetime.date``. NaN when no prior US session.
    """
    ret = pd.Series(spy_returns).dropna().sort_index()
    if ret.empty:
        return {pd.Timestamp(d).date(): float("nan") for d in kr_dates}
    ridx = pd.DatetimeIndex(ret.index).normalize()
    vals = ret.to_numpy(dtype=float)
    out: Dict = {}
    for d in kr_dates:
        dd = pd.Timestamp(d).normalize()
        pos = int(ridx.searchsorted(dd, side="left")) - 1   # last US date strictly < dd
        out[pd.Timestamp(d).date()] = float(vals[pos]) if pos >= 0 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# synthetic generator (labelled demo / smoke only — NOT a real edge)
# --------------------------------------------------------------------------- #
def make_synthetic_minutes(n_days: int = 60, bars_per_day: int = 78, seed: int = 0,
                           start: str = "2023-01-02", base_price: float = 50_000.0,
                           overnight: Optional[Dict] = None) -> pd.DataFrame:
    """A random-walk minute panel for pipeline smoke tests / the runner demo. If
    ``overnight`` is given, gaps the open by ~overnight*beta so the gap-reversal
    plumbing has events — but returns are otherwise a driftless random walk, so any
    measured edge is noise (the honest point of a synthetic run)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    rows, idx = [], []
    px = base_price
    for d in days:
        on = (overnight or {}).get(pd.Timestamp(d).date(), 0.0)
        gap = (on or 0.0) * 1.2                          # open gaps with the overnight move
        px = px * (1.0 + gap)
        t0 = pd.Timestamp(d) + pd.Timedelta(hours=9)
        for b in range(bars_per_day):
            step = rng.normal(0.0, 0.0008)
            o = px
            c = px * (1.0 + step)
            hi = max(o, c) * (1.0 + abs(rng.normal(0, 0.0004)))
            lo = min(o, c) * (1.0 - abs(rng.normal(0, 0.0004)))
            rows.append((o, hi, lo, c, float(rng.integers(100, 1000))))
            idx.append(t0 + pd.Timedelta(minutes=b))
            px = c
    df = pd.DataFrame(rows, columns=_OHLCV, index=pd.DatetimeIndex(idx, name="timestamp"))
    return df
