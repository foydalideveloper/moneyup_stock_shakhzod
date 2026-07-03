"""Binance PUBLIC funding / perp / spot data for the cash-and-carry study (no key).

Cash-and-carry = long spot + short perpetual to harvest the **funding rate** (a
structural premium perp longs pay shorts when the perp trades above spot), not a
forecast. This loader fetches what the simulation needs, all from public Binance
REST (no API key):

* funding-rate history + perp **mark price** at each funding time
  (``fapi.binance.com/fapi/v1/fundingRate``, paid every 8h);
* 8-hour **spot** closes (``api.binance.com/api/v3/klines``).

They align on the 8h funding boundaries (00:00 / 08:00 / 16:00 UTC) into one
frame indexed by ``time`` with columns ``funding_rate, perp, spot_close``, cached
to ``data/funding_<symbol>.csv``. The HTTP fetchers are injectable so the loader
is unit-tested offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from tagent.config import DATA_DIR

FAPI = "https://fapi.binance.com"
SPOT_API = "https://api.binance.com"
_EIGHT_H_MS = 8 * 60 * 60 * 1000

# BTC, ETH + a few large alts (used by the original single-window study).
DEFAULT_COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# A broad liquid-USDT-perp universe for the multi-year cross-sectional study.
# (All have multi-year Binance perp history and deep books. MATIC was migrated to
#  POL by Binance in 2024, so the universe uses POLUSDT.)
LIQUID_COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "DOTUSDT", "LINKUSDT", "POLUSDT", "LTCUSDT", "TRXUSDT", "ATOMUSDT",
    "ETCUSDT", "BCHUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "NEARUSDT", "OPUSDT",
    "INJUSDT",
]


def _floor_8h(ms: int) -> int:
    return (int(ms) // _EIGHT_H_MS) * _EIGHT_H_MS


def fetch_live_funding(symbols=None, session=None) -> dict:
    """Live funding snapshot for the universe in ONE public call (no key).

    Binance ``fapi/v1/premiumIndex`` (no symbol) returns every perp's last
    funding rate, mark price and next funding time. Returns
    ``{symbol: {funding_rate, mark, next_funding_time}}`` filtered to ``symbols``.
    """
    import requests
    s = session or requests
    rows = s.get(f"{FAPI}/fapi/v1/premiumIndex", timeout=10).json()
    if not isinstance(rows, list):
        raise ValueError(f"Binance premiumIndex error: {rows}")
    want = set(symbols) if symbols else None
    out = {}
    for d in rows:
        sym = d.get("symbol")
        if want is not None and sym not in want:
            continue
        try:
            out[sym] = {"funding_rate": float(d["lastFundingRate"]),
                        "mark": float(d.get("markPrice") or "nan"),
                        "next_funding_time": int(d.get("nextFundingTime", 0))}
        except (TypeError, ValueError, KeyError):
            continue
    return out


def fetch_funding(symbol: str, limit: int = 1000, session=None,
                  max_pages: int = 40, start_ms: Optional[int] = None) -> pd.DataFrame:
    """Public funding-rate history (rate + perp mark price) for `symbol`.

    A no-param call only returns the most recent ~200 intervals, so we walk
    FORWARD from ``start_ms`` (default ``now − limit × 8h``) page-by-page to
    accumulate the full window — funding is highly regime-dependent, so MORE
    history (multiple years, several market cycles) makes for a fairer test.
    Pass ``start_ms`` to reach back to the perp's listing.
    """
    import time
    import requests
    s = session or requests
    cur_start = int(start_ms) if start_ms is not None \
        else int(time.time() * 1000) - (limit + 10) * _EIGHT_H_MS
    out: list = []
    for _ in range(max_pages):
        rows = s.get(f"{FAPI}/fapi/v1/fundingRate",
                     params={"symbol": symbol, "limit": 1000, "startTime": cur_start},
                     timeout=10).json()
        if not isinstance(rows, list) or not rows:
            break
        out += rows
        if len(rows) < 1000:
            break                               # reached the present
        cur_start = int(rows[-1]["fundingTime"]) + 1
    if not out:
        raise ValueError(f"Binance funding error for {symbol}: empty")
    df = pd.DataFrame({
        "time": pd.to_datetime([_floor_8h(r["fundingTime"]) for r in out],
                               unit="ms", utc=True),
        "funding_rate": [float(r["fundingRate"]) for r in out],
        "perp": [float(r.get("markPrice") or "nan") for r in out],
    })
    return df.drop_duplicates("time").set_index("time").sort_index()


def fetch_spot_8h(symbol: str, limit: int = 1000, session=None,
                  start_ms: Optional[int] = None, max_pages: int = 40) -> pd.DataFrame:
    """Public 8-hour spot price AT each funding boundary for `symbol`.

    Uses the kline **open** (price at openTime = the funding boundary) so it
    aligns in time with the perp mark price at the same instant — essential for
    the delta-neutral basis to actually cancel. When ``start_ms`` is given we
    page FORWARD (klines cap at 1000/call) to pull multi-year history; otherwise
    a single recent window of ``limit`` bars is returned (original behaviour).
    """
    import requests
    s = session or requests
    url = f"{SPOT_API}/api/v3/klines"
    if start_ms is None:
        rows = s.get(url, params={"symbol": symbol, "interval": "8h", "limit": limit},
                     timeout=10).json()
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Binance spot klines error for {symbol}: {rows}")
    else:
        rows, cur = [], int(start_ms)
        for _ in range(max_pages):
            page = s.get(url, params={"symbol": symbol, "interval": "8h",
                                      "startTime": cur, "limit": 1000}, timeout=10).json()
            if not isinstance(page, list) or not page:
                break
            rows += page
            if len(page) < 1000:
                break
            cur = int(page[-1][0]) + _EIGHT_H_MS
        if not rows:
            raise ValueError(f"Binance spot klines error for {symbol}: empty")
    df = pd.DataFrame({
        "time": pd.to_datetime([k[0] for k in rows], unit="ms", utc=True),
        "spot_close": [float(k[1]) for k in rows],     # OPEN = price at the boundary
    })
    return df.drop_duplicates("time").set_index("time").sort_index()


def merge_carry(funding: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """Inner-join funding (rate + perp) with spot closes on the 8h boundary."""
    df = funding.join(spot, how="inner").dropna(subset=["funding_rate", "perp", "spot_close"])
    return df[["funding_rate", "perp", "spot_close"]].sort_index()


def funding_csv_path(symbol: str, data_dir: Optional[Path] = None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / f"funding_{symbol.upper()}.csv"


def load_carry(symbol: str, cache: bool = True, data_dir: Optional[Path] = None,
               fetch_funding_fn: Optional[Callable] = None,
               fetch_spot_fn: Optional[Callable] = None,
               limit: int = 1000) -> pd.DataFrame:
    """Load the merged carry frame for `symbol` (cached CSV, or fetched live).

    Columns: ``funding_rate`` (per-8h), ``perp`` (mark price), ``spot_close``;
    indexed by funding time. The two fetchers default to the public Binance REST
    calls and are injectable for tests.
    """
    path = funding_csv_path(symbol, data_dir)
    if cache and path.exists():
        return pd.read_csv(path, index_col="time", parse_dates=True)
    ff = fetch_funding_fn or fetch_funding
    fs = fetch_spot_fn or fetch_spot_8h
    df = merge_carry(ff(symbol, limit=limit), fs(symbol, limit=limit))
    if cache and not df.empty:
        df.to_csv(path, index_label="time")
    return df


def load_carry_history(symbol: str, years: float = 3.0, cache: bool = True,
                       data_dir: Optional[Path] = None,
                       fetch_funding_fn: Optional[Callable] = None,
                       fetch_spot_fn: Optional[Callable] = None,
                       now_ms: Optional[int] = None) -> pd.DataFrame:
    """Merged carry frame reaching back ``years`` (multi-cycle), cached separately.

    Pages both funding and spot FORWARD from ``now − years`` so the study spans
    bull/quiet/bear regimes rather than just the recent quiet window. Fetchers are
    injectable (they receive ``start_ms``) so this is unit-tested offline.
    """
    import time
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    path = base / f"funding_hist_{symbol.upper()}.csv"
    if cache and path.exists():
        return pd.read_csv(path, index_col="time", parse_dates=True)
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    start_ms = _floor_8h(now - int(years * 365 * 24 * 60 * 60 * 1000))
    ff = fetch_funding_fn or fetch_funding
    fs = fetch_spot_fn or fetch_spot_8h
    df = merge_carry(ff(symbol, start_ms=start_ms), fs(symbol, start_ms=start_ms))
    if cache and not df.empty:
        base.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index_label="time")
    return df
