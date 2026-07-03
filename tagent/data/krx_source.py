"""Real Korean-market data via pykrx (free, no Kiwoom key required).

pykrx scrapes KRX (data.krx.co.kr) directly, so it needs no broker account:

* **OHLCV** (`get_market_ohlcv_by_date`) is open access — works out of the box.
* **Short-selling** (`get_shorting_volume_by_date` / `get_shorting_balance_by_date`)
  now requires a *free* KRX website account. pykrx reads the credentials from the
  ``KRX_ID`` / ``KRX_PW`` environment variables. Without them those endpoints
  return empty and the fetch raises a clear error (caught by the download script).

Everything here mirrors the existing offline layer:

* :func:`get_krx_history` returns the SAME schema as
  :func:`tagent.data.historical.load_history` (open, high, low, close, volume,
  indexed by ``timestamp``) and caches to ``data/<TICKER>_<interval>.csv`` — so
  ``load_history("005930")`` just works afterwards.
* :func:`krx_short_fetch` returns a ``fetch(symbol)`` callable matching
  :data:`tagent.data.short_selling.FetchFn`, so you load KR short data with the
  existing normalizer: ``load_short_selling("005930", fetch=krx_short_fetch(...))``.

pykrx is imported lazily so the package (and the test suite) import cleanly when
pykrx isn't installed; tests inject a fake module via :func:`_import_pykrx`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tagent.config import DATA_DIR
from tagent.data.short_selling import FetchFn, short_csv_path

# KRX returns Korean column headers. Map the OHLCV ones to our canonical schema.
_OHLCV_RENAME = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
}
_OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Short-selling column headers (pykrx):
#   get_shorting_volume_by_date  -> 공매도(short volume), 매수(total volume), 비중(%)
#   get_shorting_balance_by_date -> 공매도잔고(balance shares), 공매도금액(balance value KRW), ...
_SHORT_VOLUME_COL = "공매도"
_TOTAL_VOLUME_COL = "매수"
_SHORT_BALANCE_COL = "공매도잔고"
_SHORT_VALUE_COL = "공매도금액"


def _import_pykrx():
    """Lazy import of pykrx's ``stock`` module (patched out in tests)."""
    try:
        from pykrx import stock
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pykrx not installed. Run: pip install -r requirements.txt") from e
    return stock


def ensure_krx_login() -> bool:
    """Export KRX_ID/KRX_PW into the environment so pykrx can auto-login.

    pykrx reads ``KRX_ID`` / ``KRX_PW`` from ``os.environ`` (lazily, at first
    gated call) to unlock the per-ticker 공매도 / 투자자별(수급) endpoints. We
    read them from Settings (.env) and set them if present. Returns True when
    credentials are available (not whether KRX actually accepts them). Secrets
    are never logged.
    """
    from tagent.config import SETTINGS
    if SETTINGS.has_krx_login():
        os.environ.setdefault("KRX_ID", SETTINGS.krx_id)
        os.environ.setdefault("KRX_PW", SETTINGS.krx_pw)
        return True
    return bool(os.getenv("KRX_ID") and os.getenv("KRX_PW"))


def _yyyymmdd(d) -> str:
    """Accept 'YYYY-MM-DD', 'YYYYMMDD', or a datetime -> KRX's 'YYYYMMDD'."""
    return pd.Timestamp(d).strftime("%Y%m%d")


def _normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_OHLCV_RENAME)
    missing = [c for c in _OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"KRX OHLCV missing columns {missing}; got {list(raw.columns)}")
    df = df[_OHLCV_COLS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    for c in _OHLCV_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[df["close"] > 0]            # KRX prints zero rows on non-trading days
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def get_krx_history(ticker: str, start, end, interval: str = "1d",
                    save: bool = True, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Daily KR OHLCV via pykrx, in the canonical open/high/low/close/volume schema.

    `ticker` is the 6-digit KRX code (e.g. "005930" = Samsung Electronics).
    Caches to ``data/<TICKER>_<interval>.csv`` so :func:`load_history` can read it.
    """
    stock = _import_pykrx()
    raw = stock.get_market_ohlcv_by_date(_yyyymmdd(start), _yyyymmdd(end), str(ticker))
    df = _normalize_ohlcv(raw)
    if save and not df.empty:
        base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        df.to_csv(base / f"{str(ticker).upper()}_{interval}.csv")
    return df


def _fetch_chunked(fn, ticker: str, start, end, chunk_years: int = 1) -> "pd.DataFrame":
    """Call a pykrx by-date function over [start, end] in yearly chunks and concat.

    The KRX 공매도 endpoints error on long ranges (~>2y) but work fine year by
    year, so we split the request and stitch the daily rows back together. Empty
    or failing chunks are skipped (a year with no data shouldn't abort the rest).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    lo = start
    while lo <= end:
        hi = min(pd.Timestamp(year=lo.year, month=12, day=31), end)
        try:
            d = fn(_yyyymmdd(lo), _yyyymmdd(hi), str(ticker))
            if d is not None and not d.empty:
                parts.append(d)
        except Exception:
            pass  # skip a bad/empty chunk; other years still load
        lo = pd.Timestamp(year=lo.year + 1, month=1, day=1)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _fetch_krx_short_raw(stock, ticker: str, start, end,
                         with_balance: bool = True) -> pd.DataFrame:
    """Pull + merge KRX short volume and balance into a canonical-named frame.

    ``with_balance=False`` skips the 공매도잔고 call (which doubles the request
    count) — useful for large universe pulls where the slow KRX endpoint makes
    balance too expensive. short_balance/short_value are then left NaN.
    """
    vol = _fetch_chunked(stock.get_shorting_volume_by_date, ticker, start, end)
    if vol is None or vol.empty:
        raise ValueError(
            f"No KRX short-selling volume for {ticker}. The 공매도 endpoints need "
            "a working KRX account in KRX_ID / KRX_PW env vars.")

    out = pd.DataFrame(index=pd.to_datetime(vol.index))
    out["short_volume"] = pd.to_numeric(vol[_SHORT_VOLUME_COL].values, errors="coerce")
    out["volume"] = pd.to_numeric(vol[_TOTAL_VOLUME_COL].values, errors="coerce")

    # Balance/value share the same daily date index; align by it (best effort).
    bal = (_fetch_chunked(stock.get_shorting_balance_by_date, ticker, start, end)
           if with_balance else None)
    if bal is not None and not bal.empty:
        bal = bal.copy()
        bal.index = pd.to_datetime(bal.index)
        out["short_balance"] = pd.to_numeric(bal[_SHORT_BALANCE_COL], errors="coerce")
        out["short_value"] = pd.to_numeric(bal[_SHORT_VALUE_COL], errors="coerce")
    else:
        out["short_balance"] = np.nan
        out["short_value"] = np.nan

    out.index.name = "date"
    return out.sort_index()


def krx_short_fetch(start, end, cache: bool = True,
                    data_dir: Optional[Path] = None,
                    with_balance: bool = True) -> FetchFn:
    """Build a ``fetch(symbol)`` for :func:`load_short_selling`, bound to a window.

    The returned callable pulls KR short-selling data for ``[start, end]`` and
    (when ``cache``) writes ``data/<SYMBOL>_short.csv`` so later default CSV loads
    round-trip. Matches the existing ``fetch=`` interface, so the canonical
    normalization (alias resolution, ratio math) is reused unchanged.
    """
    def _fetch(symbol: str) -> pd.DataFrame:
        path = short_csv_path(symbol, data_dir)
        if cache and path.exists():
            return pd.read_csv(path)
        ensure_krx_login()          # 공매도 endpoint is gated -> needs KRX_ID/KRX_PW
        stock = _import_pykrx()
        raw = _fetch_krx_short_raw(stock, symbol, start, end, with_balance=with_balance)
        if cache and not raw.empty:
            raw.to_csv(path, index_label="date")
        return raw

    return _fetch
