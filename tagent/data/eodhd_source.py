"""EODHD (eodhd.com) intraday minute-bar client — KR minute-data vendor evaluation.

API: ``GET https://eodhd.com/api/intraday/{SYMBOL}.{EXCH}`` with query params
``api_token``, ``interval`` (1m / 5m / 1h), ``fmt=json``, ``from`` / ``to`` (UNIX
seconds, UTC). The 1-minute interval allows at most 120 days per request, so we page
OLDER history in 120-day windows. Korean stocks use the ``.KO`` exchange suffix
(KOSPI), e.g. ``005930.KO``.

Bars are returned in the intraday backtester's schema (timestamp index + OHLCV).
EODHD timestamps are UTC; KR bars are stored as naive KST to line up with the Kiwoom
bars we already have, so a saved ``data/<SYM>_1m.csv`` feeds straight into
``intraday_history.load_intraday``.

``requests`` is lazy + injectable for offline tests; the api_token travels only in the
query string and is never logged. HTTP/auth errors are surfaced (not masked as empty).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from tagent.config import DATA_DIR

INTRADAY_URL = "https://eodhd.com/api/intraday/{symbol}"
VALID_INTERVALS = {"1m", "5m", "1h"}
MAX_DAYS = {"1m": 120, "5m": 600, "1h": 7200}          # EODHD per-request window caps
_OHLCV = ["open", "high", "low", "close", "volume"]
_KST = timezone(timedelta(hours=9))


class EodhdError(RuntimeError):
    """Raised on an EODHD HTTP/auth error or an error envelope."""


def build_request(symbol: str, exchange: str = "KO", interval: str = "1m",
                  from_ts: Optional[int] = None, to_ts: Optional[int] = None,
                  api_token: str = "") -> Tuple[str, dict]:
    """(url, params) for one intraday request. ``symbol`` may already include the
    exchange (``005930.KO``) or be bare (``005930`` + ``exchange='KO'``)."""
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval {interval} not in {sorted(VALID_INTERVALS)}")
    full = symbol if "." in str(symbol) else f"{symbol}.{exchange}"
    url = INTRADAY_URL.format(symbol=full)
    params = {"api_token": api_token, "interval": interval, "fmt": "json"}
    if from_ts is not None:
        params["from"] = int(from_ts)
    if to_ts is not None:
        params["to"] = int(to_ts)
    return url, params


def _utc_unix_to_kst_naive(unix_s) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(float(unix_s), tz=timezone.utc).astimezone(
            _KST).replace(tzinfo=None)
    except (TypeError, ValueError, OSError):
        return None


def _utc_str_to_kst_naive(s) -> Optional[datetime]:
    try:
        dt = datetime.strptime(str(s)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.astimezone(_KST).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def parse_intraday(data) -> List[dict]:
    """EODHD intraday JSON (a list of bars) -> [{timestamp, open, high, low, close,
    volume}] with timestamps converted to naive KST. Raises EodhdError on an error
    envelope (a dict instead of a list)."""
    if isinstance(data, dict):
        raise EodhdError(f"EODHD error: {data.get('message') or data.get('error') or data}")
    if not isinstance(data, list):
        raise EodhdError("EODHD returned an unexpected (non-list) intraday payload")
    rows: List[dict] = []
    for k in data:
        if not isinstance(k, dict):
            continue
        ts = (_utc_unix_to_kst_naive(k.get("timestamp")) if k.get("timestamp") is not None
              else _utc_str_to_kst_naive(k.get("datetime")))
        if ts is None:
            continue
        c = k.get("close")
        if c is None:
            continue                                   # EODHD emits null OHLC for empty minutes
        rows.append({"timestamp": ts,
                     "open": float(k.get("open") or 0), "high": float(k.get("high") or 0),
                     "low": float(k.get("low") or 0), "close": float(c),
                     "volume": float(k.get("volume") or 0)})
    return rows


def bars_to_df(rows: List[dict]) -> pd.DataFrame:
    """Rows -> backtester schema: timestamp index, OHLCV float, sorted, de-duped,
    zero-close dropped."""
    if not rows:
        return pd.DataFrame(columns=_OHLCV, index=pd.DatetimeIndex([], name="timestamp"))
    df = pd.DataFrame(rows).dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"] > 0]
    df.index.name = "timestamp"
    return df[_OHLCV].astype(float)


def _requests():
    import requests
    return requests


def fetch_minutes(symbol: str, *, api_token: str, exchange: str = "KO", interval: str = "1m",
                  session=None, end_ts: Optional[int] = None, max_windows: int = 16,
                  pause: float = 0.0) -> Tuple[pd.DataFrame, dict]:
    """Fetch intraday bars for ``symbol``, paging OLDER in per-request windows up to
    ``max_windows``. Returns (df, info) with windows/rows/date-range. ``api_token`` is
    query-only and never logged; HTTP/auth errors raise EodhdError."""
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval {interval} not in {sorted(VALID_INTERVALS)}")
    sess = session or _requests()
    win = MAX_DAYS[interval] * 86400
    to_ts = int(end_ts if end_ts is not None
                else datetime.now(tz=timezone.utc).timestamp())
    all_rows: List[dict] = []
    windows = 0
    prev_oldest: Optional[int] = None
    while windows < max_windows:
        from_ts = to_ts - win
        url, params = build_request(symbol, exchange, interval, from_ts, to_ts, api_token)
        resp = sess.get(url, params=params, timeout=30)
        status = getattr(resp, "status_code", 200)
        text = getattr(resp, "text", "") or ""
        try:
            data = resp.json()
        except Exception:
            data = None
        if status >= 400:                                  # surface plan/auth errors verbatim
            msg = (data.get("message") if isinstance(data, dict) else None) or text[:160] or "rejected"
            raise EodhdError(f"HTTP {status}: {msg}")
        if data is None:
            raise EodhdError(f"EODHD returned non-JSON (HTTP {status}): {text[:120]}")
        rows = parse_intraday(data)
        windows += 1
        if not rows:
            break                                      # window empty -> history exhausted
        all_rows.extend(rows)
        oldest = min(int(k["timestamp"]) for k in data
                     if isinstance(k, dict) and k.get("timestamp") is not None)
        if prev_oldest is not None and oldest >= prev_oldest:
            break                                      # no progress -> stop
        prev_oldest = oldest
        to_ts = oldest - 1
        if pause:
            import time as _t
            _t.sleep(pause)
    df = bars_to_df(all_rows)
    info = {"windows": windows, "rows": int(len(df)),
            "start": df.index.min().isoformat() if len(df) else None,
            "end": df.index.max().isoformat() if len(df) else None,
            "days": int(df.index.normalize().nunique()) if len(df) else 0}
    return df, info


def minute_csv_path(symbol: str, data_dir=None) -> Path:
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / f"{str(symbol).upper()}_1m.csv"


def save_minutes(df: pd.DataFrame, symbol: str, data_dir=None) -> Path:
    """Write data/<SYM>_1m.csv in the canonical (timestamp + OHLCV) schema."""
    path = minute_csv_path(symbol, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="timestamp")
    return path
