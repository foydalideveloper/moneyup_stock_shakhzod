"""iTick (itick.org) minute-bar (OHLCV) client — vendor evaluation for intraday.

API: ``GET https://api.itick.org/stock/kline`` with headers ``accept`` + ``token``
and query params ``region`` (market), ``code``, ``kType`` (1=1min, 2=5min, 3=15min,
4=30min, 5=1hr, 8=1day), ``limit``, and optional ``et`` (end timestamp ms). Response:
``{"code":0,"msg":null,"data":[{"t":<ms>,"o","h","l","c","v","tu"}]}``.

We page OLDER history by setting ``et`` to the oldest bar's timestamp minus one, and
save into the intraday backtester's schema (timestamp index + OHLCV) so a
``data/<SYM>_1m.csv`` feeds straight into ``intraday_history.load_intraday``.

KR caveat: Korea/KOSPI is NOT in iTick's documented region list — this client lets us
attempt ``region="KR"`` and report the exact server response. Timestamps are epoch ms
(UTC); KR bars are stored as naive KST to line up with the Kiwoom bars we already have.

``requests`` is lazy + injectable, so tests parse canned payloads with no network.
The token travels only in the request header and is never logged.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from tagent.config import DATA_DIR

KLINE_URL = "https://api.itick.org/stock/kline"
_KTYPE_FOR_MIN = {1: 1, 5: 2, 15: 3, 30: 4, 60: 5}     # interval-minutes -> iTick kType
_OHLCV = ["open", "high", "low", "close", "volume"]
_KST = timezone(timedelta(hours=9))


class ItickError(RuntimeError):
    """Raised when iTick returns a non-zero code (bad region/key/limit, etc.)."""


def build_request(code: str, region: str = "KR", interval: int = 1, limit: int = 1000,
                  et: Optional[int] = None, token: str = "") -> Tuple[str, dict, dict]:
    """(url, headers, params) for one kline page. ``et`` (ms) pages older history."""
    ktype = _KTYPE_FOR_MIN.get(int(interval))
    if ktype is None:
        raise ValueError(f"interval {interval}min not in {sorted(_KTYPE_FOR_MIN)}")
    headers = {"accept": "application/json", "token": token}
    params = {"region": region, "code": str(code), "kType": ktype, "limit": int(limit)}
    if et is not None:
        params["et"] = int(et)
    return KLINE_URL, headers, params


def _to_kst_naive(ms) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).astimezone(
            _KST).replace(tzinfo=None)
    except (TypeError, ValueError, OSError):
        return None


def parse_klines(data: dict) -> List[dict]:
    """iTick kline JSON -> [{timestamp, open, high, low, close, volume}]. Raises
    ItickError on a non-zero response code."""
    if not isinstance(data, dict):
        raise ItickError("iTick returned a non-dict response")
    code = data.get("code")
    if code is None and "data" not in data:               # error envelope, e.g. 401 {"message":...}
        raise ItickError(f"iTick request failed: {data.get('message') or data}")
    if str(code) not in ("0", "None"):
        raise ItickError(f"iTick error code={code}: {data.get('msg') or data.get('message')}")
    rows: List[dict] = []
    for k in (data.get("data") or []):
        if not isinstance(k, dict):
            continue
        ts = _to_kst_naive(k.get("t"))
        if ts is None:
            continue
        rows.append({"timestamp": ts,
                     "open": float(k.get("o", 0) or 0), "high": float(k.get("h", 0) or 0),
                     "low": float(k.get("l", 0) or 0), "close": float(k.get("c", 0) or 0),
                     "volume": float(k.get("v", 0) or 0)})
    return rows


def bars_to_df(rows: List[dict]) -> pd.DataFrame:
    """Rows -> backtester schema: timestamp index, OHLCV float, sorted, de-duped,
    zero-close bars dropped."""
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


def fetch_klines(code: str, *, token: str, region: str = "KR", interval: int = 1,
                 limit: int = 1000, max_pages: int = 20, session=None,
                 pause: float = 0.0) -> Tuple[pd.DataFrame, dict]:
    """Fetch minute bars for ``code``, paging OLDER via ``et`` up to ``max_pages``.
    Returns (df, info) with pages/rows/date-range. ``token`` is header-only, not logged.
    """
    sess = session or _requests()
    all_rows: List[dict] = []
    et: Optional[int] = None
    pages = 0
    seen_oldest: Optional[int] = None
    while pages < max_pages:
        url, headers, params = build_request(code, region, interval, limit, et, token)
        resp = sess.get(url, headers=headers, params=params, timeout=15)
        status = getattr(resp, "status_code", 200)
        try:
            data = resp.json()
        except Exception as e:
            raise ItickError(f"iTick returned non-JSON (HTTP {status}): {e}")
        if status >= 400:                                  # surface auth/HTTP errors clearly
            msg = data.get("message") if isinstance(data, dict) else None
            raise ItickError(f"HTTP {status}: {msg or 'request rejected'}")
        rows = parse_klines(data)              # raises on non-zero code
        pages += 1
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ms = min(int(pd.Timestamp(r["timestamp"]).tz_localize(_KST).timestamp() * 1000)
                        for r in rows)
        if seen_oldest is not None and oldest_ms >= seen_oldest:
            break                              # no progress -> stop (avoid loop)
        seen_oldest = oldest_ms
        et = oldest_ms - 1
        if len(rows) < limit:
            break                              # last (partial) page
        import time as _t
        if pause:
            _t.sleep(pause)
    df = bars_to_df(all_rows)
    info = {"pages": pages, "rows": int(len(df)),
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
