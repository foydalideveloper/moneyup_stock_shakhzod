"""Kiwoom 주식분봉차트조회 (ka10080) — minute-bar REST fetch with continuation.

Pulls stock minute bars over Kiwoom's REST API and returns them in the exact schema
the intraday backtester expects (timestamp index + open/high/low/close/volume), so a
saved ``data/<SYM>_<interval>m.csv`` feeds straight into
:func:`tagent.data.intraday_history.load_intraday`.

API (openapi.kiwoom.com → 차트):
  * ``POST {base}/api/dostk/chart`` with headers ``api-id: ka10080`` +
    ``authorization: Bearer <token>``,
  * body ``{stk_cd, tic_scope, upd_stkpc_tp}`` (tic_scope = 1/3/5/10/15/30/60 min;
    upd_stkpc_tp = "1" applies adjusted prices),
  * **continuation (연속조회):** the response carries ``cont-yn`` / ``next-key``
    headers; resend with ``cont-yn: Y`` + ``next-key: <value>`` to page through
    older history until ``cont-yn`` is no longer ``Y``.

Field parsing is DEFENSIVE: the per-bar OHLCV keys follow Kiwoom's standard naming
(``cntr_tm``, ``cur_prc``, ``open_pric``, ``high_pric``, ``low_pric``, ``trde_qty``)
but we accept documented aliases and fall back to the first list in the payload, so
a minor naming difference surfaces as parsed data rather than a crash.

Mock-first: ``requests`` is injected in tests so nothing hits the network. The token
lives only in the Authorization header and is never logged.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from tagent.config import KIWOOM_REST_URLS
from tagent.feeds.kiwoom_feed import _to_number

CHART_ENDPOINT = "/api/dostk/chart"
MINUTE_API_ID = "ka10080"
VALID_INTERVALS = (1, 3, 5, 10, 15, 30, 60)
_OHLCV = ["open", "high", "low", "close", "volume"]

# Response array keys (standard first, wrapper alias second), then per-bar field aliases.
_ARRAY_KEYS = ("stk_min_pole_chart_qry", "chart")
_TIME_KEYS = ("cntr_tm", "cntr_dt", "dt", "stck_cntg_hour")
_OPEN_KEYS = ("open_pric", "opn_prc", "open")
_HIGH_KEYS = ("high_pric", "hgpr", "high")
_LOW_KEYS = ("low_pric", "lwpr", "low")
_CLOSE_KEYS = ("cur_prc", "close_pric", "stck_prpr", "close")
_VOL_KEYS = ("trde_qty", "cntg_vol", "volume")


class KiwoomMinuteError(RuntimeError):
    """Raised when the ka10080 request is rejected by the server."""


def _requests():
    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise KiwoomMinuteError("requests not installed. Run: pip install -r requirements.txt") from e
    return requests


def build_minute_request(base_url: str, token: str, stk_cd: str, interval: int = 1,
                         upd_stkpc_tp: str = "1", cont_yn: str = "N", next_key: str = "",
                         base_date: str = "") -> Tuple[str, dict, dict]:
    """Construct (url, headers, body) for one ka10080 page. ``cont_yn``/``next_key``
    carry the continuation cursor from the previous response's headers."""
    if int(interval) not in VALID_INTERVALS:
        raise ValueError(f"interval {interval} not in {VALID_INTERVALS}")
    url = f"{base_url}{CHART_ENDPOINT}"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "authorization": f"Bearer {token}",
        "api-id": MINUTE_API_ID,
        "cont-yn": cont_yn,
        "next-key": next_key,
    }
    body = {"stk_cd": str(stk_cd), "tic_scope": str(int(interval)), "upd_stkpc_tp": str(upd_stkpc_tp)}
    if base_date:
        body["date"] = str(base_date)                    # optional 기준일자 (recent addition)
    return url, headers, body


def _pick(bar: dict, keys: Sequence[str]):
    for k in keys:
        v = bar.get(k)
        if v not in (None, ""):
            return v
    return None


def _parse_ts(raw) -> Optional[datetime]:
    s = "".join(ch for ch in str(raw).strip() if ch.isdigit())
    for ln, fmt in ((14, "%Y%m%d%H%M%S"), (12, "%Y%m%d%H%M"), (8, "%Y%m%d")):
        if len(s) >= ln:
            try:
                return datetime.strptime(s[:ln], fmt)
            except ValueError:
                continue
    return None


def _first_list(data: dict) -> list:
    for k in _ARRAY_KEYS:
        v = data.get(k)
        if isinstance(v, list):
            return v
    for v in data.values():                              # fallback: any list in the payload
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


def parse_minute_bars(data: dict) -> List[dict]:
    """ka10080 JSON -> list of {timestamp, open, high, low, close, volume} rows.

    Prices arrive as sign-prefixed Korean integer strings ("+74100"); ``_to_number``
    strips the sign. Bars with no valid time are skipped."""
    rows: List[dict] = []
    for bar in _first_list(data):
        if not isinstance(bar, dict):
            continue
        ts = _parse_ts(_pick(bar, _TIME_KEYS))
        if ts is None:
            continue
        rows.append({
            "timestamp": ts,
            "open": _to_number(_pick(bar, _OPEN_KEYS)),
            "high": _to_number(_pick(bar, _HIGH_KEYS)),
            "low": _to_number(_pick(bar, _LOW_KEYS)),
            "close": _to_number(_pick(bar, _CLOSE_KEYS)),
            "volume": _to_number(_pick(bar, _VOL_KEYS)),
        })
    return rows


def bars_to_df(rows: List[dict]) -> pd.DataFrame:
    """Rows -> the backtester's schema: timestamp index, OHLCV float columns, sorted,
    de-duplicated, zero-price (non-trading) bars dropped."""
    if not rows:
        return pd.DataFrame(columns=_OHLCV, index=pd.DatetimeIndex([], name="timestamp"))
    df = pd.DataFrame(rows).dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df["close"] > 0]                             # drop empty/non-trading rows
    df.index.name = "timestamp"
    return df[_OHLCV].astype(float)


def _cont_headers(resp) -> Tuple[str, str]:
    h = getattr(resp, "headers", {}) or {}
    cont = str(h.get("cont-yn") or h.get("cont_yn") or "N").strip().upper()
    nxt = (h.get("next-key") or h.get("next_key") or "").strip()
    return cont, nxt


def fetch_minute_bars(stk_cd: str, interval: int = 1, *, auth, session=None,
                      env: str = "mock", base_url: Optional[str] = None,
                      upd_stkpc_tp: str = "1", max_pages: int = 100, pause: float = 0.0,
                      base_date: str = "") -> Tuple[pd.DataFrame, dict]:
    """Fetch minute bars for ``stk_cd``, following continuation up to ``max_pages``.

    ``auth`` is a :class:`tagent.feeds.kiwoom_auth.KiwoomAuth` (its token is used and
    never logged). Returns ``(df, info)`` where ``info`` has pages / rows / date range.
    """
    base = base_url or KIWOOM_REST_URLS.get(env.lower(), KIWOOM_REST_URLS["mock"])
    token = auth.get_token()
    sess = session or _requests()
    all_rows: List[dict] = []
    cont_yn, next_key, pages = "N", "", 0
    while pages < max_pages:
        url, headers, body = build_minute_request(
            base, token, stk_cd, interval, upd_stkpc_tp, cont_yn, next_key, base_date)
        resp = sess.post(url, headers=headers, json=body, timeout=10)
        try:
            data = resp.json()
        except Exception as e:
            raise KiwoomMinuteError(f"ka10080 returned non-JSON: {e}")
        code = data.get("return_code")
        if str(code) not in ("0", "None"):
            raise KiwoomMinuteError(
                f"ka10080 rejected (return_code={code}): {data.get('return_msg')}")
        rows = parse_minute_bars(data)
        all_rows.extend(rows)
        pages += 1
        cont, nxt = _cont_headers(resp)
        if cont != "Y" or not nxt or not rows:
            break
        cont_yn, next_key = "Y", nxt
        if pause:
            time.sleep(pause)
    df = bars_to_df(all_rows)
    info = {"pages": pages, "rows": int(len(df)),
            "start": df.index.min().isoformat() if len(df) else None,
            "end": df.index.max().isoformat() if len(df) else None,
            "days": int(df.index.normalize().nunique()) if len(df) else 0}
    return df, info


def minute_csv_path(symbol: str, interval: int = 1, data_dir=None) -> "Path":
    from pathlib import Path
    from tagent.config import DATA_DIR
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / f"{str(symbol).upper()}_{int(interval)}m.csv"


def save_minutes(df: pd.DataFrame, symbol: str, interval: int = 1, data_dir=None) -> "Path":
    """Write to data/<SYM>_<interval>m.csv in the canonical (timestamp + OHLCV) schema
    that the intraday backtester reads (1-min files load directly)."""
    path = minute_csv_path(symbol, interval, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="timestamp")
    return path
