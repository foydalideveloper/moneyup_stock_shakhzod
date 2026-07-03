"""REAL market-data price table (Kiwoom, with pykrx fallback) for the daily YouTube report.

``fetch_price_table(tickers)`` returns one grounded row per ticker:
    {name, ticker, current, today_open, prev_close, week_ago_close, month_ago_close, change_pct,
     source}
where current + today_open come from Kiwoom real-time (minute TR ka10080), and
prev_close / week_ago_close (~5 trading days) / month_ago_close (~21 trading days) come from Kiwoom
daily bars (ka10081), falling back to pykrx if the Kiwoom daily TR is unavailable. ``change_pct`` is
vs prev_close. ANY field that can't be fetched is set to None — we NEVER invent a number.

This is REAL market data and is kept SEPARATE from the video-grounded insights (tagged
``source="키움/KRX"``). Each ticker's row is cached ~60s to respect Kiwoom rate limits. Works under
mock env; the token rides only in the Authorization header and is never logged. ``requests`` is lazy
and the fetchers are injectable, so the unit tests run fully mocked (no network).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from tagent.config import KIWOOM_REST_URLS

_KST = timezone(timedelta(hours=9))
_DEBUG_COUNT = [0]


def _debug(msg: str) -> None:
    """One-line diagnostic to stderr (first few only), opt-in via YT_PRICES_DEBUG. Never prints
    secrets (tokens/keys) — only TR ids, return codes, row counts."""
    if os.getenv("YT_PRICES_DEBUG") and _DEBUG_COUNT[0] < 8:
        _DEBUG_COUNT[0] += 1
        print(msg, file=sys.stderr)
CHART_ENDPOINT = "/api/dostk/chart"
DAILY_API_ID = "ka10081"                                  # 주식일봉차트조회
CACHE_TTL_SECONDS = 60.0

# defensive field aliases for the ka10081 daily-bar payload
_DAILY_ARRAY = ("stk_dt_pole_chart_qry", "stk_dt_pole_chart", "chart")
_DATE_KEYS = ("dt", "date", "cntr_dt", "stck_bsop_date")
_OPEN_KEYS = ("open_pric", "opn_prc", "stck_oprc", "open")
_CLOSE_KEYS = ("cur_prc", "close_pric", "clpr", "stck_clpr", "close")

_PRICE_CACHE: Dict[str, Tuple[float, dict]] = {}         # "env:ticker" -> (epoch, row)

ROW_FIELDS = ("name", "ticker", "current", "today_open", "prev_close",
              "week_ago_close", "month_ago_close", "change_pct")


class KiwoomPriceError(RuntimeError):
    """Raised when the daily-bar TR is rejected by the server."""


def _requests():
    import requests  # lazy
    return requests


def _num(raw) -> Optional[float]:
    """Kiwoom sign-prefixed integer string -> positive float magnitude; None if blank/garbage."""
    s = str(raw if raw is not None else "").strip().replace(",", "").lstrip("+-")
    if not s:
        return None
    try:
        return abs(float(s))
    except ValueError:
        return None


def _today_kst(now=None) -> date:
    if isinstance(now, datetime):
        dt = now if now.tzinfo else now.replace(tzinfo=_KST)
        return dt.astimezone(_KST).date()
    if isinstance(now, date):
        return now
    return datetime.now(_KST).date()


def _ymd(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def _pick(row: dict, keys: Sequence[str]):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return v
    return None


# --------------------------------------------------------------------------- #
# Kiwoom daily bars (ka10081) + pykrx fallback
# --------------------------------------------------------------------------- #
def parse_daily_bars(data: dict) -> List[dict]:
    """ka10081 JSON -> [{date:int(YYYYMMDD), open, close}] ascending by date (defensive)."""
    rows: List[dict] = []
    lst = None
    if isinstance(data, dict):
        for k in _DAILY_ARRAY:
            if isinstance(data.get(k), list):
                lst = data[k]
                break
        if lst is None:
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    lst = v
                    break
    for bar in lst or []:
        if not isinstance(bar, dict):
            continue
        ds = "".join(ch for ch in str(_pick(bar, _DATE_KEYS) or "") if ch.isdigit())[:8]
        if len(ds) != 8:
            continue
        rows.append({"date": int(ds), "open": _num(_pick(bar, _OPEN_KEYS)),
                     "close": _num(_pick(bar, _CLOSE_KEYS))})
    rows.sort(key=lambda r: r["date"])
    return rows


def fetch_daily_closes(ticker: str, *, auth, session=None, env: str = "mock",
                       base_url: Optional[str] = None, now=None) -> List[dict]:
    """Kiwoom ka10081 daily bars for ``ticker`` -> [{date, open, close}] ascending. Raises
    KiwoomPriceError on a rejected TR (the caller then falls back to pykrx)."""
    base = base_url or KIWOOM_REST_URLS.get(env.lower(), KIWOOM_REST_URLS["mock"])
    token = auth.get_token()
    sess = session or _requests()
    headers = {"Content-Type": "application/json;charset=UTF-8", "authorization": f"Bearer {token}",
               "api-id": DAILY_API_ID, "cont-yn": "N", "next-key": ""}
    body = {"stk_cd": str(ticker), "base_dt": str(_ymd(_today_kst(now))), "upd_stkpc_tp": "1"}
    resp = sess.post(f"{base}{CHART_ENDPOINT}", headers=headers, json=body, timeout=10)
    try:
        data = resp.json()
    except Exception as e:
        raise KiwoomPriceError(f"{DAILY_API_ID} returned non-JSON: {e}")
    code = data.get("return_code")
    _debug(f"[prices] {DAILY_API_ID} env={env} ticker={ticker} return_code={code} "
           f"keys={list(data)[:6] if isinstance(data, dict) else type(data).__name__}")
    if str(code) not in ("0", "None"):
        raise KiwoomPriceError(f"{DAILY_API_ID} rejected (return_code={code}): {data.get('return_msg')}")
    rows = parse_daily_bars(data)
    _debug(f"[prices] {DAILY_API_ID} ticker={ticker} parsed {len(rows)} daily bars")
    return rows


def _pykrx_daily(ticker: str, now=None) -> List[dict]:
    """pykrx daily OHLCV fallback -> [{date, open, close}] ascending. [] on any failure."""
    try:
        from pykrx import stock
        end = _today_kst(now)
        start = end - timedelta(days=45)
        df = stock.get_market_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), str(ticker))
        out = []
        for idx, r in df.iterrows():
            ds = "".join(ch for ch in str(idx) if ch.isdigit())[:8]
            if len(ds) == 8:
                out.append({"date": int(ds), "open": _num(r.get("시가")), "close": _num(r.get("종가"))})
        out.sort(key=lambda r: r["date"])
        return out
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Kiwoom real-time current + today_open (minute TR ka10080)
# --------------------------------------------------------------------------- #
def _minute_current_open(ticker: str, *, auth, session=None, env: str = "mock",
                         base_url: Optional[str] = None) -> Tuple[Optional[float], Optional[float]]:
    """(current, today_open) from the latest trading day's minute bars; (None, None) on empty."""
    try:
        from tagent.data.kiwoom_minute import fetch_minute_bars
        df, _ = fetch_minute_bars(ticker, interval=1, auth=auth, session=session, env=env,
                                  base_url=base_url, max_pages=1)
        if df is None or df.empty:
            return None, None
        last_day = df.index[-1].date()
        day = df[[ts.date() == last_day for ts in df.index]]
        if day.empty:
            return None, None
        return float(day["close"].iloc[-1]), float(day["open"].iloc[0])
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# one row (every field independently None on failure — never invented)
# --------------------------------------------------------------------------- #
def _anchors(daily_rows: List[dict], today_int: int):
    """(prev_close, week_ago_close, month_ago_close, today_open, today_close) from daily bars.
    Anchors are COMPLETED sessions strictly before today; today's (partial) bar gives open/close."""
    comp = [r for r in daily_rows if r["date"] < today_int]
    today_row = next((r for r in daily_rows if r["date"] == today_int), None)
    prev_close = comp[-1]["close"] if comp else None
    week_ago = comp[-5]["close"] if len(comp) >= 5 else None       # ~5 trading days ago
    month_ago = comp[-21]["close"] if len(comp) >= 21 else None    # ~21 trading days ago
    d_open = today_row["open"] if today_row else None
    d_close = today_row["close"] if today_row else None
    return prev_close, week_ago, month_ago, d_open, d_close


def build_price_row(ticker: str, name: str, *, minute, daily_rows, today_int: int) -> dict:
    """Assemble one grounded row. ``minute`` = (current, open) from real-time; ``daily_rows`` =
    [{date,open,close}]. Missing fields stay None; change_pct is vs prev_close."""
    prev_close, week_ago, month_ago, d_open, d_close = _anchors(daily_rows or [], today_int)
    m_current, m_open = (minute or (None, None))
    current = m_current if m_current is not None else d_close
    today_open = m_open if m_open is not None else d_open
    change_pct = None
    if current is not None and prev_close not in (None, 0):
        change_pct = round((current - prev_close) / prev_close * 100.0, 2)
    return {"name": name, "ticker": ticker, "current": current, "today_open": today_open,
            "prev_close": prev_close, "week_ago_close": week_ago, "month_ago_close": month_ago,
            "change_pct": change_pct, "source": "키움/KRX"}


def _names_and_codes(tickers, names_override=None):
    """Accept a list of codes or a {code: name} dict; resolve display names."""
    from tagent.news.youtube_report import DEFAULT_GIANTS, LINKED_GLOBALS
    from tagent.news.youtube_source import TICKER_NAMES
    if isinstance(tickers, dict):
        codes, names = list(tickers.keys()), dict(tickers)
    else:
        codes = [str(t) for t in (tickers or [])]
        names = {}
    over = names_override or {}

    def nm(code):
        return (over.get(code) or names.get(code) or DEFAULT_GIANTS.get(code)
                or LINKED_GLOBALS.get(code) or TICKER_NAMES.get(code) or code)
    return codes, {c: nm(c) for c in codes}


def fetch_price_table(tickers, auth=None, now=None, *, session=None, env: Optional[str] = None,
                      base_url: Optional[str] = None, minute_fn: Optional[Callable] = None,
                      daily_fn: Optional[Callable] = None, pykrx_fn: Optional[Callable] = None,
                      names: Optional[dict] = None, clock: Optional[Callable[[], float]] = None,
                      use_cache: bool = True) -> List[dict]:
    """One real-data row per ticker (see module docstring). ``auth`` is a KiwoomAuth (None -> Kiwoom
    skipped, pykrx still tried for the daily closes). Rows are cached ~60s per (env, ticker). Fetchers
    are injectable for tests. Never invents a number — unfetchable fields stay None."""
    env = env or (getattr(auth, "env", None) or "mock")
    clk = clock or time.time
    today_int = _ymd(_today_kst(now))

    # default fetchers (real Kiwoom / pykrx) unless injected
    if minute_fn is None:
        minute_fn = (lambda t: _minute_current_open(t, auth=auth, session=session, env=env,
                                                     base_url=base_url)) if auth is not None \
            else (lambda t: (None, None))
    if daily_fn is None:
        def daily_fn(t):
            if auth is None:
                return []
            try:
                return fetch_daily_closes(t, auth=auth, session=session, env=env,
                                          base_url=base_url, now=now)
            except Exception:
                return []
    if pykrx_fn is None:
        pykrx_fn = lambda t: _pykrx_daily(t, now=now)

    codes, name_map = _names_and_codes(tickers, names)
    out: List[dict] = []
    for code in codes:
        key = f"{env}:{code}"
        if use_cache:
            hit = _PRICE_CACHE.get(key)
            if hit and (clk() - hit[0]) < CACHE_TTL_SECONDS:
                out.append(hit[1])
                continue
        try:
            minute = minute_fn(code)
        except Exception:
            minute = (None, None)
        try:
            daily_rows = daily_fn(code) or []
        except Exception:
            daily_rows = []
        if not daily_rows:                                   # Kiwoom daily unavailable -> pykrx
            try:
                daily_rows = pykrx_fn(code) or []
                _debug(f"[prices] {code} Kiwoom daily empty -> pykrx returned {len(daily_rows)} rows")
            except Exception as e:
                daily_rows = []
                _debug(f"[prices] {code} pykrx fallback failed: {str(e)[:80]}")
        row = build_price_row(code, name_map.get(code, code), minute=minute,
                              daily_rows=daily_rows, today_int=today_int)
        _debug(f"[prices] {code} current={row['current']} prev={row['prev_close']} chg={row['change_pct']}")
        if use_cache:
            _PRICE_CACHE[key] = (clk(), row)
        out.append(row)
    return out
