# -*- coding: utf-8 -*-
"""Kiwoom real-time intraday quote feed (Task 6) — ISOLATED.

The 5090 box has the Kiwoom **REST API** credentials in .env (KIWOOM_APP_KEY / KIWOOM_SECRET_KEY /
KIWOOM_ENV); the legacy 32-bit OpenAPI+ OCX is NOT installed and the creds are REST keys, so this wires
the REST API (the supported path for these keys). Quotes only — no orders, fully read-only.

Setup (.env):
    KIWOOM_APP_KEY=...        # REST app key   (never in chat/code — read from .env)
    KIWOOM_SECRET_KEY=...     # REST secret key
    KIWOOM_ENV=mock           # 'mock' -> https://mockapi.kiwoom.com ; anything else -> https://api.kiwoom.com

Interface:
    from moneyup_advisor.live.kiwoom_feed import get_quote, KiwoomFeed
    q = get_quote("005930")   # -> {ticker,name,price,change,change_pct,volume,open,high,low,prev_close,ts,source,ok}

CLI (also the acceptance demo):
    python -m moneyup_advisor.live.kiwoom_feed 005930 000660 035420 086520 028300

Isolation: standalone module. Imports only requests + python-dotenv. Does NOT import or touch the
collector, pipeline, daily report, playbook, or dashboard. Login + token refresh + reconnect-on-401 are
handled internally; on any error get_quote returns {"ok": False, "error": ...} so callers degrade to
`pending` rather than crash.
"""
from __future__ import annotations
import datetime as _dt
import os
import threading
import time
from typing import Dict, List, Optional

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_MOCK_ENVS = ("mock", "모의", "dev", "demo", "paper", "sim", "test")
_KST = _dt.timezone(_dt.timedelta(hours=9))

# majors (required) + a few of his theme names as peers
WATCHLIST: List[str] = ["005930", "000660", "035420", "086520", "028300",
                        "079900", "000250", "196170"]


def _ts() -> str:
    return _dt.datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S")


def _to_int(s, signed: bool = False) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip()
    if s in ("", "0", "-", "+"):
        return 0
    try:
        return int(s) if signed else int(s.lstrip("+-") or 0)
    except ValueError:
        return None


def _to_float(s) -> Optional[float]:
    if s is None or str(s).strip() in ("", "-", "+"):
        return None
    try:
        return float(str(s).strip())
    except ValueError:
        return None


class KiwoomFeed:
    """Thread-safe Kiwoom REST quote client with cached OAuth token + reconnect."""

    def __init__(self, env: Optional[str] = None, app_key: Optional[str] = None,
                 secret_key: Optional[str] = None, cache_ttl: float = 2.0, timeout: int = 20):
        self.env = (env or os.getenv("KIWOOM_ENV") or "").lower()
        self.base = "https://mockapi.kiwoom.com" if self.env in _MOCK_ENVS else "https://api.kiwoom.com"
        self._ak = app_key or os.getenv("KIWOOM_APP_KEY")
        self._sk = secret_key or os.getenv("KIWOOM_SECRET_KEY")
        if not (self._ak and self._sk):
            raise RuntimeError("Kiwoom credentials missing — set KIWOOM_APP_KEY / KIWOOM_SECRET_KEY in .env")
        self.timeout = timeout
        self._cache_ttl = cache_ttl
        self._tok: Optional[str] = None
        self._tok_exp: float = 0.0
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._sess = requests.Session()
        self._min_interval = 0.35           # self-throttle: stay under the REST rate limit (~3/s)
        self._last_call = 0.0
        self._throttle_lock = threading.Lock()

    # ---- login / token ----------------------------------------------------- #
    @staticmethod
    def _parse_exp(s) -> float:
        try:                                            # 'YYYYMMDDHHMMSS' in KST
            d = _dt.datetime.strptime(str(s), "%Y%m%d%H%M%S").replace(tzinfo=_KST)
            return d.timestamp()
        except Exception:
            return time.time() + 6 * 3600

    def login(self, force: bool = False) -> str:
        """Fetch/refresh the OAuth token. Cached until 60s before expiry. Raises on hard failure."""
        with self._lock:
            if self._tok and not force and time.time() < self._tok_exp - 60:
                return self._tok
            r = self._sess.post(self.base + "/oauth2/token",
                                json={"grant_type": "client_credentials", "appkey": self._ak, "secretkey": self._sk},
                                timeout=self.timeout)
            r.raise_for_status()
            j = r.json()
            if j.get("return_code") not in (0, None):
                raise RuntimeError(f"Kiwoom token error: {j.get('return_msg')}")
            self._tok = j.get("token") or j.get("access_token")
            if not self._tok:
                raise RuntimeError("Kiwoom token missing in response")
            self._tok_exp = self._parse_exp(j.get("expires_dt"))
            return self._tok

    def _throttle(self):
        with self._throttle_lock:
            gap = time.time() - self._last_call
            if gap < self._min_interval:
                time.sleep(self._min_interval - gap)
            self._last_call = time.time()

    def _call(self, api_id: str, body: dict, path: str = "/api/dostk/stkinfo", retries: int = 3) -> dict:
        headers = {"api-id": api_id, "Content-Type": "application/json;charset=UTF-8"}
        last_exc = None
        for attempt in range(retries + 1):
            headers["Authorization"] = f"Bearer {self.login()}"
            self._throttle()
            try:
                r = self._sess.post(self.base + path, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as e:      # transient network → backoff + retry
                last_exc = e
                if attempt < retries:
                    time.sleep(0.6 * (attempt + 1)); continue
                raise
            if r.status_code in (401, 403) and attempt < retries:   # token expired/revoked → re-login
                self.login(force=True); continue
            if r.status_code == 429 and attempt < retries:          # rate-limited → honor Retry-After / backoff
                wait = _to_float(r.headers.get("Retry-After")) or (0.7 * (attempt + 1))
                time.sleep(max(0.5, wait)); continue
            r.raise_for_status()
            return r.json()
        raise last_exc or RuntimeError("Kiwoom _call exhausted retries")

    # ---- quotes ------------------------------------------------------------ #
    def get_quote(self, ticker: str, use_cache: bool = True) -> dict:
        """Live intraday quote for one ticker. Never raises — returns {'ok': False, ...} on failure."""
        ticker = str(ticker).strip().zfill(6)
        now = time.time()
        if use_cache and ticker in self._cache and now - self._cache[ticker][0] < self._cache_ttl:
            return self._cache[ticker][1]
        try:
            d = self._call("ka10001", {"stk_cd": ticker})       # 주식기본정보요청
            if d.get("return_code") not in (0, None):
                q = {"ticker": ticker, "ok": False, "error": d.get("return_msg") or "non-zero return_code",
                     "ts": _ts(), "source": self._src()}
            else:
                price = _to_int(d.get("cur_prc"))
                chg = _to_int(d.get("pred_pre"), signed=True)
                q = {"ticker": ticker, "name": d.get("stk_nm"), "price": price,
                     "change": chg, "change_pct": _to_float(d.get("flu_rt")),
                     "volume": _to_int(d.get("trde_qty")), "open": _to_int(d.get("open_pric")),
                     "high": _to_int(d.get("high_pric")), "low": _to_int(d.get("low_pric")),
                     "prev_close": (price - chg) if (price is not None and chg is not None) else None,
                     "source": self._src(), "ts": _ts(), "ok": True}
        except Exception as e:
            q = {"ticker": ticker, "ok": False, "error": f"{type(e).__name__}: {e}",
                 "ts": _ts(), "source": self._src()}
        self._cache[ticker] = (now, q)
        return q

    def get_quotes(self, tickers: List[str], gap: float = 0.12) -> Dict[str, dict]:
        """Batch quotes (sequential, gentle on the rate limit)."""
        out = {}
        for t in tickers:
            out[str(t).strip().zfill(6)] = self.get_quote(t)
            time.sleep(gap)
        return out

    def _src(self) -> str:
        return "kiwoom-mock" if self.base.startswith("https://mock") else "kiwoom-real"


# ---- module-level singleton + convenience -------------------------------- #
_FEED: Optional[KiwoomFeed] = None
_FEED_LOCK = threading.Lock()


def feed() -> KiwoomFeed:
    global _FEED
    if _FEED is None:
        with _FEED_LOCK:
            if _FEED is None:
                _FEED = KiwoomFeed()
    return _FEED


def get_quote(ticker: str) -> dict:
    return feed().get_quote(ticker)


def get_quotes(tickers: Optional[List[str]] = None) -> Dict[str, dict]:
    return feed().get_quotes(tickers or WATCHLIST)


if __name__ == "__main__":
    import sys
    tks = [a for a in sys.argv[1:] if a.isdigit()] or WATCHLIST
    f = KiwoomFeed()
    print(f"# Kiwoom feed — base={f.base} env={f.env!r}  ({_ts()} KST)")
    for t, q in f.get_quotes(tks).items():
        if q.get("ok"):
            print(f"  {t} {str(q['name'] or ''):10} price={q['price']:>9,} "
                  f"chg%={q['change_pct']:>6} vol={q['volume']:>13,} src={q['source']}")
        else:
            print(f"  {t} ERR {q.get('error')}")
