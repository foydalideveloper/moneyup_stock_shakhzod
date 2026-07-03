"""Local web dashboard backend (FastAPI).

Two INDEPENDENT markets, two independent panels:

* **US stocks (Alpaca)** -> ``/signals`` (model BUY/HOLD/SELL per symbol) and
  ``/account`` (paper equity, day P&L, cash, positions).
* **Crypto (Binance via CryptoFeed)** -> ``/orderbook`` (10-level ladder) and
  ``/memory`` (vanished-level tags + absorption/spoof/imbalance). Requesting a
  symbol that isn't subscribed yet calls ``CryptoFeed.subscribe_symbol`` to start
  it.

The FastAPI app is built by :func:`create_app` from two injected *sources*, so
endpoints are unit-tested with fakes (no Alpaca, no network). The runner script
wires the real Alpaca + CryptoFeed sources.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from tagent.feeds.crypto_feed import normalize_symbol

_STATIC = Path(__file__).resolve().parent / "static" / "dashboard.html"

# Supported chart timeframes. Binance has no native 10m, so it is resampled from
# 1m klines; Alpaca takes an (amount, unit). `days` sizes the lookback so each
# timeframe yields ~120 bars.
TIMEFRAMES = {
    "1m":  {"minutes": 1,    "binance": "1m",  "alpaca": (1, "Minute"), "days": 5},
    "10m": {"minutes": 10,   "binance": None,  "alpaca": (10, "Minute"), "days": 12},
    "30m": {"minutes": 30,   "binance": "30m", "alpaca": (30, "Minute"), "days": 25},
    "1h":  {"minutes": 60,   "binance": "1h",  "alpaca": (1, "Hour"),   "days": 45},
    "1d":  {"minutes": 1440, "binance": "1d",  "alpaca": (1, "Day"),    "days": 400},
}


# --------------------------------------------------------------------------- #
# YouTube report (Phase 4) — helpers for the dashboard generate/download section
# --------------------------------------------------------------------------- #
import re as _re

_VIDEO_ID_RE = _re.compile(r"(?:v=|/shorts/|youtu\.be/|/embed/|/live/)([A-Za-z0-9_-]{11})")


def resolve_video_id(url: str) -> str:
    """Extract an 11-char YouTube video id from a URL or a bare id ("" if none)."""
    s = str(url or "").strip()
    if _re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    m = _VIDEO_ID_RE.search(s)
    return m.group(1) if m else ""


def safe_report_name(name: str):
    """A validated report filename (basename only, .docx/.pdf), or None. Refuses any path component
    or traversal so the file endpoint can ONLY ever serve from data/reports."""
    n = str(name or "")
    if not n or n != Path(n).name or "/" in n or "\\" in n or ".." in n:
        return None
    if Path(n).suffix.lower() not in (".docx", ".pdf"):
        return None
    return n


def _coerce_watchlist(watchlist):
    """A {code: name} dict from a list of codes / a dict / None (None -> default top-20)."""
    from tagent.news.youtube_report import DEFAULT_GIANTS
    if isinstance(watchlist, dict) and watchlist:
        return dict(watchlist)
    if isinstance(watchlist, (list, tuple)) and watchlist:
        return {str(c): DEFAULT_GIANTS.get(str(c), str(c)) for c in watchlist}
    return None


def _report_translate_fn(lang):
    """A KO->EN BATCH translator on the FAST interactive model (one call for the whole report, so EN
    isn't ~2x KO), or None when not English / no key."""
    if lang != "en":
        return None
    try:
        from tagent.config import SETTINGS
        if not SETTINGS.has_gemini_key():
            return None
        from tagent.gemini import GeminiClient, translate_batch_fn
        model = getattr(SETTINGS, "gemini_interactive_model", None) or SETTINGS.gemini_model
        return translate_batch_fn(GeminiClient(SETTINGS.gemini_api_key, model=model))
    except Exception:
        return None


def _default_report_builder(lang="ko", end=None, watchlist=None):
    """Real window report: build_youtube_report (yesterday 00:00 KST -> end/now) with Kiwoom prices."""
    from tagent.news.youtube_report import build_youtube_report
    return build_youtube_report(start=None, end=end, lang=lang, watchlist=_coerce_watchlist(watchlist),
                                translate_fn=_report_translate_fn(lang), prices_fn="auto")


def _video_metadata(video_id: str) -> dict:
    """(title, channel, published_at) for a single video via yt-dlp metadata; defensive defaults."""
    import subprocess
    import sys
    title, channel, pub = video_id, "YouTube", ""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--skip-download", "--no-warnings", "--no-playlist",
             "--print", "%(title)s\t%(channel)s\t%(upload_date)s",
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=60)
        parts = (out.stdout or "").strip().split("\t")
        if parts and parts[0]:
            title = parts[0]
        if len(parts) > 1 and parts[1]:
            channel = parts[1]
        if len(parts) > 2 and len(parts[2]) == 8:
            pub = f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:8]}T00:00:00Z"
    except Exception:
        pass
    if not pub:
        pub = datetime.now(timezone.utc).isoformat()
    return {"title": title, "channel": channel, "published_at": pub}


def _video_cache_path(vid: str, data_dir=None) -> Path:
    """Where one video's (segments + extracted insights) live, keyed by video_id."""
    from tagent.config import DATA_DIR
    base = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
    return base / "youtube_cache" / f"{vid}.json"


def _load_video_cache(vid: str, data_dir=None) -> Optional[dict]:
    """The cached entry for ``vid`` (segments + insights both present), or None on miss / bad file."""
    import json
    p = _video_cache_path(vid, data_dir)
    if not p.exists():
        return None
    try:
        e = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(e, dict) and e.get("segments") is not None and e.get("insights") is not None:
        return e
    return None


def _save_video_cache(entry: dict, data_dir=None) -> Path:
    """Persist one video's entry so EN/KO (and repeat requests) reuse ONE extraction."""
    import json
    p = _video_cache_path(entry["video_id"], data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    return p


def _default_report_video(url="", lang="ko", watchlist=None, *, transcribe_fn=None, extract_fn=None,
                          metadata_fn=None, prices_fn="auto", data_dir=None, use_cache=True,
                          refresh=False):
    """Real single-video report. To avoid contradictory advice across languages, the transcript AND
    extracted insights are CACHED per video_id and reused: KO and EN (and any repeat request) render
    from ONE deterministic insight set — only the prose is translated, so the stock list and the
    BUY/SELL/WATCH calls are identical across languages. Extraction uses the STRONGER batch model
    (gemini-2.5-pro) for specific, named, well-grounded insights — transcription dominates first-run
    time, so the extra latency is marginal and it is cached afterwards; flash is kept for TRANSLATION.
    ``refresh=True`` busts a cached (e.g. weak) result and re-extracts, overwriting the cache.
    ``transcribe_fn``/``extract_fn``/``metadata_fn`` are injectable for tests."""
    import time as _t
    from tagent.news.youtube_report import build_report_from_videos
    vid = resolve_video_id(url)
    if not vid:
        raise ValueError("could not resolve a YouTube video id from the URL")

    entry = None if refresh else (_load_video_cache(vid, data_dir) if use_cache else None)
    if entry is not None:                                       # cache hit -> NO transcribe / NO extract
        print(f"[yt-report] {vid}: reused cached extraction "
              f"({len(entry.get('segments') or [])} seg, {len(entry.get('insights') or [])} insights) "
              f"· lang={lang}")
    else:                                                      # cache miss / refresh -> transcribe + extract
        from tagent.config import SETTINGS
        meta = (metadata_fn or _video_metadata)(vid)
        # STRONGER model for extraction quality (named stocks, target prices); flash stays on translation
        model = getattr(SETTINGS, "gemini_extract_model", None) or getattr(SETTINGS, "gemini_interactive_model", None)
        if transcribe_fn is None or extract_fn is None:        # build the real source only when needed
            from tagent.news.youtube_source import YouTubeSource
            from tagent.gemini import build_extractor
            _ex = build_extractor(SETTINGS, model=model)        # provider per LLM_PROVIDER (gemini|openai)
            src = YouTubeSource(SETTINGS.youtube_api_key or "UNUSED", extract_fn=_ex, enable_fallbacks=True,
                                proxy_url=SETTINGS.youtube_proxy_url, webshare=SETTINGS.webshare_proxy(),
                                pace_seconds=SETTINGS.youtube_transcript_pace_seconds)
            transcribe_fn = transcribe_fn or src.fetch_video_transcript   # GPU Whisper per WHISPER_DEVICE
            extract_fn = extract_fn or src._extract_fn
        vmeta = {"video_id": vid, "video_title": meta["title"], "published_at": meta["published_at"]}
        t0 = _t.time()
        segments, _method = transcribe_fn(vid)
        t_tr = _t.time() - t0
        t0 = _t.time()
        try:
            insights = (extract_fn(segments, vmeta, meta["channel"]) or []) if extract_fn else []
        except Exception:
            insights = []
        t_ex = _t.time() - t0
        entry = {"video_id": vid, "channel": meta["channel"], "title": meta["title"],
                 "published_at": meta["published_at"], "fetched_at": meta["published_at"],
                 "segments": [{"start": s.get("start"), "text": s.get("text", "")} for s in segments],
                 "insights": insights}
        if use_cache:
            _save_video_cache(entry, data_dir)
        from tagent.news.youtube_source import _whisper_device_compute
        dev, _ct = _whisper_device_compute()
        print(f"[yt-report] {vid}: transcribe {t_tr:.1f}s ({len(segments)} seg, whisper={dev}) · "
              f"extract {t_ex:.1f}s ({len(insights)} insights, model={model}) · "
              f"{'re-extracted (refresh)' if refresh else 'cached'}")

    return build_report_from_videos([entry], watchlist=_coerce_watchlist(watchlist), lang=lang,
                                    translate_fn=_report_translate_fn(lang), prices_fn=prices_fn)


def normalize_interval(interval: str) -> str:
    """Map a requested interval to a supported timeframe code (default 1m)."""
    return interval if interval in TIMEFRAMES else "1m"


# --------------------------------------------------------------------------- #
# pure helpers (candle classification + crypto microstructure decision)
# --------------------------------------------------------------------------- #
def candle_type(o: float, h: float, l: float, c: float) -> dict:
    """Classify a single OHLC bar: bullish / bearish / doji + body fraction."""
    o, h, l, c = float(o), float(h), float(l), float(c)
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    body_frac = body / rng
    if body_frac < 0.1:
        kind = "doji"
    elif c > o:
        kind = "bullish"
    else:
        kind = "bearish"
    return {"type": kind, "body_pct": round(body_frac * 100.0, 1),
            "up": c >= o}


def crypto_decision(imbalance: float, absorption: float, spoof: float) -> dict:
    """Order-book microstructure decision (the crypto agent has no price model).

    Strong bid-side depth that is being *absorbed* (not spoofed) -> BUY; strong
    ask-side / lots of pulled liquidity -> SELL; otherwise HOLD. `confidence`
    scales with the imbalance magnitude.
    """
    imb, ab, sp = float(imbalance), float(absorption), float(spoof)
    if imb > 0.15 and ab >= 0.5 and sp < 0.5:
        action = "BUY"
    elif imb < -0.15 or sp > 0.6:
        action = "SELL"
    else:
        action = "HOLD"
    return {"action": action, "confidence": round(min(abs(imb), 1.0), 4)}


def _candles_from_ohlcv(df, interval: str, symbol: str, market: str,
                        limit: int = 120) -> dict:
    """OHLCV DataFrame -> the /candles JSON shape (last `limit` bars)."""
    df = df.tail(limit)
    candles = [{
        "t": idx.isoformat(), "o": float(r["open"]), "h": float(r["high"]),
        "l": float(r["low"]), "c": float(r["close"]), "v": float(r["volume"]),
    } for idx, r in df.iterrows()]
    return {"symbol": symbol, "market": market, "interval": interval, "candles": candles}


def _ta_from_candles(candles: list, symbol: str, market: str) -> dict:
    """Run the TA overlay (tagent.ta) on a /candles candle list."""
    import pandas as pd
    from tagent.ta import analyze
    if not candles or len(candles) < 12:
        return {"symbol": symbol, "market": market, "lines": [], "levels": [],
                "pattern": "n/a", "breakout": None, "pivots": [], "candle_patterns": [],
                "explanation": "Not enough bars for analysis yet."}
    df = pd.DataFrame(candles).rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    out = analyze(df[["open", "high", "low", "close", "volume"]])
    out["symbol"], out["market"] = symbol, market
    return out


def _hhmmss(iso: Optional[str]) -> str:
    """'HH:MM:SS' from an ISO timestamp, or '' if absent."""
    if not iso or len(iso) < 19:
        return iso or ""
    return iso[11:19]


def _breakout_text(symbol: str, bk: dict) -> str:
    """Human-readable alert line for a breakout event."""
    where = "above the upper trendline" if bk["direction"] == "up" else "below the lower trendline"
    bias = "bullish" if bk["direction"] == "up" else "bearish"
    price = bk.get("price")
    pstr = f"{price:,.2f}" if isinstance(price, (int, float)) else str(price)
    return (f"{symbol} broke {where} at {_hhmmss(bk.get('t'))}, price {pstr}, "
            f"{bk.get('vol_x')}x volume -> {bias}")


class AlertLog:
    """Persistent, de-duplicated feed of fired alerts (newest-first on read).

    Shared across both market panels so the dashboard shows one combined feed.
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._items: List[dict] = []
        self._seen = set()

    def add(self, *, symbol: str, market: str, kind: str, text: str,
            ts: str, meta: Optional[dict] = None) -> bool:
        """Append an alert; returns False (and does nothing) if already logged."""
        meta = meta or {}
        key = (symbol, market, kind, meta.get("t"), meta.get("direction"))
        if key in self._seen:
            return False
        self._seen.add(key)
        self._items.append({"ts": ts, "symbol": symbol, "market": market,
                            "kind": kind, "text": text, "meta": meta})
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity:]
        return True

    def recent(self, limit: int = 50, symbol: Optional[str] = None) -> List[dict]:
        items = self._items
        if symbol and str(symbol).lower() != "all":
            sym = str(symbol).upper()
            items = [x for x in items if x["symbol"].upper() == sym]
        return list(reversed(items[-limit:]))


class TaTracker:
    """Runs the TA overlay with per-symbol memory so the chart is verifiable:

    * stamps every pivot / trendline endpoint with its bar **timestamp** (stable
      across polls even as the window slides),
    * applies **pattern hysteresis** so the label stops flickering,
    * reports *when* and *why* the TA last changed (``updated_at`` + ``reason``),
    * fires **breakout alerts** into a shared :class:`AlertLog`.
    """

    def __init__(self, alert_log: Optional[AlertLog] = None,
                 min_confirm: int = 2, clock=None):
        self.alert_log = alert_log
        self.min_confirm = min_confirm
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._state: dict = {}

    def analyze(self, candles: list, symbol: str, market: str, tf: str = "1m") -> dict:
        from tagent.ta import stabilize_pattern
        out = _ta_from_candles(candles, symbol, market)
        out["interval"] = tf
        times = [c.get("t") for c in candles]

        def t_at(i):
            return times[i] if isinstance(i, int) and 0 <= i < len(times) else None

        # enrich coordinates with bar timestamps (so dots/lines are verifiable)
        for p in out.get("pivots", []):
            p["t"] = t_at(p.get("index"))
        for L in out.get("lines", []):
            L["pivots"] = [[i, pr, t_at(i)] for i, pr in L.get("pivots", [])]
            L["points"] = [[i, pr, t_at(i)] for i, pr in L.get("points", [])]
        bk = out.get("breakout")
        if bk:
            bk["t"] = t_at(bk.get("index"))
        for cp in out.get("candle_patterns", []):          # stamp pattern bars for the chart
            cp["t"] = t_at(cp.get("index"))
            cp["price"] = float(candles[cp["index"]]["c"]) if 0 <= cp.get("index", -1) < len(candles) else None

        now_iso = self._clock().replace(microsecond=0).isoformat()
        out["server_time"] = now_iso

        key = (market, symbol, tf)
        st = self._state.get(key)
        first = st is None
        if first:
            st = {"shown": None, "cand": None, "n": 0, "pivots": set(),
                  "updated_at": now_iso, "reason": "first analysis",
                  "breakout_key": None}

        # pattern hysteresis
        pstate, pat_changed = stabilize_pattern(
            {"shown": st["shown"], "cand": st["cand"], "n": st["n"]},
            out.get("pattern", "n/a"), self.min_confirm)

        # why did it change? diff confirmed pivots by (type, bar-time)
        new_piv = {(p["type"], p.get("t")) for p in out.get("pivots", [])}
        added = new_piv - st["pivots"]
        changed = first
        reason = st["reason"]
        if added and not first:
            kinds = {k for k, _ in added}
            if {"high", "low"} <= kinds:
                reason = "new swing high & low confirmed"
            elif "high" in kinds:
                reason = "new swing high confirmed"
            else:
                reason = "new swing low confirmed"
            changed = True
        if pat_changed and not first:
            reason = f"pattern changed -> {pstate['shown']}"
            changed = True

        # breakout alert (dedup on bar-time + direction)
        bk_key = (bk.get("t"), bk.get("direction")) if bk else None
        if bk and bk_key != st["breakout_key"]:
            reason = f"{bk['direction']} breakout"
            changed = True
            if self.alert_log is not None:
                self.alert_log.add(
                    symbol=symbol, market=market, kind="breakout",
                    text=_breakout_text(symbol, bk), ts=now_iso,
                    meta={"t": bk.get("t"), "direction": bk["direction"],
                          "price": bk.get("price"), "vol_x": bk.get("vol_x")})

        updated_at = now_iso if changed else st["updated_at"]
        self._state[key] = {"shown": pstate["shown"], "cand": pstate["cand"],
                            "n": pstate["n"], "pivots": new_piv,
                            "updated_at": updated_at, "reason": reason,
                            "breakout_key": bk_key if bk else st["breakout_key"]}

        if pstate["shown"] is not None:
            out["pattern"] = pstate["shown"]
        out["updated_at"] = updated_at
        out["reason"] = reason
        return out

    def alerts(self, limit: int = 50, symbol: Optional[str] = None) -> List[dict]:
        return self.alert_log.recent(limit, symbol) if self.alert_log else []


# --------------------------------------------------------------------------- #
# pure signal mapping
# --------------------------------------------------------------------------- #
def stock_signal(symbol: str, proba: float, price: float, risk,
                 buy_threshold: float = 0.55, sell_threshold: float = 0.45) -> dict:
    """Map the model's P(good long entry) to BUY/HOLD/SELL + risk sizing.

    BUY above buy_threshold (sized/stopped by the risk layer), SELL below
    sell_threshold, HOLD in between. `confidence` is the raw probability (0..1).
    """
    if price <= 0:
        return {"symbol": symbol, "price": 0.0, "action": "HOLD",
                "confidence": round(float(proba), 4), "qty": 0,
                "stop": 0.0, "target": 0.0}
    if proba >= buy_threshold:
        stop, take = risk.stop_take_prices(price, "buy")
        qty = risk.position_size(price, stop)
        action = "BUY"
    elif proba <= sell_threshold:
        stop = take = 0.0
        qty = 0
        action = "SELL"
    else:
        stop = take = 0.0
        qty = 0
        action = "HOLD"
    return {
        "symbol": symbol, "price": round(float(price), 4), "action": action,
        "confidence": round(float(proba), 4), "qty": int(qty),
        "stop": round(float(stop), 4), "target": round(float(take), 4),
    }


# --------------------------------------------------------------------------- #
# real sources (Alpaca + CryptoFeed)
# --------------------------------------------------------------------------- #
class AlpacaStockSource:
    """US-stock signals + paper account from Alpaca (read-only; places no orders)."""

    def __init__(self, predictor, trading_client, data_client, risk,
                 buy_threshold: float = 0.55, sell_threshold: float = 0.45,
                 lookback_days: int = 200, ta_tracker: Optional["TaTracker"] = None):
        self.predictor = predictor
        self.trading = trading_client
        self.data = data_client
        self.risk = risk
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.lookback_days = lookback_days
        self.ta_tracker = ta_tracker or TaTracker()

    def _recent_bars(self, symbol: str):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        import pandas as pd
        start = datetime.now(timezone.utc) - timedelta(days=self.lookback_days * 2 + 15)
        df = self.data.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start)).df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].tail(
            self.lookback_days)

    def _latest_price(self, symbol: str) -> float:
        from alpaca.data.requests import StockLatestTradeRequest
        t = self.data.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol))[symbol]
        return float(getattr(t, "price", 0) or 0)

    def signals(self, symbols: List[str]) -> List[dict]:
        out = []
        for s in symbols:
            try:
                bars = self._recent_bars(s)
                proba = self.predictor.predict_proba_latest(bars)
                try:
                    price = self._latest_price(s) or float(bars["close"].iloc[-1])
                except Exception:
                    price = float(bars["close"].iloc[-1])
                out.append(stock_signal(s, proba, price, self.risk,
                                        self.buy_threshold, self.sell_threshold))
            except Exception as e:
                out.append({"symbol": s, "action": "ERROR", "error": str(e)[:120],
                            "price": 0.0, "confidence": 0.0, "qty": 0})
        return out

    def candles(self, symbol: str, interval: str = "1m") -> dict:
        """Recent OHLC bars from Alpaca at the requested timeframe."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        import pandas as pd
        tf = normalize_interval(interval)
        cfg = TIMEFRAMES[tf]
        amount, unit = cfg["alpaca"]
        timeframe = TimeFrame(amount, getattr(TimeFrameUnit, unit))
        start = datetime.now(timezone.utc) - timedelta(days=cfg["days"])
        df = self.data.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=timeframe, start=start)).df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        return _candles_from_ohlcv(df, tf, symbol.upper(), "us")

    def ta(self, symbol: str, interval: str = "1m") -> dict:
        """Rule-based TA overlay (trendlines / S/R / pattern / breakout) recomputed
        on the chosen timeframe's bars, with hysteresis + alerts."""
        tf = normalize_interval(interval)
        return self.ta_tracker.analyze(self.candles(symbol, tf).get("candles", []),
                                       symbol.upper(), "us", tf=tf)

    def alerts(self, limit: int = 50, symbol: Optional[str] = None) -> List[dict]:
        return self.ta_tracker.alerts(limit, symbol)

    def analysis(self, symbol: str) -> dict:
        """The agent's computed view: daily features (model inputs), candle type,
        the model probability and the BUY/HOLD/SELL decision."""
        from tagent.features import make_features
        bars = self._recent_bars(symbol)
        feats = make_features(bars, dropna=True)
        row = feats.iloc[-1]
        proba = float(self.predictor.predict_proba_latest(bars))
        try:
            price = self._latest_price(symbol) or float(bars["close"].iloc[-1])
        except Exception:
            price = float(bars["close"].iloc[-1])
        last = bars.iloc[-1]
        sig = stock_signal(symbol, proba, price, self.risk,
                           self.buy_threshold, self.sell_threshold)
        return {
            "symbol": symbol.upper(), "market": "us", "price": round(price, 4),
            "candle": candle_type(last["open"], last["high"], last["low"], last["close"]),
            "rsi": round(float(row["rsi_14"]), 1),
            "momentum": round(float(row["mom_5"]) * 100.0, 2),
            "ret_1": round(float(row["ret_1"]) * 100.0, 3),
            "macd_hist": round(float(row["macd_hist"]), 4),
            "probability": round(proba, 4),
            "action": sig["action"], "confidence": round(proba, 4),
            "qty": sig["qty"], "stop": sig["stop"], "target": sig["target"],
            "decision_source": "ML model (daily technical features)",
        }

    def account(self) -> dict:
        a = self.trading.get_account()
        equity = float(a.equity)
        last = float(getattr(a, "last_equity", a.equity) or equity)
        positions = []
        try:
            for p in self.trading.get_all_positions():
                positions.append({
                    "symbol": p.symbol, "qty": float(p.qty),
                    "market_value": round(float(p.market_value), 2),
                    "unrealized_pl": round(float(p.unrealized_pl), 2),
                })
        except Exception:
            pass
        return {
            "equity": round(equity, 2), "cash": round(float(a.cash), 2),
            "day_pnl": round(equity - last, 2),
            "day_pnl_pct": round((equity - last) / last * 100.0, 3) if last else 0.0,
            "positions": positions,
        }


class CryptoSource:
    """Order book + memory from a live CryptoFeed (starts it on first use)."""

    def __init__(self, feed, autostart: bool = True,
                 ta_tracker: Optional["TaTracker"] = None, model_dir=None,
                 ml_interval: str = "1h"):
        self.feed = feed
        self._thread: Optional[threading.Thread] = None
        self.ta_tracker = ta_tracker or TaTracker()
        self.model_dir = model_dir
        self.ml_interval = ml_interval
        self._ml_cache: dict = {}        # pair -> CryptoMLPredictor (or None if absent)
        if autostart:
            self.start()

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self.feed.run, daemon=True)
            self._thread.start()

    def _ensure(self, symbol: str) -> str:
        pair = normalize_symbol(symbol)
        if pair not in self.feed.subscriptions:
            self.feed.subscribe_symbol(pair)
        return pair

    def orderbook(self, symbol: str) -> dict:
        try:
            pair = self._ensure(symbol)
        except ValueError as e:
            return {"error": str(e), "symbol": symbol}
        mem = self.feed.books.get(pair)
        ob = getattr(mem, "_prev", None) if mem else None
        if ob is None:
            return {"symbol": pair, "asks": [], "bids": [], "status": "subscribing"}
        return {
            "symbol": pair, "status": "live",
            "asks": [[p, q] for p, q in ob.asks[:10]],   # ascending (best first)
            "bids": [[p, q] for p, q in ob.bids[:10]],   # descending (best first)
            "best_ask": ob.asks[0][0] if ob.asks else None,
            "best_bid": ob.bids[0][0] if ob.bids else None,
            "ts": ob.timestamp.isoformat(),
        }

    def _klines_df(self, pair: str, interval: str, limit: int):
        """Fetch public Binance klines (no API key) into an OHLCV DataFrame."""
        import requests
        import pandas as pd
        url = "https://api.binance.com/api/v3/klines"
        r = requests.get(url, params={"symbol": pair, "interval": interval,
                                      "limit": limit}, timeout=8)
        rows = r.json()
        if not isinstance(rows, list):
            raise ValueError(f"binance klines error: {rows}")
        idx = pd.to_datetime([k[0] for k in rows], unit="ms", utc=True)
        return pd.DataFrame({
            "open": [float(k[1]) for k in rows], "high": [float(k[2]) for k in rows],
            "low": [float(k[3]) for k in rows], "close": [float(k[4]) for k in rows],
            "volume": [float(k[5]) for k in rows],
        }, index=idx)

    def recent_ticks(self, symbol: str, limit: int = 500):
        """Raw recent TRADES (aggTrades) from public Binance — a genuine tick stream
        (time-ms, price, qty), no API key. Feeds the intraday show-don't-tell demo."""
        import requests
        pair = normalize_symbol(symbol)
        r = requests.get("https://api.binance.com/api/v3/aggTrades",
                         params={"symbol": pair, "limit": min(1000, int(limit))}, timeout=8)
        rows = r.json()
        if not isinstance(rows, list):
            raise ValueError(f"binance aggTrades error: {rows}")
        return [(int(x["T"]), float(x["p"]), float(x["q"])) for x in rows]

    def candles(self, symbol: str, interval: str = "1m", limit: int = 120) -> dict:
        try:
            pair = self._ensure(symbol)
        except ValueError as e:
            return {"error": str(e), "symbol": symbol}
        tf = normalize_interval(interval)
        cfg = TIMEFRAMES[tf]
        if cfg["binance"]:
            df = self._klines_df(pair, cfg["binance"], limit)
        else:                                            # 10m: resample from 1m klines
            base = self._klines_df(pair, "1m", min(1000, limit * cfg["minutes"]))
            df = base.resample(f"{cfg['minutes']}min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"}).dropna().tail(limit)
        return _candles_from_ohlcv(df, tf, pair, "crypto", limit)

    def ta(self, symbol: str, interval: str = "1m") -> dict:
        tf = normalize_interval(interval)
        c = self.candles(symbol, tf)
        if "error" in c:
            return c
        return self.ta_tracker.analyze(c.get("candles", []), c["symbol"], "crypto", tf=tf)

    def alerts(self, limit: int = 50, symbol: Optional[str] = None) -> List[dict]:
        return self.ta_tracker.alerts(limit, symbol)

    def _ml_predictor(self, pair: str):
        """Load (and cache) the per-coin crypto ML model, or None if not trained."""
        if pair not in self._ml_cache:
            from tagent.crypto_ml import CryptoMLPredictor, crypto_model_path
            try:
                if crypto_model_path(pair, self.model_dir).exists():
                    self._ml_cache[pair] = CryptoMLPredictor.load(pair, self.model_dir)
                else:
                    self._ml_cache[pair] = None
            except Exception:
                self._ml_cache[pair] = None
        return self._ml_cache[pair]

    def ml(self, symbol: str) -> dict:
        """Crypto ML agent: P(good entry) from the per-coin LightGBM on recent
        klines (+ live order-book micro), mapped to BUY/HOLD/SELL. Mirrors US ML."""
        from tagent.crypto_ml import crypto_ml_decision
        try:
            pair = self._ensure(symbol)
        except ValueError as e:
            return {"error": str(e), "symbol": symbol}
        pred = self._ml_predictor(pair)
        if pred is None:
            return {"symbol": pair, "market": "crypto", "action": None,
                    "error": "no crypto-ml model — train with scripts/train_crypto_ml.py"}
        try:
            df = self._klines_df(pair, self.ml_interval, 200)
            micro = self.feed.book_features(pair) or {}
            proba = pred.predict_proba_latest(df, micro)
            dec = crypto_ml_decision(proba)
            return {"symbol": pair, "market": "crypto", "price": float(df["close"].iloc[-1]),
                    "decision_source": "ML model (kline tech + order-book micro)", **dec}
        except Exception as e:
            return {"symbol": pair, "market": "crypto", "action": None, "error": str(e)[:120]}

    def funding(self, symbol: str) -> dict:
        """Live 8h funding rate + next funding time (public Binance premiumIndex),
        for the delta-neutral funding-carry agent. No API key."""
        import requests
        try:
            pair = normalize_symbol(symbol)
        except ValueError as e:
            return {"error": str(e), "symbol": symbol}
        try:
            d = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                             params={"symbol": pair}, timeout=8).json()
            if not isinstance(d, dict) or "lastFundingRate" not in d:
                return {"error": f"funding fetch failed for {pair}", "symbol": pair}
            return {"symbol": pair, "market": "crypto",
                    "funding_rate": float(d["lastFundingRate"]),
                    "funding_time": int(d.get("nextFundingTime", 0)),
                    "mark": float(d.get("markPrice") or "nan")}
        except Exception as e:
            return {"error": str(e)[:120], "symbol": pair}

    def analysis(self, symbol: str) -> dict:
        """The order-book agent's view: candle type + RSI/momentum from klines,
        plus depth imbalance / absorption / spoof and a microstructure decision."""
        from tagent.features import make_features
        try:
            pair = self._ensure(symbol)
        except ValueError as e:
            return {"error": str(e), "symbol": symbol}
        feats = self.feed.book_features(pair) or {}
        imb = float(feats.get("depth_imbalance", 0.0))
        ab = float(feats.get("absorption_ratio", 0.0))
        sp = float(feats.get("spoof_ratio", 0.0))
        out = {
            "symbol": pair, "market": "crypto",
            "imbalance": round(imb, 4), "absorption": round(ab, 4), "spoof": round(sp, 4),
            "decision_source": "order-book memory (imbalance/absorption/spoof)",
        }
        try:
            df = self._klines_df(pair, "1m", 60)
            f = make_features(df, dropna=True).iloc[-1]
            last = df.iloc[-1]
            out.update({
                "price": round(float(last["close"]), 8),
                "candle": candle_type(last["open"], last["high"], last["low"], last["close"]),
                "rsi": round(float(f["rsi_14"]), 1),
                "momentum": round(float(f["mom_5"]) * 100.0, 2),
            })
        except Exception:
            out.update({"price": None, "candle": None, "rsi": None, "momentum": None})
        dec = crypto_decision(imb, ab, sp)
        out.update({"action": dec["action"], "confidence": dec["confidence"],
                    "probability": None})
        return out

    def memory(self, symbol: str) -> dict:
        try:
            pair = self._ensure(symbol)
        except ValueError as e:
            return {"error": str(e), "symbol": symbol}
        feats = self.feed.book_features(pair) or {}
        mem = self.feed.books.get(pair)
        vanished = []
        if mem is not None:
            for p, q, r, ts in list(mem.vanished_asks)[-10:]:
                vanished.append({"side": "ask", "price": p, "qty": q, "reason": r})
            for p, q, r, ts in list(mem.vanished_bids)[-10:]:
                vanished.append({"side": "bid", "price": p, "qty": q, "reason": r})
        return {**feats, "symbol": pair, "vanished": vanished}


class IntradayDemoSource:
    """Show-don't-tell: compute candle analysis straight from a RAW tick stream.

    ``tick_provider(symbol) -> [(time, price, volume), ...]`` supplies raw ticks
    (Kiwoom 0B executions in production; public Binance aggTrades for the always-on
    live demo). ``linked_provider(symbol) -> {name: return}`` supplies the linked US
    peers/index returns for the safety kill switch. Everything is computed by the
    pure functions in :mod:`tagent.intraday_demo`; this just wires data to them.
    """

    def __init__(self, tick_provider, linked_provider=None, interval: str = "1min",
                 windows=(5, 20, 60), threshold: float = -0.03):
        self.tick_provider = tick_provider
        self.linked_provider = linked_provider
        self.interval = interval
        self.windows = tuple(windows)
        self.threshold = threshold

    def analyze(self, symbol: str, interval: Optional[str] = None) -> dict:
        from tagent.intraday_demo import analyze_ticks
        ticks = self.tick_provider(symbol) or []
        linked = {}
        if self.linked_provider is not None:
            try:
                linked = self.linked_provider(symbol) or {}
            except Exception:
                linked = {}
        out = analyze_ticks(ticks, linked_returns=linked, interval=interval or self.interval,
                            windows=self.windows, threshold=self.threshold)
        out["symbol"] = str(symbol).upper()
        return out


# --------------------------------------------------------------------------- #
# app factory
# --------------------------------------------------------------------------- #
def poll_scorecard(engine, stock_source, crypto_source,
                   stock_symbols=(), crypto_symbols=()) -> None:
    """One forward-test pass: feed each source's current call into the scorecard.

    De-dup inside the engine means this is safe to call every few seconds — only
    genuinely new signals are logged / traded. Used by the overnight poller and
    callable directly in tests.
    """
    try:
        for s in stock_source.signals(list(stock_symbols)):
            if s.get("action") and s.get("price") and s["action"] != "ERROR":
                engine.observe("us-ml", s["symbol"], s["action"], s["price"])
    except Exception:
        pass
    for sym in crypto_symbols:
        try:
            a = crypto_source.analysis(sym)
            if not a.get("error") and a.get("price") is not None and a.get("action"):
                engine.observe("crypto-ob", a["symbol"], a["action"], a["price"])
        except Exception:
            pass
        try:                                              # crypto ML agent (per-coin model)
            m = crypto_source.ml(sym)
            if m and not m.get("error") and m.get("price") is not None and m.get("action"):
                engine.observe("crypto-ml", m["symbol"], m["action"], m["price"])
        except Exception:
            pass
        try:                                              # funding-carry agent (delta-neutral)
            fnd = crypto_source.funding(sym)
            if (fnd and not fnd.get("error") and fnd.get("funding_time")
                    and hasattr(engine, "accrue_funding")):
                engine.accrue_funding(fnd["symbol"], fnd["funding_rate"], fnd["funding_time"])
        except Exception:
            pass
    # TA chart-drawing scored per market: us-ta and crypto-ta (each its own book)
    for src, syms, name in ((stock_source, stock_symbols, "us-ta"),
                            (crypto_source, crypto_symbols, "crypto-ta")):
        for sym in syms:
            try:
                t = src.ta(sym)
                if not isinstance(t, dict):
                    continue
                tsym = t.get("symbol", sym)
                bk = t.get("breakout")
                if bk and bk.get("price") is not None:
                    direction = "bullish" if bk["direction"] == "up" else "bearish"
                    engine.observe(name, tsym, direction, bk["price"], tag="breakout")
                # each famous candlestick pattern is its own tagged signal stream
                for cp in t.get("candle_patterns", []):
                    if cp.get("price") is not None and cp.get("direction"):
                        engine.observe(name, tsym, cp["direction"], cp["price"], tag=cp["name"])
            except Exception:
                pass


def create_app(stock_source, crypto_source, static_path: Optional[Path] = None,
               scorecard=None, funding_live=None, momentum_live=None,
               intraday_source=None, news=None, leadlag=None, pead=None,
               desk=None, desk_session=None, trend_core=None, forward_ops=None, media=None,
               briefing=None, feed=None, orders=None, report_builder=None, report_renderer=None,
               report_video=None, reports_dir=None):
    """Build the FastAPI app from two injected sources (real or fake).

    ``funding_live`` / ``momentum_live`` are optional zero-arg callables returning a
    live paper runner's status (see tagent.funding_live.load_live_status and
    tagent.momentum_live.load_live_status). ``intraday_source`` is an optional
    IntradayDemoSource exposing ``analyze(symbol, interval)``. ``news`` is an optional
    zero-arg callable returning the news/alerts payload (tagent.news.alerts).
    """
    from fastapi import FastAPI, Query
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    app = FastAPI(title="trading-agent dashboard")
    static_path = static_path or _STATIC

    def _safe(fn):
        try:
            return fn()
        except Exception as e:  # never 500 the polling frontend
            return JSONResponse({"error": str(e)[:200]}, status_code=200)

    @app.get("/signals")
    def signals(symbols: str = Query("")):
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        return _safe(lambda: {"signals": stock_source.signals(syms)})

    @app.get("/account")
    def account():
        return _safe(stock_source.account)

    def _src(market: str):
        return crypto_source if str(market).lower() == "crypto" else stock_source

    @app.get("/candles")
    def candles(symbol: str = Query("AAPL"), market: str = Query("us"),
                interval: str = Query("1m")):
        return _safe(lambda: _src(market).candles(symbol, interval))

    @app.get("/analysis")
    def analysis(symbol: str = Query("AAPL"), market: str = Query("us")):
        return _safe(lambda: _src(market).analysis(symbol))

    @app.get("/ta")
    def ta(symbol: str = Query("AAPL"), market: str = Query("us"),
           interval: str = Query("1m")):
        return _safe(lambda: _src(market).ta(symbol, interval))

    @app.get("/alerts")
    def alerts(limit: int = Query(50), market: str = Query("us"),
               symbol: str = Query("all")):
        return _safe(lambda: {"alerts": _src(market).alerts(limit, symbol)})

    @app.get("/scorecard")
    def scorecard_endpoint(symbol: str = Query("all")):
        if scorecard is None:
            return {"enabled": False, "sources": {}}
        return _safe(lambda: {"enabled": True, **scorecard.scorecard(symbol)})

    @app.get("/funding_live")
    def funding_live_endpoint():
        if funding_live is None:
            return {"enabled": False, "held": []}
        return _safe(funding_live)

    @app.get("/momentum_live")
    def momentum_live_endpoint():
        if momentum_live is None:
            return {"enabled": False, "held": []}
        return _safe(momentum_live)

    @app.get("/intraday_demo")
    def intraday_demo_endpoint(symbol: str = Query("BTCUSDT"), interval: str = Query("")):
        if intraday_source is None:
            return {"enabled": False}
        return _safe(lambda: {"enabled": True, **intraday_source.analyze(symbol, interval or None)})

    @app.get("/news")
    def news_endpoint():
        if news is None:
            return {"enabled": False, "alerts": []}
        return _safe(news)

    @app.get("/leadlag")
    def leadlag_endpoint():
        if leadlag is None:
            return {"enabled": False}
        return _safe(leadlag)

    @app.get("/pead")
    def pead_endpoint():
        if pead is None:
            return {"enabled": False}
        return _safe(pead)

    @app.get("/desk")
    def desk_endpoint():
        if desk is None:
            return {"enabled": False}
        return _safe(desk)

    @app.get("/desk_session")
    def desk_session_endpoint(symbols: str = Query("")):
        if desk_session is None:
            return {"enabled": False, "rows": []}
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        return _safe(lambda: {"enabled": True, "rows": desk_session(syms)})

    @app.get("/trend_core")
    def trend_core_endpoint():
        if trend_core is None:
            return {"enabled": False}
        return _safe(trend_core)

    @app.get("/forward_ops")
    def forward_ops_endpoint():
        if forward_ops is None:
            return {"enabled": False, "candidates": []}
        return _safe(forward_ops)

    @app.get("/media")
    def media_endpoint():
        # Media briefing (YouTube/TV) — AWARENESS ONLY, display-only. Deliberately NOT
        # connected to any trading/HALT logic; the dashboard renders it as informational.
        if media is None:
            return {"enabled": False, "items": [], "signal": False}
        return _safe(media)

    @app.get("/briefing")
    def briefing_endpoint():
        # Daily 4-report briefing (newspaper / Kiwoom / YouTube / recommendation) — INFORMATIONAL,
        # every claim cited; refreshed by the daily auto-advance. Display-only.
        if briefing is None:
            return {"enabled": False, "reports": {}, "breaking": []}
        return _safe(briefing)

    @app.get("/feed")
    def feed_endpoint(since: str = Query("")):
        # Continuous intraday news FEED — newest-first, dual timestamps (detected_at/published_at),
        # every item cited. ``since`` (the latest detected_at the client saw) marks NEW items.
        if feed is None:
            return {"enabled": False, "items": [], "n": 0, "note": "feed not wired"}
        return _safe(lambda: feed(since or None))

    @app.get("/orders")
    def orders_endpoint():
        # Live order-execution log (Kiwoom API) — proof the agent places + cancels orders. MOCK
        # unless badged LIVE; display-only.
        if orders is None:
            return {"enabled": False, "orders": [], "n": 0}
        return _safe(orders)

    @app.get("/orderbook")
    def orderbook(symbol: str = Query("BTCUSDT")):
        return _safe(lambda: crypto_source.orderbook(symbol))

    @app.get("/memory")
    def memory(symbol: str = Query("BTCUSDT")):
        return _safe(lambda: crypto_source.memory(symbol))

    # ---- YouTube report: generate (window / single video) + download (Phase 4) ---------- #
    from fastapi import Body
    from tagent.config import DATA_DIR
    from tagent.news.report_render import render_report, report_to_html

    _reports_dir = Path(reports_dir) if reports_dir else (Path(DATA_DIR) / "reports")
    _builder = report_builder or _default_report_builder
    _video_builder = report_video or _default_report_video
    _renderer = report_renderer or (lambda rep, lang, out_dir, basename:
                                    render_report(rep, lang=lang, out_dir=out_dir, basename=basename))

    def _report_response(report, files, lang):
        """The shared JSON shape: inline HTML preview + download urls + meta (built from the SAME
        grounded report dict, so the preview can't drift from the files)."""
        from pathlib import Path as _P
        return {"meta": report.get("meta", {}), "html_preview": report_to_html(report, lang),
                "docx_url": f"/youtube_report/file/{_P(files['docx']).name}",
                "pdf_url": f"/youtube_report/file/{_P(files['pdf']).name}"}

    @app.post("/youtube_report/window")
    def youtube_report_window(body: dict = Body(default={})):
        def run():
            lang = "en" if str((body or {}).get("lang", "ko")).lower() == "en" else "ko"
            report = _builder(lang=lang, end=(body or {}).get("end") or None,
                              watchlist=(body or {}).get("watchlist") or None)
            files = _renderer(report, lang, str(_reports_dir), "youtube_report")
            return _report_response(report, files, lang)
        return _safe(run)

    @app.post("/youtube_report/video")
    def youtube_report_video(body: dict = Body(default={})):
        def run():
            import time as _t
            lang = "en" if str((body or {}).get("lang", "ko")).lower() == "en" else "ko"
            url = str((body or {}).get("url", "")).strip()
            if not url:
                return {"error": "url required"}
            refresh = bool((body or {}).get("refresh"))         # cache-bust: re-extract a weak cached result
            report = _video_builder(url=url, lang=lang, watchlist=(body or {}).get("watchlist") or None,
                                    refresh=refresh)
            t0 = _t.time()
            files = _renderer(report, lang, str(_reports_dir), "youtube_video")
            print(f"[yt-report] render {(_t.time() - t0):.1f}s (docx+pdf)")
            return _report_response(report, files, lang)
        return _safe(run)

    @app.get("/youtube_report/file/{name:path}")
    def youtube_report_file(name: str):
        # serve a generated report ONLY from data/reports — validate the name, no path traversal
        safe = safe_report_name(name)
        if not safe:
            return JSONResponse({"error": "invalid filename"}, status_code=400)
        root = _reports_dir.resolve()
        path = (root / safe).resolve()
        if path.parent != root or not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        media = ("application/pdf" if path.suffix.lower() == ".pdf"
                 else "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        return FileResponse(str(path), filename=safe, media_type=media)

    @app.get("/", response_class=HTMLResponse)
    def index():
        if static_path.exists():
            return FileResponse(str(static_path))
        return HTMLResponse("<h1>dashboard.html missing</h1>", status_code=500)

    return app
