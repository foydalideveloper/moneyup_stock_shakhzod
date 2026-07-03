"""Run the local web dashboard (FastAPI + one static page).

Open http://localhost:8000 — two independent search boxes:
  * US stock search -> ML signals table + Alpaca paper-account summary.
  * Crypto search    -> live 10-level order book + memory (vanished levels,
                        absorption / spoofing / imbalance) from CryptoFeed.

US signals + account need ALPACA_API_KEY / ALPACA_SECRET_KEY in .env and a
trained model (scripts/train_model.py). Crypto needs no key. If the Alpaca keys
or model are missing, the crypto panel still works and the US panel shows errors.

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --host 0.0.0.0 --port 8000
"""

import argparse
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tagent.config import SETTINGS  # noqa: E402
from tagent.dashboard import (  # noqa: E402
    AlertLog, AlpacaStockSource, CryptoSource, IntradayDemoSource, TaTracker,
    create_app, poll_scorecard)
from tagent.feeds.crypto_feed import CryptoFeed  # noqa: E402
from tagent.data.intraday_history import load_spy_daily_returns  # noqa: E402
from tagent.funding_live import load_live_status  # noqa: E402
from tagent.leadlag_live import load_live_status as load_leadlag_status  # noqa: E402
from tagent.daily_desk import load_desk_snapshot, session_signals  # noqa: E402
from tagent.trend_core_live import load_live_status as load_trend_core_status  # noqa: E402
from tagent.forward_ops import load_advance_state, load_forward_ops  # noqa: E402
from tagent.pead_live import load_live_status as load_pead_status  # noqa: E402
from tagent.news.alerts import NewsAlertConfig, build_news_payload  # noqa: E402
from tagent.news.finnhub_source import DEFAULT_LINKED_US, FinnhubSource  # noqa: E402
from tagent.news.opendart_source import OpenDartSource  # noqa: E402
from tagent.news.naver_source import NaverNewsSource  # noqa: E402
from tagent.intraday_feed import (  # noqa: E402
    SourceThrottle, feed_view, from_kiwoom_cards, from_news_items, from_youtube_claims,
    poll_and_persist, refilter_feed)
from tagent.news.source_whitelist import filter_top_sources, parse_whitelist  # noqa: E402
from tagent.order_log import order_panel  # noqa: E402
from tagent.youtube_audit import AuditWriter  # noqa: E402
from tagent.news.youtube_source import (  # noqa: E402
    YouTubeSource, build_media_payload, channels_from_env)
from tagent.momentum_live import (  # noqa: E402
    MomentumLiveTrader, MonthlyRebalanceScheduler, load_live_status as load_momentum_status)
from tagent.scorecard import Scorecard  # noqa: E402


class _BrokenStockSource:
    """Graceful stand-in when Alpaca keys / the model are missing or the US market is closed — keeps
    the app up and the US panel shows 'unavailable' instead of crashing. (US ML is a retired demo;
    this just degrades cleanly across every endpoint the dashboard calls.)"""
    NOTE = "US data unavailable (market closed / no Alpaca keys)"

    def __init__(self, msg): self.msg = msg

    def signals(self, symbols):
        return [{"symbol": s, "action": "UNAVAILABLE", "unavailable": True, "note": self.NOTE,
                 "price": 0.0, "confidence": 0.0, "qty": 0} for s in symbols]

    def account(self):
        return {"unavailable": True, "note": self.NOTE, "positions": []}

    def analysis(self, symbol):
        return {"symbol": str(symbol).upper(), "market": "us", "unavailable": True,
                "note": self.NOTE, "price": None, "action": "UNAVAILABLE"}

    def candles(self, symbol, interval="1m"):
        return {"symbol": str(symbol).upper(), "market": "us", "interval": interval,
                "unavailable": True, "note": self.NOTE, "candles": []}

    def ta(self, symbol, interval="1m"):
        return {"symbol": str(symbol).upper(), "market": "us", "interval": interval,
                "unavailable": True, "note": self.NOTE, "lines": [], "levels": [], "pivots": [],
                "breakout": None, "candle_patterns": []}

    def alerts(self, limit=50, symbol=None):
        return []


_NEWS_KR_WATCHLIST = ["005930", "000660", "005380", "035420", "051910", "005490",
                      "000270", "207940"]


def _gemini_client():
    """A GeminiClient if GEMINI_API_KEY is set, else None (callers fall back to the grounded default)."""
    if not SETTINGS.has_gemini_key():
        return None
    try:
        from tagent.gemini import GeminiClient
        return GeminiClient(SETTINGS.gemini_api_key, model=SETTINGS.gemini_model)
    except Exception:
        return None


def _naver_relevance_fn():
    g = _gemini_client()
    if g is None:
        return None
    try:
        from tagent.gemini import naver_relevance_fn
        return naver_relevance_fn(g)
    except Exception:
        return None


def _kr_whitelist():
    return parse_whitelist(SETTINGS.source_whitelist_kr, [])


def _us_whitelist():
    return parse_whitelist(SETTINGS.source_whitelist_us, [])


def _source_filter():
    """The top-source whitelist as an item->bool predicate (KR by domain, US by Finnhub publisher)."""
    from tagent.news.source_whitelist import is_top_source
    kr, us = _kr_whitelist(), _us_whitelist()
    return lambda it: is_top_source(it, kr_whitelist=kr, us_whitelist=us)


def _build_news():
    """Zero-arg news payload callable for the dashboard (cached ~60s to respect free
    API rate limits). None if neither key is set. Secrets are never logged."""
    if not (SETTINGS.has_opendart_key() or SETTINGS.has_finnhub_key() or SETTINGS.has_naver_keys()):
        return None
    import datetime as _dt
    import time as _time
    od = OpenDartSource(SETTINGS.opendart_api_key) if SETTINGS.has_opendart_key() else None
    fh = FinnhubSource(SETTINGS.finnhub_api_key) if SETTINGS.has_finnhub_key() else None
    nv = (NaverNewsSource(SETTINGS.naver_client_id, SETTINGS.naver_client_secret)
          if SETTINGS.has_naver_keys() else None)
    cache = {"ts": 0.0, "payload": None}

    def _overnight():
        try:
            r = load_spy_daily_returns()
            return float(r.iloc[-1]) if len(r) else None
        except Exception:
            return None

    def payload():
        now = _time.time()
        if cache["payload"] is not None and now - cache["ts"] < 60.0:
            return cache["payload"]
        today = _dt.date.today()
        kr = us = []
        try:
            if od:
                # query by corp_code (cached stock->corp map) so each watchlist name
                # reliably returns its own disclosures, not the all-market recent slice
                kr = od.recent_disclosures_for(
                    _NEWS_KR_WATCHLIST,
                    (today - _dt.timedelta(days=14)).strftime("%Y%m%d"),
                    today.strftime("%Y%m%d"))
        except Exception:
            kr = []
        try:
            if nv:                                       # 한경/매경 articles w/ real links (relevance-filtered)
                kr = list(kr) + nv.watchlist_news(_NEWS_KR_WATCHLIST, per_symbol=4,
                                                  relevance_fn=_naver_relevance_fn())
        except Exception:
            pass
        try:
            if fh:
                us = fh.market_news(DEFAULT_LINKED_US,
                                    (today - _dt.timedelta(days=3)).isoformat(),
                                    today.isoformat(), per_symbol=6)
        except Exception:
            us = []
        kr = filter_top_sources(kr, kr_whitelist=_kr_whitelist(), us_whitelist=_us_whitelist())
        us = filter_top_sources(us, kr_whitelist=_kr_whitelist(), us_whitelist=_us_whitelist())
        p = build_news_payload(kr, us, overnight_return=_overnight(), cfg=NewsAlertConfig())
        cache["ts"], cache["payload"] = now, p
        return p

    return payload


def _build_media():
    """Zero-arg media-briefing payload callable (YouTube/TV) for the dashboard, cached ~5 min
    (the YouTube Data API quota is small + transcripts are heavy). AWARENESS ONLY — never fed
    into trading/HALT logic. None if no key. The API key is never logged."""
    if not SETTINGS.has_youtube_key():
        return None
    import time as _time
    from tagent.gemini import build_extractor
    extract_fn = build_extractor(SETTINGS)               # provider per LLM_PROVIDER (gemini|openai)
    src = YouTubeSource(SETTINGS.youtube_api_key, extract_fn=extract_fn,
                        enable_fallbacks=True, proxy_url=SETTINGS.youtube_proxy_url,
                        webshare=SETTINGS.webshare_proxy(),
                        pace_seconds=SETTINGS.youtube_transcript_pace_seconds)  # api->yt-dlp->whisper
    channels = channels_from_env(SETTINGS.youtube_channels)
    cache = {"ts": 0.0, "payload": None}

    def payload():
        now = _time.time()
        if cache["payload"] is not None and now - cache["ts"] < 300.0:
            return cache["payload"]
        try:
            items = src.media_briefing(channels, audit=AuditWriter())   # log the raw fetch (audit)
        except Exception:
            items = []
        p = build_media_payload(items)
        cache["ts"], cache["payload"] = now, p
        return p

    return payload


def _build_desk_session():
    """In-session monitor backed by the REAL Kiwoom minute feed (ka10080, mock-first).
    Decision-support only; cached ~60s for the Kiwoom rate limit. None without keys."""
    if not SETTINGS.has_kiwoom_keys():
        return None
    import time as _time
    from tagent.data.kiwoom_minute import fetch_minute_bars
    from tagent.feeds.kiwoom_auth import KiwoomAuth
    base, env = SETTINGS.kiwoom_rest_url(), SETTINGS.kiwoom_env
    auth = KiwoomAuth(SETTINGS.kiwoom_app_key, SETTINGS.kiwoom_secret_key, env=env, base_url=base)
    cache = {"ts": 0.0, "rows": {}}

    def session(symbols):
        now = _time.time()
        armed = (load_leadlag_status() or {}).get("armed")
        rows = []
        for s in symbols:
            if s in cache["rows"] and now - cache["ts"] < 60.0:
                rows.append(cache["rows"][s])
                continue
            try:
                df, _ = fetch_minute_bars(s, interval=1, auth=auth, env=env, base_url=base, max_pages=1)
                row = session_signals(s, df, us_armed=armed, safety=None)
            except Exception as e:
                row = {"symbol": s, "error": str(e)[:70], "price": None,
                       "label": "decision-support (no proven intraday edge)"}
            cache["rows"][s] = row
            rows.append(row)
        cache["ts"] = now
        return rows

    return session


def _build_stock_source(tracker):
    if not SETTINGS.has_alpaca_keys():
        return _BrokenStockSource("Set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")
    try:
        from tagent.ml.predict import Predictor
        from tagent.risk import RiskManager, RiskParams
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        predictor = Predictor.load()
        trading = TradingClient(SETTINGS.alpaca_api_key, SETTINGS.alpaca_secret_key,
                                paper=True)
        data = StockHistoricalDataClient(SETTINGS.alpaca_api_key, SETTINGS.alpaca_secret_key)
        equity = SETTINGS.account_equity
        try:
            equity = float(trading.get_account().equity)
        except Exception:
            pass
        risk = RiskManager(RiskParams(
            account_equity=equity, risk_per_trade_pct=SETTINGS.risk_per_trade_pct,
            max_position_pct=SETTINGS.max_position_pct, stop_loss_pct=SETTINGS.stop_loss_pct,
            take_profit_pct=SETTINGS.take_profit_pct, max_daily_loss_pct=SETTINGS.max_daily_loss_pct))
        return AlpacaStockSource(predictor, trading, data, risk, ta_tracker=tracker)
    except FileNotFoundError:
        return _BrokenStockSource("No trained model — run scripts/train_model.py")
    except Exception as e:
        return _BrokenStockSource(f"Alpaca init failed: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                    help="initial crypto subscriptions")
    ap.add_argument("--watch-stocks", nargs="+", default=["AAPL", "NVDA", "TSLA"],
                    help="stocks the overnight scorecard forward-tests")
    ap.add_argument("--horizon-min", type=int, default=15,
                    help="minutes before a signal is scored correct/wrong")
    ap.add_argument("--trade-pct", type=float, default=1.0,
                    help="fraction of the $10k base used per simulated trade")
    ap.add_argument("--reset-scorecard", action="store_true",
                    help="WIPE scorecard state + signal log (otherwise it resumes)")
    ap.add_argument("--momentum-top-n", type=int, default=100,
                    help="point-in-time universe size for the auto KR-momentum rebalance")
    ap.add_argument("--no-auto-rebalance", action="store_true",
                    help="disable the monthly KR-momentum auto-rebalance scheduler")
    ap.add_argument("--no-auto-advance", action="store_true",
                    help="disable the daily forward-ops auto-advance (trend-core shakedown + auditions)")
    ap.add_argument("--no-feed", action="store_true",
                    help="disable the continuous intraday news-feed poller")
    args = ap.parse_args()

    # one tracker + alert log shared by both panels -> one combined alerts feed
    tracker = TaTracker(alert_log=AlertLog())
    stock_source = _build_stock_source(tracker)
    crypto_source = CryptoSource(CryptoFeed(symbols=args.symbols), autostart=True,
                                 ta_tracker=tracker)

    # signal scorecard: forward-tests the dashboard's calls overnight. ALWAYS
    # resumes from data/scorecard_state.json unless --reset-scorecard is passed.
    scorecard = Scorecard(horizon_s=args.horizon_min * 60, trade_pct=args.trade_pct,
                          reset=args.reset_scorecard)

    def _scorecard_loop():
        import time
        while True:
            poll_scorecard(scorecard, stock_source, crypto_source,
                           args.watch_stocks, args.symbols)
            time.sleep(20)

    if scorecard.owner:                  # only the owning instance forward-tests
        threading.Thread(target=_scorecard_loop, daemon=True).start()
    else:
        print("  Scorecard: another instance owns the state; running read-only.")
    # show-don't-tell intraday demo: candles computed from RAW ticks (live Binance
    # aggTrades here; the same pipeline ingests Kiwoom 0B executions in production).
    # The safety light uses a systemic proxy (BTC's recent window return) standing in
    # for the linked US peers/index kill switch.
    def _linked_returns(_symbol):
        try:
            t = crypto_source.recent_ticks("BTCUSDT", limit=1000)
            if len(t) >= 2:
                return {"BTCUSDT": t[-1][1] / t[0][1] - 1.0}
        except Exception:
            pass
        return {}

    # 1s bars so the 5/20/60 MAs populate from ~1000 raw aggTrades (~2 min of BTC).
    intraday_source = IntradayDemoSource(
        tick_provider=lambda s: crypto_source.recent_ticks(s, limit=1000),
        linked_provider=_linked_returns, interval="1s")

    # the live paper runners (run_funding_live.py / run_momentum_live.py) write state we read.
    # the trend-core + forward-ops panels also carry the daily auto-advance stamp ("last
    # advanced / next advance"), so just keeping the dashboard up shows it advancing.
    def _trend_core_status():
        st = load_trend_core_status()
        if isinstance(st, dict) and st.get("enabled"):
            st = {**st, "advance": load_advance_state()}
        return st

    def _forward_ops_status():
        d = load_forward_ops()
        if isinstance(d, dict):
            d = {**d, "advance": load_advance_state()}
        return d

    def _briefing_status():
        # the daily 4-report briefing, persisted each morning by the auto-advance; read-only here
        from tagent.daily_briefing import load_briefing
        import datetime as _d
        b = load_briefing(_d.date.today().isoformat())
        if not b or not b.get("reports"):
            return {"enabled": False, "reports": {}, "breaking": [],
                    "note": "not generated yet — runs on the daily auto-advance"}
        return {"enabled": True, **b, "advance": load_advance_state()}

    def _feed_view(since=None):
        # continuous intraday feed for today (read-only here; the poller thread appends to it)
        import datetime as _d
        return feed_view(_d.date.today().isoformat(), since=since)

    app = create_app(stock_source, crypto_source, scorecard=scorecard,
                     funding_live=load_live_status, momentum_live=load_momentum_status,
                     intraday_source=intraday_source, news=_build_news(),
                     leadlag=load_leadlag_status, pead=load_pead_status,
                     desk=load_desk_snapshot, desk_session=_build_desk_session(),
                     trend_core=_trend_core_status, forward_ops=_forward_ops_status,
                     media=_build_media(), briefing=_briefing_status, feed=_feed_view,
                     orders=order_panel)

    # auto-rebalance the KR momentum paper book once per new month — so just keeping
    # the dashboard running performs the monthly update (the manual script still works;
    # the year-month guard in step() prevents any double-rebalance).
    if not args.no_auto_rebalance:
        from scripts.run_momentum_live import rebalance_once

        def _momentum_rebalance():
            try:
                st, info = rebalance_once(MomentumLiveTrader(), top_n=args.momentum_top_n)
                print(f"  [momentum] auto-rebalance: {info}")
            except Exception as e:
                print(f"  [momentum] auto-rebalance failed: {str(e)[:80]}")

        sched = MonthlyRebalanceScheduler(
            rebalance_fn=_momentum_rebalance,
            month_done_fn=lambda: load_momentum_status().get("last_month"))

        def _momentum_loop():
            import time
            while True:
                try:
                    sched.tick()
                except Exception:
                    pass
                time.sleep(3600)        # check hourly; throttle + month-guard limit real work

        threading.Thread(target=_momentum_loop, daemon=True).start()

    # auto-advance the forward operations once per new trading day — so just keeping the
    # dashboard running advances the trend-core ops shakedown AND every candidate audition
    # (the manual scripts still work; the once/day guard + idempotent runners prevent any
    # double-count). PAPER/mock only — nothing here places an order.
    if not args.no_auto_advance:
        import datetime as _dtm

        from scripts.run_forward_ops import advance_once as _advance_forward_ops
        from scripts.run_trend_core_live import advance_once as _advance_trend_core
        from tagent.forward_ops import DailyAdvanceScheduler, write_advance_state

        def _advance_all():
            ran, as_of = [], None
            try:
                st = _advance_trend_core()
                ran.append("trend_core")
                as_of = st.get("as_of")
            except Exception as e:
                print(f"  [advance] trend-core skipped: {str(e)[:80]}")
            try:
                _advance_forward_ops()
                ran.append("forward_ops")
            except Exception as e:
                print(f"  [advance] forward-ops skipped: {str(e)[:80]}")
            try:                                          # refresh the daily 4-report briefing
                from scripts.run_daily_briefing import build_and_persist
                _, binfo = build_and_persist(_NEWS_KR_WATCHLIST)
                ran.append("briefing")
                print(f"  [advance] briefing refreshed: {binfo['counts']}")
            except Exception as e:
                print(f"  [advance] briefing skipped: {str(e)[:80]}")
            day = _dtm.datetime.now(_dtm.timezone.utc).strftime("%Y-%m-%d")
            write_advance_state(day, as_of=as_of, ran=ran)
            print(f"  [advance] {day}: advanced {ran or 'nothing'} (data as of {as_of})")

        advance_sched = DailyAdvanceScheduler(
            advance_fn=_advance_all,
            day_done_fn=lambda: load_advance_state().get("last_advanced"))

        def _advance_loop():
            import time
            while True:
                try:
                    advance_sched.tick()
                except Exception:
                    pass
                time.sleep(3600)        # check hourly; throttle + day-guard limit real work

        threading.Thread(target=_advance_loop, daemon=True).start()

    # continuous intraday news FEED: re-fetch every source on a paced, per-source-throttled
    # interval; dedupe vs today's feed; APPEND only genuinely-new items (each with detected_at +
    # published_at + a real link). Throttle respects Naver/Finnhub quota + the YouTube IP-block.
    if not args.no_feed:
        import datetime as _fdt

        _kr_wl, _us_wl = _kr_whitelist(), _us_whitelist()

        def _wl(items):                                       # top-source whitelist for feed news
            return filter_top_sources(items, kr_whitelist=_kr_wl, us_whitelist=_us_wl)

        def _feed_fetchers():
            fx = {}
            if SETTINGS.has_naver_keys():
                nv = NaverNewsSource(SETTINGS.naver_client_id, SETTINGS.naver_client_secret)
                _rel = _naver_relevance_fn()
                fx["naver"] = lambda: _wl(from_news_items(
                    nv.watchlist_news(_NEWS_KR_WATCHLIST, per_symbol=3, relevance_fn=_rel), "naver"))
            if SETTINGS.has_finnhub_key():
                fh = FinnhubSource(SETTINGS.finnhub_api_key)

                def _fh():
                    t = _fdt.date.today()
                    return _wl(from_news_items(fh.market_news(
                        DEFAULT_LINKED_US, (t - _fdt.timedelta(days=2)).isoformat(),
                        t.isoformat(), per_symbol=4), "finnhub"))
                fx["finnhub"] = _fh
            if SETTINGS.has_opendart_key():
                od = OpenDartSource(SETTINGS.opendart_api_key)

                def _od():
                    t = _fdt.date.today()
                    return from_news_items(od.recent_disclosures_for(
                        _NEWS_KR_WATCHLIST, (t - _fdt.timedelta(days=3)).strftime("%Y%m%d"),
                        t.strftime("%Y%m%d")), "opendart")
                fx["opendart"] = _od
            if SETTINGS.has_youtube_key():
                from tagent.news.youtube_source import YouTubeSource, channels_from_env
                from tagent.gemini import build_extractor
                _yex = build_extractor(SETTINGS)         # provider per LLM_PROVIDER (gemini|openai)
                yt = YouTubeSource(SETTINGS.youtube_api_key, extract_fn=_yex,
                                   enable_fallbacks=True, proxy_url=SETTINGS.youtube_proxy_url,
                                   webshare=SETTINGS.webshare_proxy(),
                                   pace_seconds=SETTINGS.youtube_transcript_pace_seconds)
                fx["youtube"] = lambda: from_youtube_claims(
                    yt.media_briefing(channels_from_env(SETTINGS.youtube_channels), audit=AuditWriter()))
            if SETTINGS.has_kiwoom_keys():
                def _kw():
                    from tagent.daily_briefing import build_kiwoom_briefing
                    from tagent.feeds.kiwoom_auth import KiwoomAuth
                    from tagent.kiwoom_report import build_kiwoom_report
                    auth = KiwoomAuth(SETTINGS.kiwoom_app_key, SETTINGS.kiwoom_secret_key,
                                      env=SETTINGS.kiwoom_env, base_url=SETTINGS.kiwoom_rest_url())
                    pl = build_kiwoom_report(_NEWS_KR_WATCHLIST[:4], auth=auth, env=SETTINGS.kiwoom_env,
                                             base_url=SETTINGS.kiwoom_rest_url())
                    return from_kiwoom_cards(build_kiwoom_briefing(pl).get("items", []))
                fx["kiwoom"] = _kw
            return fx

        feed_throttle = SourceThrottle()
        feed_fetchers = _feed_fetchers()

        try:                                                 # one-time cleanup: re-filter today's feed
            _rinfo = refilter_feed(_fdt.date.today().isoformat(), kr_whitelist=_kr_wl, us_whitelist=_us_wl)
            if _rinfo.get("dropped"):
                print(f"  [feed] re-filtered today's feed: dropped {_rinfo['dropped']} non-top-source/"
                      f"tangential items ({_rinfo['after']} kept)")
        except Exception:
            pass

        def _feed_loop():
            import time
            while True:
                try:
                    day = _fdt.date.today().isoformat()
                    det = _fdt.datetime.now(_fdt.timezone.utc).isoformat()
                    res = poll_and_persist(feed_fetchers, day, detected_at=det, throttle=feed_throttle)
                    if res.get("appended"):
                        print(f"  [feed] +{res['appended']} new items (fetched {res['fetched']})")
                except Exception as e:
                    print(f"  [feed] poll error: {str(e)[:80]}")
                time.sleep(60)              # check each minute; per-source throttle paces real fetches

        if feed_fetchers:
            threading.Thread(target=_feed_loop, daemon=True).start()
            print(f"  Feed:       intraday poller on {sorted(feed_fetchers)} "
                  "(paced; appends to data/briefings/<date>/feed.jsonl)")

    import uvicorn
    print(f"\n  Dashboard:  http://{args.host}:{args.port}")
    print(f"  Scorecard:  forward-testing {args.watch_stocks} + {args.symbols}, "
          f"{args.horizon_min}-min horizon -> data/signal_log.csv\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
