"""Daily 4-REPORT briefing — newspaper / Kiwoom 수급 / YouTube / recommendation.

Fetches each source (honest WALL per report if one is unreachable — never filled with generic
text), assembles the grounded+cited briefing, persists it under data/briefings/<date>/
(accumulating, never overwriting prior days), and prints it. Mock-first; secrets are never
printed. INFORMATIONAL only — not a validated edge, not a trading signal.

Usage:
    python scripts/run_daily_briefing.py
    python scripts/run_daily_briefing.py --symbols 005930 000660 035420
"""

import argparse
import datetime as _dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tagent.config import SETTINGS  # noqa: E402
from tagent.daily_briefing import build_daily_briefing, persist_briefing, render_briefing_text  # noqa: E402

KR_WATCH = ["005930", "000660", "035420", "051910", "005490", "000270", "207940"]


def _gemini_client():
    """A GeminiClient if GEMINI_API_KEY is set, else None (callers fall back to the grounded default)."""
    if not SETTINGS.has_gemini_key():
        return None
    try:
        from tagent.gemini import GeminiClient
        return GeminiClient(SETTINGS.gemini_api_key, model=SETTINGS.gemini_model)
    except Exception:
        return None


def _newspaper_items(symbols):
    """(naver_items, finnhub_items, wall). Each item already cites its real article URL."""
    naver, finnhub, walls = [], [], []
    today = _dt.date.today()
    if SETTINGS.has_naver_keys():
        try:
            from tagent.news.naver_source import NaverNewsSource
            g = _gemini_client()
            rel = None
            if g is not None:
                from tagent.gemini import naver_relevance_fn
                rel = naver_relevance_fn(g)              # Gemini judges substantive relevance
            naver = NaverNewsSource(SETTINGS.naver_client_id, SETTINGS.naver_client_secret) \
                .watchlist_news(symbols, per_symbol=4, relevance_fn=rel)
        except Exception as e:
            walls.append(f"Naver: {str(e)[:60]}")
    if SETTINGS.has_finnhub_key():
        try:
            from tagent.news.finnhub_source import DEFAULT_LINKED_US, FinnhubSource
            finnhub = FinnhubSource(SETTINGS.finnhub_api_key).market_news(
                DEFAULT_LINKED_US, (today - _dt.timedelta(days=3)).isoformat(), today.isoformat(),
                per_symbol=5)
        except Exception as e:
            walls.append(f"Finnhub: {str(e)[:60]}")
    wall = "; ".join(walls) if (walls and not naver and not finnhub) else None
    return naver, finnhub, wall


def _youtube_claims():
    if not SETTINGS.has_youtube_key():
        return [], "YOUTUBE_API_KEY 미설정"
    # FIRST: replay today's already-logged insights (populated by scripts/run_youtube_batch.py) —
    # fast, no re-transcription. Whisper is slow, so we don't re-fetch when the log already has them.
    try:
        from tagent.youtube_audit import load_insights_from_log
        logged = load_insights_from_log()
        if logged:
            return logged, None
    except Exception:
        pass
    try:
        from tagent.news.youtube_source import YouTubeSource, channels_from_env
        from tagent.gemini import build_extractor
        extract_fn = build_extractor(SETTINGS)            # provider per LLM_PROVIDER (gemini|openai)
        from tagent.youtube_audit import AuditWriter
        src = YouTubeSource(SETTINGS.youtube_api_key, extract_fn=extract_fn,
                            enable_fallbacks=True, proxy_url=SETTINGS.youtube_proxy_url,
                            webshare=SETTINGS.webshare_proxy(),
                            pace_seconds=SETTINGS.youtube_transcript_pace_seconds)  # api->yt-dlp->whisper
        return src.media_briefing(channels_from_env(SETTINGS.youtube_channels),
                                  audit=AuditWriter()), None    # log the raw fetch (audit)
    except Exception as e:
        return [], f"YouTube: {str(e)[:60]}"


def _kiwoom_payload(symbols):
    if not SETTINGS.has_kiwoom_keys():
        return None, "KIWOOM 키 미설정"
    try:
        from tagent.feeds.kiwoom_auth import KiwoomAuth
        from tagent.kiwoom_report import build_kiwoom_report
        auth = KiwoomAuth(SETTINGS.kiwoom_app_key, SETTINGS.kiwoom_secret_key,
                          env=SETTINGS.kiwoom_env, base_url=SETTINGS.kiwoom_rest_url())
        return build_kiwoom_report(symbols, auth=auth, env=SETTINGS.kiwoom_env,
                                   base_url=SETTINGS.kiwoom_rest_url()), None
    except Exception as e:
        return None, f"Kiwoom: {str(e)[:60]}"


def _prices(symbols):
    """Latest daily % change per stock (for the price-vs-news consistency guard)."""
    out = {}
    try:
        from tagent.stock_momentum import load_stock_panel
        for s in symbols:
            try:
                df = load_stock_panel("kr", symbols=[s], fields=["close"], min_bars=2).get(s)
                c = df["close"].dropna() if df is not None and "close" in df else None
                if c is not None and len(c) >= 2 and c.iloc[-2]:
                    out[s] = float(c.iloc[-1] / c.iloc[-2] - 1.0) * 100.0
            except Exception:
                continue
    except Exception:
        pass
    return out


def _momentum_panel(symbols):
    """Cached KR closes for the momentum view, refreshing missing names best-effort."""
    from tagent.stock_momentum import load_stock_panel
    panel = {}
    for s in symbols:
        try:
            panel.update(load_stock_panel("kr", symbols=[s], fields=["close"], min_bars=2))
        except Exception:
            continue
    missing = [s for s in symbols if s not in panel]
    if missing:                                          # refresh the panels if missing (best-effort)
        try:
            from tagent.data.krx_source import get_krx_history
            for s in missing:
                try:
                    get_krx_history(s)
                    panel.update(load_stock_panel("kr", symbols=[s], fields=["close"], min_bars=2))
                except Exception:
                    continue
        except Exception:
            pass
    return panel


def _recommendation_inputs(symbols):
    """(our_view momentum picks, providers [broker, 13F])."""
    from tagent.recommendation import momentum_picks, naver_broker_picks, whale_13f_picks
    panel = _momentum_panel(symbols)
    our = momentum_picks(symbols, panel=panel or None, top_n=5)
    providers = []
    if SETTINGS.has_naver_keys():
        try:
            from tagent.news.naver_source import NaverNewsSource
            nv = NaverNewsSource(SETTINGS.naver_client_id, SETTINGS.naver_client_secret)
            providers.append(naver_broker_picks(nv, symbols))
        except Exception:
            pass
    providers.append(whale_13f_picks(symbols))           # cached holdings; empty -> no picks
    return our, providers


def assemble_live(symbols, today=None):
    """Fetch every source (walling failures), build the 4-report briefing payload. No persistence."""
    today = today or _dt.date.today().isoformat()
    naver, finnhub, news_wall = _newspaper_items(symbols)
    claims, yt_wall = _youtube_claims()
    kiwoom, kw_wall = _kiwoom_payload(symbols)
    our_view, providers = _recommendation_inputs(symbols)
    walls = {k: v for k, v in (("newspaper", news_wall), ("youtube", yt_wall),
                               ("kiwoom", kw_wall)) if v}
    from tagent.news.source_whitelist import is_top_source, parse_whitelist
    kr = parse_whitelist(SETTINGS.source_whitelist_kr, [])
    us = parse_whitelist(SETTINGS.source_whitelist_us, [])
    return build_daily_briefing(
        symbols=symbols, today=today, naver_items=naver, finnhub_items=finnhub,
        youtube_claims=claims, kiwoom_payload=kiwoom, providers=providers, our_view=our_view,
        kiwoom_picks=[], prices=_prices(symbols), walls=walls,
        source_filter=lambda it: is_top_source(it, kr_whitelist=kr, us_whitelist=us))


def build_and_persist(symbols, today=None):
    """Assemble the live briefing and persist it (used by the dashboard's daily auto-advance)."""
    payload = assemble_live(symbols, today=today)
    info = persist_briefing(payload)
    return payload, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=KR_WATCH)
    args = ap.parse_args()
    payload, info = build_and_persist(args.symbols)
    print(render_briefing_text(payload))
    print(f"\n-> persisted to {info['dir']}  counts={info['counts']} (accumulates daily; history kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
