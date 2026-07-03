"""GROUNDED daily YouTube report BUILDER — data layer only (no docx/pdf/email/dashboard yet).

A daily report of what top Korean finance YouTube channels said about GIANT KOREAN stocks, grounded
100% in the real transcripts already captured in ``data/youtube_fetch_log/<date>/fetch.jsonl`` — we
do NOT re-transcribe and we NEVER fabricate. This exists to fix a rival report that copy-pasted fake
claims ("KODEX 200 signed AI infrastructure agreements with NVIDIA"), invented prices/catalysts,
contradicted itself, reused one templated sentence across stocks, and even rendered Japanese under a
"Korean" heading.

HARD anti-hallucination rules (enforced in code here, and instructed in the Gemini prompt for any
LLM step):
  * Every claim carries a VERBATIM quote that is present in that video's transcript + timestamp +
    deeplink; an insight whose quote is not found verbatim is DROPPED.
  * Recommendation Action is BUY/SELL/HOLD only when an analyst expressed a directional view in the
    transcript quote; otherwise Action="관심(WATCH)". Never BUY/SELL by default.
  * The Reason is specific to that stock and cites channel + speaker + quote + timestamp; an
    identical reason shared across two different stocks is treated as a templating bug and DROPPED.
  * No prices/percentages unless they appear in the transcript; an insight summary that introduces a
    number absent from the transcript falls back to citing the verbatim quote (never invents).
  * Catalysts/schedule entries are included ONLY when explicitly mentioned in a transcript (with
    quote + link); otherwise the section is empty. No invented events.
  * Universe = giant Korean names (configurable via YOUTUBE_REPORT_WATCHLIST); linked globals
    (NVIDIA/Broadcom/AMD/Micron/TSMC) only when tied to a giant in the SAME video; everything else
    (other tickers, ETFs like KODEX/SOXX, US-only small caps) is dropped.
  * Korean is PRIMARY; English is produced by a BATCH translate_fn ([str]->[str], one call for the
    whole report, e.g. tagent.gemini.translate_batch_fn). A KO field detected as Japanese is rejected
    (the rival shipped Japanese under a Korean heading).

Pure / offline by default: builds deterministically from the logged grounded insights, so the unit
tests need no network or LLM. Secrets live only in .env.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tagent.config import DATA_DIR
from tagent.youtube_audit import latest_entries_by_video

_KST = timezone(timedelta(hours=9))

# Giant-Korean universe (default top-20). Configurable via YOUTUBE_REPORT_WATCHLIST ("code=name,...").
DEFAULT_GIANTS: Dict[str, str] = {
    "005930": "삼성전자", "000660": "SK하이닉스", "373220": "LG에너지솔루션",
    "207940": "삼성바이오로직스", "005380": "현대차", "000270": "기아", "035420": "NAVER",
    "035720": "카카오", "006400": "삼성SDI", "051910": "LG화학", "068270": "셀트리온",
    "005490": "POSCO홀딩스", "012330": "현대모비스", "018260": "삼성SDS", "017670": "SK텔레콤",
    "105560": "KB금융", "055550": "신한지주", "028260": "삼성물산", "034020": "두산에너빌리티",
    "042700": "한미반도체",
}
# Linked globals — included ONLY when tied to a Korean giant in the same video.
LINKED_GLOBALS: Dict[str, str] = {
    "NVDA": "NVIDIA", "AVGO": "브로드컴", "AMD": "AMD", "MU": "마이크론", "TSM": "TSMC",
}

# High-impact ("sensitive") catalyst categories worth acting on.
SENSITIVE_CATEGORIES = {"M&A/deal", "regulatory/geopolitical", "earnings"}

# Directional-view vocabulary (scanned in the VERBATIM quote only).
_BUY_KW = ["매수", "사라", "사야", "비중확대", "비중 확대", "담아", "분할매수", "저점매수",
           "적극 매수", "적극매수", "buy", "overweight", "accumulate", "add"]
_SELL_KW = ["매도", "팔아", "팔라", "비중축소", "비중 축소", "익절", "손절", "차익실현",
            "줄여", "정리", "sell", "underweight", "reduce", "trim"]
_HOLD_KW = ["보유", "홀드", "유지", "관망", "hold"]

# Schedule / event language — a catalyst is logged ONLY if the quote actually mentions one.
_CATALYST_KW = ["예정", "일정", "예상", "발표", "출시", "공개", "컨퍼런스", "콘퍼런스", "행사",
                "실적발표", "어닝", "상장", "공모", "배당", "만기", "다음 주", "다음주", "내달",
                "이달", "차주", "분기 실적", "gtc", "ces", "wwdc"]
_DATE_RE = re.compile(r"\d+\s*월\s*\d+\s*일|\d+\s*분기|\bq[1-4]\b|\d{1,2}/\d{1,2}")

# Japanese kana — Korean prose never contains these; their presence is the rival's "Japanese under a
# Korean heading" bug.
_KANA_RE = re.compile(r"[぀-ヿ]")
_NUM_RE = re.compile(r"\d[\d.]*")


# --------------------------------------------------------------------------- #
# small pure helpers
# --------------------------------------------------------------------------- #
def _norm(s) -> str:
    """Whitespace-insensitive, case-folded form for 'verbatim present in transcript' checks."""
    return re.sub(r"\s+", "", str(s or "")).lower()


def looks_japanese(text) -> bool:
    """True if KO-primary prose contains Japanese kana (a rejection signal)."""
    return bool(_KANA_RE.search(str(text or "")))


def _coerce_dt(value, assume_tz=_KST) -> Optional[datetime]:
    """Parse a datetime / ISO string / epoch into an aware datetime; a naive value is assumed to be
    in ``assume_tz``. Returns None if unparseable."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=assume_tz)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (ValueError, OSError):
            return None
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=assume_tz)
    except ValueError:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if len(digits) >= 8:
            try:
                return datetime.strptime(digits[:8], "%Y%m%d").replace(tzinfo=assume_tz)
            except ValueError:
                return None
        return None


def _kst_iso(value) -> str:
    dt = _coerce_dt(value, timezone.utc)
    return dt.astimezone(_KST).isoformat() if dt else ""


def report_watchlist(watchlist: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The giant-Korean universe {ticker: name}: explicit ``watchlist`` wins, else
    YOUTUBE_REPORT_WATCHLIST (.env, "code=name,code=name" or bare "code,code"), else DEFAULT_GIANTS."""
    if watchlist:
        return dict(watchlist)
    env = os.getenv("YOUTUBE_REPORT_WATCHLIST", "").strip()
    if not env:
        return dict(DEFAULT_GIANTS)
    out: Dict[str, str] = {}
    for part in env.split(","):
        part = part.strip()
        if not part:
            continue
        code, _, name = part.partition("=")
        code = code.strip()
        if code:
            out[code] = name.strip() or DEFAULT_GIANTS.get(code) or LINKED_GLOBALS.get(code) or code
    return out or dict(DEFAULT_GIANTS)


# --------------------------------------------------------------------------- #
# load + window filter (read the fetch logs; do NOT re-transcribe)
# --------------------------------------------------------------------------- #
def _all_log_entries(data_dir=None) -> List[dict]:
    """Every logged video across all date folders, deduped by video id (keep the latest fetch). The
    log folder is named by FETCH date, which can differ from published_at, so we scan all folders and
    filter by published_at in the caller."""
    return list(latest_entries_by_video(data_dir).values())


def _videos_in_window(start, end, data_dir=None) -> List[dict]:
    """Logged videos whose ``published_at`` is within [start, end] (inclusive), newest first."""
    s = _coerce_dt(start, _KST)
    e = _coerce_dt(end, _KST)
    su = s.astimezone(timezone.utc) if s else None
    eu = e.astimezone(timezone.utc) if e else None
    out = []
    for v in _all_log_entries(data_dir):
        pub = _coerce_dt(v.get("published_at"), timezone.utc)
        if pub is None:
            continue
        if su and pub < su:
            continue
        if eu and pub > eu:
            continue
        out.append(v)
    out.sort(key=lambda v: str(v.get("published_at", "")), reverse=True)
    return out


# --------------------------------------------------------------------------- #
# grounding + universe filter
# --------------------------------------------------------------------------- #
def _ticker(ins: dict) -> str:
    return str(ins.get("ticker") or ins.get("stock") or "").strip()


def _quote_grounded(quote: str, transcript_norm: str) -> bool:
    """The quote must be present VERBATIM (whitespace-insensitive) in the transcript."""
    q = _norm(quote)
    return len(q) >= 4 and q in transcript_norm


def _ungrounded_number(text: str, transcript: str) -> bool:
    """True if ``text`` contains a digit-number that does NOT appear in the transcript (digit form) —
    i.e. an invented price/percentage. (Korean spoken numbers stay in the quote, never invented.)"""
    tdigits = re.sub(r"[,\s]", "", str(transcript or ""))
    for m in _NUM_RE.findall(str(text or "")):
        if m.replace(",", "") not in tdigits:
            return True
    return False


def _kept_insights(video: dict, giants: Dict[str, str], giant_only: bool = True):
    """Grounded insights for one video. Returns (kept, stats) where each kept insight is annotated
    with channel/title/published_at and a grounded ``reason``. Always drops ungrounded quotes and
    Japanese-contaminated prose. When ``giant_only`` (the daily WINDOW report) it also keeps only the
    giant universe (linked globals only if a giant is also discussed here); when False (the
    SINGLE-VIDEO report) it keeps EVERY substantively-discussed stock, correctly named."""
    segs = video.get("segments", [])
    transcript = " ".join(str(s.get("text", "")) for s in segs)
    tnorm = _norm(transcript)
    channel = video.get("channel", "")
    title = video.get("title", "")
    pub = video.get("published_at", "")
    vid = video.get("video_id", "")

    grounded, stats = [], {"dropped_ungrounded": 0, "rejected_japanese": 0}
    for ins in video.get("insights", []):
        quote = str(ins.get("quote", ""))
        if not _quote_grounded(quote, tnorm):              # fabricated / not in transcript -> DROP
            stats["dropped_ungrounded"] += 1
            continue
        summary = str(ins.get("summary", ""))
        if looks_japanese(summary):                        # Japanese under a Korean heading -> REJECT
            stats["rejected_japanese"] += 1
            continue
        # number-grounding: a summary number must be in the transcript, else cite the verbatim quote
        reason = summary if (summary and not _ungrounded_number(summary, transcript)) else quote
        grounded.append({**ins, "channel": ins.get("channel") or channel, "video_title": title,
                         "video_id": ins.get("video_id") or vid, "published_at": ins.get("published_at") or pub,
                         "_reason": reason})

    if not giant_only:                                     # single-video: keep EVERY discussed stock
        return grounded, stats
    tickers = {_ticker(i) for i in grounded}
    has_giant = any(t in giants for t in tickers)
    kept = []
    for i in grounded:
        t = _ticker(i)
        if t in giants:
            kept.append(i)
        elif t in LINKED_GLOBALS and has_giant:            # linked global only if tied to a giant here
            kept.append(i)
        # else: other tickers / ETFs (KODEX/SOXX) / US-only -> dropped
    return kept, stats


# --------------------------------------------------------------------------- #
# recommendation fields (grounded in the quote)
# --------------------------------------------------------------------------- #
_TARGET_KW = ["목표가", "목표주가", "적정가", "적정주가", "target price", "price target"]


def _norm_action(a) -> Optional[str]:
    """Map an explicit analyst call (Korean/English) to BUY/SELL/HOLD/WATCH, or None if not one."""
    s = str(a or "").strip().lower()
    if not s:
        return None
    if any(k in s for k in ("buy", "매수", "비중확대", "비중 확대", "overweight", "accumulate")):
        return "BUY"
    if any(k in s for k in ("sell", "매도", "비중축소", "비중 축소", "underweight", "reduce")):
        return "SELL"
    if any(k in s for k in ("hold", "보유", "홀드", "중립", "neutral")):
        return "HOLD"
    if any(k in s for k in ("watch", "관심")):
        return "관심(WATCH)"
    return None


def _action(quote: str, ins: Optional[dict] = None) -> str:
    """BUY/SELL/HOLD when the analyst expressed a directional view; else WATCH. An explicit insight
    ``action`` wins; otherwise a target/fair price (목표가/적정가) or buy wording -> BUY, sell -> SELL."""
    if ins is not None:
        a = _norm_action(ins.get("action"))
        if a:
            return a
        if ins.get("target_price"):                  # a stated target/fair price is a directional call
            return "BUY"
    q = str(quote or "").lower()
    if any(k in q for k in _SELL_KW):
        return "SELL"
    if any(k in q for k in _BUY_KW) or any(k in q for k in _TARGET_KW):
        return "BUY"
    if any(k in q for k in _HOLD_KW):
        return "HOLD"
    return "관심(WATCH)"


def _is_directional(ins: dict) -> bool:
    """True if the insight carries a grounded directional call (BUY/SELL/HOLD), not WATCH."""
    return _action(ins.get("quote", ""), ins) != "관심(WATCH)"


def _count_sentences(text: str) -> int:
    return len([p for p in re.split(r"(?<=[.!?。…])\s|다\.\s*|다$", str(text or "")) if p.strip()])


def _synthesize(insights, max_sentences: int = 4) -> str:
    """A grounded SYNTHESIS: combine an insight set's most important DISTINCT summaries into one
    multi-sentence block. Only restates verified insight text (no new facts) — highest-importance
    first, deduped, until ~max_sentences."""
    parts, seen, n = [], set(), 0
    for i in sorted(insights, key=lambda x: float(x.get("importance", 0.0) or 0.0), reverse=True):
        s = (i.get("_reason") or i.get("summary") or "").strip()
        if not s:
            continue
        k = _norm(s)
        if k in seen:
            continue
        seen.add(k)
        parts.append(s)
        n += max(1, _count_sentences(s))
        if n >= max_sentences:
            break
    return " ".join(parts).strip()


def _conviction(ins: dict) -> str:
    score = max(float(ins.get("importance", 0.0) or 0.0) / 5.0,
                abs(float(ins.get("sentiment_score", 0.0) or 0.0)))
    return "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"


def _has_catalyst(quote: str) -> bool:
    q = str(quote or "").lower()
    return bool(_DATE_RE.search(q)) or any(k in q for k in _CATALYST_KW)


def _name(ticker: str, giants: Dict[str, str], ins: dict) -> str:
    return giants.get(ticker) or LINKED_GLOBALS.get(ticker) or ins.get("stock_name") or ins.get("stock") or ticker


def _rec(ins: dict, giants: Dict[str, str]) -> dict:
    t = _ticker(ins)
    return {
        "stock": _name(t, giants, ins), "ticker": t,
        "action": _action(ins.get("quote", ""), ins), "conviction": _conviction(ins),
        "reason": ins.get("_reason") or ins.get("summary", ""), "target_price": ins.get("target_price") or "",
        "quote": ins.get("quote", ""), "channel": ins.get("channel", ""),
        "speaker": ins.get("speaker", ""), "timestamp": ins.get("timestamp_mmss", ""),
        "deeplink": ins.get("deeplink", ""),
    }


# --------------------------------------------------------------------------- #
# the builder
# --------------------------------------------------------------------------- #
def build_youtube_report(start=None, end=None, watchlist: Optional[Dict[str, str]] = None,
                         lang: str = "ko", data_dir=None, now=None,
                         translate_fn: Optional[Callable[[str], str]] = None,
                         prices_fn=None) -> dict:
    """Build the grounded daily YouTube report (structured dict, ready to render later).

    Window [start, end] in KST (default end=now, start=yesterday 00:00 KST); videos are included by
    ``published_at`` within the window. Reads the fetch logs (no re-transcription). Korean is primary;
    pass ``translate_fn`` — a BATCH translator ``[str]->[str]`` (e.g. ``tagent.gemini.translate_batch_fn
    (client)``) that renders ALL prose in ONE call — to also emit English (EN won't cost ~2x KO).

    ``prices_fn`` controls the REAL market-data price table (kept SEPARATE from the video insights):
    a callable ``codes -> rows`` is used directly; the sentinel ``"auto"`` fetches via Kiwoom when
    KIWOOM keys exist (else ``[]``); ``None`` (default) leaves ``prices`` empty (offline/test-safe)."""
    nowk = _coerce_dt(now, _KST) or datetime.now(_KST)
    end_dt = _coerce_dt(end, _KST) or nowk
    if start is not None:
        start_dt = _coerce_dt(start, _KST)
    else:
        y = (nowk - timedelta(days=1))
        start_dt = datetime(y.year, y.month, y.day, 0, 0, tzinfo=_KST)   # yesterday 00:00 KST

    giants = report_watchlist(watchlist)
    videos = _videos_in_window(start_dt, end_dt, data_dir)
    return _assemble_report(videos, giants, start_dt=start_dt, end_dt=end_dt, nowk=nowk,
                            lang=lang, translate_fn=translate_fn, prices_fn=prices_fn)


def build_report_from_videos(videos, watchlist: Optional[Dict[str, str]] = None, lang: str = "ko",
                             now=None, translate_fn: Optional[Callable[[str], str]] = None,
                             prices_fn=None, start=None, end=None) -> dict:
    """Build the SAME report structure from an in-memory list of video entries (each
    {video_id, channel, title, published_at, segments, insights}) — used by the single-video path.
    Reuses all grounding / universe / anti-hallucination rules. ``prices_fn`` default None -> []."""
    videos = list(videos or [])
    nowk = _coerce_dt(now, _KST) or datetime.now(_KST)
    end_dt = _coerce_dt(end, _KST) or nowk
    pub = _coerce_dt(videos[0].get("published_at"), timezone.utc) if videos else None
    start_dt = _coerce_dt(start, _KST) or pub or nowk      # window header = the video's publish time
    giants = report_watchlist(watchlist)
    return _assemble_report(videos, giants, start_dt=start_dt, end_dt=end_dt, nowk=nowk,
                            lang=lang, translate_fn=translate_fn, prices_fn=prices_fn, single_video=True)


def _assemble_report(videos, giants, *, start_dt, end_dt, nowk, lang, translate_fn, prices_fn,
                     single_video=False) -> dict:
    """Grounding -> universe filter -> sections, shared by the window and single-video builders. The
    WINDOW report keeps the giant-only universe + prices for all giants; the SINGLE-VIDEO report keeps
    EVERY discussed stock and prices ONLY the stocks the video actually mentions."""
    sources, per_stock = [], {}
    all_kept: List[dict] = []
    dropped_ungrounded = rejected_japanese = 0
    channels = set()
    for v in videos:
        kept, stats = _kept_insights(v, giants, giant_only=not single_video)
        dropped_ungrounded += stats["dropped_ungrounded"]
        rejected_japanese += stats["rejected_japanese"]
        channels.add(v.get("channel", ""))
        sources.append({"channel": v.get("channel", ""), "title": v.get("title", ""),
                        "published_at_kst": _kst_iso(v.get("published_at")),
                        "url": f"https://www.youtube.com/watch?v={v.get('video_id')}" if v.get("video_id") else "",
                        "n_insights": len(kept)})
        for i in kept:
            all_kept.append(i)
            per_stock.setdefault(_name(_ticker(i), giants, i), []).append(i)

    all_kept.sort(key=lambda i: float(i.get("importance", 0.0) or 0.0), reverse=True)

    # §4 시세 (price table) REMOVED from the report — no longer built, passed, or rendered.
    # _price_table / report_prices / Kiwoom / pykrx are left intact (dormant) so no imports break.
    prices = []

    # recommendations: one per stock — PREFER a grounded directional call (BUY/SELL/HOLD) over a WATCH
    # one (so a stated 목표가/매수·매도 surfaces), then by importance; reject templated/duplicate reasons.
    best_by_ticker: Dict[str, dict] = {}
    for i in all_kept:                                        # all_kept is importance-desc
        t = _ticker(i)
        if not t:
            continue
        cur = best_by_ticker.get(t)
        if cur is None or (_is_directional(i) and not _is_directional(cur)):
            best_by_ticker[t] = i                            # upgrade WATCH -> directional grounded call
    recs = [_rec(i, giants) for i in best_by_ticker.values()]
    reason_owners: Dict[str, set] = {}
    for r in recs:
        reason_owners.setdefault(_norm(r["reason"]), set()).add(r["ticker"])
    dup_reasons = {k for k, owners in reason_owners.items() if len(owners) > 1}   # shared across stocks
    recs = [r for r in recs if _norm(r["reason"]) not in dup_reasons]            # templating bug -> drop

    # §1 핵심 요약 — an executive SYNTHESIS (grounded): an overall overview + a multi-sentence,
    # source-linked synthesis per major stock, built ONLY from that stock's grounded insights.
    by_ticker_insights: Dict[str, list] = {}
    for i in all_kept:
        t = _ticker(i)
        if t in giants:
            by_ticker_insights.setdefault(t, []).append(i)
    # rank stocks by their strongest insight; each entry synthesizes its top points + strongest link
    ranked = sorted(by_ticker_insights.items(),
                    key=lambda kv: max(float(x.get("importance", 0.0) or 0.0) for x in kv[1]), reverse=True)
    summary = []
    for t, ins_list in ranked[:8]:
        top = max(ins_list, key=lambda x: float(x.get("importance", 0.0) or 0.0))
        summary.append({"stock": _name(t, giants, top), "ticker": t,
                        "text": _synthesize(ins_list, max_sentences=4),     # 3-4 sentence grounded synthesis
                        "channel": top.get("channel", ""), "timestamp": top.get("timestamp_mmss", ""),
                        "deeplink": top.get("deeplink", "")})
    # overall market overview: a factual scope line + the top grounded points (no new facts)
    n_stocks = len(by_ticker_insights)
    if all_kept:
        lead = (f"이번 구간 {n_stocks}개 핵심 종목에서 {len(all_kept)}건의 자막 근거 인사이트가 수집되었습니다."
                if lang == "ko" else
                f"This window collected {len(all_kept)} caption-grounded insights across {n_stocks} key stocks.")
        overview = (lead + " " + _synthesize(all_kept, max_sentences=3)).strip()
    else:
        overview = ""

    # §3 민감·중요 뉴스 — grounded high-impact items that are DISTINCT from §1 (no repeats). Any item
    # whose text is already headlined in the §1 overview/synthesis is dropped here.
    s1_corpus = _norm(overview + " " + " ".join(b.get("text", "") for b in summary))
    sensitive_news, seen_sn = [], set()
    for i in all_kept:
        if i.get("category") not in SENSITIVE_CATEGORIES:
            continue
        text = i.get("_reason") or i.get("summary", "")
        tn = _norm(text)
        if not tn or tn in seen_sn or (s1_corpus and tn in s1_corpus):   # dedup vs §3 itself AND §1
            continue
        seen_sn.add(tn)
        sensitive_news.append(
            {"stock": _name(_ticker(i), giants, i), "ticker": _ticker(i), "category": i.get("category", ""),
             "text": text, "quote": i.get("quote", ""), "channel": i.get("channel", ""),
             "timestamp": i.get("timestamp_mmss", ""), "deeplink": i.get("deeplink", "")})

    # catalysts: ONLY when the quote explicitly mentions a schedule/event (else empty)
    catalysts = [
        {"stock": _name(_ticker(i), giants, i), "ticker": _ticker(i),
         "event": i.get("_reason") or i.get("summary", ""), "quote": i.get("quote", ""),
         "channel": i.get("channel", ""), "timestamp": i.get("timestamp_mmss", ""),
         "deeplink": i.get("deeplink", "")}
        for i in all_kept if _has_catalyst(i.get("quote", ""))]

    report = {
        "meta": {
            "generated_at_kst": nowk.isoformat(), "window_start": start_dt.astimezone(_KST).isoformat(),
            "window_end": end_dt.astimezone(_KST).isoformat(), "lang": lang,
            "n_videos": len(sources), "n_insights": len(all_kept),
            "channels": sorted(c for c in channels if c),
            "dropped_ungrounded": dropped_ungrounded, "rejected_japanese": rejected_japanese,
            # single-video reports use the video's publish time as the header (not a zero-length window)
            "single_video": bool(single_video),
            "video_published_kst": (sources[0]["published_at_kst"] if (single_video and sources) else ""),
        },
        "overview": overview, "summary": summary, "recommendations": recs,
        "sensitive_news": sensitive_news, "per_stock": per_stock, "catalysts": catalysts,
        "sources": sources,
        "prices": prices,                                     # REAL market data (키움/KRX), separate
        "en": (_english(overview, summary, recs, sensitive_news, catalysts, per_stock, translate_fn)
               if translate_fn else None),
    }
    return report


def _price_table(codes, prices_fn) -> List[dict]:
    """The REAL price table for the universe. A callable wins; "auto" fetches via Kiwoom when keys
    exist; None -> [] (offline/test-safe). Never raises into the report (returns [] on any error)."""
    if callable(prices_fn):
        try:
            return list(prices_fn(codes) or [])
        except Exception:
            return []
    if prices_fn == "auto":
        try:
            from tagent.config import SETTINGS
            if not SETTINGS.has_kiwoom_keys():
                return []
            from tagent.feeds.kiwoom_auth import KiwoomAuth
            from tagent.news.report_prices import fetch_price_table
            auth = KiwoomAuth(SETTINGS.kiwoom_app_key, SETTINGS.kiwoom_secret_key,
                              env=SETTINGS.kiwoom_env, base_url=SETTINGS.kiwoom_rest_url())
            return fetch_price_table(codes, auth=auth)
        except Exception:
            return []
    return []


def _english(overview, summary, recs, sensitive_news, catalysts, per_stock, tr) -> dict:
    """English mirror of EVERY rendered prose field — overview, §1 synthesis, recommendation reason +
    cited quote, sensitive-news text + quote, catalysts, AND the §5 per-stock summaries + quotes — so
    the English report is actually English (not Korean body under English headings). Numbers / links /
    tickers / timestamps stay intact. ``tr`` is a BATCH translator (``[str] -> [str]``): ALL prose is
    translated in ONE call (EN doesn't cost ~2x KO)."""
    texts = [overview]
    texts += [b.get("text", "") for b in summary]
    for r in recs:
        texts += [r.get("reason", ""), r.get("quote", "")]
    for n in sensitive_news:
        texts += [n.get("text", ""), n.get("quote", "")]
    for c in catalysts:
        texts += [c.get("event", ""), c.get("quote", "")]
    for _stock, lst in per_stock.items():
        for it in lst:
            texts += [it.get("summary", ""), it.get("quote", "")]
    try:
        en = tr(texts)
    except Exception:
        en = texts
    if not isinstance(en, list) or len(en) != len(texts):
        en = texts                                            # shape mismatch -> keep originals (safe net)
    g = iter(en)
    out = {
        "overview": next(g),
        "summary": [{**b, "text": next(g)} for b in summary],
        "recommendations": [{**r, "reason": next(g), "quote": next(g)} for r in recs],
        "sensitive_news": [{**n, "text": next(g), "quote": next(g)} for n in sensitive_news],
        "catalysts": [{**c, "event": next(g), "quote": next(g)} for c in catalysts],
    }
    out["per_stock"] = {stock: [{**it, "summary": next(g), "quote": next(g)} for it in lst]
                        for stock, lst in per_stock.items()}
    return out
