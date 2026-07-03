"""Daily 4-REPORT briefing — grounded, cited, deduped, persisted.

Assembles four reports on top of the already-built grounded engines, fixing the failure modes
of a naive briefing:

  #1 NEWSPAPER     — 한경/매경 (Naver) + Finnhub (US); every item cites its REAL article URL;
                     breaking macro/geopolitical/war shocks surfaced at the TOP; per-stock the
                     SPECIFIC cited reason; a price-vs-news CONSISTENCY GUARD (never "up" on a
                     down day); no boilerplate closing paragraph.
  #2 KIWOOM        — the new ``tagent.kiwoom_report`` one-line interpretations (공매도 / 수급 /
                     프로그램), NOT raw price tables; empty TRs flagged ("needs live account").
  #3 YOUTUBE       — the grounded/timestamped ``youtube_source`` claims (verbatim quote + mm:ss
                     deep-link); "AI 잠재력" filler suppressed; "no coverage" when nothing stated.
  #4 RECOMMENDATION— a DIFFERENTIATED blend: KR broker picks + our momentum/LLM view + US-house
                     / 13F views, each pick CITED, marked distinct-from-Kiwoom, with source
                     agreement / disagreement.

PRINCIPLE (all four): every surfaced claim carries a citation; no source → "no coverage", never
invent. Claims are de-duplicated across all four reports via one shared seen-set. Dates are
validated (no future / stale-recycled). If a source is unreachable the report is a WALL (honest),
not filled with generic text. INFORMATIONAL — not a validated edge, not a trading signal.

This module is PURE/assembly (the runner does the network I/O and passes materials in), so the
whole briefing unit-tests with synthetic data and no network. Persistence ACCUMULATES per day
under ``data/briefings/<date>/`` and never overwrites prior days.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from tagent.config import DATA_DIR
from tagent.news.youtube_source import WATCHLIST_TICKERS

MAX_AGE_DAYS = 14                       # older than this = stale/recycled -> dropped

NEWS_LABEL = "신문 — 한경/매경(Naver)+Finnhub · 기사 원문 링크 인용"
YT_LABEL = "유튜브/TV — 근거 인용(verbatim quote + mm:ss 점프링크), 없으면 no coverage"
REC_LABEL = "추천 — 국내 증권사 · 자체 모멘텀 · 미국 13F/하우스 블렌드 (출처 인용)"
BRIEFING_LABEL = "일일 4-리포트 브리핑 — 모든 주장 출처 인용 · 미보유 시 no coverage (정보용, 신호 아님)"


# --------------------------------------------------------------------------- #
# shared helpers — dates, dedup, source, breaking, price consistency
# --------------------------------------------------------------------------- #
def _as_date(today) -> date:
    if today is None:
        return datetime.now(timezone.utc).date()
    if isinstance(today, datetime):
        return today.date()
    if isinstance(today, date):
        return today
    s = "".join(ch for ch in str(today) if ch.isdigit())[:8]
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return datetime.now(timezone.utc).date()


def _date_of_ts(ts) -> Optional[date]:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None


def _date_of_iso(iso) -> Optional[date]:
    s = "".join(ch for ch in str(iso or "") if ch.isdigit())[:8]
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None


def valid_date(when, today, *, iso: bool = False) -> bool:
    """True if ``when`` is not in the future and not older than MAX_AGE_DAYS (anti-recycle).
    Undated items are kept (some feeds omit a timestamp)."""
    today = _as_date(today)
    d = _date_of_iso(when) if iso else _date_of_ts(when)
    if d is None:
        return True
    return not (d > today) and (today - d).days <= MAX_AGE_DAYS


def claim_key(item: dict) -> str:
    """Stable cross-report dedup key: the real link if any, else symbol+headline/quote."""
    link = (item.get("url") or item.get("deeplink") or item.get("source_link") or "").strip().lower()
    if link:
        return "u:" + link
    kind = item.get("_kind", "")
    body = str(item.get("title") or item.get("quote") or item.get("symbol") or "").strip().lower()
    return f"k:{kind}:{body[:160]}"


class SeenSet:
    """Tracks claim keys already surfaced — so a claim appears in ONE report only."""
    def __init__(self):
        self._seen: set = set()

    def add(self, item: dict) -> bool:
        k = claim_key(item)
        if k in self._seen:
            return False
        self._seen.add(k)
        return True


def citation(item: dict) -> str:
    """The citation backing a claim — a URL/deep-link for external sources, or an explicit
    provenance tag for our own data (Kiwoom TR / momentum engine). Empty -> uncited."""
    return (item.get("url") or item.get("deeplink") or item.get("source_link")
            or item.get("cite") or "").strip()


# ---- BREAKING classifier: PRECISE matching + a market-wide relevance gate ----
# Loose substring matching mis-fired ("software" -> "war"); we use whole-word/phrase patterns for
# war/geo, an exclusion list for metaphorical "war", and require a market-wide (not single-company)
# shock. Macro indicators (CPI/FOMC/inflation/rate) are inherently market-wide and stay permissive.

# Kinetic-war terms (EN whole-word; KO distinctive substrings). Metaphorical "war" is excluded below.
_WAR_WORDS = [r"\bwar\b", r"\bwars\b", r"\bwarfare\b", r"\bwartime\b", r"\binvasion\b",
              r"\binvade[sd]?\b", r"\bmissile\b", r"\bairstrike\b", r"\bair strike\b",
              r"\bshelling\b", r"\bceasefire\b", r"\bcombat\b"]
_WAR_KO = ["전쟁", "전면전", "공습", "침공", "포격", "미사일", "교전", "개전", "내전", "무력충돌"]
# Phrases where "war"/"공격" is metaphorical or a proper name -> stripped before testing war terms.
_WAR_EXCLUDE = [
    r"\bprice war\b", r"\bbidding war\b", r"\bwar chest\b", r"\bwar of words\b", r"\bwar room\b",
    r"\bturf war\b", r"\b(ai|chip|chips|tech|trade|tariff|talent|culture|format|patent|streaming|"
    r"pricing|bid|cola|console|browser|app|platform|content|cloud) war\b",
    r"\bwar-?gam\w*", r"\bwarrior\w*", r"\bstar wars\b", r"\bwarren\b", r"\bwarner\b",
    r"\bwarrant\w*", r"\bwarehouse\w*", r"\bwarm\w*", r"\bwary\b",
    # KOREAN metaphorical "전쟁" (chip/price/trade/talent war etc.) — stripped before the 전쟁 check.
    r"반도체\s*전쟁", r"칩\s*전쟁", r"메모리\s*전쟁", r"가격\s*전쟁", r"무역\s*전쟁", r"관세\s*전쟁",
    r"인재\s*전쟁", r"특허\s*전쟁", r"치킨\s*전쟁", r"플랫폼\s*전쟁", r"환율\s*전쟁", r"문화\s*전쟁",
    r"패권\s*전쟁", r"점유율\s*전쟁", r"ai\s*전쟁", r"기술\s*전쟁",
]
# Real geopolitical POLICY terms (whole-word EN; KO substrings).
_GEO_WORDS = [r"\btariffs?\b", r"\bsanctions?\b", r"\bexport controls?\b", r"\bexport ban\b",
              r"\bembargo\b", r"\bchips act\b", r"\bgeopolitical\b"]
# 무역전쟁 is EXCLUDED above as a metaphor; real trade policy is caught by 관세/제재/수출규제/미중갈등.
_GEO_KO = ["관세", "제재", "수출규제", "수출통제", "수출금지", "보복관세", "지정학", "미중 갈등",
           "미중갈등", "반도체법", "무역분쟁"]
# Macro shock keywords (kept as-is, distinctive substrings) + a few precise EN words ('fed' must be
# whole-word so it doesn't fire on 'feed'/'federal').
_MACRO_SUB = ["금리", "연준", "fomc", "cpi", "인플레이션", "inflation", "유가", "oil shock",
              "환율 급", "폭락", "급락", "crash", "circuit breaker", "서킷브레이커", "panic",
              "recession", "경기침체"]
_MACRO_WORDS = [r"\bfed\b", r"\brate (hike|cut|decision)\b"]
# Inherently market-wide indicators that OVERRIDE the single-company gate.
_MACRO_INDICATOR_SUB = ["cpi", "fomc", "inflation", "금리", "연준", "인플레이션", "물가", "기준금리",
                        "circuit breaker", "서킷브레이커", "recession", "경기침체"]
# Watchlist company-name tokens (KO/EN names, not bare codes) for the single-company gate.
_COMPANY_TOKENS = sorted({a for a in WATCHLIST_TICKERS if not a.replace(" ", "").isdigit()},
                         key=len, reverse=True)


def _is_war(low: str) -> bool:
    cleaned = low
    for p in _WAR_EXCLUDE:                                # drop metaphorical 'war'/names first
        cleaned = re.sub(p, " ", cleaned)
    return any(re.search(p, cleaned) for p in _WAR_WORDS) or any(k in cleaned for k in _WAR_KO)


def _is_geo(low: str) -> bool:
    return any(re.search(p, low) for p in _GEO_WORDS) or any(k in low for k in _GEO_KO)


def _is_macro(low: str) -> bool:
    return any(k in low for k in _MACRO_SUB) or any(re.search(p, low) for p in _MACRO_WORDS)


def breaking_category(text: str) -> Optional[str]:
    """Precise breaking category (war / geopolitical / macro) or None — whole-word EN matching +
    metaphorical-war exclusions, so 'software' / 'price war' / 'AI war' are NOT war."""
    low = str(text or "").lower()
    if _is_war(low):
        return "war"
    if _is_geo(low):
        return "geopolitical"
    if _is_macro(low):
        return "macro"
    return None


def is_market_wide(text: str) -> bool:
    """Relevance gate: a genuine market-wide shock, NOT a single-company headline. Strong macro
    indicators (CPI/FOMC/inflation/rate/연준) override; otherwise a headline naming a specific
    watchlist company is treated as single-company (e.g. a Broadcom or NAVER-cloud story)."""
    low = str(text or "").lower()
    if any(k in low for k in _MACRO_INDICATOR_SUB) or re.search(r"\bfed\b|\brate (hike|cut|decision)\b", low):
        return True
    return not any(tok in low for tok in _COMPANY_TOKENS)


def capture_breaking(news_items: Sequence[dict], today, limit: int = 6) -> List[dict]:
    """Genuine market-wide geopolitical/war/macro shocks, deduped, newest-first, cited only.
    A precise category AND the market-wide gate are both required (kills single-company / metaphor)."""
    out, seen = [], set()
    for it in news_items:
        if not it.get("url") or not valid_date(it.get("ts"), today):
            continue
        blob = f"{it.get('title', '')} {it.get('summary', '')}"
        cat = breaking_category(blob)
        if not cat or not is_market_wide(blob):          # precise category + market-wide relevance
            continue
        k = it["url"].strip().lower()
        if k in seen:
            continue
        seen.add(k)
        out.append({**it, "breaking_category": cat, "cite": it["url"]})
    out.sort(key=lambda a: a.get("ts", 0), reverse=True)
    return out[:limit]


def reconcile_direction(sentiment: str, price_change: Optional[float]) -> dict:
    """Price-vs-news CONSISTENCY GUARD: never let a bullish headline read as "up" on a down day.
    Returns the actual price direction + a mismatch note when the news skews against the tape."""
    if price_change is None:
        return {"consistent": True, "price_dir": None, "note": ""}
    pdir = "up" if price_change > 0 else ("down" if price_change < 0 else "flat")
    mismatch = ((sentiment == "bullish" and price_change < 0) or
                (sentiment == "bearish" and price_change > 0))
    if mismatch:
        kdir = "하락" if pdir == "down" else "상승"
        return {"consistent": False, "price_dir": pdir,
                "note": f"뉴스 {sentiment}이나 주가 {kdir} ({price_change:+.1f}%) — 방향 불일치"}
    return {"consistent": True, "price_dir": pdir, "note": ""}


def _wall(name: str, today, msg: str, label: str = "") -> dict:
    return {"report": name, "date": str(_as_date(today)), "label": label, "wall": str(msg),
            "items": [], "n": 0, "note": "source unreachable — reported as a wall, not filled"}


# --------------------------------------------------------------------------- #
# #1 NEWSPAPER
# --------------------------------------------------------------------------- #
def build_newspaper_report(symbols: Sequence[str], *, naver_items: Sequence[dict] = (),
                           finnhub_items: Sequence[dict] = (), prices: Optional[Dict[str, float]] = None,
                           today=None, seen: Optional[SeenSet] = None, wall: Optional[str] = None,
                           source_filter: Optional[Callable[[dict], bool]] = None) -> dict:
    today = _as_date(today)
    if wall:
        return _wall("newspaper", today, wall, NEWS_LABEL)
    prices = prices or {}
    seen = seen if seen is not None else SeenSet()
    allnews = [x for x in list(naver_items) + list(finnhub_items)
               if x.get("url") and valid_date(x.get("ts"), today)         # cited + dated only
               and (source_filter is None or source_filter(x))]           # top-source whitelist

    breaking = [b for b in capture_breaking(allnews, today) if seen.add(b)]

    by_sym: Dict[str, List[dict]] = {}
    for it in sorted(allnews, key=lambda a: a.get("ts", 0), reverse=True):
        by_sym.setdefault(str(it.get("symbol") or ""), []).append(it)

    items, no_cov = [], []
    for sym in symbols:
        cand = by_sym.get(str(sym), [])
        cand = [c for c in cand if c.get("url")]
        if not cand:
            no_cov.append(str(sym))                                     # honest: no coverage
            continue
        best = cand[0]
        rec = reconcile_direction(best.get("sentiment", "neutral"), prices.get(str(sym)))
        line = {"symbol": str(sym), "reason": best.get("title", ""), "url": best["url"],
                "cite": best["url"], "source": best.get("source", ""), "time": best.get("time", ""),
                "sentiment": best.get("sentiment", "neutral"), "price_change": prices.get(str(sym)),
                "price_dir": rec["price_dir"], "consistent": rec["consistent"], "note": rec["note"],
                "_kind": "news"}
        if seen.add(line):                                             # dedupe vs breaking/others
            items.append(line)
    return {"report": "newspaper", "date": str(today), "label": NEWS_LABEL, "wall": None,
            "breaking": breaking, "items": items, "n": len(items), "no_coverage": no_cov}


# --------------------------------------------------------------------------- #
# #2 KIWOOM (one-line interpretations; no price tables; empty TRs flagged)
# --------------------------------------------------------------------------- #
def build_kiwoom_briefing(kiwoom_payload: Optional[dict] = None, *, today=None,
                          seen: Optional[SeenSet] = None, wall: Optional[str] = None) -> dict:
    today = _as_date(today)
    if wall:
        return _wall("kiwoom", today, wall, (kiwoom_payload or {}).get("label", "Kiwoom 수급"))
    payload = kiwoom_payload or {}
    seen = seen if seen is not None else SeenSet()
    cards = []
    for c in payload.get("stocks", []):
        sym = str(c.get("symbol") or "")
        lines = []
        for blk, tag in (("short", "공매도"), ("supply", "수급"), ("program", "프로그램")):
            b = c.get(blk) or {}
            cite = "kiwoom:" + str((b.get("info") or {}).get("api_id", blk))
            if b.get("text"):
                lines.append({"kind": blk, "tag": tag, "text": b["text"],
                              "status": b.get("status"), "cite": cite})
            elif b.get("empty"):                                       # honest empty-TR flag
                lines.append({"kind": blk, "tag": tag, "text": None, "empty": True,
                              "note": b.get("note", "데이터 없음"), "cite": cite})
        if seen.add({"symbol": sym, "_kind": "kiwoom"}):              # dedupe per stock
            cards.append({"symbol": sym, "lines": lines})
    return {"report": "kiwoom", "date": payload.get("generated") or str(today),
            "label": payload.get("label", "Kiwoom 수급·공매도·프로그램"), "env": payload.get("env"),
            "wall": None, "items": cards, "n": len(cards),
            "empty_endpoints": payload.get("empty_endpoints", []), "note": payload.get("note", "")}


# --------------------------------------------------------------------------- #
# #3 YOUTUBE (grounded claims; no price table; "no coverage" when empty)
# --------------------------------------------------------------------------- #
def build_youtube_briefing(claims: Sequence[dict] = (), *, today=None,
                           seen: Optional[SeenSet] = None, wall: Optional[str] = None) -> dict:
    today = _as_date(today)
    if wall:
        return _wall("youtube", today, wall, YT_LABEL)
    seen = seen if seen is not None else SeenSet()
    items = []
    for c in claims:
        if not c.get("deeplink"):                                     # no source -> never surface
            continue
        if not valid_date(c.get("published_at"), today, iso=True):
            continue
        row = {**c, "cite": c["deeplink"], "_kind": "youtube"}
        if seen.add(row):
            items.append(row)
    items.sort(key=lambda r: (str(r.get("published_at", "")), r.get("importance", 0.0)), reverse=True)
    return {"report": "youtube", "date": str(today), "label": YT_LABEL, "wall": None,
            "items": items, "n": len(items),
            "note": "" if items else "no coverage — 워치리스트 종목 관련 발언 없음"}


# --------------------------------------------------------------------------- #
# #4 RECOMMENDATION (differentiated blend; cited; agreement/disagreement)
# --------------------------------------------------------------------------- #
def build_recommendation_report(symbols: Sequence[str] = (), *, providers: Sequence[dict] = (),
                                our_view: Sequence[dict] = (), kiwoom_picks: Sequence[str] = (),
                                today=None, seen: Optional[SeenSet] = None,
                                wall: Optional[str] = None) -> dict:
    """``providers``: [{house, source, picks:[{symbol, reason, url, side}]}] (KR brokers + US
    houses/13F). ``our_view``: [{symbol, reason, side, cite}] (our momentum/LLM). External picks
    MUST carry a real url or they are dropped (never invented)."""
    today = _as_date(today)
    if wall:
        return _wall("recommendation", today, wall, REC_LABEL)
    seen = seen if seen is not None else SeenSet()
    agg: Dict[str, dict] = {}
    for prov in providers:
        house, src = prov.get("house", ""), prov.get("source", "")
        for p in prov.get("picks", []):
            sym, url = str(p.get("symbol") or ""), (p.get("url") or "").strip()
            if not sym or not url:                                    # no source -> skip
                continue
            ph = str(p.get("house") or house or "")                   # per-pick broker attribution
            a = agg.setdefault(sym, {"houses": [], "sides": set(), "reasons": [], "our": None})
            a["houses"].append(ph)
            a["sides"].add(str(p.get("side", "buy")))
            a["reasons"].append({"house": ph, "source": p.get("source") or src,
                                 "reason": p.get("reason", ""), "url": url,
                                 "side": str(p.get("side", "buy"))})
    for v in our_view:
        sym = str(v.get("symbol") or "")
        if not sym:
            continue
        a = agg.setdefault(sym, {"houses": [], "sides": set(), "reasons": [], "our": None})
        a["our"] = {"reason": v.get("reason", ""), "side": str(v.get("side", "buy")),
                    "cite": v.get("cite", "momentum-engine")}
        a["sides"].add(str(v.get("side", "buy")))

    kset = {str(x) for x in kiwoom_picks}
    items = []
    for sym, a in agg.items():
        cite = next((r["url"] for r in a["reasons"]), "") or (a["our"] or {}).get("cite", "")
        if not cite:                                                  # uncited -> drop
            continue
        sides = sorted(a["sides"])
        item = {"symbol": sym, "houses": sorted({h for h in a["houses"] if h}),
                "sides": sides, "agreement": "동의" if len(sides) == 1 else "이견",
                "distinct_from_kiwoom": sym not in kset, "reasons": a["reasons"],
                "our_view": a["our"], "cite": cite, "_kind": "rec",
                "n_sources": len({h for h in a["houses"] if h}) + (1 if a["our"] else 0)}
        if seen.add({"symbol": sym, "_kind": "rec"}):
            items.append(item)
    items.sort(key=lambda r: (r["n_sources"], r["distinct_from_kiwoom"]), reverse=True)
    return {"report": "recommendation", "date": str(today), "label": REC_LABEL, "wall": None,
            "items": items, "n": len(items), "kiwoom_picks": sorted(kset),
            "note": "" if items else "no coverage — 인용 가능한 추천 출처 없음"}


# --------------------------------------------------------------------------- #
# assemble — one shared breaking capture + cross-report dedup
# --------------------------------------------------------------------------- #
def build_daily_briefing(*, symbols: Sequence[str], today=None, naver_items: Sequence[dict] = (),
                         finnhub_items: Sequence[dict] = (), youtube_claims: Sequence[dict] = (),
                         kiwoom_payload: Optional[dict] = None, providers: Sequence[dict] = (),
                         our_view: Sequence[dict] = (), kiwoom_picks: Sequence[str] = (),
                         prices: Optional[Dict[str, float]] = None,
                         walls: Optional[Dict[str, str]] = None,
                         source_filter: Optional[Callable[[dict], bool]] = None) -> dict:
    """The four reports assembled with ONE shared seen-set (cross-report dedup) and ONE shared
    breaking-news capture surfaced at the top. ``walls`` maps a report name to an unreachable
    message so that source is reported as a wall instead of being faked. ``source_filter`` (the
    top-source whitelist) gates the newspaper items."""
    today = _as_date(today)
    walls = walls or {}
    seen = SeenSet()
    news = build_newspaper_report(symbols, naver_items=naver_items, finnhub_items=finnhub_items,
                                  prices=prices, today=today, seen=seen, wall=walls.get("newspaper"),
                                  source_filter=source_filter)
    youtube = build_youtube_briefing(youtube_claims, today=today, seen=seen,
                                     wall=walls.get("youtube"))
    kiwoom = build_kiwoom_briefing(kiwoom_payload, today=today, seen=seen, wall=walls.get("kiwoom"))
    rec = build_recommendation_report(symbols, providers=providers, our_view=our_view,
                                      kiwoom_picks=kiwoom_picks, today=today, seen=seen,
                                      wall=walls.get("recommendation"))
    return {"date": str(today), "label": BRIEFING_LABEL, "breaking": news.get("breaking", []),
            "reports": {"newspaper": news, "kiwoom": kiwoom, "youtube": youtube,
                        "recommendation": rec}}


# --------------------------------------------------------------------------- #
# persistence — accumulate per day under data/briefings/<date>/, never overwrite history
# --------------------------------------------------------------------------- #
def briefing_root(data_dir=None) -> Path:
    return (Path(data_dir) if data_dir is not None else Path(DATA_DIR)) / "briefings"


def briefing_day_dir(day, data_dir=None) -> Path:
    return briefing_root(data_dir) / str(_as_date(day))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _item_key(it: dict) -> str:
    return (it.get("url") or it.get("deeplink") or it.get("cite") or it.get("symbol")
            or json.dumps(it, ensure_ascii=False, sort_keys=True))[:240]


def _merge_report(old: dict, new: dict) -> dict:
    """Union items by key (latest wins), KEEPING any prior items not in this run — so a same-day
    re-run accumulates and never loses history. ``runs`` counts the re-runs."""
    by = {_item_key(i): i for i in (old or {}).get("items", [])}
    for i in new.get("items", []):
        by[_item_key(i)] = i
    merged = {**new, "items": list(by.values()), "runs": int((old or {}).get("runs", 0)) + 1,
              "first_seen": (old or {}).get("first_seen", new.get("date"))}
    merged["n"] = len(merged["items"])
    return merged


def persist_briefing(payload: dict, data_dir=None) -> dict:
    """Write each report to data/briefings/<date>/<name>.json, ACCUMULATING within the day and
    leaving every prior day's folder untouched. Returns per-report item counts."""
    day = payload.get("date") or str(_as_date(None))
    d = briefing_day_dir(day, data_dir)
    counts = {}
    for name, rep in (payload.get("reports") or {}).items():
        path = d / f"{name}.json"
        merged = _merge_report(_load_json(path), rep)
        _write_json(path, merged)
        counts[name] = merged["n"]
    # breaking: union by url across same-day runs
    bpath = d / "breaking.json"
    bold = {b.get("url"): b for b in _load_json(bpath).get("items", [])}
    for b in payload.get("breaking", []):
        bold[b.get("url")] = b
    _write_json(bpath, {"date": day, "items": list(bold.values())})
    _write_json(d / "briefing.json", {"date": day, "label": payload.get("label"),
                                      "reports": list((payload.get("reports") or {})), "counts": counts})
    # append-only run index across all days
    idx = briefing_root(data_dir) / "index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    with idx.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"date": day, "counts": counts}, ensure_ascii=False) + "\n")
    return {"dir": str(d), "date": day, "counts": counts}


def load_briefing(day, data_dir=None) -> dict:
    """Read back a persisted day's reports (or {} if absent)."""
    d = briefing_day_dir(day, data_dir)
    if not d.exists():
        return {}
    out = {"date": str(_as_date(day)), "reports": {}}
    for name in ("newspaper", "kiwoom", "youtube", "recommendation"):
        rep = _load_json(d / f"{name}.json")
        if rep:
            out["reports"][name] = rep
    out["breaking"] = _load_json(d / "breaking.json").get("items", [])
    return out


# --------------------------------------------------------------------------- #
# render — plain text (only cited lines; "no coverage" honest; walls reported)
# --------------------------------------------------------------------------- #
def render_briefing_text(payload: dict) -> str:
    out = [f"=== {payload.get('label', '일일 브리핑')} · {payload.get('date', '')} ==="]
    br = payload.get("breaking", [])
    if br:
        out.append("\n[속보 BREAKING]")
        for b in br:
            out.append(f"  ⚑ [{b.get('breaking_category')}] {b.get('title', '')}  ({b.get('url')})")
    reports = payload.get("reports", {})

    def _wallnote(rep):
        return f"  · WALL: {rep['wall']}" if rep.get("wall") else None

    news = reports.get("newspaper", {})
    out.append("\n[#1 신문]")
    if _wallnote(news):
        out.append(_wallnote(news))
    else:
        for it in news.get("items", []):
            tail = f"  ⚠ {it['note']}" if not it.get("consistent") else ""
            out.append(f"  · {it['symbol']}: {it['reason']}  ({it['url']}){tail}")
        if news.get("no_coverage"):
            out.append(f"  · no coverage: {', '.join(news['no_coverage'])}")

    kiwoom = reports.get("kiwoom", {})
    out.append("\n[#2 키움 수급]")
    if _wallnote(kiwoom):
        out.append(_wallnote(kiwoom))
    else:
        for c in kiwoom.get("items", []):
            lines = [f"{ln['tag']}: {ln['text']}" for ln in c.get("lines", []) if ln.get("text")]
            empties = [ln['tag'] for ln in c.get("lines", []) if ln.get("empty")]
            txt = " · ".join(lines) if lines else "특이 수급 신호 없음"
            if empties:
                txt += f"  (빈 TR: {', '.join(empties)} — 실계좌 필요)"
            out.append(f"  · {c['symbol']}: {txt}")

    yt = reports.get("youtube", {})
    out.append("\n[#3 유튜브/TV]")
    if _wallnote(yt):
        out.append(_wallnote(yt))
    elif yt.get("items"):
        for it in yt["items"]:
            out.append(f"  · {it.get('stock_name', it.get('stock'))} — {it.get('channel')}: "
                       f"\"{it.get('quote')}\" [{it.get('timestamp_mmss')} → {it.get('deeplink')}]")
    else:
        out.append(f"  · {yt.get('note', 'no coverage')}")

    rec = reports.get("recommendation", {})
    out.append("\n[#4 추천 블렌드]")
    if _wallnote(rec):
        out.append(_wallnote(rec))
    elif rec.get("items"):
        for it in rec["items"]:
            houses = ", ".join(it["houses"]) + ("+자체" if it.get("our_view") else "")
            mark = "★자체차별" if it["distinct_from_kiwoom"] else "키움중복"
            out.append(f"  · {it['symbol']} [{it['agreement']}·{mark}] {houses}  ({it['cite']})")
    else:
        out.append(f"  · {rec.get('note', 'no coverage')}")
    return "\n".join(out)
