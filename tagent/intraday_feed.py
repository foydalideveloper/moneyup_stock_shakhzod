"""Continuous intraday news FEED — paced polling, dual timestamps, append-only per trading day.

Throughout the trading day the poller re-fetches every source on a paced, per-source-throttled
interval (Naver 한경/매경, Finnhub US, YouTube, OpenDART, Kiwoom flows), dedupes against
already-seen items (by URL/id), and APPENDS only genuinely-new items to today's feed
(``data/briefings/<date>/feed.jsonl``). Each item carries BOTH timestamps:

  * ``detected_at``  — OUR poller's server clock when we FIRST saw the item (never changes after),
  * ``published_at`` — the source's own publish time (Naver pubDate / Finnhub datetime / YouTube
                       video publish; YouTube also keeps the ``[mm:ss]`` moment),

plus a real link (article URL or YouTube deep-link). GROUNDED: no link -> no item, never invented;
external news needs a real http URL; our-own signals (Kiwoom flow) carry a provenance cite.

Pure / injectable: ``poll_once`` takes fetcher callables + a clock, so the whole feed unit-tests
with synthetic data and no network. Append-only persistence preserves the first-seen detected_at.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from tagent.daily_briefing import _as_date, briefing_day_dir

# The boss is in Korea — store timestamps in UTC internally, but DISPLAY everything in KST.
_KST = timezone(timedelta(hours=9))


def _to_dt(value) -> Optional[datetime]:
    """Parse a UTC ISO string / epoch into an aware UTC datetime (None if unparseable)."""
    if value in (None, ""):
        return None
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
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if len(digits) >= 8:
            try:
                return datetime.strptime(digits[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None


def kst_hhmm(value) -> str:
    """A UTC timestamp rendered as KST HH:MM (Asia/Seoul). "" if unparseable."""
    dt = _to_dt(value)
    return dt.astimezone(_KST).strftime("%H:%M") if dt else ""


def kst_stamp(value) -> str:
    """A UTC timestamp rendered as KST 'MM-DD HH:MM' (Asia/Seoul); a date-only value -> KST date."""
    dt = _to_dt(value)
    if dt is None:
        return ""
    sv = str(value)
    if isinstance(value, str) and "T" not in sv and ":" not in sv:   # date-only -> no fake time
        return dt.astimezone(_KST).strftime("%Y-%m-%d")
    return dt.astimezone(_KST).strftime("%m-%d %H:%M")

# Per-source minimum seconds between REAL fetches (throttle to respect quotas / the YouTube
# IP-block). Configurable; YouTube is the longest because its transcript endpoint rate-limits hard.
DEFAULT_POLL_SECONDS = 720                 # ~12 min default cadence
DEFAULT_THROTTLE = {"naver": 600, "finnhub": 600, "opendart": 900, "kiwoom": 900, "youtube": 1800}


def _is_http(u: str) -> bool:
    return str(u or "").lower().startswith(("http://", "https://"))


def _iso_to_ts(iso: str) -> float:
    s = "".join(ch for ch in str(iso or "") if ch.isdigit())
    for ln, fmt in ((14, "%Y%m%d%H%M%S"), (8, "%Y%m%d")):
        if len(s) >= ln:
            try:
                return datetime.strptime(s[:ln], fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    return 0.0


def _hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]


def make_item(source: str, *, link: str, title: str = "", symbol: str = "", published_at: str = "",
              published_ts: float = 0.0, link_type: Optional[str] = None, mmss: str = "",
              external: bool = True, extra: Optional[dict] = None) -> Optional[dict]:
    """Canonical feed item (pre-detection). Returns None when UNGROUNDED — no link at all, or an
    external-news item without a real http URL — so nothing is ever invented."""
    link = str(link or "").strip()
    if not link:
        return None
    if external and not _is_http(link):
        return None
    lt = link_type or ("url" if _is_http(link) else "provenance")
    item_id = ("u:" + link.lower() if _is_http(link)
               else f"s:{source}:{symbol}:{_hash(title or link)}")
    return {"id": item_id, "source": source, "title": str(title)[:300], "symbol": str(symbol or ""),
            "published_at": str(published_at or ""), "published_ts": float(published_ts or 0.0),
            "link": link, "link_type": lt, "mmss": str(mmss or ""), "extra": dict(extra or {})}


# --------------------------------------------------------------------------- #
# adapters — each source's normalized items -> feed items (dual timestamps)
# --------------------------------------------------------------------------- #
def from_news_items(items: Sequence[dict], source: str) -> List[dict]:
    """Naver / Finnhub / OpenDART normalized items (carry url + ts + time) -> feed items. The
    publisher (Finnhub ``source``) is carried for the US source whitelist."""
    out = []
    for it in items or []:
        mi = make_item(source, link=it.get("url", ""), title=it.get("title", ""),
                       symbol=it.get("symbol", ""), published_at=it.get("time", ""),
                       published_ts=float(it.get("ts", 0) or 0.0),
                       extra={"publisher": it.get("publisher", "")})
        if mi:
            mi["publisher"] = it.get("publisher", "")
            out.append(mi)
    return out


def from_youtube_claims(claims: Sequence[dict]) -> List[dict]:
    """Grounded YouTube insights (summary + supporting quote span + deeplink + mm:ss) -> feed items.
    The title is the insight SUMMARY (the actual point), with the quote span kept in extra."""
    out = []
    for c in claims or []:
        mi = make_item("youtube", link=c.get("deeplink", ""),
                       title=c.get("summary") or c.get("quote", ""),
                       symbol=c.get("stock", ""), published_at=c.get("published_at", ""),
                       published_ts=_iso_to_ts(c.get("published_at", "")),
                       link_type="deeplink", mmss=c.get("timestamp_mmss", ""),
                       extra={"channel": c.get("channel", ""), "video_title": c.get("video_title", ""),
                              "quote": c.get("quote", ""), "category": c.get("category", "")})
        if mi:
            out.append(mi)
    return out


def from_kiwoom_cards(cards: Sequence[dict]) -> List[dict]:
    """Kiwoom 수급 one-line interpretations -> feed items (our-own data; provenance cite, not http)."""
    out = []
    for c in cards or []:
        for ln in c.get("lines", []):
            if not ln.get("text"):
                continue
            mi = make_item("kiwoom", link=ln.get("cite", ""), external=False, link_type="provenance",
                           title=f"{c.get('symbol', '')} {ln['text']}", symbol=c.get("symbol", ""),
                           extra={"kind": ln.get("kind")})
            if mi:
                out.append(mi)
    return out


# --------------------------------------------------------------------------- #
# paced polling — per-source throttle + dedupe + detected_at stamping
# --------------------------------------------------------------------------- #
class SourceThrottle:
    """Per-source min-interval pacer (epoch-seconds clock). Keeps each source from being fetched
    more often than its quota allows."""

    def __init__(self, intervals: Optional[Dict[str, float]] = None, clock: Optional[Callable] = None):
        self.intervals = dict(DEFAULT_THROTTLE)
        if intervals:
            self.intervals.update(intervals)
        self._last: Dict[str, float] = {}
        self._clock = clock or (lambda: datetime.now(timezone.utc).timestamp())

    def due(self, source: str, now: Optional[float] = None) -> bool:
        now = self._clock() if now is None else now
        last = self._last.get(source)
        return last is None or (now - last) >= self.intervals.get(source, DEFAULT_POLL_SECONDS)

    def mark(self, source: str, now: Optional[float] = None) -> None:
        self._last[source] = self._clock() if now is None else now


def poll_once(fetchers: Dict[str, Callable[[], List[dict]]], seen_ids: set, *, detected_at: str,
              throttle: SourceThrottle, now_ts: Optional[float] = None) -> dict:
    """Fetch each DUE source, dedupe vs ``seen_ids`` (mutated), stamp ``detected_at`` on genuinely
    new items. A throttled source is skipped (not fetched). Returns the new items + which sources
    were fetched/skipped. Pure aside from the injected fetchers."""
    new_items, fetched, skipped = [], [], []
    for source, fetch in fetchers.items():
        if not throttle.due(source, now_ts):
            skipped.append(source)
            continue
        try:
            items = fetch() or []
        except Exception:
            items = []
        throttle.mark(source, now_ts)
        fetched.append(source)
        for it in items:
            if not it or not it.get("id") or it["id"] in seen_ids:
                continue
            seen_ids.add(it["id"])
            new_items.append({**it, "detected_at": detected_at})
    new_items.sort(key=lambda r: (r.get("published_ts", 0.0), r.get("title", "")), reverse=True)
    return {"new": new_items, "n_new": len(new_items), "fetched": fetched, "skipped": skipped}


# --------------------------------------------------------------------------- #
# append-only persistence (preserves first-seen detected_at) + NEW marking
# --------------------------------------------------------------------------- #
def feed_path(day, data_dir=None) -> Path:
    return briefing_day_dir(day, data_dir) / "feed.jsonl"


def load_feed(day, data_dir=None) -> List[dict]:
    p = feed_path(day, data_dir)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def seen_ids_for(day, data_dir=None) -> set:
    return {it.get("id") for it in load_feed(day, data_dir) if it.get("id")}


def append_feed(items: Sequence[dict], day, data_dir=None) -> int:
    """APPEND-ONLY: write each new item as one JSONL line. Existing lines are never rewritten, so a
    first-seen detected_at is preserved for the life of the day."""
    items = [it for it in (items or []) if it]
    if not items:
        return 0
    p = feed_path(day, data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(items)


def refilter_feed(day, *, kr_whitelist=None, us_whitelist=None, relevance_ok: Optional[Callable] = None,
                  data_dir=None) -> dict:
    """One-time cleanup: REWRITE today's feed.jsonl keeping only top-source + relevant news items
    (cited links preserved), so pre-fix tangential items disappear. Non-news (youtube/kiwoom) kept.
    ``relevance_ok(item)->bool`` defaults to the strict 'KR name is the title subject' check."""
    from tagent.news.source_whitelist import is_top_source

    if relevance_ok is None:
        from tagent.news.naver_source import title_subject
        from tagent.news.youtube_source import TICKER_NAMES

        def relevance_ok(it):                                # only re-judge KR newspaper items
            if str(it.get("source") or "") != "naver":
                return True
            name = TICKER_NAMES.get(str(it.get("symbol") or ""), "")
            return (not name) or title_subject({"title": it.get("title", "")}, name)

    items = load_feed(day, data_dir)
    kept = [it for it in items
            if is_top_source(it, kr_whitelist=kr_whitelist, us_whitelist=us_whitelist) and relevance_ok(it)]
    path = feed_path(day, data_dir)
    if path.exists():
        with path.open("w", encoding="utf-8") as fh:        # rewrite (deliberate cleanup, UTF-8)
            for it in kept:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return {"before": len(items), "after": len(kept), "dropped": len(items) - len(kept)}


def poll_and_persist(fetchers: Dict[str, Callable[[], List[dict]]], day, *, detected_at: str,
                     throttle: SourceThrottle, now_ts: Optional[float] = None, data_dir=None) -> dict:
    """One paced poll cycle: dedupe vs today's persisted feed, append only the new items."""
    seen = seen_ids_for(day, data_dir)
    res = poll_once(fetchers, seen, detected_at=detected_at, throttle=throttle, now_ts=now_ts)
    res["appended"] = append_feed(res["new"], day, data_dir)
    return res


def mark_new(items: Sequence[dict], since: Optional[str]) -> List[dict]:
    """Flag items detected since the caller's last view (``since`` = the latest detected_at it saw)."""
    return [{**it, "is_new": bool(since) and str(it.get("detected_at", "")) > str(since)}
            for it in items]


def feed_view(day, *, since: Optional[str] = None, limit: int = 200, data_dir=None) -> dict:
    """Newest-first feed for the dashboard — sorted by detected_at (when WE saw it), NEW-marked
    against ``since``. Honest 'no items' when the feed is empty."""
    items = load_feed(day, data_dir)
    items.sort(key=lambda r: (str(r.get("detected_at", "")), r.get("published_ts", 0.0)), reverse=True)
    items = mark_new(items[:limit], since)
    # storage stays UTC; add KST display fields so the boss (Asia/Seoul) sees local wall-clock time
    items = [{**i, "detected_kst": kst_hhmm(i.get("detected_at")),
              "published_kst": kst_stamp(i.get("published_at"))} for i in items]
    latest = items[0]["detected_at"] if items else (since or "")
    return {"enabled": True, "date": str(_as_date(day)), "tz": "Asia/Seoul (KST)", "n": len(items),
            "items": items, "latest_detected_at": latest,
            "n_new": sum(1 for i in items if i.get("is_new")),
            "note": "" if items else "no items yet today"}
