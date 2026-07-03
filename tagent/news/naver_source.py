"""Naver News (KR newspapers — 한국경제 / 매일경제 / etc.) client with REAL article links.

API (Naver Search, openapi.naver.com): ``GET /v1/search/news.json?query=&display=&sort=date``
with headers ``X-Naver-Client-Id`` / ``X-Naver-Client-Secret``. Each item carries
``originallink`` (the publisher's REAL article URL) and ``link`` (Naver's mirror); we cite the
``originallink`` so every newspaper item points at its actual source. Titles/descriptions arrive
with <b> highlight tags and HTML entities, which we strip.

Each item gets a RULE-BASED bullish/bearish/neutral tag (transparent — not a price model). The
client credentials travel only in request headers and are never logged. ``requests`` is lazy +
the session is injectable, so tests parse canned payloads with no network.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

from tagent.news.youtube_source import TICKER_NAMES, media_sentiment

NEWS_URL = "https://openapi.naver.com/v1/search/news.json"

# Watchlist code -> the Korean query term Naver searches best on (its common name).
WATCHLIST_QUERY = dict(TICKER_NAMES)


def _strip_html(s: str) -> str:
    """Remove <b> highlight tags + unescape entities from a Naver title/description."""
    return html.unescape(re.sub(r"<[^>]+>", "", str(s or ""))).strip()


def _parse_pubdate(pub: str) -> float:
    """RFC-822 pubDate ('Mon, 09 Jun 2026 09:00:00 +0900') -> epoch seconds (0.0 on failure)."""
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(str(pub or ""))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


def _sentiment(text: str) -> str:
    score = media_sentiment(text)
    return "bullish" if score > 0.15 else ("bearish" if score < -0.15 else "neutral")


def is_relevant(item: dict, name: str) -> bool:
    """True if the exact company ``name`` appears in the article's title or summary — the
    relevance guard that drops off-topic Naver hits. Names <2 chars are not filtered (too noisy)."""
    nm = str(name or "").strip()
    if len(nm) < 2:
        return True
    return nm in f"{item.get('title', '')} {item.get('summary', '')}"


def title_subject(item: dict, name: str) -> bool:
    """STRICTER relevance: the company ``name`` is in the TITLE (the article's subject), not merely
    mentioned in the body — so a passing-name / promo / same-name piece is dropped."""
    nm = str(name or "").strip()
    if len(nm) < 2:
        return True
    return nm in str(item.get("title", ""))


def normalize_news(item: dict, symbol: str = "") -> dict:
    """Raw Naver news item -> normalized alert dict. ``url`` is the REAL article link
    (originallink, falling back to Naver's link)."""
    title = _strip_html(item.get("title"))
    desc = _strip_html(item.get("description"))
    url = str(item.get("originallink") or item.get("link") or "").strip()
    ts = _parse_pubdate(item.get("pubDate"))
    return {
        "source": "naver", "market": "kr",
        "time": (datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""), "ts": ts,
        "symbol": str(symbol or "").strip(), "title": title, "summary": desc,
        "type": "news", "sentiment": _sentiment(f"{title} {desc}"),
        "important": False, "url": url,
    }


class NaverNewsSource:
    """Search KR newspaper articles for the watchlist, each with its real article URL.
    Credentials live only in request headers; never logged."""

    def __init__(self, client_id: str, client_secret: str, session=None):
        self.client_id = client_id
        self.client_secret = client_secret
        self._session = session

    def _get(self, params: dict):
        session = self._session
        if session is None:
            import requests  # lazy
            session = requests
        headers = {"X-Naver-Client-Id": self.client_id, "X-Naver-Client-Secret": self.client_secret}
        resp = session.get(NEWS_URL, params=params, headers=headers, timeout=10)
        try:
            return resp.json()
        except Exception:
            return None

    def search(self, query: str, display: int = 10, sort: str = "date") -> List[dict]:
        """Recent articles for ``query`` (newest first). Only items WITH a real link are kept —
        no source link -> not surfaced."""
        data = self._get({"query": query, "display": display, "sort": sort})
        items = (data or {}).get("items") if isinstance(data, dict) else None
        out = [normalize_news(x) for x in (items or []) if isinstance(x, dict)]
        out = [a for a in out if a["url"]]                 # every item must cite a real article URL
        out.sort(key=lambda a: a["ts"], reverse=True)
        return out

    def watchlist_news(self, codes: Sequence[str], per_symbol: int = 5,
                       queries: Optional[Dict[str, str]] = None,
                       names: Optional[Dict[str, str]] = None,
                       relevance_fn: Optional[Callable[[list, str], list]] = None) -> List[dict]:
        """Recent newspaper articles per watchlist code, tagged with its symbol, newest first.

        RELEVANCE-FILTERED so tangential hits (incidental mentions, promos, same-name articles like
        a '035420 NAVER → FC온라인' or '삼성 → 노타 수상' piece) are dropped. ``relevance_fn(items, name)``
        (e.g. a Gemini judge) decides which articles are SUBSTANTIVELY about the stock; without it the
        STRICT fallback keeps only articles whose TITLE (the subject) contains the company name."""
        queries = queries or WATCHLIST_QUERY
        names = names or WATCHLIST_QUERY
        out: List[dict] = []
        for code in codes:
            nm = names.get(str(code), str(code))
            q = queries.get(str(code), str(code))
            try:
                arts = self.search(q, display=max(per_symbol * 3, per_symbol))   # over-fetch, then filter
            except Exception:
                continue
            if relevance_fn is not None:
                try:
                    arts = relevance_fn(arts, nm)
                except Exception:
                    arts = [a for a in arts if title_subject(a, nm)]
            else:
                arts = [a for a in arts if title_subject(a, nm)]   # strict: name is the TITLE subject
            for a in arts[:per_symbol]:
                out.append({**a, "symbol": str(code), "matched_name": nm})
        out.sort(key=lambda a: a["ts"], reverse=True)
        return out
