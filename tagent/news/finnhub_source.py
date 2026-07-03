"""Finnhub (US news + sentiment) client — for the US peers/indices linked to KR names.

API (finnhub.io):
  * ``GET /api/v1/company-news?symbol=&from=&to=&token=`` -> recent company headlines,
  * ``GET /api/v1/news-sentiment?symbol=&token=`` -> a company news-sentiment score
    (premium on some plans; handled gracefully if empty).

Each headline gets a RULE-BASED sentiment score in [-1, 1] from keyword rules (so the
alert layer is transparent and never a price model). The token travels only in the
query string and is never logged; ``requests`` is lazy + injectable for offline tests.

Default linkage: KR semis (Samsung 005930 / SK Hynix 000660) track US chip names +
broad indices; this is used by the kill-switch to read "is the linked US tape weak?".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
NEWS_SENTIMENT_URL = "https://finnhub.io/api/v1/news-sentiment"

# US peers / indices linked to each KR symbol (semis-heavy watchlist + broad tape).
DEFAULT_LINKED_US = ["SPY", "QQQ", "NVDA", "MU", "AVGO"]
LINKED_PEERS: Dict[str, List[str]] = {
    "005930": ["NVDA", "MU", "AVGO", "QQQ", "SPY"],   # Samsung Electronics (memory/semis)
    "000660": ["MU", "NVDA", "AVGO", "QQQ", "SPY"],   # SK Hynix (memory)
    "005380": ["TM", "GM", "F", "SPY"],               # Hyundai Motor (autos)
    "207940": ["LLY", "NVO", "XBI", "SPY"],           # Samsung Biologics (biotech)
}

_BULLISH_KW = ["beat", "beats", "surge", "soar", "record", "upgrade", "raises", "raise",
               "jumps", "jump", "rally", "tops", "outperform", "buyback", "strong", "wins"]
_BEARISH_KW = ["miss", "misses", "plunge", "falls", "fall", "cut", "cuts", "downgrade",
               "lowers", "warning", "warns", "lawsuit", "recall", "slump", "slumps",
               "weak", "layoff", "layoffs", "probe", "halt", "tumble", "tumbles", "drop"]


def headline_sentiment(text: str) -> float:
    """Rule-based sentiment score in [-1, 1] from bullish/bearish keyword counts."""
    t = str(text or "").lower()
    bull = sum(1 for k in _BULLISH_KW if k in t)
    bear = sum(1 for k in _BEARISH_KW if k in t)
    if bull == bear:
        return 0.0
    return max(-1.0, min(1.0, (bull - bear) / float(bull + bear)))


def sentiment_label(score: float) -> str:
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def _iso(epoch_s) -> str:
    try:
        return datetime.fromtimestamp(float(epoch_s), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def normalize_news(item: dict, symbol: str) -> dict:
    """Raw Finnhub company-news item -> normalized alert dict with a rule-based score."""
    headline = item.get("headline", "")
    score = headline_sentiment(f"{headline} {item.get('summary', '')}")
    ts = float(item.get("datetime") or 0)
    return {
        "source": "finnhub", "market": "us",
        "time": _iso(ts), "ts": ts,
        "symbol": str(symbol).upper(), "title": headline, "headline": headline,
        "type": "news", "sentiment_score": round(score, 3),
        "sentiment": sentiment_label(score), "important": abs(score) >= 0.5,
        "publisher": item.get("source", ""), "url": item.get("url", ""),
    }


def aggregate_sentiment(items: Sequence[dict]) -> float:
    """Mean rule-based sentiment_score across news items (0.0 if none)."""
    scores = [float(x.get("sentiment_score", 0.0)) for x in items if "sentiment_score" in x]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


class FinnhubSource:
    """Fetch + tag US company news (and optional company sentiment). Key never logged."""

    def __init__(self, api_key: str, session=None):
        self.api_key = api_key
        self._session = session

    def _get(self, url: str, params: dict):
        session = self._session
        if session is None:
            import requests  # lazy
            session = requests
        resp = session.get(url, params={**params, "token": self.api_key}, timeout=10)
        try:
            return resp.json()
        except Exception:
            return None

    def company_news(self, symbol: str, from_: str, to: str, limit: int = 20) -> List[dict]:
        """Recent headlines for ``symbol`` over [from_, to] (YYYY-MM-DD), newest first."""
        data = self._get(COMPANY_NEWS_URL, {"symbol": str(symbol).upper(), "from": from_, "to": to})
        rows = data if isinstance(data, list) else []
        out = [normalize_news(x, symbol) for x in rows if isinstance(x, dict)]
        out.sort(key=lambda a: a["ts"], reverse=True)
        return out[:limit]

    def news_sentiment(self, symbol: str) -> Optional[float]:
        """Finnhub company news-sentiment score mapped to [-1, 1], or None if absent
        (the endpoint is premium on some plans)."""
        data = self._get(NEWS_SENTIMENT_URL, {"symbol": str(symbol).upper()})
        if not isinstance(data, dict):
            return None
        cns = data.get("companyNewsScore")
        if cns is None:
            sent = data.get("sentiment") or {}
            bp, brp = sent.get("bullishPercent"), sent.get("bearishPercent")
            if bp is None and brp is None:
                return None
            return round(float(bp or 0.0) - float(brp or 0.0), 4)
        return round((float(cns) - 0.5) * 2.0, 4)         # 0..1 (0.5 neutral) -> -1..1

    def market_news(self, symbols: Sequence[str], from_: str, to: str,
                    per_symbol: int = 8) -> List[dict]:
        """Company news across a set of linked US symbols, newest first."""
        out: List[dict] = []
        for s in symbols:
            try:
                out += self.company_news(s, from_, to, limit=per_symbol)
            except Exception:
                continue
        out.sort(key=lambda a: a["ts"], reverse=True)
        return out
